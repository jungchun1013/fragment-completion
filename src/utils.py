"""Shared utilities for fragment-completion experiments.

Contains:
  - Feature extraction helpers (embed_pil, extract_patch_features, encoder geometry)
  - Ground-truth mask and patch index helpers
  - Progressive segmentation (patch-level 2-means + IoU)
  - Plotting and I/O
  - JSON helpers (fix_json_keys, extract_val, extract_std)
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from wrappers.encoder import BaseEncoder

from .masking import get_mask_levels, get_visibility_ratio

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_pil(encoder: BaseEncoder, pil, transform) -> torch.Tensor:
    """Extract [D] feature vector from a PIL image."""
    img_t = transform(pil).unsqueeze(0).to(encoder.device)
    feat = encoder.extract_features(img_t)  # [1, D]
    return feat[0].cpu()


# ---------------------------------------------------------------------------
# Encoder geometry
# ---------------------------------------------------------------------------

def get_encoder_geometry(encoder: BaseEncoder) -> tuple[int, int]:
    """Return (image_size, patch_size) for the encoder's native resolution."""
    model = encoder.model

    # timm models (DINOv2, SigLIP, ViT-sup, ResNet, MAE-FT)
    if hasattr(model, "patch_embed"):
        pe = model.patch_embed
        if hasattr(pe, "img_size"):
            img_sz = pe.img_size
            img_size = img_sz[0] if isinstance(img_sz, (tuple, list)) else img_sz
        else:
            img_size = 224
        if hasattr(pe, "patch_size"):
            ps = pe.patch_size
            patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        else:
            patch_size = pe.proj.kernel_size[0]
        return img_size, patch_size

    # HuggingFace models (MAE, I-JEPA, NEPA)
    if hasattr(model, "config"):
        cfg = model.config
        img_size = getattr(cfg, "image_size", 224)
        patch_size = getattr(cfg, "patch_size", 16)
        return img_size, patch_size

    # OpenCLIP CLIP
    if hasattr(model, "visual"):
        v = model.visual
        if hasattr(v, "trunk") and hasattr(v.trunk, "patch_embed"):
            pe = v.trunk.patch_embed
            img_sz = pe.img_size
            img_size = img_sz[0] if isinstance(img_sz, (tuple, list)) else img_sz
            ps = pe.patch_size
            patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
            return img_size, patch_size
        if hasattr(v, "conv1"):
            patch_size = v.conv1.kernel_size[0]
            return 224, patch_size

    return 224, 16


def get_patch_grid_size(encoder: BaseEncoder) -> tuple[int, int]:
    """Return (gh, gw) — the spatial grid of patches for the encoder."""
    img_size, patch_size = get_encoder_geometry(encoder)
    g = img_size // patch_size
    return (g, g)


# ---------------------------------------------------------------------------
# Patch feature extraction — multi-encoder support
# ---------------------------------------------------------------------------

_transform_cache: dict[str, object] = {}


@torch.no_grad()
def extract_patch_features(encoder: BaseEncoder, image_pil: Image.Image) -> torch.Tensor:
    """Extract patch-level features from a single image.

    Returns [N_patches, D] tensor (excluding CLS/prefix tokens).
    """
    enc_name = encoder.name
    if enc_name not in _transform_cache:
        from wrappers.processor import to_transform
        _transform_cache[enc_name] = to_transform(encoder.processor)
    transform = _transform_cache[enc_name]
    img_t = transform(image_pil).unsqueeze(0).to(encoder.device)

    model = encoder.model
    name = encoder.name

    # HuggingFace MAE
    if name == "MAE":
        output = model(pixel_values=img_t)
        return output.last_hidden_state[:, 1:][0].cpu()

    # HuggingFace I-JEPA (no CLS token — all tokens are patches)
    if name == "I-JEPA":
        output = model(pixel_values=img_t)
        return output.last_hidden_state[0].cpu()

    # NEPA (has CLS token at position 0 — skip it)
    if name == "NEPA":
        output = model(pixel_values=img_t)
        return output.last_hidden_state[0, 1:].cpu()

    # OpenCLIP CLIP
    if name == "CLIP":
        visual = model.visual
        if hasattr(visual, "trunk") and hasattr(visual.trunk, "forward_features"):
            features = visual.trunk.forward_features(img_t)
            n_prefix = getattr(visual.trunk, "num_prefix_tokens", 1)
            return features[0, n_prefix:].cpu()
        if hasattr(visual, "transformer"):
            x = visual.conv1(img_t)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls_tok = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tok, x], dim=1)
            x = x + visual.positional_embedding.unsqueeze(0)
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)
            return x[0, 1:].cpu()
        raise RuntimeError("Could not extract CLIP patch features")

    # timm-style ViT (DINOv2, SigLIP, ViT-sup, MAE-FT, etc.)
    if hasattr(model, "forward_features"):
        features = model.forward_features(img_t)
        if hasattr(model, "num_prefix_tokens"):
            n_prefix = model.num_prefix_tokens
        elif hasattr(model, "cls_token") and model.cls_token is not None:
            n_prefix = 1
        else:
            n_prefix = 0
        return features[0, n_prefix:].cpu()

    # DINO-v1 (torch.hub) — only reached if forward_features is absent
    if hasattr(model, "get_intermediate_layers"):
        outputs = model.get_intermediate_layers(img_t, n=1)
        return outputs[0][0, 1:].cpu()

    raise RuntimeError(f"Patch feature extraction not supported for {encoder.name}")


# ---------------------------------------------------------------------------
# Ground-truth mask and patch indices
# ---------------------------------------------------------------------------

def get_foreground_mask(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """Binary mask: 1 = foreground (non-white), 0 = background."""
    bg = np.all(image >= threshold, axis=-1)  # [H, W]
    return (~bg).astype(np.float32)


def get_object_patch_indices(
    image: np.ndarray, gh: int, gw: int, threshold: int = 250,
) -> list[int]:
    """Return flat indices of encoder patches that overlap with non-white pixels."""
    H, W = image.shape[:2]
    patch_h = H / gh
    patch_w = W / gw

    object_indices = []
    for idx in range(gh * gw):
        i, j = divmod(idx, gw)
        r0, r1 = int(i * patch_h), int((i + 1) * patch_h)
        c0, c1 = int(j * patch_w), int((j + 1) * patch_w)
        region = image[r0:r1, c0:c1]
        if np.any(region < threshold):
            object_indices.append(idx)

    return object_indices


# ---------------------------------------------------------------------------
# Progressive masking at patch level + 2-means + IoU
# ---------------------------------------------------------------------------

def progressive_segment_iou(
    patch_feats: torch.Tensor,
    gh: int, gw: int,
    object_indices: list[int],
    gt_mask: np.ndarray,
    seed: int = 42,
) -> list[float]:
    """Run progressive masking on patch features and compute IoU at each level.

    Returns list of 8 IoU values.
    """
    H, W = gt_mask.shape
    N = gh * gw
    feats_np = patch_feats.float().numpy()

    rng = random.Random(seed)
    shuffled_obj = list(object_indices)
    rng.shuffle(shuffled_obj)

    ious = []
    for L in range(1, 9):
        P = 0.7 ** (8.0 - L)
        num_reveal = max(1, int(P * len(shuffled_obj)))
        revealed = shuffled_obj[:num_reveal]

        if len(revealed) < 2:
            ious.append(0.0)
            continue

        revealed_feats = feats_np[revealed]
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        revealed_labels = kmeans.fit_predict(revealed_feats)

        label_flat = np.full(N, -1, dtype=np.int32)
        for idx, lab in zip(revealed, revealed_labels):
            label_flat[idx] = lab
        label_grid = label_flat.reshape(gh, gw)

        label_img = np.array(
            Image.fromarray(label_grid.astype(np.int32).astype(np.int16))
                 .resize((W, H), Image.NEAREST)
        ).astype(np.int32)

        iou_best = 0.0
        for fg_label in [0, 1]:
            pred_fg = (label_img == fg_label).astype(np.float32)
            intersection = (pred_fg * gt_mask).sum()
            union = ((pred_fg + gt_mask) > 0).sum()
            iou = intersection / (union + 1e-8)
            iou_best = max(iou_best, iou)

        ious.append(float(iou_best))

    return ious


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def fix_json_keys(d):
    """Recursively convert string keys to int where possible (for JSON roundtrip)."""
    if not isinstance(d, dict):
        return d
    try:
        return {int(k): v for k, v in d.items()}
    except (ValueError, TypeError):
        return {k: fix_json_keys(v) for k, v in d.items()}


def extract_val(v):
    """Extract mean from a {mean, std} dict or return the value directly."""
    return v["mean"] if isinstance(v, dict) and "mean" in v else v


def extract_std(v):
    """Extract std from a {mean, std} dict or return 0."""
    return v["std"] if isinstance(v, dict) and "std" in v else 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _visibility_x() -> list[float]:
    """X-axis values: visibility ratio for each level."""
    return [get_visibility_ratio(L) for L in get_mask_levels()]


def plot_metric_vs_masking(
    results: dict[str, dict],
    ylabel: str,
    title: str,
    save_path: Path,
    colors: dict[str, str] | None = None,
) -> None:
    """Line plot with one line per encoder, x = visibility ratio."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array(_visibility_x())
    levels = get_mask_levels()

    for enc_name, values in results.items():
        c = colors.get(enc_name) if colors else None
        v0 = values[levels[0]]
        has_std = isinstance(v0, dict) and "std" in v0
        if has_std:
            y = np.array([values[L]["mean"] for L in levels])
            std = np.array([values[L]["std"] for L in levels])
            line, = ax.plot(x, y, marker="o", label=enc_name, color=c)
            ax.fill_between(x, y - std, y + std, alpha=0.15, color=line.get_color())
        else:
            y = [values[L] for L in levels]
            ax.plot(x, y, marker="o", label=enc_name, color=c)

    ax.set_xlabel("Visibility (fraction of FG patches shown)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_completion_summary(
    gestalt: dict[str, dict[int, float]] | None,
    mnemonic: dict[str, dict] | None,
    semantic: dict[str, dict] | None,
    save_path: Path,
    colors: dict[str, str] | None = None,
) -> None:
    """1xN subplot summary figure."""
    panels = []
    if gestalt:
        panels.append(("Gestalt (IoU)", gestalt, "IoU"))
    if mnemonic:
        sim_data = {k: v["similarity"] for k, v in mnemonic.items()}
        ret_data = {k: v["retrieval"] for k, v in mnemonic.items()}
        panels.append(("Mnemonic (Similarity)", sim_data, "Cosine Sim"))
        panels.append(("Mnemonic (Retrieval@1)", ret_data, "Accuracy"))
    if semantic:
        proto_data = {k: v["prototype_acc"] for k, v in semantic.items()}
        panels.append(("Semantic (Prototype)", proto_data, "Accuracy"))
        zs_data = {k: v["zeroshot_acc"] for k, v in semantic.items() if "zeroshot_acc" in v}
        if zs_data:
            panels.append(("Semantic (Zero-shot)", zs_data, "Accuracy"))

    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    x = np.array(_visibility_x())
    levels = get_mask_levels()
    for ax, (title, data, ylabel) in zip(axes, panels):
        for enc_name, values in data.items():
            c = colors.get(enc_name) if colors else None
            v0 = values[levels[0]]
            has_std = isinstance(v0, dict) and "std" in v0
            if has_std:
                y = np.array([values[L]["mean"] for L in levels])
                std = np.array([values[L]["std"] for L in levels])
                line, = ax.plot(x, y, marker="o", label=enc_name, color=c)
                ax.fill_between(x, y - std, y + std, alpha=0.15, color=line.get_color())
            else:
                y = [values[L] for L in levels]
                ax.plot(x, y, marker="o", label=enc_name, color=c)
        ax.set_xlabel("Visibility")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, path: Path) -> None:
    """Save results dict as JSON with type conversion."""
    def _convert(obj):
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, float)):
            return round(float(obj), 6)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"  Saved: {path}")
