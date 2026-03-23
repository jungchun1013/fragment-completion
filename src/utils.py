"""Plotting and I/O utilities for completion experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .masking import get_mask_levels, get_visibility_ratio

"""Progressive fragment-completion segmentation via patch features.

Algorithm (mirrors fragment_v2 logic at the encoder patch level):
1. Extract ALL patch features from the original image
2. Identify "object patches" — those overlapping non-white pixels
3. At each level L, reveal P = 0.7^(8-L) of object patches (exponential schedule)
4. Run 2-means on only the revealed patches
5. Assign cluster labels, map back to full grid, compute IoU

Output: a plot of mean IoU vs fragmentation level (1–8).
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from wrappers.encoder import BaseEncoder

# --------------------------------------------------------------------------- #
# Ground-truth mask from original image
# --------------------------------------------------------------------------- #

def _get_foreground_mask(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """Binary mask: 1 = foreground (non-white), 0 = background."""
    bg = np.all(image >= threshold, axis=-1)  # [H, W]
    return (~bg).astype(np.float32)


# --------------------------------------------------------------------------- #
# Patch feature extraction — multi-encoder support
# --------------------------------------------------------------------------- #

_transform_cache: dict[str, object] = {}


def _get_patch_grid_size(encoder: BaseEncoder) -> tuple[int, int]:
    """Return (gh, gw) — the spatial grid of patches for the encoder."""
    model = encoder.model
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        return tuple(model.patch_embed.grid_size)
    if hasattr(model, "config") and hasattr(model.config, "image_size"):
        ps = model.config.patch_size
        img_sz = model.config.image_size
        g = img_sz // ps
        return (g, g)
    # CLIP visual
    if hasattr(model, "visual"):
        v = model.visual
        if hasattr(v, "conv1"):
            # infer from conv1 kernel
            ps = v.conv1.kernel_size[0]
            # need input size — assume 224
            g = 224 // ps
            return (g, g)
    return (14, 14)  # fallback


@torch.no_grad()
def _extract_patch_features(encoder: BaseEncoder, image_pil: Image.Image) -> torch.Tensor:
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

    # HuggingFace I-JEPA / NEPA (no CLS token — all tokens are patches)
    if name in ("I-JEPA", "NEPA"):
        output = model(pixel_values=img_t)
        return output.last_hidden_state[0].cpu()

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
    # Must come before DINO-v1 check: DINOv2 (timm) also has get_intermediate_layers
    # but that method already strips the CLS token, causing off-by-one.
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


# --------------------------------------------------------------------------- #
# Identify object patches in the encoder's patch grid
# --------------------------------------------------------------------------- #

def _get_object_patch_indices(
    image: np.ndarray, gh: int, gw: int, threshold: int = 250,
) -> list[int]:
    """Return flat indices of encoder patches that overlap with non-white pixels.

    Maps the original image (H, W) onto the encoder's (gh, gw) patch grid.
    A patch is an "object patch" if any pixel in its region is non-white.
    """
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


# --------------------------------------------------------------------------- #
# Progressive masking at patch level + 2-means + IoU
# --------------------------------------------------------------------------- #

def _progressive_segment_iou(
    patch_feats: torch.Tensor,
    gh: int, gw: int,
    object_indices: list[int],
    gt_mask: np.ndarray,
    seed: int = 42,
) -> list[float]:
    """Run progressive masking on patch features and compute IoU at each level.

    At each level L (1-8):
      - Reveal P = 0.7^(8-L) fraction of object patches
      - 2-means on revealed patches only
      - Map labels back to full image, compute IoU

    Returns list of 8 IoU values.
    """
    H, W = gt_mask.shape
    N = gh * gw
    feats_np = patch_feats.float().numpy()

    # Shuffle object indices with fixed seed
    rng = random.Random(seed)
    shuffled_obj = list(object_indices)
    rng.shuffle(shuffled_obj)

    ious = []
    for L in range(1, 9):
        P = 0.7 ** (8.0 - L)
        num_reveal = max(1, int(P * len(shuffled_obj)))
        revealed = shuffled_obj[:num_reveal]

        if len(revealed) < 2:
            # Not enough patches for 2-means
            ious.append(0.0)
            continue

        # Extract features of revealed patches
        revealed_feats = feats_np[revealed]

        # 2-means clustering on revealed patches
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        revealed_labels = kmeans.fit_predict(revealed_feats)  # [num_reveal]

        # Build full label grid: -1 = background (unrevealed/white)
        label_flat = np.full(N, -1, dtype=np.int32)
        for idx, lab in zip(revealed, revealed_labels):
            label_flat[idx] = lab
        label_grid = label_flat.reshape(gh, gw)

        # Upsample to original resolution
        # Use float for interpolation: -1 stays as bg
        label_img = np.array(
            Image.fromarray(label_grid.astype(np.int32).astype(np.int16))
                 .resize((W, H), Image.NEAREST)
        ).astype(np.int32)

        # Try both cluster assignments for foreground
        iou_best = 0.0
        for fg_label in [0, 1]:
            pred_fg = (label_img == fg_label).astype(np.float32)
            intersection = (pred_fg * gt_mask).sum()
            union = ((pred_fg + gt_mask) > 0).sum()
            iou = intersection / (union + 1e-8)
            iou_best = max(iou_best, iou)

        ious.append(float(iou_best))

    return ious



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
    """Line plot with one line per encoder, x = visibility ratio.

    Args:
        results: {encoder_name: {level: float_or_dict}}
                 Values can be plain floats or {"mean": ..., "std": ...}.
        colors: optional {label: color_string} mapping.
    """
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


def save_results(results: dict, path: Path) -> None:
    """Save results dict as JSON."""
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
