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

from models.encoder import BaseEncoder

from .config import (
    PLOT_STYLE as PS,
    ENCODER_COLORS,
    ENCODER_DISPLAY_ORDER,
    IMAGE_TYPES,
    IMAGE_TYPE_ALPHA,
)
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
def extract_patch_features(
    encoder: BaseEncoder,
    image_pil: Image.Image,
    transform=None,
) -> torch.Tensor:
    """Extract patch-level features from a single image.

    Args:
        encoder: Vision encoder.
        image_pil: PIL image (should already be at encoder's target resolution
                   if passing a normalize-only *transform*).
        transform: Optional transform override. Pass a normalize-only transform
                   (from ``get_normalize_transform``) for pre-prepared images.
                   If None, uses the encoder's full transform (with resize/crop).

    Returns:
        [N_patches, D] tensor (excluding CLS/prefix tokens).
    """
    if transform is None:
        enc_name = encoder.name
        if enc_name not in _transform_cache:
            from models.processor import to_transform
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

def make_fig(
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kw,
) -> tuple:
    """Create figure + axes where each axes area is PS['subplot_size'] inches.

    Uses a two-pass approach: create a throwaway figure with constrained_layout,
    measure the actual axes size, then rescale the figure so each axes hits the
    target size exactly.
    """
    sw, sh = PS["subplot_size"]
    # Initial guess — generous so constrained_layout has room
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(sw * ncols + (ncols - 1) * 0.2 + 1,
                 sh * nrows + (nrows - 1) * 0.2 + 2),
        constrained_layout=True,
        **subplot_kw,
    )
    # Force layout computation
    fig.canvas.draw()
    # Measure first axes
    ax0 = axes.flat[0] if hasattr(axes, "flat") else axes
    pos = ax0.get_position()
    actual_w = pos.width * fig.get_figwidth()
    actual_h = pos.height * fig.get_figheight()
    # Scale figure so axes hit target size
    fig.set_size_inches(
        fig.get_figwidth() * (sw / actual_w),
        fig.get_figheight() * (sh / actual_h),
    )
    return fig, axes


def _visibility_x() -> list[float]:
    """X-axis values: visibility ratio for each level."""
    return [get_visibility_ratio(L) for L in get_mask_levels()]


def _apply_style(ax, title: str, ylabel: str) -> None:
    """Apply unified plot style to an axis."""
    ax.set_xlabel("Visibility", fontsize=PS["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=PS["label_fontsize"])
    ax.set_title(title, fontsize=PS["subplot_title_fontsize"], fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=PS["tick_labelsize"], width=PS["tick_width"])
    for spine in ax.spines.values():
        spine.set_linewidth(PS["tick_width"])
    ax.grid(True, alpha=0.3)


def _plot_line(ax, x, values, levels, enc_name, color=None):
    """Plot a single encoder line with optional std band."""
    v0 = values[levels[0]]
    has_std = isinstance(v0, dict) and "std" in v0
    if has_std:
        y = np.array([values[L]["mean"] for L in levels])
        std = np.array([values[L]["std"] for L in levels])
        line, = ax.plot(x, y, marker=PS["marker"], markersize=PS["markersize"],
                        linewidth=PS["linewidth"], label=enc_name, color=color)
        ax.fill_between(x, y - std, y + std, alpha=PS["std_alpha"],
                        color=line.get_color())
    else:
        y = [values[L] for L in levels]
        ax.plot(x, y, marker=PS["marker"], markersize=PS["markersize"],
                linewidth=PS["linewidth"], label=enc_name, color=color)


def plot_metric_vs_masking(
    results: dict[str, dict],
    ylabel: str,
    title: str,
    save_path: Path,
    colors: dict[str, str] | None = None,
) -> None:
    """Line plot with one line per encoder, x = visibility ratio."""
    fig, ax = make_fig(1, 1)
    x = np.array(_visibility_x())
    levels = get_mask_levels()

    for enc_name, values in results.items():
        c = colors.get(enc_name) if colors else None
        _plot_line(ax, x, values, levels, enc_name, color=c)

    _apply_style(ax, title, ylabel)
    fig.legend(*ax.get_legend_handles_labels(),
               loc="outside lower center", ncol=len(results),
               fontsize=PS["legend_fontsize"], frameon=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_completion_summary(
    gestalt: dict[str, dict[int, float]] | None,
    mnemonic: dict[str, dict] | None,
    semantic: dict[str, dict] | None,
    save_path: Path,
    colors: dict[str, str] | None = None,
) -> None:
    """1xN subplot summary figure with one shared legend below."""
    panels = []
    if gestalt:
        panels.append(("Gestalt (IoU)", gestalt, "IoU"))
    if mnemonic:
        sim_data = {k: v["similarity"] for k, v in mnemonic.items() if "similarity" in v}
        r1_data = {k: v["retrieval_r1"] for k, v in mnemonic.items() if "retrieval_r1" in v}
        if sim_data:
            panels.append(("Mnemonic (Similarity)", sim_data, "Cosine Sim"))
        if r1_data:
            panels.append(("Mnemonic (R@1)", r1_data, "R@1"))
    if semantic:
        proto_data = {k: v["prototype_acc"] for k, v in semantic.items()}
        panels.append(("Semantic (Prototype)", proto_data, "Accuracy"))
        zs_data = {k: v["zeroshot_acc"] for k, v in semantic.items() if "zeroshot_acc" in v}
        if zs_data:
            panels.append(("Semantic (Zero-shot)", zs_data, "Accuracy"))

    if not panels:
        return

    n_panels = len(panels)
    fig, axes = make_fig(1, n_panels)
    if n_panels == 1:
        axes = [axes]

    x = np.array(_visibility_x())
    levels = get_mask_levels()
    for ax, (title, data, ylabel) in zip(axes, panels):
        for enc_name, values in data.items():
            c = colors.get(enc_name) if colors else None
            _plot_line(ax, x, values, levels, enc_name, color=c)
        _apply_style(ax, title, ylabel)

    # One shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncol=len(labels), fontsize=PS["legend_fontsize"],
               frameon=True)
    fig.suptitle("Fragment Completion Summary", fontsize=PS["suptitle_fontsize"],
                 fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_all_encoders_summary(
    data: dict[str, dict[str, dict]],
    save_dir: Path,
) -> None:
    """Combined all-encoders plot: color=encoder, alpha=image_type.

    Args:
        data: {encoder_display: {img_type: {metric: {level: {mean, std}}}}}
        save_dir: directory to save plots into
    """
    import matplotlib.lines as mlines

    levels = get_mask_levels()
    x = np.array(_visibility_x())

    encoders = [e for e in ENCODER_DISPLAY_ORDER if e in data]
    if not encoders:
        encoders = sorted(data.keys())

    # Build panel items: list of (enc, img_type, values)
    def _items(metric):
        items = []
        for enc in encoders:
            for img_type in IMAGE_TYPES:
                if img_type in data.get(enc, {}) and metric in data[enc][img_type]:
                    items.append((enc, img_type, data[enc][img_type][metric]))
        return items

    panels = [
        ("Gestalt (IoU)",          "IoU",        _items("gestalt_iou")),
        ("Mnemonic (Similarity)",  "Cosine Sim", _items("mnemonic_similarity")),
        ("Mnemonic (Retrieval@1)", "Accuracy",   _items("mnemonic_retrieval")),
        ("Semantic (Prototype)",   "Accuracy",   _items("semantic_prototype")),
    ]
    zs = _items("semantic_zeroshot")
    if zs:
        panels.append(("Semantic (Zero-shot)", "Accuracy", zs))

    save_dir.mkdir(parents=True, exist_ok=True)

    def _plot_panel(ax, items, title, ylabel):
        for enc, img_type, values in items:
            color = ENCODER_COLORS.get(enc, (0, 0, 0))
            alpha = IMAGE_TYPE_ALPHA.get(img_type, 1.0)
            v0 = values[levels[0]]
            has_std = isinstance(v0, dict) and "std" in v0
            if has_std:
                y = np.array([values[L]["mean"] for L in levels])
                std = np.array([values[L]["std"] for L in levels])
                ax.plot(x, y, marker=PS["marker"], markersize=PS["markersize"],
                        linewidth=PS["linewidth"], color=(*color[:3], alpha))
                ax.fill_between(x, y - std, y + std,
                                color=(*color[:3], alpha * PS["std_alpha"]))
            else:
                y = [values[L] for L in levels]
                ax.plot(x, y, marker=PS["marker"], markersize=PS["markersize"],
                        linewidth=PS["linewidth"], color=(*color[:3], alpha))
        _apply_style(ax, title, ylabel)

    def _make_legend():
        enc_handles = [
            mlines.Line2D([], [], color=ENCODER_COLORS.get(e, "black"),
                           marker=PS["marker"], markersize=PS["markersize"],
                           linewidth=PS["linewidth"], label=e)
            for e in encoders
        ]
        type_handles = [
            mlines.Line2D([], [], color=(0.3, 0.3, 0.3, IMAGE_TYPE_ALPHA[t]),
                           marker=PS["marker"], markersize=PS["markersize"],
                           linewidth=PS["linewidth"], label=t)
            for t in IMAGE_TYPES
        ]
        return enc_handles, type_handles

    # --- Combined N-subplot figure ---
    n = len(panels)
    fig, axes = make_fig(1, n)
    if n == 1:
        axes = [axes]
    for ax, (title, ylabel, items) in zip(axes, panels):
        _plot_panel(ax, items, title, ylabel)

    enc_h, type_h = _make_legend()
    leg1 = fig.legend(handles=enc_h, loc="outside lower center",
                      ncol=len(enc_h), fontsize=PS["legend_fontsize"],
                      frameon=False)
    fig.legend(handles=type_h, loc="outside lower center",
               ncol=len(type_h), fontsize=PS["legend_fontsize"],
               frameon=False)
    fig.add_artist(leg1)
    fig.suptitle("Fragment Completion — All Encoders",
                 fontsize=PS["suptitle_fontsize"], fontweight="bold")
    save = save_dir / "completion_summary.png"
    fig.savefig(save, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save}")

    # --- Individual metric plots ---
    fname_map = ["gestalt_iou", "mnemonic_similarity", "mnemonic_retrieval",
                 "semantic_prototype", "semantic_zeroshot"]
    for (title, ylabel, items), fname in zip(panels, fname_map):
        fig, ax = make_fig(1, 1)
        _plot_panel(ax, items, title, ylabel)
        enc_h, type_h = _make_legend()
        leg1 = fig.legend(handles=enc_h, loc="outside lower center",
                          ncol=len(enc_h), fontsize=PS["legend_fontsize"],
                          frameon=False)
        fig.legend(handles=type_h, loc="outside lower center",
                   ncol=len(type_h), fontsize=PS["legend_fontsize"],
                   frameon=False)
        fig.add_artist(leg1)
        sp = save_dir / f"{fname}.png"
        fig.savefig(sp, dpi=PS["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {sp}")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_json(data: dict, path: Path) -> None:
    """Save dict as JSON (compact utility for experiments)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: round(float(o), 6)
                  if isinstance(o, (np.floating, float)) else int(o)
                  if isinstance(o, (np.integer,)) else o)


def compute_retrieval_metrics(
    query: torch.Tensor,
    gallery: torch.Tensor,
    gt_ids: torch.Tensor,
) -> dict[str, float]:
    """Compute R@1, R@5, MRR for query→gallery retrieval.

    Args:
        query: [N, D] L2-normalised query embeddings.
        gallery: [M, D] L2-normalised gallery embeddings.
        gt_ids: [N] ground-truth gallery index per query.

    Returns:
        Dict with recall_at_1, recall_at_5, mrr.
    """
    sims = query @ gallery.T  # [N, M]
    ranks = sims.argsort(dim=1, descending=True)
    n = query.shape[0]
    r1 = r5 = mrr = 0.0
    for i in range(n):
        rank_list = ranks[i].tolist()
        gt = int(gt_ids[i])
        pos = rank_list.index(gt) if gt in rank_list else len(rank_list)
        if pos < 1:
            r1 += 1
        if pos < 5:
            r5 += 1
        mrr += 1.0 / (pos + 1)
    return {
        "recall_at_1": r1 / n,
        "recall_at_5": r5 / n,
        "mrr": mrr / n,
    }


def compute_category_accuracy(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    gt_cat_ids: list[int],
) -> float:
    """Top-1 accuracy: query → nearest prototype matches gt category.

    Args:
        query: [N, D] query embeddings.
        prototypes: [C, D] prototype embeddings (one per category).
        gt_cat_ids: [N] ground-truth category index per query.

    Returns:
        Accuracy in [0, 1].
    """
    sims = query @ prototypes.T  # [N, C]
    preds = sims.argmax(dim=1).tolist()
    correct = sum(p == g for p, g in zip(preds, gt_cat_ids))
    return correct / len(gt_cat_ids)


def compute_exemplar_accuracy(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_cats: list[int],
    gallery_cats: list[int],
    k: int = 10,
) -> float:
    """k-NN exemplar accuracy: majority vote among k nearest neighbours.

    Args:
        query: [N, D] query embeddings.
        gallery: [M, D] gallery embeddings.
        query_cats: [N] category id per query.
        gallery_cats: [M] category id per gallery item.
        k: Number of neighbours.

    Returns:
        Accuracy in [0, 1].
    """
    sims = query @ gallery.T  # [N, M]
    _, topk_idx = sims.topk(k + 1, dim=1)  # +1 to skip self-match
    n = query.shape[0]
    correct = 0
    for i in range(n):
        neighbours = [gallery_cats[j] for j in topk_idx[i].tolist() if j != i][:k]
        if not neighbours:
            continue
        # Majority vote
        from collections import Counter
        pred = Counter(neighbours).most_common(1)[0][0]
        if pred == query_cats[i]:
            correct += 1
    return correct / n


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
