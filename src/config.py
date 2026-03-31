"""Centralized configuration for fragment-completion experiments.

Single source of truth for encoder metadata, image types, colors, and result paths.
"""

from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Encoder metadata
# ---------------------------------------------------------------------------

EncoderMeta = namedtuple("EncoderMeta", ["display", "dir_name"])

# Registry key → (display name, directory name)
ENCODER_META: dict[str, EncoderMeta] = {
    "clip":    EncoderMeta("CLIP",           "clip"),
    "dino":    EncoderMeta("DINO-v1",        "dino_v1"),
    "dinov2":  EncoderMeta("DINOv2",         "dinov2"),
    "mae":     EncoderMeta("MAE",            "mae"),
    "mae_ft":  EncoderMeta("MAE-FT",         "mae_ft"),
    "ijepa":   EncoderMeta("I-JEPA",         "i_jepa"),
    "vit_sup": EncoderMeta("ViT-supervised", "vit_supervised"),
    "siglip":  EncoderMeta("SigLIP",         "siglip"),
    "simclr":  EncoderMeta("SimCLR",         "simclr"),
    "resnet":  EncoderMeta("ResNet-50",      "resnet_50"),
    "nepa":    EncoderMeta("NEPA",           "nepa"),
    "llava":   EncoderMeta("LLaVA",          "llava"),
    "qwen2vl": EncoderMeta("Qwen2-VL",      "qwen2vl"),
}

# Canonical plot ordering (subset used in most experiments)
ENCODER_DISPLAY_ORDER = ["CLIP", "DINO-v1", "DINOv2", "MAE", "I-JEPA", "ViT-supervised", "NEPA"]

# ---------------------------------------------------------------------------
# Reverse lookups
# ---------------------------------------------------------------------------

_DISPLAY_TO_REGISTRY = {m.display: k for k, m in ENCODER_META.items()}
_DISPLAY_TO_DIR = {m.display: m.dir_name for m in ENCODER_META.values()}
_DIR_TO_DISPLAY = {m.dir_name: m.display for m in ENCODER_META.values()}


def display_to_registry(display: str) -> str:
    """'CLIP' → 'clip', 'DINO-v1' → 'dino'."""
    return _DISPLAY_TO_REGISTRY[display]


def display_to_dir(display: str) -> str:
    """'CLIP' → 'clip', 'DINO-v1' → 'dino_v1'."""
    return _DISPLAY_TO_DIR[display]


def dir_to_display(dir_name: str) -> str:
    """'dino_v1' → 'DINO-v1', 'clip' → 'CLIP'."""
    return _DIR_TO_DISPLAY[dir_name]


def registry_to_dir(registry_key: str) -> str:
    """'dino' → 'dino_v1', 'ijepa' → 'i_jepa'."""
    return ENCODER_META[registry_key].dir_name


def registry_to_display(registry_key: str) -> str:
    """'dino' → 'DINO-v1', 'clip' → 'CLIP'."""
    return ENCODER_META[registry_key].display


# ---------------------------------------------------------------------------
# Image types
# ---------------------------------------------------------------------------

IMAGE_TYPES = ["original", "gray", "lined"]

IMAGE_TYPE_COLORS = {
    "original": "#1f77b4",
    "gray":     "#555555",
    "lined":    "#b0b0b0",
}

IMAGE_TYPE_ALPHA = {
    "original": 1.0,
    "gray":     0.7,
    "lined":    0.4,
}

# ---------------------------------------------------------------------------
# Encoder colors (tab10, keyed by display name)
# ---------------------------------------------------------------------------

_TAB10 = plt.cm.tab10.colors
ENCODER_COLORS = {enc: _TAB10[i] for i, enc in enumerate(ENCODER_DISPLAY_ORDER)}

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

PLOT_STYLE = {
    "linewidth": 2,
    "std_alpha": 0.2,
    "tick_width": 2,
    "tick_labelsize": 14,
    "label_fontsize": 16,
    "legend_fontsize": 14,
    "legend_loc": "outside lower center",  # legend below the plot
    "subplot_title_fontsize": 18,
    "suptitle_fontsize": 20,
    "marker": "o",
    "markersize": 6,
    "dpi": 150,
    "subplot_size": (6.4, 4.8),  # (w, h) desired axes area per subplot
}

# ---------------------------------------------------------------------------
# Result paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("results")


def results_for_image_type(img_type: str, root: Path = RESULTS_ROOT) -> Path:
    return root / "image_types" / img_type


def results_for_encoder(name: str, root: Path = RESULTS_ROOT) -> Path:
    """Accept display name, registry key, or dir name."""
    if name in _DISPLAY_TO_DIR:
        dir_name = _DISPLAY_TO_DIR[name]
    elif name in ENCODER_META:
        dir_name = ENCODER_META[name].dir_name
    else:
        dir_name = name  # assume it's already a dir name
    return root / "encoders" / dir_name


def results_all_encoders(root: Path = RESULTS_ROOT) -> Path:
    return root / "all_encoders"


def results_visualizations(root: Path = RESULTS_ROOT) -> Path:
    return root / "visualizations"
