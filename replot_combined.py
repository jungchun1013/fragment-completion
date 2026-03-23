"""Combined replot: 5 subplots + individual plots, all encoders.

For each encoder:
  - original: full color (alpha=1.0)
  - gray:     alpha=0.7
  - lined:    alpha=0.4

Outputs:
  results/all_encoders/completion_summary.png   (5-subplot combined)
  results/all_encoders/gestalt_iou.png          (individual)
  results/all_encoders/mnemonic_similarity.png
  results/all_encoders/mnemonic_retrieval.png
  results/all_encoders/semantic_prototype.png
  results/all_encoders/semantic_zeroshot.png
  results/all_encoders/results.json             (aggregated)
  results/{encoder}/...                         (per-encoder, 3 image types)

Usage:
    python replot_combined.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from src.masking import get_mask_levels, get_visibility_ratio

RESULTS_ROOT = Path("results")

# Encoder display order and standard matplotlib tab10 colors
ENCODERS = ["CLIP", "DINO-v1", "DINOv2", "MAE", "I-JEPA", "ViT-supervised"]
TAB10 = plt.cm.tab10.colors
ENCODER_COLORS = {enc: TAB10[i] for i, enc in enumerate(ENCODERS)}

IMAGE_TYPES = ["original", "gray", "lined"]
IMAGE_TYPE_ALPHA = {"original": 1.0, "gray": 0.7, "lined": 0.4}

# Per-encoder plot colors (blue / dark gray / light gray)
PER_ENCODER_COLORS = {
    "original": "#1f77b4",
    "gray":     "#555555",
    "lined":    "#b0b0b0",
}

# Which result dirs to merge per image type
DIR_CANDIDATES = {
    "original": ["fragment_v2_original_1", "fragment_v2_original"],
    "gray":     ["fragment_v2_gray_1",     "fragment_v2_gray"],
    "lined":    ["fragment_v2_lined_1",    "fragment_v2_lined"],
}


def _fix_keys(d):
    if not isinstance(d, dict):
        return d
    try:
        return {int(k): v for k, v in d.items()}
    except (ValueError, TypeError):
        return {k: _fix_keys(v) for k, v in d.items()}


def _val(v):
    return v["mean"] if isinstance(v, dict) and "mean" in v else v


def _std(v):
    return v["std"] if isinstance(v, dict) and "std" in v else 0.0


def load_merged():
    merged = {}
    for img_type, dirs in DIR_CANDIDATES.items():
        merged[img_type] = {"gestalt": {}, "mnemonic": {}, "semantic": {}}
        for dirname in dirs:
            path = RESULTS_ROOT / dirname / "results.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            for task in ("gestalt", "mnemonic", "semantic"):
                if task in data:
                    for enc, vals in data[task].items():
                        if enc not in merged[img_type][task]:
                            merged[img_type][task][enc] = _fix_keys(vals)
            print(f"  Loaded {path} -> {list(data.get('gestalt', {}).keys())}")
    return merged


# --------------------------------------------------------------------------- #
#  Plotting helpers
# --------------------------------------------------------------------------- #

def _get_xy(values, levels):
    x = np.array([get_visibility_ratio(L) for L in levels])
    y = np.array([_val(values[L]) for L in levels])
    std = np.array([_std(values[L]) for L in levels])
    return x, y, std


def _plot_panel(ax, panel_data, title, ylabel):
    """Plot one panel for the combined figure (color=encoder, alpha=image_type)."""
    levels = get_mask_levels()
    for enc, img_type, values in panel_data:
        color = ENCODER_COLORS.get(enc, "black")
        alpha = IMAGE_TYPE_ALPHA[img_type]
        x, y, std = _get_xy(values, levels)
        ax.plot(x, y, marker="o", markersize=4, linewidth=1.5,
                color=(*color[:3], alpha))
        if np.any(std > 0):
            ax.fill_between(x, y - std, y + std, color=(*color[:3], alpha * 0.2))
    ax.set_xlabel("Visibility", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def _make_legend_handles(encoders):
    enc_handles = []
    for enc in encoders:
        c = ENCODER_COLORS[enc]
        enc_handles.append(mlines.Line2D([], [], color=c, marker="o", markersize=5,
                                          linewidth=2, label=enc))
    type_handles = []
    for img_type in IMAGE_TYPES:
        a = IMAGE_TYPE_ALPHA[img_type]
        type_handles.append(mlines.Line2D([], [], color=(0.3, 0.3, 0.3, a),
                                           marker="o", markersize=5, linewidth=2,
                                           label=img_type))
    return enc_handles + [mlines.Line2D([], [], color="none", label="")] + type_handles


def plot_individual(panel_data, title, ylabel, save_path, encoders):
    """Single plot with same styling as combined panels + legend."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_panel(ax, panel_data, title, ylabel)

    handles = _make_legend_handles(encoders)
    ax.legend(handles=handles, loc="best", fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_encoder(enc, merged, out_dir):
    """Per-encoder directory: lines = image types, colored blue/dark-gray/light-gray."""
    from src.utils import plot_metric_vs_masking, plot_completion_summary

    enc_gestalt = {}
    enc_mnemonic = {}
    enc_semantic = {}

    for img_type in IMAGE_TYPES:
        if enc in merged[img_type]["gestalt"]:
            enc_gestalt[img_type] = merged[img_type]["gestalt"][enc]
        if enc in merged[img_type]["mnemonic"]:
            enc_mnemonic[img_type] = merged[img_type]["mnemonic"][enc]
        if enc in merged[img_type]["semantic"]:
            enc_semantic[img_type] = merged[img_type]["semantic"][enc]

    if enc_gestalt:
        plot_metric_vs_masking(enc_gestalt, "IoU", f"{enc} — Gestalt (IoU)",
                               out_dir / "gestalt_iou.png", colors=PER_ENCODER_COLORS)
    if enc_mnemonic:
        sim = {k: v["similarity"] for k, v in enc_mnemonic.items()}
        ret = {k: v["retrieval"] for k, v in enc_mnemonic.items()}
        plot_metric_vs_masking(sim, "Cosine Similarity", f"{enc} — Mnemonic (Similarity)",
                               out_dir / "mnemonic_similarity.png", colors=PER_ENCODER_COLORS)
        plot_metric_vs_masking(ret, "Top-1 Accuracy", f"{enc} — Mnemonic (Retrieval)",
                               out_dir / "mnemonic_retrieval.png", colors=PER_ENCODER_COLORS)
    if enc_semantic:
        proto = {k: v["prototype_acc"] for k, v in enc_semantic.items()}
        plot_metric_vs_masking(proto, "Accuracy", f"{enc} — Semantic (Prototype)",
                               out_dir / "semantic_prototype.png", colors=PER_ENCODER_COLORS)
        zs = {k: v["zeroshot_acc"] for k, v in enc_semantic.items() if "zeroshot_acc" in v}
        if zs:
            plot_metric_vs_masking(zs, "Accuracy", f"{enc} — Semantic (Zero-shot)",
                                   out_dir / "semantic_zeroshot.png", colors=PER_ENCODER_COLORS)
    plot_completion_summary(enc_gestalt or None, enc_mnemonic or None, enc_semantic or None,
                            out_dir / "completion_summary.png", colors=PER_ENCODER_COLORS)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    merged = load_merged()

    found = set()
    for img_type in IMAGE_TYPES:
        for task_data in merged[img_type].values():
            found.update(task_data.keys())
    encoders = [e for e in ENCODERS if e in found]
    print(f"  Encoders: {encoders}")

    # Build panel data
    def _build_items(task, metric=None):
        items = []
        for enc in encoders:
            for img_type in IMAGE_TYPES:
                d = merged[img_type][task].get(enc)
                if d is None:
                    continue
                vals = d.get(metric) if metric else d
                if vals is None:
                    continue
                items.append((enc, img_type, vals))
        return items

    panels = [
        ("Gestalt (IoU)",          "IoU",        _build_items("gestalt")),
        ("Mnemonic (Similarity)",  "Cosine Sim", _build_items("mnemonic", "similarity")),
        ("Mnemonic (Retrieval@1)", "Accuracy",   _build_items("mnemonic", "retrieval")),
        ("Semantic (Prototype)",   "Accuracy",   _build_items("semantic", "prototype_acc")),
    ]
    zs_items = _build_items("semantic", "zeroshot_acc")
    if zs_items:
        panels.append(("Semantic (Zero-shot)", "Accuracy", zs_items))

    all_dir = RESULTS_ROOT / "all_encoders"
    all_dir.mkdir(parents=True, exist_ok=True)

    # --- Combined 5-subplot figure ---
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (title, ylabel, items) in zip(axes, panels):
        _plot_panel(ax, items, title, ylabel)

    handles = _make_legend_handles(encoders)
    fig.legend(handles=handles, loc="lower center",
               ncol=len(encoders) + 1 + len(IMAGE_TYPES),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fragment Completion — All Encoders", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save = all_dir / "completion_summary.png"
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {save}")

    # --- Individual plots in all_encoders/ ---
    fname_map = [
        "gestalt_iou", "mnemonic_similarity", "mnemonic_retrieval",
        "semantic_prototype", "semantic_zeroshot",
    ]
    for (title, ylabel, items), fname in zip(panels, fname_map):
        plot_individual(items, title, ylabel, all_dir / f"{fname}.png", encoders)

    # --- Per-encoder directories ---
    for enc in encoders:
        dir_name = enc.lower().replace("-", "_").replace(" ", "_")
        enc_dir = RESULTS_ROOT / dir_name
        enc_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  === {enc} -> {enc_dir} ===")
        plot_per_encoder(enc, merged, enc_dir)

    # --- Per image-type directories (lines = encoders, standard colors) ---
    for img_type in IMAGE_TYPES:
        it_dir = RESULTS_ROOT / img_type
        it_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  === {img_type} -> {it_dir} ===")

        it_gestalt = merged[img_type]["gestalt"]
        it_mnemonic = merged[img_type]["mnemonic"]
        it_semantic = merged[img_type]["semantic"]

        from src.utils import plot_metric_vs_masking, plot_completion_summary
        enc_colors = {e: "#{:02x}{:02x}{:02x}".format(
            int(ENCODER_COLORS[e][0]*255), int(ENCODER_COLORS[e][1]*255), int(ENCODER_COLORS[e][2]*255))
            for e in encoders if e in it_gestalt or e in it_mnemonic or e in it_semantic}

        if it_gestalt:
            plot_metric_vs_masking(it_gestalt, "IoU", f"{img_type} — Gestalt (IoU)",
                                   it_dir / "gestalt_iou.png", colors=enc_colors)
        if it_mnemonic:
            sim = {k: v["similarity"] for k, v in it_mnemonic.items()}
            ret = {k: v["retrieval"] for k, v in it_mnemonic.items()}
            plot_metric_vs_masking(sim, "Cosine Similarity", f"{img_type} — Mnemonic (Similarity)",
                                   it_dir / "mnemonic_similarity.png", colors=enc_colors)
            plot_metric_vs_masking(ret, "Top-1 Accuracy", f"{img_type} — Mnemonic (Retrieval)",
                                   it_dir / "mnemonic_retrieval.png", colors=enc_colors)
        if it_semantic:
            proto = {k: v["prototype_acc"] for k, v in it_semantic.items()}
            plot_metric_vs_masking(proto, "Accuracy", f"{img_type} — Semantic (Prototype)",
                                   it_dir / "semantic_prototype.png", colors=enc_colors)
            zs = {k: v["zeroshot_acc"] for k, v in it_semantic.items() if "zeroshot_acc" in v}
            if zs:
                plot_metric_vs_masking(zs, "Accuracy", f"{img_type} — Semantic (Zero-shot)",
                                       it_dir / "semantic_zeroshot.png", colors=enc_colors)
        plot_completion_summary(it_gestalt or None, it_mnemonic or None, it_semantic or None,
                                it_dir / "completion_summary.png", colors=enc_colors)

    # --- Aggregated results.json ---
    agg = {}
    for img_type in IMAGE_TYPES:
        for task, task_data in merged[img_type].items():
            for enc, vals in task_data.items():
                if enc not in agg:
                    agg[enc] = {}
                if task not in agg[enc]:
                    agg[enc][task] = {}
                agg[enc][task][img_type] = vals
    agg_path = RESULTS_ROOT / "results.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Saved: {agg_path}")


if __name__ == "__main__":
    main()
