#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vis_category.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-30-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Category readout comparison plot: CLIP vs DINOv2.

Two Tab20c hue groups (one per model), 4 dark-to-light shades per model
for 4 readout methods.

Usage:
    uv run python analysis/vis_category.py
    uv run python analysis/vis_category.py --results path/to/results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────
PS = {
    "linewidth": 2,
    "tick_width": 2,
    "tick_labelsize": 14,
    "label_fontsize": 16,
    "legend_fontsize": 14,
    "suptitle_fontsize": 20,
    "marker": "o",
    "markersize": 6,
    "dpi": 150,
}

# Tab20c hue groups: each group has 4 shades (dark → light)
# Group 0 (blue-ish):  indices 0,1,2,3
# Group 1 (orange-ish): indices 4,5,6,7
TAB20C = plt.colormaps.get_cmap("tab20c")

MODEL_HUE = {
    "clip_L14": 0,   # blue group: indices 0–3
    "dinov2":   1,    # orange group: indices 4–7
}

# (json_key, display_label, marker, shade_index 0=darkest)
READOUTS = [
    ("category_acc",  "Category Text",    "o", 0),
    ("img_proto_acc", "Image Prototype",  "s", 1),
    ("txt_proto_acc", "Text Prototype",   "^", 2),
    ("instance_r1",   "Instance Text R@1", "D", 3),
]

MODEL_DISPLAY = {
    "clip_L14": "CLIP",
    "dinov2":   "DINOv2",
}


def _visibility_ratio(level: int) -> float:
    """P = 0.7^(8 - L)."""
    return 0.7 ** (8 - level)


def plot_category_readouts(data: dict, save_path: Path) -> None:
    """Plot category readouts for all models.

    Args:
        data: {model_key: {level_str: {metric: value}}}.
        save_path: Output PNG path.
    """
    levels = list(range(1, 9))
    vis_x = [_visibility_ratio(L) for L in levels]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_key in ["clip_L14", "dinov2"]:
        if model_key not in data:
            continue
        hue_group = MODEL_HUE[model_key]
        model_name = MODEL_DISPLAY[model_key]

        for metric_key, label, marker, shade_idx in READOUTS:
            color_idx = hue_group * 4 + shade_idx
            color = TAB20C(color_idx)

            vals = []
            has_data = True
            for L in levels:
                level_data = data[model_key].get(str(L), {})
                if metric_key not in level_data:
                    has_data = False
                    break
                vals.append(level_data[metric_key])

            if not has_data:
                continue

            ax.plot(
                vis_x, vals,
                marker=marker,
                linewidth=PS["linewidth"],
                color=color,
                label=f"{model_name} — {label}",
                markersize=PS["markersize"],
            )

    ax.set_xlabel("Visibility Ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel("Accuracy", fontsize=PS["label_fontsize"])
    ax.set_title(
        "Category Classification: 4 Readout Methods",
        fontsize=PS["suptitle_fontsize"], fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=PS["tick_labelsize"], width=PS["tick_width"])
    for spine in ax.spines.values():
        spine.set_linewidth(PS["tick_width"])
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Category readout comparison plot")
    parser.add_argument(
        "--results",
        default="results/ground_retrieval/frag/all/results.json",
        help="Path to results JSON with both models",
    )
    parser.add_argument(
        "--out",
        default="results/ground_retrieval/frag/all/category_readouts.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    plot_category_readouts(data, Path(args.out))


if __name__ == "__main__":
    main()
