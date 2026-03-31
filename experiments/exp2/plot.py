#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Exp2 plotting: regenerate plots from saved results JSON files.

Reads JSON results produced by clip_interp, dinov2_interp, or ground_retrieval
and regenerates the corresponding plots without re-running experiments.

Usage:
    uv run python -m experiments.exp2.plot retrieval --results-dir results/exp2/ground_retrieval/coco_subset/dinov2
    uv run python -m experiments.exp2.plot retrieval --results-dir results/exp2/ground_retrieval/frag/clip_L14
    uv run python -m experiments.exp2.plot zeroshot --results results/exp2/clip_interp/zeroshot/results.json
    uv run python -m experiments.exp2.plot probe --results results/exp2/clip_interp/probing/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import PLOT_STYLE as PS
from src.masking import get_mask_levels, get_visibility_ratio

# ---------------------------------------------------------------------------
# Tab20c task colors (consistent across all retrieval plots)
# ---------------------------------------------------------------------------

_TAB20C = plt.cm.tab20c.colors

# Task definitions: (candidate_keys, display_label, color)
# candidate_keys supports old and new key names for backwards compatibility.
# Row 1 (legend): Retrieval tasks.  Row 2: Categorization tasks.
# Blues: dark = conceptual, light = perceptual.
RETRIEVAL_TASKS = [
    # Retrieval
    (["image_r1"],                         "Image Retrieval",         _TAB20C[9]),
    (["exemplar_acc"],                     "Image k-NN Exemplar",     _TAB20C[3]),
    (["instance_r1"],                      "Concept Retrieval",       _TAB20C[5]),
    (["img_proto_acc"],                    "Image Mean Prototype",    _TAB20C[2]),
    (["category_acc", "supercat_acc"],     "Category Retrieval",      _TAB20C[0]),
    (["concept_proto_acc", "txt_proto_acc"], "Concept Mean Prototype", _TAB20C[1]),
]


def _find_key(d: dict, candidates: list[str]) -> str | None:
    """Return the first key from candidates that exists in d."""
    for k in candidates:
        if k in d:
            return k
    return None


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_zeroshot_accuracy(
    results: dict,
    save_path: Path,
    title: str = "Zero-Shot Accuracy vs. Visibility",
) -> None:
    """Plot accuracy across masking levels from zeroshot results.json."""
    levels = get_mask_levels()
    vis = [get_visibility_ratio(L) for L in levels]
    means = [results[str(L)]["mean"] for L in levels]
    stds = [results[str(L)]["std"] for L in levels]

    fig, ax = plt.subplots(figsize=PS["subplot_size"])
    ax.errorbar(
        vis, means, yerr=stds,
        marker=PS["marker"], markersize=PS["markersize"],
        linewidth=PS["linewidth"], capsize=3,
    )
    ax.set_xlabel("Visibility Ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel("Accuracy", fontsize=PS["label_fontsize"])
    ax.set_title(title, fontsize=PS["subplot_title_fontsize"])
    ax.tick_params(labelsize=PS["tick_labelsize"], width=PS["tick_width"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_probe_heatmap(
    results: dict,
    save_path: Path,
    title: str = "Probe Accuracy (Layer × Masking Level)",
) -> None:
    """Plot probing accuracy heatmap from probing results.json."""
    acc_matrix = np.array(results["accuracy"])  # [num_layers, 8]
    levels = get_mask_levels()
    vis_labels = [f"L{L}\n({get_visibility_ratio(L):.0%})" for L in levels]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(acc_matrix, aspect="auto", cmap="Blues")
    ax.set_xlabel("Masking Level", fontsize=PS["label_fontsize"])
    ax.set_ylabel("Layer", fontsize=PS["label_fontsize"])
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(vis_labels, fontsize=10)
    ax.set_yticks(range(acc_matrix.shape[0]))
    ax.set_title(title, fontsize=PS["subplot_title_fontsize"])
    fig.colorbar(im, ax=ax, label="Accuracy")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_task_comparison(
    data: dict,
    save_path: Path,
    title: str = "Task Comparison",
) -> None:
    """Plot all retrieval tasks on one figure.

    Args:
        data: Level-keyed results dict, e.g. {"1": {"image_r1": 0.5, ...}, ...}
        save_path: Output .png path.
        title: Plot title.
    """
    levels = get_mask_levels()
    vis = [get_visibility_ratio(L) for L in levels]

    fig, ax = plt.subplots(figsize=PS["subplot_size"])
    handles = []
    for keys, label, color in RETRIEVAL_TASKS:
        k = _find_key(data["1"], keys)
        if k is None:
            continue
        vals = [data[str(L)][k] for L in levels]
        h, = ax.plot(
            vis, vals,
            marker=PS["marker"], markersize=PS["markersize"],
            linewidth=PS["linewidth"], color=color, label=label,
        )
        handles.append(h)

    ax.set_xlabel("Visibility Ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel("Accuracy / R@1", fontsize=PS["label_fontsize"])
    ax.set_title(title, fontsize=PS["subplot_title_fontsize"], fontweight="bold")
    ax.tick_params(labelsize=PS["tick_labelsize"], width=PS["tick_width"])
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(PS["tick_width"])

    fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3, fontsize=PS["legend_fontsize"], frameon=True,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_retrieval(
    results_dir: Path,
    title: str | None = None,
) -> None:
    """Plot task comparison for each results_*.json in results_dir/retrieval/.

    Args:
        results_dir: Model results directory (e.g. .../coco_subset/dinov2).
        title: Override plot title; default derived from directory name.
    """
    retrieval_dir = results_dir / "retrieval"
    if not retrieval_dir.exists():
        print(f"  [skip] No retrieval dir: {retrieval_dir}")
        return

    result_files = sorted(retrieval_dir.glob("results_*.json"))
    if not result_files:
        print(f"  [skip] No result files in {retrieval_dir}")
        return

    for rfile in result_files:
        with open(rfile) as f:
            data = json.load(f)

        # Derive image type from filename: results_original.json -> original
        img_type = rfile.stem.replace("results_", "")
        enc_name = results_dir.name  # e.g. "dinov2", "clip_L14"
        default_title = f"{enc_name} — {img_type}"

        out_path = retrieval_dir / f"task_comparison_{img_type}.png"
        plot_task_comparison(data, out_path, title or default_title)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI for regenerating exp2 plots from saved JSON results."""
    parser = argparse.ArgumentParser(description="Exp2: Regenerate plots from JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("zeroshot", help="Replot zeroshot accuracy")
    p.add_argument("--results", type=str, required=True, help="Path to results.json")
    p.add_argument("--out", type=str, default=None, help="Output .png path")

    p = sub.add_parser("probe", help="Replot probe heatmap")
    p.add_argument("--results", type=str, required=True, help="Path to results.json")
    p.add_argument("--out", type=str, default=None, help="Output .png path")

    p = sub.add_parser("retrieval", help="Replot retrieval task comparison")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Model results dir (e.g. results/exp2/ground_retrieval/coco_subset/dinov2)")
    p.add_argument("--title", type=str, default=None, help="Override plot title")

    args = parser.parse_args()
    cmd = args.command

    if cmd == "zeroshot":
        results_path = Path(args.results)
        with open(results_path) as f:
            data = json.load(f)
        out = Path(args.out) if args.out else results_path.parent / "accuracy_vs_visibility.png"
        plot_zeroshot_accuracy(data, out)

    elif cmd == "probe":
        results_path = Path(args.results)
        with open(results_path) as f:
            data = json.load(f)
        out = Path(args.out) if args.out else results_path.parent / "probe_accuracy_heatmap.png"
        plot_probe_heatmap(data, out)

    elif cmd == "retrieval":
        plot_retrieval(Path(args.results_dir), title=args.title)


if __name__ == "__main__":
    main()
