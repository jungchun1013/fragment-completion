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
    uv run python -m experiments.exp2.plot retrieval --results-dir results/exp2/ground_retrieval/coco_subset
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


def plot_retrieval_by_task(
    results_dir: Path,
    save_path: Path,
    title: str = "Retrieval Accuracy by Task",
) -> None:
    """Plot retrieval metrics from ground_retrieval results directory.

    Reads all results_*.json files under results_dir/retrieval/.
    """
    retrieval_dir = results_dir / "retrieval"
    if not retrieval_dir.exists():
        print(f"  [skip] No retrieval dir: {retrieval_dir}")
        return

    levels = get_mask_levels()
    vis = [get_visibility_ratio(L) for L in levels]

    # Collect all result files
    result_files = sorted(retrieval_dir.glob("results_*.json"))
    if not result_files:
        print(f"  [skip] No result files in {retrieval_dir}")
        return

    # Use first result file
    with open(result_files[0]) as f:
        data = json.load(f)

    # Plot key metrics
    metric_keys = ["image_r1", "category_acc", "img_proto_acc", "txt_proto_acc"]
    metric_labels = ["Image R@1", "Category Acc", "Img Proto Acc", "Txt Proto Acc"]

    fig, ax = plt.subplots(figsize=PS["subplot_size"])
    for key, label in zip(metric_keys, metric_labels):
        vals = [data[str(L)].get(key, 0) for L in levels]
        ax.plot(
            vis, vals, marker=PS["marker"], markersize=PS["markersize"],
            linewidth=PS["linewidth"], label=label,
        )
    ax.set_xlabel("Visibility Ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel("Score", fontsize=PS["label_fontsize"])
    ax.set_title(title, fontsize=PS["subplot_title_fontsize"])
    ax.legend(fontsize=PS["legend_fontsize"])
    ax.tick_params(labelsize=PS["tick_labelsize"], width=PS["tick_width"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


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

    p = sub.add_parser("retrieval", help="Replot retrieval curves")
    p.add_argument("--results-dir", type=str, required=True, help="Path to model results dir")
    p.add_argument("--out", type=str, default=None, help="Output .png path")

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
        results_dir = Path(args.results_dir)
        out = Path(args.out) if args.out else results_dir / "retrieval_by_task.png"
        plot_retrieval_by_task(results_dir, out)


if __name__ == "__main__":
    main()
