#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Exp1 plotting: generate plots from unified results.json.

All plots are derived from a single unified results.json. Each output subdirectory
gets plots + a results.json subset (same structure, filtered by encoder or image type).

Standalone usage:
    uv run python -m experiments.exp1.plot --results results/exp1/results.json
    uv run python -m experiments.exp1.plot --results results/exp1/results.json --encoders CLIP MAE
    uv run python -m experiments.exp1.plot --results results/exp1/results.json --image-types original gray
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    ENCODER_COLORS,
    PLOT_STYLE as PS,
    results_for_encoder,
    results_for_image_type,
)
from src.masking import get_mask_levels, get_visibility_ratio
from src.utils import (
    extract_val,
    extract_std,
    fix_json_keys,
    plot_completion_summary,
    plot_metric_vs_masking,
    save_results,
)


# ---------------------------------------------------------------------------
# Core plotting (operates on per-view dicts)
# ---------------------------------------------------------------------------

def _plot_gestalt(all_gestalt: dict, out_dir: Path) -> None:
    """Plot gestalt IoU and silhouette score."""
    plot_metric_vs_masking(
        all_gestalt,
        "IoU",
        "Gestalt Completion (Segmentation IoU)",
        out_dir / "gestalt" / "gestalt_iou.png",
    )
    sil_data = {}
    for enc, vals in all_gestalt.items():
        sil_data[enc] = {
            L: {
                "mean": vals[L]["silhouette_mean"],
                "std": vals[L]["silhouette_std"],
            }
            for L in vals
        }
    plot_metric_vs_masking(
        sil_data,
        "Silhouette Score",
        "Gestalt Completion (Cluster Separation)",
        out_dir / "gestalt" / "gestalt_silhouette.png",
    )


def _plot_mnemonic(all_mnemonic: dict, out_dir: Path) -> None:
    """Plot mnemonic similarity, K-choice retrieval, and full-rank metrics."""
    sim_data = {k: v["similarity"] for k, v in all_mnemonic.items()}
    ret_data = {k: v["retrieval"] for k, v in all_mnemonic.items()}
    plot_metric_vs_masking(
        sim_data,
        "Cosine Similarity",
        "Mnemonic Completion (Embedding Similarity)",
        out_dir / "mnemonic" / "mnemonic_similarity.png",
    )
    plot_metric_vs_masking(
        ret_data,
        "Top-1 Accuracy",
        "Mnemonic Completion (K-choice Retrieval)",
        out_dir / "mnemonic" / "mnemonic_retrieval.png",
    )
    for metric_key, ylabel, title_suffix, fname in [
        ("retrieval_r1", "R@1", "Full-Rank R@1", "mnemonic_retrieval_full_rank.png"),
        ("retrieval_r5", "R@5", "Full-Rank R@5", "mnemonic_retrieval_r5.png"),
        ("retrieval_mrr", "MRR", "Full-Rank MRR", "mnemonic_retrieval_mrr.png"),
    ]:
        metric_data = {
            k: v[metric_key]
            for k, v in all_mnemonic.items()
            if metric_key in v
        }
        if metric_data:
            plot_metric_vs_masking(
                metric_data,
                ylabel,
                f"Mnemonic Completion ({title_suffix})",
                out_dir / "mnemonic" / fname,
            )


def _plot_semantic(all_semantic: dict, out_dir: Path) -> None:
    """Plot semantic prototype and zero-shot accuracy."""
    proto_data = {k: v["prototype_acc"] for k, v in all_semantic.items()}
    plot_metric_vs_masking(
        proto_data,
        "Accuracy",
        "Semantic Completion (Prototype Classification)",
        out_dir / "semantic" / "semantic_prototype.png",
    )
    zs_data = {
        k: v["zeroshot_acc"]
        for k, v in all_semantic.items()
        if "zeroshot_acc" in v
    }
    if zs_data:
        plot_metric_vs_masking(
            zs_data,
            "Accuracy",
            "Semantic Completion (CLIP Zero-shot)",
            out_dir / "semantic" / "semantic_zeroshot.png",
        )


def _plot_similarity(
    all_similarity: dict,
    all_mnemonic: dict | None,
    all_semantic: dict | None,
    out_dir: Path,
) -> None:
    """Plot similarity analysis: accuracy + raw similarity + diff.

    Layout per task (mnemonic / semantic):
      col 0: Task accuracy (R@1 or prototype acc) for reference
      col 1: Raw similarity (target vs all)
      col 2: Similarity difference (target - all)

    Args:
        all_similarity: {encoder: {mnemonic_target, mnemonic_all, semantic_same_cat, semantic_all_cat}}
        all_mnemonic: mnemonic results for accuracy reference (optional)
        all_semantic: semantic results for accuracy reference (optional)
    """
    levels = get_mask_levels()
    vis = [get_visibility_ratio(L) for L in levels]
    encoders = sorted(all_similarity.keys())

    has_mnem = any("mnemonic_target" in v for v in all_similarity.values())
    has_sem = any("semantic_same_cat" in v for v in all_similarity.values())
    if not has_mnem and not has_sem:
        return

    rows = []
    if has_mnem:
        rows.append("mnemonic")
    if has_sem:
        rows.append("semantic")

    fig, axes = plt.subplots(
        len(rows), 3,
        figsize=(PS["subplot_size"][0] * 3, PS["subplot_size"][1] * len(rows)),
    )
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    for row_idx, task in enumerate(rows):
        ax_acc, ax_raw, ax_diff = axes[row_idx]

        for enc in encoders:
            sim = all_similarity[enc]
            color = ENCODER_COLORS.get(enc, None)

            if task == "mnemonic":
                tgt_key, all_key = "mnemonic_target", "mnemonic_all"
                acc_label = "R@1"
                task_label = "Mnemonic"
            else:
                tgt_key, all_key = "semantic_same_cat", "semantic_all_cat"
                acc_label = "Proto Acc"
                task_label = "Semantic"

            if tgt_key not in sim or all_key not in sim:
                continue

            tgt_vals = [extract_val(sim[tgt_key][L]) for L in levels]
            all_vals = [extract_val(sim[all_key][L]) for L in levels]
            diff_vals = [t - a for t, a in zip(tgt_vals, all_vals)]
            tgt_std = [extract_std(sim[tgt_key][L]) for L in levels]
            all_std = [extract_std(sim[all_key][L]) for L in levels]
            diff_std = [np.sqrt(s1**2 + s2**2) for s1, s2 in zip(tgt_std, all_std)]

            # Col 0: Task accuracy
            if task == "mnemonic" and all_mnemonic and enc in all_mnemonic:
                r1_key = "retrieval_r1"
                if r1_key in all_mnemonic[enc]:
                    acc_vals = [extract_val(all_mnemonic[enc][r1_key][L]) for L in levels]
                    ax_acc.plot(vis, acc_vals, "o-", label=enc, color=color,
                                linewidth=PS["linewidth"], markersize=PS["markersize"])
            elif task == "semantic" and all_semantic and enc in all_semantic:
                if "prototype_acc" in all_semantic[enc]:
                    acc_vals = [extract_val(all_semantic[enc]["prototype_acc"][L]) for L in levels]
                    ax_acc.plot(vis, acc_vals, "o-", label=enc, color=color,
                                linewidth=PS["linewidth"], markersize=PS["markersize"])

            # Col 1: Raw similarity (target solid, all dashed)
            ax_raw.plot(vis, tgt_vals, "o-", label=f"{enc} target", color=color,
                        linewidth=PS["linewidth"], markersize=PS["markersize"])
            ax_raw.plot(vis, all_vals, "s--", color=color,
                        linewidth=PS["linewidth"] * 0.7, markersize=PS["markersize"] * 0.7,
                        alpha=0.6)

            # Col 2: Similarity diff
            ax_diff.plot(vis, diff_vals, "o-", label=enc, color=color,
                         linewidth=PS["linewidth"], markersize=PS["markersize"])
            ax_diff.fill_between(
                vis,
                [m - s for m, s in zip(diff_vals, diff_std)],
                [m + s for m, s in zip(diff_vals, diff_std)],
                alpha=PS["std_alpha"], color=color,
            )

        # Formatting
        ax_acc.set_title(f"{task_label} {acc_label}", fontsize=PS["subplot_title_fontsize"])
        ax_acc.set_ylabel(acc_label, fontsize=PS["label_fontsize"])
        ax_raw.set_title(f"{task_label} Similarity", fontsize=PS["subplot_title_fontsize"])
        ax_raw.set_ylabel("Cosine Similarity", fontsize=PS["label_fontsize"])
        ax_diff.set_title(f"{task_label} Diff (target − all)", fontsize=PS["subplot_title_fontsize"])
        ax_diff.set_ylabel("Similarity Difference", fontsize=PS["label_fontsize"])
        ax_diff.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        for ax in (ax_acc, ax_raw, ax_diff):
            ax.set_xlabel("Visibility", fontsize=PS["label_fontsize"])
            ax.tick_params(labelsize=PS["tick_labelsize"], width=PS["tick_width"])
            ax.grid(True, alpha=0.3)

        ax_acc.legend(fontsize=PS["legend_fontsize"] * 0.8)

    fig.suptitle("Similarity Analysis", fontsize=PS["suptitle_fontsize"], fontweight="bold")
    fig.tight_layout()
    save_path = out_dir / "similarity" / "similarity_analysis.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_view(
    all_gestalt: dict | None,
    all_mnemonic: dict | None,
    all_semantic: dict | None,
    all_similarity: dict | None,
    out_dir: Path,
) -> None:
    """Generate all plots for one view (by image-type or by encoder)."""
    if all_gestalt:
        _plot_gestalt(all_gestalt, out_dir)
    if all_mnemonic:
        _plot_mnemonic(all_mnemonic, out_dir)
    if all_semantic:
        _plot_semantic(all_semantic, out_dir)
    if all_similarity:
        _plot_similarity(all_similarity, all_mnemonic, all_semantic, out_dir)
    plot_completion_summary(
        all_gestalt or None,
        all_mnemonic or None,
        all_semantic or None,
        out_dir / "completion_summary.png",
    )


# ---------------------------------------------------------------------------
# Unified JSON → per-view dicts
# ---------------------------------------------------------------------------

def _unified_to_views(
    enc_data: dict,
) -> dict[str, dict[str, dict]]:
    """Convert unified {encoder: {img_type: {metric: ...}}} to per-image-type views.

    Returns:
        {img_type: {"gestalt": {enc: data}, "mnemonic": {enc: data}, "semantic": {enc: data}}}
    """
    # Discover all image types
    all_img_types: set[str] = set()
    for per_type in enc_data.values():
        all_img_types.update(per_type.keys())

    results_by_type: dict[str, dict] = {
        img_type: {"gestalt": {}, "mnemonic": {}, "semantic": {}, "similarity": {}}
        for img_type in all_img_types
    }

    for enc_display, per_type in enc_data.items():
        for img_type, metrics in per_type.items():
            if img_type not in results_by_type:
                continue
            r = results_by_type[img_type]
            if "gestalt_iou" in metrics:
                r["gestalt"][enc_display] = metrics["gestalt_iou"]
            if "mnemonic_similarity" in metrics and "mnemonic_retrieval" in metrics:
                mnem: dict = {
                    "similarity": metrics["mnemonic_similarity"],
                    "retrieval": metrics["mnemonic_retrieval"],
                }
                for k in ("retrieval_r1", "retrieval_r5", "retrieval_mrr"):
                    if f"mnemonic_{k}" in metrics:
                        mnem[k] = metrics[f"mnemonic_{k}"]
                r["mnemonic"][enc_display] = mnem
            if "semantic_prototype" in metrics:
                sem: dict = {"prototype_acc": metrics["semantic_prototype"]}
                if "semantic_zeroshot" in metrics:
                    sem["zeroshot_acc"] = metrics["semantic_zeroshot"]
                r["semantic"][enc_display] = sem
            # Similarity analysis
            sim_keys = ("mnemonic_target", "mnemonic_all",
                        "semantic_same_cat", "semantic_all_cat")
            sim_data = {}
            for sk in sim_keys:
                json_key = f"similarity_{sk}"
                if json_key in metrics:
                    sim_data[sk] = metrics[json_key]
            if sim_data:
                r["similarity"][enc_display] = sim_data

    return results_by_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_from_json(
    results_path: Path,
    out_root: Path | None = None,
    encoders: list[str] | None = None,
    image_types: list[str] | None = None,
) -> None:
    """Read unified results.json, generate plots per image-type and per encoder.

    Each output subdirectory receives:
      - All relevant plots
      - A results.json subset (same unified structure, filtered)

    Args:
        results_path: Path to unified results.json.
        out_root: Output root (defaults to results_path parent).
        encoders: Subset of encoders to include (None = all).
        image_types: Subset of image types to include (None = all).
    """
    if out_root is None:
        out_root = results_path.parent

    with open(results_path) as f:
        raw = fix_json_keys(json.load(f))

    enc_data: dict = raw.get("encoders", raw)

    # Filter encoders
    if encoders:
        enc_data = {k: v for k, v in enc_data.items() if k in encoders}

    # Filter image types
    if image_types:
        enc_data = {
            enc: {it: m for it, m in per_type.items() if it in image_types}
            for enc, per_type in enc_data.items()
        }

    # Convert to per-image-type views for plotting
    results_by_type = _unified_to_views(enc_data)

    # --- Per image-type: lines = encoders ---
    for img_type, r in results_by_type.items():
        out_dir = results_for_image_type(img_type, root=out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Plotting: image_types/{img_type}")
        _plot_view(r["gestalt"], r["mnemonic"], r["semantic"], r["similarity"], out_dir)
        # Save subset: only this image type, all encoders
        subset = {
            "encoders": {
                enc: {img_type: enc_data[enc][img_type]}
                for enc in enc_data
                if img_type in enc_data[enc]
            }
        }
        save_results(subset, out_dir / "results.json")

    # --- Per encoder: lines = image types (only when multiple types) ---
    discovered_types = list(results_by_type.keys())
    if len(discovered_types) > 1:
        all_enc_names: set[str] = set()
        for r in results_by_type.values():
            for task in ("gestalt", "mnemonic", "semantic"):
                all_enc_names.update(r[task].keys())

        for enc_display in sorted(all_enc_names):
            enc_dir = results_for_encoder(enc_display, root=out_root)
            enc_dir.mkdir(parents=True, exist_ok=True)

            enc_gestalt: dict = {}
            enc_mnemonic: dict = {}
            enc_semantic: dict = {}
            enc_similarity: dict = {}
            for img_type in discovered_types:
                r = results_by_type[img_type]
                if enc_display in r["gestalt"]:
                    enc_gestalt[img_type] = r["gestalt"][enc_display]
                if enc_display in r["mnemonic"]:
                    enc_mnemonic[img_type] = r["mnemonic"][enc_display]
                if enc_display in r["semantic"]:
                    enc_semantic[img_type] = r["semantic"][enc_display]
                if enc_display in r["similarity"]:
                    enc_similarity[img_type] = r["similarity"][enc_display]

            print(f"  Plotting: encoders/{enc_display} (lines = image types)")
            _plot_view(enc_gestalt, enc_mnemonic, enc_semantic, enc_similarity, enc_dir)
            # Save subset: only this encoder, all image types
            if enc_display in enc_data:
                subset = {"encoders": {enc_display: enc_data[enc_display]}}
                save_results(subset, enc_dir / "results.json")

    print("  Plotting complete.")



def main() -> None:
    """CLI for standalone plotting from unified results.json."""
    parser = argparse.ArgumentParser(description="Exp1: Plot from results.json")
    parser.add_argument(
        "--results",
        type=str,
        default="results/exp1/results.json",
        help="Path to unified results.json",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--encoders", nargs="+", default=None, help="Subset of encoders")
    parser.add_argument("--image-types", nargs="+", default=None, help="Subset of image types")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_root = Path(args.out_dir) if args.out_dir else None
    plot_from_json(results_path, out_root, args.encoders, args.image_types)


if __name__ == "__main__":
    main()
