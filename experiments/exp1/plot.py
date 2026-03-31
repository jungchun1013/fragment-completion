#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Exp1 plotting: generate plots from results.json or in-memory results.

Standalone usage:
    uv run python -m experiments.exp1.plot --results results/exp1/results.json
    uv run python -m experiments.exp1.plot --results results/exp1/results.json --encoders CLIP MAE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import results_for_encoder, results_for_image_type
from src.masking import get_mask_levels
from src.utils import (
    fix_json_keys,
    plot_completion_summary,
    plot_metric_vs_masking,
    save_results,
)


def _plot_group(
    all_gestalt: dict | None,
    all_mnemonic: dict | None,
    all_semantic: dict | None,
    out_dir: Path,
) -> None:
    """Generate all plots for one grouping (image-type or encoder)."""
    if all_gestalt:
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

    if all_mnemonic:
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
        # Full-rank retrieval metrics (R@1, R@5, MRR)
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

    if all_semantic:
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

    plot_completion_summary(
        all_gestalt or None,
        all_mnemonic or None,
        all_semantic or None,
        out_dir / "completion_summary.png",
    )

    all_results: dict = {}
    if all_gestalt:
        all_results["gestalt"] = all_gestalt
    if all_mnemonic:
        all_results["mnemonic"] = all_mnemonic
    if all_semantic:
        all_results["semantic"] = all_semantic
    save_results(all_results, out_dir / "results.json")


def plot_from_results(
    results_by_type: dict[str, dict],
    similarity_by_encoder: dict[str, dict[str, dict]] | None = None,
    image_types: list[str] | None = None,
    out_root: Path = Path("results/exp1"),
) -> None:
    """Generate plots from in-memory experiment results.

    Called by run.py --plot or directly.
    """
    if image_types is None:
        image_types = list(results_by_type.keys())

    # Per image-type plots
    for img_type in image_types:
        out_dir = results_for_image_type(img_type, root=out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = results_by_type[img_type]
        print(f"\n  Plotting: {img_type}")
        _plot_group(r["gestalt"], r["mnemonic"], r["semantic"], out_dir)

    # Per-encoder plots (lines = image types) — only when multiple types
    if len(image_types) > 1:
        all_enc_names: set[str] = set()
        for img_type in image_types:
            for task in ("gestalt", "mnemonic", "semantic"):
                all_enc_names.update(results_by_type[img_type][task].keys())

        for enc_display in sorted(all_enc_names):
            enc_dir = results_for_encoder(enc_display, root=out_root)
            enc_dir.mkdir(parents=True, exist_ok=True)

            enc_gestalt: dict = {}
            enc_mnemonic: dict = {}
            enc_semantic: dict = {}
            for img_type in image_types:
                r = results_by_type[img_type]
                if enc_display in r["gestalt"]:
                    enc_gestalt[img_type] = r["gestalt"][enc_display]
                if enc_display in r["mnemonic"]:
                    enc_mnemonic[img_type] = r["mnemonic"][enc_display]
                if enc_display in r["semantic"]:
                    enc_semantic[img_type] = r["semantic"][enc_display]

            print(f"  Plotting: {enc_display} (lines = image types)")
            _plot_group(enc_gestalt, enc_mnemonic, enc_semantic, enc_dir)

    print("  Plotting complete.")


def plot_from_json(
    results_path: Path,
    out_root: Path | None = None,
    encoders: list[str] | None = None,
    image_types: list[str] | None = None,
) -> None:
    """Generate plots by reading results.json.

    Args:
        results_path: Path to results.json.
        out_root: Output directory (defaults to results_path parent).
        encoders: Subset of encoders to plot (None = all).
        image_types: Subset of image types to plot (None = all).
    """
    if out_root is None:
        out_root = results_path.parent

    with open(results_path) as f:
        data = fix_json_keys(json.load(f))

    enc_data = data.get("encoders", data)

    # Filter encoders
    if encoders:
        enc_data = {k: v for k, v in enc_data.items() if k in encoders}

    # Discover image types
    all_img_types: set[str] = set()
    for enc_metrics in enc_data.values():
        all_img_types.update(enc_metrics.keys())
    if image_types:
        all_img_types &= set(image_types)

    # Reconstruct results_by_type from unified JSON
    results_by_type: dict[str, dict] = {
        img_type: {"gestalt": {}, "mnemonic": {}, "semantic": {}}
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
                    json_key = f"mnemonic_{k}"
                    if json_key in metrics:
                        mnem[k] = metrics[json_key]
                r["mnemonic"][enc_display] = mnem
            if "semantic_prototype" in metrics:
                sem: dict = {"prototype_acc": metrics["semantic_prototype"]}
                if "semantic_zeroshot" in metrics:
                    sem["zeroshot_acc"] = metrics["semantic_zeroshot"]
                r["semantic"][enc_display] = sem

    plot_from_results(results_by_type, out_root=out_root)


def main() -> None:
    """CLI for standalone plotting from results.json."""
    parser = argparse.ArgumentParser(description="Exp1: Plot from results.json")
    parser.add_argument(
        "--results",
        type=str,
        default="results/exp1/results.json",
        help="Path to results.json",
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
