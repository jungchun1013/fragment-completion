#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Experiment 1: Fragment completion (gestalt, mnemonic, semantic, similarity).

Produces a unified results.json with all metrics across encoders and image types.
Incremental save after each encoder completes.

Usage:
    uv run python -m experiments.exp1.run
    uv run python -m experiments.exp1.run --max-images 5
    uv run python -m experiments.exp1.run --encoders clip mae dino
    uv run python -m experiments.exp1.run --image-type original gray lined
    uv run python -m experiments.exp1.run --tasks gestalt mnemonic
    uv run python -m experiments.exp1.run --plot  # generate plots after experiments
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import torch

from models.registry import get_encoder
from src.config import (
    IMAGE_TYPES,
    results_for_encoder,
    results_for_image_type,
)
from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio
from src.utils import extract_val, save_results

from .gestalt import evaluate_gestalt
from .mnemonic import evaluate_mnemonic
from .semantic import evaluate_semantic
from .similarity import compute_similarity_analysis

ALL_TASKS = ["gestalt", "mnemonic", "semantic", "similarity"]


def _merge_and_save(unified: dict, results_path: Path) -> None:
    """Merge unified results into existing results.json (incremental save)."""
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f).get("encoders", {})
    else:
        existing = {}
    for enc, data in unified.items():
        if enc not in existing:
            existing[enc] = data
        else:
            for img_type, metrics in data.items():
                if img_type not in existing[enc]:
                    existing[enc][img_type] = metrics
                else:
                    existing[enc][img_type].update(metrics)
    save_results({"encoders": existing}, results_path)


def _print_summary(
    all_gestalt: dict,
    all_mnemonic: dict,
    all_semantic: dict,
) -> None:
    """Print a concise summary table to stdout."""
    levels = get_mask_levels()
    vis_header = "  ".join(f"L{L}({get_visibility_ratio(L):.2f})" for L in levels)

    if all_gestalt:
        print(f"\n  Gestalt IoU:")
        print(f"    {'Encoder':<12}  {vis_header}")
        for enc, vals in all_gestalt.items():
            row = "  ".join(f"{extract_val(vals[L]):.4f}       " for L in levels)
            print(f"    {enc:<12}  {row}")

    if all_mnemonic:
        print(f"\n  Mnemonic Similarity:")
        print(f"    {'Encoder':<12}  {vis_header}")
        for enc, vals in all_mnemonic.items():
            row = "  ".join(
                f"{extract_val(vals['similarity'][L]):.4f}       " for L in levels
            )
            print(f"    {enc:<12}  {row}")

    if all_semantic:
        print(f"\n  Semantic Prototype Acc:")
        print(f"    {'Encoder':<12}  {vis_header}")
        for enc, vals in all_semantic.items():
            row = "  ".join(
                f"{extract_val(vals['prototype_acc'][L]):.4f}       " for L in levels
            )
            print(f"    {enc:<12}  {row}")


def main() -> None:
    """Run fragment completion experiments."""
    parser = argparse.ArgumentParser(description="Exp1: Fragment completion")
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=["clip", "mae", "dino", "ijepa", "vit_sup"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fragment_v2",
        choices=["fragment_v2", "ade20k"],
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--image-type",
        nargs="+",
        type=str,
        default=["original"],
        choices=IMAGE_TYPES,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=ALL_TASKS,
        choices=ALL_TASKS,
    )
    parser.add_argument("--out-dir", type=str, default="results/exp1")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--choices", type=int, default=5)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after experiments finish",
    )
    args = parser.parse_args()

    image_types = args.image_type
    tasks = set(args.tasks)
    out_root = Path(args.out_dir)

    # Pre-load all datasets
    datasets: dict = {}
    for img_type in image_types:
        print(f"Loading dataset: {args.dataset} ({img_type}) ...")
        ds = get_dataset(args.dataset, root=args.data_root, image_type=img_type)
        datasets[img_type] = ds
        n = min(len(ds), args.max_images) if args.max_images else len(ds)
        print(f"  {len(ds)} images, {ds.num_scenes} scenes (using {n})")

    print(f"  Tasks  : {sorted(tasks)}")
    print(f"  Choices: {args.choices}")
    print(f"  Levels : {get_mask_levels()}")
    print()

    # Per image-type accumulators
    results_by_type: dict[str, dict] = {
        img_type: {"gestalt": {}, "mnemonic": {}, "semantic": {}}
        for img_type in image_types
    }
    similarity_by_encoder: dict[str, dict[str, dict]] = {}
    unified: dict[str, dict] = {}

    # Outer loop: encoder (load once), inner loop: image types
    for enc_name in args.encoders:
        print(f"\n{'=' * 60}")
        print(f"  ENCODER: {enc_name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            encoder = get_encoder(enc_name, device=args.device)
            _ = encoder.model
        except Exception as e:
            print(f"  [SKIP] {enc_name}: {e}")
            continue

        display = encoder.name
        print(
            f"  Loaded {display} (dim={encoder.feature_dim}) "
            f"on {args.device} in {time.time() - t0:.1f}s\n"
        )

        unified[display] = {}
        similarity_by_encoder[display] = {}

        for img_type in image_types:
            dataset = datasets[img_type]
            r = results_by_type[img_type]
            unified[display][img_type] = {}

            print(f"\n  --- Image type: {img_type} ---")

            if "gestalt" in tasks:
                print(f"  [gestalt] {display}, {img_type}")
                t1 = time.time()
                gestalt_result = evaluate_gestalt(
                    encoder, dataset, seed=args.seed, max_images=args.max_images,
                )
                r["gestalt"][display] = gestalt_result
                unified[display][img_type]["gestalt_iou"] = gestalt_result
                print(f"  gestalt done in {time.time() - t1:.1f}s\n")

            if "mnemonic" in tasks:
                print(f"  [mnemonic] {display}, {img_type}")
                t2 = time.time()
                mnemonic_result = evaluate_mnemonic(
                    encoder, dataset, seed=args.seed, max_images=args.max_images,
                    num_choices=args.choices,
                )
                r["mnemonic"][display] = mnemonic_result
                unified[display][img_type]["mnemonic_similarity"] = mnemonic_result[
                    "similarity"
                ]
                unified[display][img_type]["mnemonic_retrieval"] = mnemonic_result[
                    "retrieval"
                ]
                unified[display][img_type]["mnemonic_retrieval_r1"] = mnemonic_result[
                    "retrieval_r1"
                ]
                unified[display][img_type]["mnemonic_retrieval_r5"] = mnemonic_result[
                    "retrieval_r5"
                ]
                unified[display][img_type]["mnemonic_retrieval_mrr"] = mnemonic_result[
                    "retrieval_mrr"
                ]
                print(f"  mnemonic done in {time.time() - t2:.1f}s\n")

            if "semantic" in tasks:
                print(f"  [semantic] {display}, {img_type}")
                t3 = time.time()
                semantic_result = evaluate_semantic(
                    encoder, dataset, seed=args.seed, max_images=args.max_images,
                    num_choices=args.choices,
                )
                r["semantic"][display] = semantic_result
                unified[display][img_type]["semantic_prototype"] = semantic_result[
                    "prototype_acc"
                ]
                if "zeroshot_acc" in semantic_result:
                    unified[display][img_type]["semantic_zeroshot"] = semantic_result[
                        "zeroshot_acc"
                    ]
                print(f"  semantic done in {time.time() - t3:.1f}s\n")

            if "similarity" in tasks:
                print(f"  [similarity] {display}, {img_type}")
                t4 = time.time()
                sim_result = compute_similarity_analysis(
                    encoder, dataset, seed=args.seed, max_images=args.max_images,
                )
                similarity_by_encoder[display][img_type] = sim_result
                for key in (
                    "mnemonic_target",
                    "mnemonic_all",
                    "semantic_same_cat",
                    "semantic_all_cat",
                ):
                    unified[display][img_type][f"similarity_{key}"] = sim_result[key]
                print(f"  similarity done in {time.time() - t4:.1f}s\n")

        print(f"  Total for {display}: {time.time() - t0:.1f}s")

        # Incremental save after each encoder
        _merge_and_save(unified, out_root / "results.json")

        del encoder
        torch.cuda.empty_cache()

    # Print summary tables
    for img_type in image_types:
        r = results_by_type[img_type]
        print(f"\n{'#' * 60}")
        print(f"  IMAGE TYPE: {img_type}")
        print(f"{'#' * 60}")
        _print_summary(r["gestalt"], r["mnemonic"], r["semantic"])

    # Save similarity analysis per encoder
    if "similarity" in tasks:
        for enc_display, results_by_img in similarity_by_encoder.items():
            if not results_by_img:
                continue
            enc_dir = results_for_encoder(enc_display, root=out_root)
            enc_dir.mkdir(parents=True, exist_ok=True)
            sim_path = enc_dir / "mnemonic" / "similarity_analysis.json"
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sim_path, "w") as f:
                json.dump(results_by_img, f, indent=2, default=str)
            print(f"  Saved: {sim_path}")

    print(f"\nAll results saved to: {out_root.resolve()}/")

    # Optional plotting
    if args.plot:
        from .plot import plot_from_results

        plot_from_results(
            results_by_type,
            similarity_by_encoder,
            image_types=image_types,
            out_root=out_root,
        )


if __name__ == "__main__":
    main()
