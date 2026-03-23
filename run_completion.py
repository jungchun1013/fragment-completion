"""Run all fragment completion experiments: gestalt, mnemonic, semantic.

Usage:
    cd vision-encoder && source .venv/bin/activate
    uv run python run_completion.py
    uv run python run_completion.py --max-images 5        # quick test
    uv run python run_completion.py --encoders clip       # single encoder
    uv run python run_completion.py --image-type lined gray original  # all types in one run
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure unbuffered output so progress is visible in real-time
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np

from models.registry import get_encoder
from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio
from src.gestalt import evaluate_gestalt
from src.mnemonic import evaluate_mnemonic
from src.semantic import evaluate_semantic
from src.utils import (
    plot_metric_vs_masking,
    plot_completion_summary,
    save_results,
)


def _val(v):
    """Extract mean from a value that's either a float or {"mean": ..., "std": ...}."""
    return v["mean"] if isinstance(v, dict) else v


def _run_plots_and_save(all_gestalt, all_mnemonic, all_semantic, out_dir):
    """Generate plots, save JSON, and print summary for one image type."""
    levels = get_mask_levels()

    print(f"\n{'='*60}")
    print("  GENERATING PLOTS")
    print(f"{'='*60}\n")

    if all_gestalt:
        plot_metric_vs_masking(
            all_gestalt, "IoU",
            "Gestalt Completion (Segmentation IoU)",
            out_dir / "gestalt_iou.png",
        )
        # Silhouette score plot (reformat data for plot_metric_vs_masking)
        sil_data = {}
        for enc, vals in all_gestalt.items():
            sil_data[enc] = {
                L: {"mean": vals[L]["silhouette_mean"], "std": vals[L]["silhouette_std"]}
                for L in vals
            }
        plot_metric_vs_masking(
            sil_data, "Silhouette Score",
            "Gestalt Completion (Cluster Separation)",
            out_dir / "gestalt_silhouette.png",
        )

    if all_mnemonic:
        sim_data = {k: v["similarity"] for k, v in all_mnemonic.items()}
        ret_data = {k: v["retrieval"] for k, v in all_mnemonic.items()}
        plot_metric_vs_masking(
            sim_data, "Cosine Similarity",
            "Mnemonic Completion (Embedding Similarity)",
            out_dir / "mnemonic_similarity.png",
        )
        plot_metric_vs_masking(
            ret_data, "Top-1 Accuracy",
            "Mnemonic Completion (Retrieval Accuracy)",
            out_dir / "mnemonic_retrieval.png",
        )

    if all_semantic:
        proto_data = {k: v["prototype_acc"] for k, v in all_semantic.items()}
        plot_metric_vs_masking(
            proto_data, "Accuracy",
            "Semantic Completion (Prototype Classification)",
            out_dir / "semantic_prototype.png",
        )
        zs_data = {
            k: v["zeroshot_acc"]
            for k, v in all_semantic.items()
            if "zeroshot_acc" in v
        }
        if zs_data:
            plot_metric_vs_masking(
                zs_data, "Accuracy",
                "Semantic Completion (CLIP Zero-shot)",
                out_dir / "semantic_zeroshot.png",
            )

    # Summary figure
    plot_completion_summary(
        all_gestalt or None,
        all_mnemonic or None,
        all_semantic or None,
        out_dir / "completion_summary.png",
    )

    # Save JSON
    all_results = {}
    if all_gestalt:
        all_results["gestalt"] = all_gestalt
    if all_mnemonic:
        all_results["mnemonic"] = all_mnemonic
    if all_semantic:
        all_results["semantic"] = all_semantic
    save_results(all_results, out_dir / "results.json")

    # Summary table
    vis_header = "  ".join(f"L{L}({get_visibility_ratio(L):.2f})" for L in levels)

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")

    if all_gestalt:
        print(f"\n  Gestalt IoU:")
        print(f"    {'Encoder':<12}  {vis_header}")
        for enc, vals in all_gestalt.items():
            row = "  ".join(f"{_val(vals[L]):.4f}       " for L in levels)
            print(f"    {enc:<12}  {row}")

    if all_mnemonic:
        print(f"\n  Mnemonic Similarity:")
        print(f"    {'Encoder':<12}  {vis_header}")
        for enc, vals in all_mnemonic.items():
            row = "  ".join(f"{_val(vals['similarity'][L]):.4f}       " for L in levels)
            print(f"    {enc:<12}  {row}")

        print(f"\n  Mnemonic Retrieval@1:")
        print(f"    {'Encoder':<12}  {vis_header}")
        for enc, vals in all_mnemonic.items():
            row = "  ".join(f"{_val(vals['retrieval'][L]):.4f}       " for L in levels)
            print(f"    {enc:<12}  {row}")

    if all_semantic:
        print(f"\n  Semantic Prototype Acc:")
        print(f"    {'Encoder':<12}  {vis_header}")
        for enc, vals in all_semantic.items():
            row = "  ".join(f"{_val(vals['prototype_acc'][L]):.4f}       " for L in levels)
            print(f"    {enc:<12}  {row}")

        has_zs = [enc for enc, v in all_semantic.items() if "zeroshot_acc" in v]
        if has_zs:
            print(f"\n  Semantic Zero-shot Acc (CLIP):")
            for enc in has_zs:
                vals = all_semantic[enc]
                row = "  ".join(f"{_val(vals['zeroshot_acc'][L]):.4f}       " for L in levels)
                print(f"    {enc:<12}  {row}")

    print(f"\n  Outputs saved to: {out_dir.resolve()}/")
    print()


def main():
    parser = argparse.ArgumentParser(description="Fragment completion experiment")
    parser.add_argument(
        "--encoders", nargs="+", default=["clip", "mae", "dino", "ijepa", "vit_sup"],
        help="Encoder names from registry (default: clip mae dino)",
    )
    parser.add_argument(
        "--dataset", type=str, default="fragment_v2",
        choices=["fragment_v2", "ade20k"],
        help="Dataset to use (default: fragment_v2)",
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root path")
    parser.add_argument(
        "--image-type", nargs="+", type=str, default=["original"],
        choices=["original", "gray", "lined"],
        help="Image type(s) for fragment_v2 (default: original)",
    )
    parser.add_argument("--out-dir", type=str, default=f"results")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images (for quick testing)")
    parser.add_argument("--choices", type=int, default=5,
                        help="Number of candidates in retrieval/classification (default: 5)")
    args = parser.parse_args()

    image_types = args.image_type

    # Pre-load all datasets
    datasets = {}
    for img_type in image_types:
        print(f"Loading dataset: {args.dataset} ({img_type}) ...")
        ds = get_dataset(args.dataset, root=args.data_root, image_type=img_type)
        datasets[img_type] = ds
        n = min(len(ds), args.max_images) if args.max_images else len(ds)
        print(f"  {len(ds)} images, {ds.num_scenes} scenes (using {n})")

    print(f"  Choices: {args.choices}")
    print(f"  Levels : {get_mask_levels()}")
    print(f"  Vis    : {[f'{get_visibility_ratio(L):.3f}' for L in get_mask_levels()]}")
    print()

    # Per image-type accumulators
    results_by_type: dict[str, dict] = {
        img_type: {"gestalt": {}, "mnemonic": {}, "semantic": {}}
        for img_type in image_types
    }

    # ------------------------------------------------------------------ #
    #  Outer loop: encoder (load once), inner loop: image types
    # ------------------------------------------------------------------ #
    for enc_name in args.encoders:
        print(f"\n{'='*60}")
        print(f"  ENCODER: {enc_name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            encoder = get_encoder(enc_name, device=args.device)
            _ = encoder.model  # trigger lazy load
        except Exception as e:
            print(f"  [SKIP] {enc_name}: {e}")
            continue

        display = encoder.name
        print(f"  Loaded {display} (dim={encoder.feature_dim}) "
              f"on {args.device} in {time.time()-t0:.1f}s\n")

        for img_type in image_types:
            dataset = datasets[img_type]
            r = results_by_type[img_type]

            print(f"\n  --- Image type: {img_type} ---")

            # Task 1: Gestalt
            print(f"  --- Task 1/3: Gestalt ({display}, {img_type}) ---")
            t1 = time.time()
            r["gestalt"][display] = evaluate_gestalt(
                encoder, dataset, seed=args.seed, max_images=args.max_images,
                num_choices=args.choices,
            )
            print(f"  Gestalt done in {time.time()-t1:.1f}s\n")

            # Task 2: Mnemonic
            print(f"  --- Task 2/3: Mnemonic ({display}, {img_type}) ---")
            t2 = time.time()
            r["mnemonic"][display] = evaluate_mnemonic(
                encoder, dataset, seed=args.seed, max_images=args.max_images,
                num_choices=args.choices,
            )
            print(f"  Mnemonic done in {time.time()-t2:.1f}s\n")

            # Task 3: Semantic
            print(f"  --- Task 3/3: Semantic ({display}, {img_type}) ---")
            t3 = time.time()
            r["semantic"][display] = evaluate_semantic(
                encoder, dataset, seed=args.seed, max_images=args.max_images,
                num_choices=args.choices,
            )
            print(f"  Semantic done in {time.time()-t3:.1f}s\n")

        print(f"  Total for {display}: {time.time()-t0:.1f}s")

        # Free VRAM before next encoder
        del encoder
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    #  Plots & results per image type (existing)
    # ------------------------------------------------------------------ #
    for img_type in image_types:
        out_dir = Path(args.out_dir) / "image_types" / f"{img_type}"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = results_by_type[img_type]

        print(f"\n{'#'*60}")
        print(f"  IMAGE TYPE: {img_type}")
        print(f"{'#'*60}")

        _run_plots_and_save(
            r["gestalt"], r["mnemonic"], r["semantic"], out_dir,
        )

    # ------------------------------------------------------------------ #
    #  Plots per encoder: one dir per encoder, lines = image types
    # ------------------------------------------------------------------ #
    if len(image_types) > 1:
        # Collect all encoder display names that appeared
        all_enc_names = set()
        for img_type in image_types:
            for task in ("gestalt", "mnemonic", "semantic"):
                all_enc_names.update(results_by_type[img_type][task].keys())

        for enc_display in sorted(all_enc_names):
            enc_dir = Path(args.out_dir) / "encoders" / enc_display.lower().replace("-", "_").replace(" ", "_")
            enc_dir.mkdir(parents=True, exist_ok=True)

            # Re-key: {image_type: values} for this encoder
            enc_gestalt = {}
            enc_mnemonic = {}
            enc_semantic = {}
            for img_type in image_types:
                r = results_by_type[img_type]
                if enc_display in r["gestalt"]:
                    enc_gestalt[img_type] = r["gestalt"][enc_display]
                if enc_display in r["mnemonic"]:
                    enc_mnemonic[img_type] = r["mnemonic"][enc_display]
                if enc_display in r["semantic"]:
                    enc_semantic[img_type] = r["semantic"][enc_display]

            print(f"\n{'#'*60}")
            print(f"  ENCODER: {enc_display}  (lines = image types)")
            print(f"{'#'*60}")

            _run_plots_and_save(
                enc_gestalt, enc_mnemonic, enc_semantic, enc_dir,
            )


if __name__ == "__main__":
    main()
