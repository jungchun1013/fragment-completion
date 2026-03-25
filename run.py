"""Run fragment completion experiments: gestalt, mnemonic, semantic, similarity.

Produces a unified results.json with all metrics across encoders and image types.

Usage:
    uv run python run.py
    uv run python run.py --max-images 5                       # quick test
    uv run python run.py --encoders clip mae dino             # specific encoders
    uv run python run.py --image-type original gray lined     # all image types
    uv run python run.py --tasks gestalt mnemonic             # specific tasks only
"""

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
    display_to_dir,
    results_for_encoder,
    results_for_image_type,
)
from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio
from src.gestalt import evaluate_gestalt
from src.mnemonic import evaluate_mnemonic
from src.semantic import evaluate_semantic
from src.similarity import compute_similarity_analysis
from src.utils import (
    extract_val,
    plot_metric_vs_masking,
    plot_completion_summary,
    save_results,
)

ALL_TASKS = ["gestalt", "mnemonic", "semantic", "similarity"]


def _run_plots_and_save(all_gestalt, all_mnemonic, all_semantic, out_dir):
    """Generate plots, save JSON, and print summary for one grouping."""
    levels = get_mask_levels()

    if all_gestalt:
        plot_metric_vs_masking(
            all_gestalt, "IoU",
            "Gestalt Completion (Segmentation IoU)",
            out_dir / "gestalt" / "gestalt_iou.png",
        )
        sil_data = {}
        for enc, vals in all_gestalt.items():
            sil_data[enc] = {
                L: {"mean": vals[L]["silhouette_mean"], "std": vals[L]["silhouette_std"]}
                for L in vals
            }
        plot_metric_vs_masking(
            sil_data, "Silhouette Score",
            "Gestalt Completion (Cluster Separation)",
            out_dir / "gestalt" / "gestalt_silhouette.png",
        )

    if all_mnemonic:
        sim_data = {k: v["similarity"] for k, v in all_mnemonic.items()}
        ret_data = {k: v["retrieval"] for k, v in all_mnemonic.items()}
        plot_metric_vs_masking(
            sim_data, "Cosine Similarity",
            "Mnemonic Completion (Embedding Similarity)",
            out_dir / "mnemonic" / "mnemonic_similarity.png",
        )
        plot_metric_vs_masking(
            ret_data, "Top-1 Accuracy",
            "Mnemonic Completion (Retrieval Accuracy)",
            out_dir / "mnemonic" / "mnemonic_retrieval.png",
        )

    if all_semantic:
        proto_data = {k: v["prototype_acc"] for k, v in all_semantic.items()}
        plot_metric_vs_masking(
            proto_data, "Accuracy",
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
                zs_data, "Accuracy",
                "Semantic Completion (CLIP Zero-shot)",
                out_dir / "semantic" / "semantic_zeroshot.png",
            )

    plot_completion_summary(
        all_gestalt or None,
        all_mnemonic or None,
        all_semantic or None,
        out_dir / "completion_summary.png",
    )

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
            row = "  ".join(f"{extract_val(vals['similarity'][L]):.4f}       " for L in levels)
            print(f"    {enc:<12}  {row}")

    if all_semantic:
        print(f"\n  Semantic Prototype Acc:")
        print(f"    {'Encoder':<12}  {vis_header}")
        for enc, vals in all_semantic.items():
            row = "  ".join(f"{extract_val(vals['prototype_acc'][L]):.4f}       " for L in levels)
            print(f"    {enc:<12}  {row}")


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


def main():
    parser = argparse.ArgumentParser(description="Fragment completion experiments")
    parser.add_argument("--encoders", nargs="+",
                        default=["clip", "mae", "dino", "ijepa", "vit_sup"])
    parser.add_argument("--dataset", type=str, default="fragment_v2",
                        choices=["fragment_v2", "ade20k"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-type", nargs="+", type=str, default=["original"],
                        choices=IMAGE_TYPES)
    parser.add_argument("--tasks", nargs="+", type=str, default=ALL_TASKS,
                        choices=ALL_TASKS)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--choices", type=int, default=5)
    args = parser.parse_args()

    image_types = args.image_type
    tasks = set(args.tasks)
    out_root = Path(args.out_dir)

    # Pre-load all datasets
    datasets = {}
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

    # Per image-type accumulators for completion tasks
    results_by_type: dict[str, dict] = {
        img_type: {"gestalt": {}, "mnemonic": {}, "semantic": {}}
        for img_type in image_types
    }

    # Per encoder × image-type accumulators for similarity
    similarity_by_encoder: dict[str, dict[str, dict]] = {}

    # Unified results dict: {encoder_display: {img_type: {metric: {level: {mean, std}}}}}
    unified: dict[str, dict] = {}

    # Outer loop: encoder (load once), inner loop: image types
    for enc_name in args.encoders:
        print(f"\n{'='*60}")
        print(f"  ENCODER: {enc_name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            encoder = get_encoder(enc_name, device=args.device)
            _ = encoder.model
        except Exception as e:
            print(f"  [SKIP] {enc_name}: {e}")
            continue

        display = encoder.name
        print(f"  Loaded {display} (dim={encoder.feature_dim}) "
              f"on {args.device} in {time.time()-t0:.1f}s\n")

        unified[display] = {}
        similarity_by_encoder[display] = {}

        for img_type in image_types:
            dataset = datasets[img_type]
            r = results_by_type[img_type]
            unified[display][img_type] = {}

            print(f"\n  --- Image type: {img_type} ---")

            # Gestalt
            if "gestalt" in tasks:
                print(f"  [gestalt] {display}, {img_type}")
                t1 = time.time()
                gestalt_result = evaluate_gestalt(
                    encoder, dataset, seed=args.seed, max_images=args.max_images,
                )
                r["gestalt"][display] = gestalt_result
                unified[display][img_type]["gestalt_iou"] = gestalt_result
                print(f"  gestalt done in {time.time()-t1:.1f}s\n")

            # Mnemonic
            if "mnemonic" in tasks:
                print(f"  [mnemonic] {display}, {img_type}")
                t2 = time.time()
                mnemonic_result = evaluate_mnemonic(
                    encoder, dataset, seed=args.seed, max_images=args.max_images,
                    num_choices=args.choices,
                )
                r["mnemonic"][display] = mnemonic_result
                unified[display][img_type]["mnemonic_similarity"] = mnemonic_result["similarity"]
                unified[display][img_type]["mnemonic_retrieval"] = mnemonic_result["retrieval"]
                print(f"  mnemonic done in {time.time()-t2:.1f}s\n")

            # Semantic
            if "semantic" in tasks:
                print(f"  [semantic] {display}, {img_type}")
                t3 = time.time()
                semantic_result = evaluate_semantic(
                    encoder, dataset, seed=args.seed, max_images=args.max_images,
                    num_choices=args.choices,
                )
                r["semantic"][display] = semantic_result
                unified[display][img_type]["semantic_prototype"] = semantic_result["prototype_acc"]
                if "zeroshot_acc" in semantic_result:
                    unified[display][img_type]["semantic_zeroshot"] = semantic_result["zeroshot_acc"]
                print(f"  semantic done in {time.time()-t3:.1f}s\n")

            # Similarity analysis
            if "similarity" in tasks:
                print(f"  [similarity] {display}, {img_type}")
                t4 = time.time()
                sim_result = compute_similarity_analysis(
                    encoder, dataset, seed=args.seed, max_images=args.max_images,
                )
                similarity_by_encoder[display][img_type] = sim_result
                for key in ("mnemonic_target", "mnemonic_all",
                            "semantic_same_cat", "semantic_all_cat"):
                    unified[display][img_type][f"similarity_{key}"] = sim_result[key]
                print(f"  similarity done in {time.time()-t4:.1f}s\n")

        print(f"  Total for {display}: {time.time()-t0:.1f}s")

        # Save results.json immediately after each encoder
        _merge_and_save(unified, out_root / "results.json")

        del encoder
        torch.cuda.empty_cache()

    # Per image-type plots
    for img_type in image_types:
        out_dir = results_for_image_type(img_type, root=out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = results_by_type[img_type]

        print(f"\n{'#'*60}")
        print(f"  IMAGE TYPE: {img_type}")
        print(f"{'#'*60}")
        _run_plots_and_save(r["gestalt"], r["mnemonic"], r["semantic"], out_dir)

    # Per-encoder plots (lines = image types)
    if len(image_types) > 1:
        all_enc_names = set()
        for img_type in image_types:
            for task in ("gestalt", "mnemonic", "semantic"):
                all_enc_names.update(results_by_type[img_type][task].keys())

        for enc_display in sorted(all_enc_names):
            enc_dir = results_for_encoder(enc_display, root=out_root)
            enc_dir.mkdir(parents=True, exist_ok=True)

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
            _run_plots_and_save(enc_gestalt, enc_mnemonic, enc_semantic, enc_dir)

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


if __name__ == "__main__":
    main()
