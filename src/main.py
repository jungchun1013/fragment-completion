"""CLI entry point for fragment completion experiments.

Usage:
    cd vision-encoder && source .venv/bin/activate
    uv run python -m completion.main --encoders clip mae dino --tasks all
    uv run python -m completion.main --encoders clip --tasks gestalt --max-images 5
"""

import argparse
from pathlib import Path

import torch

from models.registry import get_encoder

from .dataset import get_dataset
from .gestalt import evaluate_gestalt
from .mnemonic import evaluate_mnemonic
from .semantic import evaluate_semantic
from .masking import get_mask_levels, get_visibility_ratio
from .utils import plot_metric_vs_masking, plot_completion_summary, save_results


def main():
    parser = argparse.ArgumentParser(description="Fragment completion experiment")
    parser.add_argument(
        "--encoders", nargs="+", default=["clip", "mae", "dino"],
        help="Encoder names (from registry)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["all"],
        help="Tasks to run: gestalt, mnemonic, semantic, or all",
    )
    parser.add_argument(
        "--dataset", type=str, default="fragment_v2",
        choices=["fragment_v2", "ade20k"],
        help="Dataset to use (default: fragment_v2)",
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root path")
    parser.add_argument(
        "--image-type", type=str, default="original",
        choices=["original", "gray", "lined"],
        help="Image type for fragment_v2 (default: original)",
    )
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = set(args.tasks)
    if "all" in tasks:
        tasks = {"gestalt", "mnemonic", "semantic"}

    # Load dataset once
    print(f"Loading dataset: {args.dataset}...")
    dataset = get_dataset(args.dataset, root=args.data_root, image_type=args.image_type)
    n = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    print(f"  {len(dataset)} images, {dataset.num_scenes} scenes (using {n})")
    print(f"  Levels: {get_mask_levels()}")
    print(f"  Visibility: {[f'{get_visibility_ratio(L):.3f}' for L in get_mask_levels()]}")

    # Collect results across encoders
    all_gestalt = {}
    all_mnemonic = {}
    all_semantic = {}

    for enc_name in args.encoders:
        print(f"\n{'='*60}")
        print(f"  Encoder: {enc_name}")
        print(f"{'='*60}")
        try:
            encoder = get_encoder(enc_name, device=args.device)
            _ = encoder.model  # trigger lazy load
        except Exception as e:
            print(f"  [SKIP] {enc_name}: {e}")
            continue

        display = encoder.name

        if "gestalt" in tasks:
            print(f"\n  --- Gestalt ({display}) ---")
            all_gestalt[display] = evaluate_gestalt(
                encoder, dataset, seed=args.seed, max_images=args.max_images,
            )

        if "mnemonic" in tasks:
            print(f"\n  --- Mnemonic ({display}) ---")
            all_mnemonic[display] = evaluate_mnemonic(
                encoder, dataset, seed=args.seed, max_images=args.max_images,
            )

        if "semantic" in tasks:
            print(f"\n  --- Semantic ({display}) ---")
            all_semantic[display] = evaluate_semantic(
                encoder, dataset, seed=args.seed, max_images=args.max_images,
            )

        # Free VRAM
        del encoder
        torch.cuda.empty_cache()

    # --- Plots ---
    print(f"\n{'='*60}")
    print("  Generating plots...")
    print(f"{'='*60}")

    if all_gestalt:
        plot_metric_vs_masking(all_gestalt, "IoU", "Gestalt Completion (Segmentation IoU)",
                               out_dir / "gestalt_iou.png")

    if all_mnemonic:
        sim_data = {k: v["similarity"] for k, v in all_mnemonic.items()}
        ret_data = {k: v["retrieval"] for k, v in all_mnemonic.items()}
        plot_metric_vs_masking(sim_data, "Cosine Similarity",
                               "Mnemonic Completion (Embedding Similarity)",
                               out_dir / "mnemonic_similarity.png")
        plot_metric_vs_masking(ret_data, "Top-1 Accuracy",
                               "Mnemonic Completion (Retrieval Accuracy)",
                               out_dir / "mnemonic_retrieval.png")

    if all_semantic:
        proto_data = {k: v["prototype_acc"] for k, v in all_semantic.items()}
        plot_metric_vs_masking(proto_data, "Accuracy",
                               "Semantic Completion (Prototype Classification)",
                               out_dir / "semantic_prototype.png")
        zs_data = {k: v["zeroshot_acc"] for k, v in all_semantic.items()
                   if "zeroshot_acc" in v}
        if zs_data:
            plot_metric_vs_masking(zs_data, "Accuracy",
                                   "Semantic Completion (CLIP Zero-shot)",
                                   out_dir / "semantic_zeroshot.png")

    plot_completion_summary(
        all_gestalt or None,
        all_mnemonic or None,
        all_semantic or None,
        out_dir / "completion_summary.png",
    )

    # --- Save JSON ---
    all_results = {}
    if all_gestalt:
        all_results["gestalt"] = all_gestalt
    if all_mnemonic:
        all_results["mnemonic"] = all_mnemonic
    if all_semantic:
        all_results["semantic"] = all_semantic
    save_results(all_results, out_dir / "results.json")

    # --- Print summary table ---
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for task_name, task_data in all_results.items():
        print(f"\n  {task_name}:")
        for enc_name, values in task_data.items():
            if isinstance(values, dict) and all(isinstance(v, (int, float)) for v in values.values()):
                vals_str = "  ".join(f"L{k}:{v:.4f}" for k, v in values.items())
                print(f"    {enc_name}: {vals_str}")
            else:
                for metric_name, metric_vals in values.items():
                    if isinstance(metric_vals, dict):
                        vals_str = "  ".join(f"L{k}:{v:.4f}" for k, v in metric_vals.items())
                        print(f"    {enc_name}/{metric_name}: {vals_str}")


if __name__ == "__main__":
    main()
