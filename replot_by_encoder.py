"""Re-generate plots grouped by encoder from existing results.json files.

Reads results/fragment_v2_{original,gray,lined}/results.json and produces
one directory per encoder with lines = image types.

Usage:
    python replot_by_encoder.py
    python replot_by_encoder.py --results-dir results --image-types original gray lined
"""

import argparse
import json
from pathlib import Path

from src.utils import plot_metric_vs_masking, plot_completion_summary


IMAGE_TYPES = ["original", "gray", "lined"]

# Colors for image-type lines
IMAGE_TYPE_COLORS = {
    "original": "#1f77b4",   # blue
    "gray":     "#555555",   # dark gray
    "lined":    "#b0b0b0",   # light gray
}


def main():
    parser = argparse.ArgumentParser(description="Re-plot by encoder from results.json")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="fragment_v2")
    parser.add_argument("--image-types", nargs="+", default=IMAGE_TYPES)
    args = parser.parse_args()

    results_root = Path(args.results_dir)

    # Load all results.json
    # Auto-detect: try exact match first, then with common suffixes
    data_by_type: dict[str, dict] = {}
    for img_type in args.image_types:
        # Try exact, then _1, _2, etc.
        candidates = [
            results_root / f"{args.dataset}_{img_type}" / "results.json",
        ]
        # Also try the raw img_type as a directory suffix
        for suffix in ["", "_1", "_2"]:
            candidates.append(
                results_root / f"{args.dataset}_{img_type}{suffix}" / "results.json"
            )

        path = None
        for c in candidates:
            if c.exists():
                path = c
                break

        if path is None:
            print(f"  [SKIP] No results.json found for {img_type}")
            continue

        # Use clean image type name as the label (strip _1 etc.)
        label = img_type.split("_")[0] if img_type not in ("original", "gray", "lined") else img_type
        with open(path) as f:
            data_by_type[label] = json.load(f)
        print(f"  Loaded {path} -> label={label}")

    if not data_by_type:
        print("No results found.")
        return

    # Collect all encoder names
    all_encoders = set()
    for img_type, data in data_by_type.items():
        for task_data in data.values():
            all_encoders.update(task_data.keys())

    print(f"  Encoders: {sorted(all_encoders)}")
    print(f"  Image types: {list(data_by_type.keys())}")

    # Convert string keys back to int for masking levels
    def _fix_keys(d):
        """Convert string level keys ('1','2',...) to int."""
        if not isinstance(d, dict):
            return d
        try:
            return {int(k): v for k, v in d.items()}
        except (ValueError, TypeError):
            return {k: _fix_keys(v) for k, v in d.items()}

    # Per-encoder plots
    for enc in sorted(all_encoders):
        dir_name = enc.lower().replace("-", "_").replace(" ", "_")
        enc_dir = results_root / dir_name
        enc_dir.mkdir(parents=True, exist_ok=True)

        enc_gestalt = {}
        enc_mnemonic = {}
        enc_semantic = {}

        for img_type, data in data_by_type.items():
            if enc in data.get("gestalt", {}):
                enc_gestalt[img_type] = _fix_keys(data["gestalt"][enc])
            if enc in data.get("mnemonic", {}):
                m = data["mnemonic"][enc]
                enc_mnemonic[img_type] = {
                    "similarity": _fix_keys(m["similarity"]),
                    "retrieval": _fix_keys(m["retrieval"]),
                }
            if enc in data.get("semantic", {}):
                s = data["semantic"][enc]
                enc_semantic[img_type] = {k: _fix_keys(v) for k, v in s.items()}

        print(f"\n  === {enc} -> {enc_dir} ===")

        if enc_gestalt:
            plot_metric_vs_masking(
                enc_gestalt, "IoU",
                f"{enc} — Gestalt Completion (IoU)",
                enc_dir / "gestalt_iou.png",
                colors=IMAGE_TYPE_COLORS,
            )

        if enc_mnemonic:
            sim_data = {k: v["similarity"] for k, v in enc_mnemonic.items()}
            ret_data = {k: v["retrieval"] for k, v in enc_mnemonic.items()}
            plot_metric_vs_masking(
                sim_data, "Cosine Similarity",
                f"{enc} — Mnemonic (Similarity)",
                enc_dir / "mnemonic_similarity.png",
                colors=IMAGE_TYPE_COLORS,
            )
            plot_metric_vs_masking(
                ret_data, "Top-1 Accuracy",
                f"{enc} — Mnemonic (Retrieval)",
                enc_dir / "mnemonic_retrieval.png",
                colors=IMAGE_TYPE_COLORS,
            )

        if enc_semantic:
            proto_data = {k: v["prototype_acc"] for k, v in enc_semantic.items()}
            plot_metric_vs_masking(
                proto_data, "Accuracy",
                f"{enc} — Semantic (Prototype)",
                enc_dir / "semantic_prototype.png",
                colors=IMAGE_TYPE_COLORS,
            )
            zs_data = {k: v["zeroshot_acc"] for k, v in enc_semantic.items()
                       if "zeroshot_acc" in v}
            if zs_data:
                plot_metric_vs_masking(
                    zs_data, "Accuracy",
                    f"{enc} — Semantic (Zero-shot)",
                    enc_dir / "semantic_zeroshot.png",
                    colors=IMAGE_TYPE_COLORS,
                )

        plot_completion_summary(
            enc_gestalt or None,
            enc_mnemonic or None,
            enc_semantic or None,
            enc_dir / "completion_summary.png",
            colors=IMAGE_TYPE_COLORS,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
