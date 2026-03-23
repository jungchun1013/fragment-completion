"""Similarity analysis: frag→target vs frag→all for mnemonic & semantic.

For each encoder and image type:
  Mnemonic:  avg cos-sim(frag_i, complete_i)  vs  avg cos-sim(frag_i, complete_all)
  Semantic:  avg cos-sim(frag_i, proto_same)   vs  avg cos-sim(frag_i, proto_all)

Outputs per encoder in results/encoders/{encoder}/:
  similarity_analysis.png   (2-subplot figure)
  similarity_analysis.json  (raw numbers)

Usage:
    cd fragment-completion && source .venv/bin/activate
    uv run python run_similarity_analysis.py
    uv run python run_similarity_analysis.py --encoders dinov2 clip
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
import torch.nn.functional as F

from models.registry import get_encoder
from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image


IMAGE_TYPES = ["original", "gray", "lined"]
IMAGE_TYPE_COLORS = {"original": "#1f77b4", "gray": "#555555", "lined": "#b0b0b0"}

# Map registry name → results dir name
ENCODER_DIR_MAP = {
    "CLIP": "clip",
    "DINO-v1": "dino_v1",
    "DINOv2": "dinov2",
    "MAE": "mae",
    "I-JEPA": "i_jepa",
    "ViT-supervised": "vit_supervised",
}


def _embed_pil(encoder, pil, transform):
    img_t = transform(pil).unsqueeze(0).to(encoder.device)
    feat = encoder.extract_features(img_t)  # [1, D]
    return feat[0].cpu()


@torch.no_grad()
def compute_similarity_analysis(encoder, dataset, seed=42, max_images=None):
    """Compute frag→target and frag→all similarities for mnemonic & semantic."""
    from wrappers.processor import to_transform
    transform = to_transform(encoder.processor)
    levels = get_mask_levels()
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Embed all complete images
    print(f"    Embedding {n} complete images...")
    complete_embeds = []
    cat_ids = []
    for i in range(n):
        sample = dataset[i]
        complete_embeds.append(_embed_pil(encoder, sample["image_pil"], transform))
        cat_ids.append(sample["scene_id"])
    complete_mat = F.normalize(torch.stack(complete_embeds), dim=-1)  # [N, D]

    # Build category prototypes
    num_cats = dataset.num_scenes
    D = complete_mat.shape[1]
    proto_sum = torch.zeros(num_cats, D)
    proto_count = torch.zeros(num_cats)
    for i in range(n):
        proto_sum[cat_ids[i]] += complete_mat[i]
        proto_count[cat_ids[i]] += 1
    proto_count = proto_count.clamp(min=1)
    prototypes = F.normalize(proto_sum / proto_count.unsqueeze(1), dim=-1)  # [C, D]

    mnemonic_target = {}
    mnemonic_all = {}
    semantic_same = {}
    semantic_all = {}

    for L in levels:
        # Embed masked images
        masked_embeds = []
        for i in range(n):
            sample = dataset[i]
            masked = mask_pil_image(sample["image_pil"], L, sample["seg_mask"],
                                    seed=seed, idx=i)
            masked_embeds.append(_embed_pil(encoder, masked, transform))
        masked_mat = F.normalize(torch.stack(masked_embeds), dim=-1)  # [N, D]

        # --- Mnemonic ---
        # frag→target: paired cos sim
        paired_sims = (masked_mat * complete_mat).sum(dim=-1)  # [N]
        # frag→all: each frag vs all complete, then average per frag
        all_sims = masked_mat @ complete_mat.T  # [N, N]
        avg_all_sims = all_sims.mean(dim=1)  # [N]

        mnemonic_target[L] = {"mean": float(paired_sims.mean()), "std": float(paired_sims.std())}
        mnemonic_all[L] = {"mean": float(avg_all_sims.mean()), "std": float(avg_all_sims.std())}

        # --- Semantic ---
        # frag→same category prototype
        same_cat_sims = []
        for i in range(n):
            sim = F.cosine_similarity(masked_mat[i].unsqueeze(0),
                                       prototypes[cat_ids[i]].unsqueeze(0)).item()
            same_cat_sims.append(sim)
        same_cat_arr = np.array(same_cat_sims)

        # frag→all category prototypes (average)
        frag_proto_sims = masked_mat @ prototypes.T  # [N, C]
        avg_proto_sims = frag_proto_sims.mean(dim=1)  # [N]

        semantic_same[L] = {"mean": float(same_cat_arr.mean()), "std": float(same_cat_arr.std())}
        semantic_all[L] = {"mean": float(avg_proto_sims.mean()), "std": float(avg_proto_sims.std())}

        vis = get_visibility_ratio(L)
        print(f"    L={L} vis={vis:.3f} | mnem tgt={mnemonic_target[L]['mean']:.4f} "
              f"all={mnemonic_all[L]['mean']:.4f} | sem same={semantic_same[L]['mean']:.4f} "
              f"all={semantic_all[L]['mean']:.4f}")

    return {
        "mnemonic_target": mnemonic_target,
        "mnemonic_all": mnemonic_all,
        "semantic_same_cat": semantic_same,
        "semantic_all_cat": semantic_all,
    }


def plot_similarity_analysis(results_by_type, encoder_name, save_dir):
    """Draw 2-subplot figure: Mnemonic (left), Semantic (right).

    Each subplot has solid lines (target/same-cat) and dashed lines (all)
    for each image type.
    """
    levels = get_mask_levels()
    x = np.array([get_visibility_ratio(L) for L in levels])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for img_type in IMAGE_TYPES:
        if img_type not in results_by_type:
            continue
        r = results_by_type[img_type]
        color = IMAGE_TYPE_COLORS[img_type]

        # Mnemonic
        y_tgt = np.array([r["mnemonic_target"][L]["mean"] for L in levels])
        y_all = np.array([r["mnemonic_all"][L]["mean"] for L in levels])
        std_tgt = np.array([r["mnemonic_target"][L]["std"] for L in levels])
        std_all = np.array([r["mnemonic_all"][L]["std"] for L in levels])

        ax1.plot(x, y_tgt, marker="o", markersize=4, color=color, linewidth=1.8,
                 label=f"{img_type} → target")
        ax1.fill_between(x, y_tgt - std_tgt, y_tgt + std_tgt, color=color, alpha=0.1)
        ax1.plot(x, y_all, marker="s", markersize=4, color=color, linewidth=1.8,
                 linestyle="--", label=f"{img_type} → all")
        ax1.fill_between(x, y_all - std_all, y_all + std_all, color=color, alpha=0.1)

        # Semantic
        y_same = np.array([r["semantic_same_cat"][L]["mean"] for L in levels])
        y_allc = np.array([r["semantic_all_cat"][L]["mean"] for L in levels])
        std_same = np.array([r["semantic_same_cat"][L]["std"] for L in levels])
        std_allc = np.array([r["semantic_all_cat"][L]["std"] for L in levels])

        ax2.plot(x, y_same, marker="o", markersize=4, color=color, linewidth=1.8,
                 label=f"{img_type} → same cat")
        ax2.fill_between(x, y_same - std_same, y_same + std_same, color=color, alpha=0.1)
        ax2.plot(x, y_allc, marker="s", markersize=4, color=color, linewidth=1.8,
                 linestyle="--", label=f"{img_type} → all cats")
        ax2.fill_between(x, y_allc - std_allc, y_allc + std_allc, color=color, alpha=0.1)

    ax1.set_xlabel("Visibility")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Mnemonic: frag→target vs frag→all")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Visibility")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Semantic: frag→same cat vs frag→all cats")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{encoder_name} — Similarity Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = save_dir / "similarity_analysis.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Similarity analysis: frag→target vs frag→all")
    parser.add_argument("--encoders", nargs="+",
                        default=["clip", "dino", "dinov2", "mae", "ijepa", "vit_sup"])
    parser.add_argument("--image-type", nargs="+", default=IMAGE_TYPES,
                        choices=IMAGE_TYPES)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    # Load datasets
    datasets = {}
    for img_type in args.image_type:
        print(f"Loading dataset: fragment_v2 ({img_type}) ...")
        datasets[img_type] = get_dataset("fragment_v2", image_type=img_type)
        print(f"  {len(datasets[img_type])} images, {datasets[img_type].num_scenes} scenes")

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
        dir_name = ENCODER_DIR_MAP.get(display, display.lower().replace("-", "_"))
        save_dir = Path("results") / "encoders" / dir_name

        print(f"  Loaded {display} (dim={encoder.feature_dim}) in {time.time()-t0:.1f}s")

        results_by_type = {}
        for img_type in args.image_type:
            print(f"\n  --- {img_type} ---")
            results_by_type[img_type] = compute_similarity_analysis(
                encoder, datasets[img_type], seed=args.seed, max_images=args.max_images,
            )

        # Save JSON
        save_dir.mkdir(parents=True, exist_ok=True)
        json_path = save_dir / "similarity_analysis.json"
        with open(json_path, "w") as f:
            json.dump(results_by_type, f, indent=2, default=str)
        print(f"  Saved: {json_path}")

        # Plot
        plot_similarity_analysis(results_by_type, display, save_dir)

        print(f"  Total for {display}: {time.time()-t0:.1f}s")
        del encoder
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
