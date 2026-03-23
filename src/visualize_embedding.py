"""Visualize fragment-level embeddings via t-SNE and PCA.

For a selected image, plots embeddings at all 8 masking levels alongside
all complete-image embeddings as reference context.

Usage:
    uv run python -m completion.visualize_embedding
    uv run python -m completion.visualize_embedding --image-idx 42
    uv run python -m completion.visualize_embedding --encoders clip dino
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models.registry import get_encoder
from .dataset import get_dataset
from .masking import get_mask_levels, get_visibility_ratio, mask_pil_image


@torch.no_grad()
def _collect_embeddings(encoder, dataset, image_idx: int, seed: int = 42):
    """Collect complete embeddings for all images + masked embeddings for one image.

    Returns:
        complete_embeds: [N, D] all complete image embeddings
        masked_embeds:   [8, D] selected image at each masking level
        scene_ids:       [N]   scene id per image
        levels:          [8]   masking levels
        target_scene_id: int   scene id of selected image
    """
    from wrappers.processor import to_transform
    transform = to_transform(encoder.processor)
    levels = get_mask_levels()
    n = len(dataset)

    # Embed all complete images
    complete_embeds = []
    scene_ids = []
    for i in range(n):
        sample = dataset[i]
        img_t = transform(sample["image_pil"]).unsqueeze(0).to(encoder.device)
        feat = encoder.extract_features(img_t)[0].cpu()
        complete_embeds.append(feat)
        scene_ids.append(sample["scene_id"])

    # Embed selected image at each masking level
    target_sample = dataset[image_idx]
    target_scene_id = target_sample["scene_id"]
    masked_embeds = []
    for L in levels:
        masked_pil = mask_pil_image(
            target_sample["image_pil"], L, target_sample["seg_mask"],
            seed=seed, idx=image_idx,
        )
        img_t = transform(masked_pil).unsqueeze(0).to(encoder.device)
        feat = encoder.extract_features(img_t)[0].cpu()
        masked_embeds.append(feat)

    return (
        torch.stack(complete_embeds),
        torch.stack(masked_embeds),
        scene_ids,
        levels,
        target_scene_id,
    )


def _plot_embedding(
    ax,
    coords_ref,       # [N, 2]
    coords_masked,    # [8, 2]
    scene_ids,
    target_scene_id,
    levels,
    title: str,
    show_legend: bool = True,
):
    """Plot reference cloud + masked trajectory on a single axes."""
    scene_ids_arr = np.array(scene_ids)

    # Reference: same-scene vs other-scene
    same_mask = scene_ids_arr == target_scene_id
    other_mask = ~same_mask

    ax.scatter(
        coords_ref[other_mask, 0], coords_ref[other_mask, 1],
        c="lightgray", s=20, alpha=0.5, label="other scenes", zorder=1,
    )
    ax.scatter(
        coords_ref[same_mask, 0], coords_ref[same_mask, 1],
        c="steelblue", s=40, alpha=0.8, edgecolors="navy",
        linewidths=0.5, label="same scene", zorder=2,
    )

    # Masked trajectory: color by level
    cmap = cm.get_cmap("YlOrRd", len(levels))
    for i, L in enumerate(levels):
        vis = get_visibility_ratio(L)
        color = cmap(i / (len(levels) - 1))
        marker = "*" if L == len(levels) else "o"
        size = 200 if L == len(levels) else 100
        ax.scatter(
            coords_masked[i, 0], coords_masked[i, 1],
            c=[color], s=size, edgecolors="black", linewidths=1.0,
            marker=marker, zorder=4,
            label=f"L{L} ({vis:.0%})" if show_legend else None,
        )

    # Draw trajectory line connecting masked points
    ax.plot(
        coords_masked[:, 0], coords_masked[:, 1],
        c="red", alpha=0.4, linewidth=1.5, linestyle="--", zorder=3,
    )

    # Mark complete embedding (L=8) with a star
    ax.scatter(
        coords_masked[-1, 0], coords_masked[-1, 1],
        c="red", s=250, marker="*", edgecolors="black",
        linewidths=1.0, zorder=5,
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_single_encoder(
    encoder,
    dataset,
    image_idx: int,
    out_dir: Path,
    seed: int = 42,
):
    """Generate t-SNE + PCA plot for one encoder."""
    enc_name = encoder.name
    print(f"  Collecting embeddings for {enc_name}...")

    complete_embeds, masked_embeds, scene_ids, levels, target_sid = \
        _collect_embeddings(encoder, dataset, image_idx, seed)

    # Normalize embeddings
    complete_norm = F.normalize(complete_embeds, dim=-1).numpy()
    masked_norm = F.normalize(masked_embeds, dim=-1).numpy()

    # Combine for joint projection
    all_embeds = np.concatenate([complete_norm, masked_norm], axis=0)  # [N+8, D]
    n_ref = complete_norm.shape[0]
    n_masked = masked_norm.shape[0]

    # --- PCA ---
    pca = PCA(n_components=2, random_state=seed)
    coords_pca = pca.fit_transform(all_embeds)
    pca_ref = coords_pca[:n_ref]
    pca_masked = coords_pca[n_ref:]
    var_explained = pca.explained_variance_ratio_

    # --- t-SNE ---
    perplexity = min(30, len(all_embeds) - 1)
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    coords_tsne = tsne.fit_transform(all_embeds)
    tsne_ref = coords_tsne[:n_ref]
    tsne_masked = coords_tsne[n_ref:]

    # Plot: 1 row, 2 columns (PCA | t-SNE)
    target_label = dataset.samples[image_idx]["scene_label"]
    target_id = dataset.samples[image_idx]["image_id"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _plot_embedding(
        axes[0], pca_ref, pca_masked, scene_ids, target_sid, levels,
        title=f"{enc_name} — PCA\n(var: {var_explained[0]:.1%}, {var_explained[1]:.1%})",
        show_legend=False,
    )
    _plot_embedding(
        axes[1], tsne_ref, tsne_masked, scene_ids, target_sid, levels,
        title=f"{enc_name} — t-SNE",
        show_legend=True,
    )
    axes[1].legend(
        fontsize=7, loc="upper right", ncol=2,
        framealpha=0.9, borderpad=0.5,
    )

    fig.suptitle(
        f"Fragment Completion Trajectory — {enc_name}\n"
        f"Image: {target_id} ({target_label}), {len(scene_ids)} reference embeddings",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    save_path = out_dir / f"embedding_{enc_name.lower().replace('-', '_')}_{target_id}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {
        "pca_ref": pca_ref, "pca_masked": pca_masked,
        "tsne_ref": tsne_ref, "tsne_masked": tsne_masked,
        "scene_ids": scene_ids, "target_sid": target_sid,
        "levels": levels,
    }


def visualize_combined(
    all_data: dict[str, dict],
    dataset,
    image_idx: int,
    out_dir: Path,
):
    """Generate combined figure: 2 rows (PCA, t-SNE) × N encoder columns."""
    enc_names = list(all_data.keys())
    n_enc = len(enc_names)

    fig, axes = plt.subplots(2, n_enc, figsize=(6 * n_enc, 10))
    if n_enc == 1:
        axes = axes.reshape(2, 1)

    target_label = dataset.samples[image_idx]["scene_label"]
    target_id = dataset.samples[image_idx]["image_id"]

    for col, enc_name in enumerate(enc_names):
        d = all_data[enc_name]

        # PCA row
        _plot_embedding(
            axes[0, col], d["pca_ref"], d["pca_masked"],
            d["scene_ids"], d["target_sid"], d["levels"],
            title=f"{enc_name} — PCA",
            show_legend=False,
        )

        # t-SNE row
        _plot_embedding(
            axes[1, col], d["tsne_ref"], d["tsne_masked"],
            d["scene_ids"], d["target_sid"], d["levels"],
            title=f"{enc_name} — t-SNE",
            show_legend=(col == n_enc - 1),
        )

    # Add legend to last t-SNE panel
    axes[1, -1].legend(
        fontsize=7, loc="upper right", ncol=2,
        framealpha=0.9, borderpad=0.5,
    )

    fig.suptitle(
        f"Fragment Completion Embedding Trajectories\n"
        f"Image: {target_id} ({target_label})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    save_path = out_dir / f"embedding_combined_{target_id}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize fragment-level embeddings in t-SNE and PCA"
    )
    parser.add_argument(
        "--encoders", nargs="+", default=["clip", "mae", "dino"],
        help="Encoder names",
    )
    parser.add_argument(
        "--image-idx", type=int, default=0,
        help="Image index in dataset (default: 0 = first image)",
    )
    parser.add_argument(
        "--dataset", type=str, default="fragment_v2",
        choices=["fragment_v2", "ade20k"],
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--image-type", type=str, default="original",
        choices=["original", "gray", "lined"],
    )
    parser.add_argument("--out-dir", type=str, default="output/completion")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print(f"Loading dataset: {args.dataset}...")
    dataset = get_dataset(args.dataset, root=args.data_root, image_type=args.image_type)
    print(f"  {len(dataset)} images, {dataset.num_scenes} scenes")

    idx = args.image_idx
    if idx >= len(dataset):
        print(f"  ERROR: image-idx {idx} out of range (max {len(dataset)-1})")
        return
    sample = dataset[idx]
    print(f"  Selected: idx={idx}, id={sample['image_id']}, scene={sample['scene_label']}")

    all_data = {}

    for enc_name in args.encoders:
        print(f"\n{'='*50}")
        print(f"  Encoder: {enc_name}")
        print(f"{'='*50}")

        try:
            encoder = get_encoder(enc_name, device=args.device)
            _ = encoder.model
        except Exception as e:
            print(f"  [SKIP] {enc_name}: {e}")
            continue

        data = visualize_single_encoder(
            encoder, dataset, idx, out_dir, seed=args.seed,
        )
        all_data[encoder.name] = data

        del encoder
        torch.cuda.empty_cache()

    # Combined figure
    if len(all_data) > 1:
        print(f"\n  Generating combined figure...")
        visualize_combined(all_data, dataset, idx, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
