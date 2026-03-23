"""Single-instance visualizations: gestalt segmentation grids and embedding trajectories.

Subcommands:
    gestalt     Segmentation grid (rows=encoders, cols=masking levels)
    embedding   PCA + t-SNE embedding trajectory
    all         Both

Usage:
    uv run python visualize.py gestalt --image-idx 0 --encoders clip mae dino
    uv run python visualize.py embedding --image-idx 0 --encoders clip mae dino
    uv run python visualize.py all --image-idx 0
"""

import argparse
import os
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models.registry import get_encoder
from src.config import results_visualizations
from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image
from src.utils import extract_patch_features


# ---------------------------------------------------------------------------
# Gestalt visualization
# ---------------------------------------------------------------------------

def _segment_image(encoder, masked_pil, gt_fg):
    """Run 2-means segmentation. Returns (pred_fg mask at GT resolution, IoU)."""
    patch_feats = extract_patch_features(encoder, masked_pil)
    N, D = patch_feats.shape
    gh = gw = int(round(N ** 0.5))
    if gh * gw != N:
        for h in range(int(N ** 0.5), 0, -1):
            if N % h == 0:
                gh, gw = h, N // h
                break

    feats_np = patch_feats.float().numpy()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feats_np)
    label_grid = labels.reshape(gh, gw)

    H, W = gt_fg.shape
    label_img = np.array(
        Image.fromarray(label_grid.astype(np.uint8)).resize((W, H), Image.NEAREST)
    )

    best_iou = 0.0
    best_pred = None
    for fg_label in [0, 1]:
        pred = (label_img == fg_label).astype(np.float32)
        intersection = (pred * gt_fg).sum()
        union = ((pred + gt_fg) > 0).sum()
        iou = intersection / (union + 1e-8)
        if iou > best_iou:
            best_iou = iou
            best_pred = pred

    return best_pred, float(best_iou)


@torch.no_grad()
def visualize_gestalt(encoder_names, dataset, image_idx, out_dir,
                      device="cuda", seed=42):
    """Generate gestalt visualization grid for one image across encoders & levels."""
    from scipy.ndimage import binary_dilation

    levels = get_mask_levels()
    n_levels = len(levels)
    display_size = (224, 224)

    sample = dataset[image_idx]
    pil = sample["image_pil"]
    seg_mask = sample["seg_mask"]
    gt_fg = (seg_mask > 0).astype(np.float32)
    image_id = sample["image_id"]

    gt_resized = np.array(
        Image.fromarray((gt_fg * 255).astype(np.uint8)).resize(display_size, Image.NEAREST)
    ) / 255.0

    enc_results = {}
    for enc_name in encoder_names:
        print(f"  Processing {enc_name}...")
        try:
            encoder = get_encoder(enc_name, device=device)
            _ = encoder.model
        except Exception as e:
            print(f"  [SKIP] {enc_name}: {e}")
            continue

        display = encoder.name
        enc_results[display] = []

        for L in levels:
            masked = mask_pil_image(pil, L, seg_mask, seed=seed, idx=image_idx)
            pred_fg, iou = _segment_image(encoder, masked, gt_fg)

            pred_resized = np.array(
                Image.fromarray((pred_fg * 255).astype(np.uint8)).resize(
                    display_size, Image.NEAREST
                )
            ) / 255.0
            masked_224 = masked.resize(display_size, Image.BILINEAR)

            enc_results[display].append({
                "level": L,
                "masked_img": np.array(masked_224),
                "pred_fg": pred_resized,
                "iou": iou,
            })
            print(f"    L={L} (vis={get_visibility_ratio(L):.0%}) IoU={iou:.3f}")

        del encoder
        torch.cuda.empty_cache()

    if not enc_results:
        print("  No encoders loaded, nothing to plot.")
        return

    # Plot
    n_enc = len(enc_results)
    n_rows = 1 + n_enc
    pred_color = np.array([1.0, 0.2, 0.2])
    gt_color = np.array([0.2, 0.8, 0.2])

    fig, axes = plt.subplots(n_rows, n_levels, figsize=(2.2 * n_levels, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Row 0: GT reference
    for col, L in enumerate(levels):
        ax = axes[0, col]
        masked = mask_pil_image(pil, L, seg_mask, seed=seed, idx=image_idx)
        masked_224 = np.array(masked.resize(display_size, Image.BILINEAR)) / 255.0
        overlay = masked_224.copy()
        boundary = binary_dilation(gt_resized > 0.5, iterations=1) & ~(gt_resized > 0.5)
        overlay[boundary] = gt_color
        ax.imshow(overlay)
        ax.set_title(f"L{L} ({get_visibility_ratio(L):.0%})", fontsize=8, fontweight="bold")
        if col == 0:
            ax.set_ylabel("GT", fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Encoder rows
    for row_idx, (enc_name, results) in enumerate(enc_results.items()):
        for col, res in enumerate(results):
            ax = axes[1 + row_idx, col]
            masked_img = res["masked_img"] / 255.0
            overlay = masked_img.copy()
            pred = res["pred_fg"]
            boundary = binary_dilation(pred > 0.5, iterations=1) & ~(pred > 0.5)
            overlay[boundary] = pred_color
            pred_mask = pred > 0.5
            overlay[pred_mask] = overlay[pred_mask] * 0.6 + pred_color * 0.4
            ax.imshow(overlay)
            ax.set_title(f"IoU={res['iou']:.3f}", fontsize=7)
            if col == 0:
                ax.set_ylabel(enc_name, fontsize=10, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Gestalt Segmentation — Image {image_id}", fontsize=11, fontweight="bold")
    fig.tight_layout()

    save_path = out_dir / f"gestalt_vis_{image_id}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Embedding visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_embeddings(encoder, dataset, image_idx, seed=42):
    """Collect complete embeddings for all images + masked embeddings for one image."""
    from wrappers.processor import to_transform
    transform = to_transform(encoder.processor)
    levels = get_mask_levels()
    n = len(dataset)

    complete_embeds = []
    scene_ids = []
    for i in range(n):
        sample = dataset[i]
        img_t = transform(sample["image_pil"]).unsqueeze(0).to(encoder.device)
        feat = encoder.extract_features(img_t)[0].cpu()
        complete_embeds.append(feat)
        scene_ids.append(sample["scene_id"])

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


def _plot_embedding(ax, coords_ref, coords_masked, scene_ids, target_scene_id,
                    levels, title, show_legend=True):
    """Plot reference cloud + masked trajectory on a single axes."""
    scene_ids_arr = np.array(scene_ids)
    same_mask = scene_ids_arr == target_scene_id
    other_mask = ~same_mask

    ax.scatter(coords_ref[other_mask, 0], coords_ref[other_mask, 1],
               c="lightgray", s=20, alpha=0.5, label="other scenes", zorder=1)
    ax.scatter(coords_ref[same_mask, 0], coords_ref[same_mask, 1],
               c="steelblue", s=40, alpha=0.8, edgecolors="navy",
               linewidths=0.5, label="same scene", zorder=2)

    cmap = cm.get_cmap("YlOrRd", len(levels))
    for i, L in enumerate(levels):
        vis = get_visibility_ratio(L)
        color = cmap(i / (len(levels) - 1))
        marker = "*" if L == len(levels) else "o"
        size = 200 if L == len(levels) else 100
        ax.scatter(coords_masked[i, 0], coords_masked[i, 1],
                   c=[color], s=size, edgecolors="black", linewidths=1.0,
                   marker=marker, zorder=4,
                   label=f"L{L} ({vis:.0%})" if show_legend else None)

    ax.plot(coords_masked[:, 0], coords_masked[:, 1],
            c="red", alpha=0.4, linewidth=1.5, linestyle="--", zorder=3)
    ax.scatter(coords_masked[-1, 0], coords_masked[-1, 1],
               c="red", s=250, marker="*", edgecolors="black",
               linewidths=1.0, zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def visualize_embedding(encoder_names, dataset, image_idx, out_dir,
                        device="cuda", seed=42):
    """Generate t-SNE + PCA embedding plots for each encoder + combined."""
    all_data = {}

    for enc_name in encoder_names:
        print(f"\n  Encoder: {enc_name}")
        try:
            encoder = get_encoder(enc_name, device=device)
            _ = encoder.model
        except Exception as e:
            print(f"  [SKIP] {enc_name}: {e}")
            continue

        print(f"  Collecting embeddings for {encoder.name}...")
        complete_embeds, masked_embeds, scene_ids, levels, target_sid = \
            _collect_embeddings(encoder, dataset, image_idx, seed)

        complete_norm = F.normalize(complete_embeds, dim=-1).numpy()
        masked_norm = F.normalize(masked_embeds, dim=-1).numpy()
        all_embeds = np.concatenate([complete_norm, masked_norm], axis=0)
        n_ref = complete_norm.shape[0]

        pca = PCA(n_components=2, random_state=seed)
        coords_pca = pca.fit_transform(all_embeds)
        var_explained = pca.explained_variance_ratio_

        perplexity = min(30, len(all_embeds) - 1)
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
        coords_tsne = tsne.fit_transform(all_embeds)

        target_label = dataset.samples[image_idx]["scene_label"]
        target_id = dataset.samples[image_idx]["image_id"]

        # Per-encoder plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _plot_embedding(axes[0], coords_pca[:n_ref], coords_pca[n_ref:],
                        scene_ids, target_sid, levels,
                        title=f"{encoder.name} — PCA\n(var: {var_explained[0]:.1%}, {var_explained[1]:.1%})",
                        show_legend=False)
        _plot_embedding(axes[1], coords_tsne[:n_ref], coords_tsne[n_ref:],
                        scene_ids, target_sid, levels,
                        title=f"{encoder.name} — t-SNE", show_legend=True)
        axes[1].legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)
        fig.suptitle(f"Embedding Trajectory — {encoder.name}\n"
                     f"Image: {target_id} ({target_label})",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()

        save_path = out_dir / f"embedding_{encoder.name.lower().replace('-', '_')}_{target_id}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

        all_data[encoder.name] = {
            "pca_ref": coords_pca[:n_ref], "pca_masked": coords_pca[n_ref:],
            "tsne_ref": coords_tsne[:n_ref], "tsne_masked": coords_tsne[n_ref:],
            "scene_ids": scene_ids, "target_sid": target_sid, "levels": levels,
        }

        del encoder
        torch.cuda.empty_cache()

    # Combined figure
    if len(all_data) > 1:
        enc_names = list(all_data.keys())
        n_enc = len(enc_names)
        fig, axes = plt.subplots(2, n_enc, figsize=(6 * n_enc, 10))
        if n_enc == 1:
            axes = axes.reshape(2, 1)

        target_label = dataset.samples[image_idx]["scene_label"]
        target_id = dataset.samples[image_idx]["image_id"]

        for col, enc_name in enumerate(enc_names):
            d = all_data[enc_name]
            _plot_embedding(axes[0, col], d["pca_ref"], d["pca_masked"],
                            d["scene_ids"], d["target_sid"], d["levels"],
                            title=f"{enc_name} — PCA", show_legend=False)
            _plot_embedding(axes[1, col], d["tsne_ref"], d["tsne_masked"],
                            d["scene_ids"], d["target_sid"], d["levels"],
                            title=f"{enc_name} — t-SNE", show_legend=(col == n_enc - 1))

        axes[1, -1].legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)
        fig.suptitle(f"Embedding Trajectories — Image: {target_id} ({target_label})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

        save_path = out_dir / f"embedding_combined_{target_id}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Single-instance visualizations")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("gestalt", "embedding", "all"):
        p = sub.add_parser(name)
        p.add_argument("--encoders", nargs="+",
                        default=["clip", "mae", "dino", "dinov2", "ijepa", "vit_sup"])
        p.add_argument("--image-idx", type=int, default=0)
        p.add_argument("--dataset", type=str, default="fragment_v2",
                        choices=["fragment_v2", "ade20k"])
        p.add_argument("--data-root", type=str, default=None)
        p.add_argument("--image-type", type=str, default="original",
                        choices=["original", "gray", "lined"])
        p.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: results/visualizations)")
        p.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else results_visualizations()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} ({args.image_type})...")
    dataset = get_dataset(args.dataset, root=args.data_root, image_type=args.image_type)
    print(f"  {len(dataset)} images")

    idx = args.image_idx
    if idx >= len(dataset):
        print(f"  ERROR: image-idx {idx} out of range (max {len(dataset)-1})")
        return
    sample = dataset[idx]
    print(f"  Selected: idx={idx}, id={sample['image_id']}")

    cmd = args.command
    if cmd in ("gestalt", "all"):
        print("\n--- Gestalt visualization ---")
        visualize_gestalt(args.encoders, dataset, idx, out_dir,
                          device=args.device, seed=args.seed)
    if cmd in ("embedding", "all"):
        print("\n--- Embedding visualization ---")
        visualize_embedding(args.encoders, dataset, idx, out_dir,
                            device=args.device, seed=args.seed)

    print("\nDone!")


if __name__ == "__main__":
    main()
