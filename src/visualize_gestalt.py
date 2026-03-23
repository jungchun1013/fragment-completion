"""Visualize gestalt segmentation: masked input, predicted mask, and GT per level.

For a selected image, shows a grid:
  rows = encoders, columns = masking levels (L1..L8)
  each cell = masked input overlaid with predicted segmentation contour

Plus a GT reference row.

Usage:
    uv run python -m completion.visualize_gestalt
    uv run python -m completion.visualize_gestalt --image-idx 0 --image-type lined
    uv run python -m completion.visualize_gestalt --image-idx 42 --encoders clip dino
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from models.registry import get_encoder
from wrappers.encoder import BaseEncoder
from tasks.fragment_segmentation import _extract_patch_features

from .dataset import get_dataset
from .masking import get_mask_levels, get_visibility_ratio, mask_pil_image

# Flush prints immediately
sys.stdout.reconfigure(line_buffering=True)


def _segment_image(
    encoder: BaseEncoder, masked_pil: Image.Image, gt_fg: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run 2-means segmentation. Returns (pred_fg mask at GT resolution, IoU)."""
    patch_feats = _extract_patch_features(encoder, masked_pil)  # [N, D]
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

    # Pick best cluster assignment
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
def visualize_gestalt(
    encoder_names: list[str],
    dataset,
    image_idx: int,
    out_dir: Path,
    device: str = "cuda",
    seed: int = 42,
):
    """Generate gestalt visualization grid for one image across encoders & levels."""
    levels = get_mask_levels()
    n_levels = len(levels)

    sample = dataset[image_idx]
    pil = sample["image_pil"]
    seg_mask = sample["seg_mask"]
    gt_fg = (seg_mask > 0).astype(np.float32)
    image_id = sample["image_id"]

    # Resize GT for display
    display_size = (224, 224)
    gt_resized = np.array(
        Image.fromarray((gt_fg * 255).astype(np.uint8)).resize(display_size, Image.NEAREST)
    ) / 255.0

    # Collect results per encoder
    enc_results: dict[str, list[dict]] = {}

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

            # Resize pred for display
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
            vis = get_visibility_ratio(L)
            print(f"    L={L} (vis={vis:.0%}) IoU={iou:.3f}")

        del encoder
        torch.cuda.empty_cache()

    if not enc_results:
        print("  No encoders loaded, nothing to plot.")
        return

    # --- Plot ---
    n_enc = len(enc_results)
    # Rows: GT row + one row per encoder; Cols: one per level
    n_rows = 1 + n_enc  # GT + encoders
    n_cols = n_levels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Color overlays
    pred_color = np.array([1.0, 0.2, 0.2])   # red for prediction
    gt_color = np.array([0.2, 0.8, 0.2])      # green for GT

    # Row 0: GT reference (same across all levels, but show masked input + GT overlay)
    for col, L in enumerate(levels):
        ax = axes[0, col]
        masked = mask_pil_image(pil, L, seg_mask, seed=seed, idx=image_idx)
        masked_224 = np.array(masked.resize(display_size, Image.BILINEAR)) / 255.0

        # Overlay GT contour
        overlay = masked_224.copy()
        # GT boundary
        from scipy.ndimage import binary_dilation
        boundary = binary_dilation(gt_resized > 0.5, iterations=1) & ~(gt_resized > 0.5)
        overlay[boundary] = gt_color

        ax.imshow(overlay)
        vis = get_visibility_ratio(L)
        ax.set_title(f"L{L} ({vis:.0%})", fontsize=8, fontweight="bold")
        if col == 0:
            ax.set_ylabel("GT", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Encoder rows: show masked input + predicted segmentation overlay
    for row_idx, (enc_name, results) in enumerate(enc_results.items()):
        for col, res in enumerate(results):
            ax = axes[1 + row_idx, col]
            masked_img = res["masked_img"] / 255.0

            # Overlay predicted segmentation
            overlay = masked_img.copy()
            pred = res["pred_fg"]
            boundary = binary_dilation(pred > 0.5, iterations=1) & ~(pred > 0.5)
            overlay[boundary] = pred_color

            # Also show filled prediction with transparency
            pred_mask = pred > 0.5
            overlay[pred_mask] = overlay[pred_mask] * 0.6 + pred_color * 0.4

            ax.imshow(overlay)
            ax.set_title(f"IoU={res['iou']:.3f}", fontsize=7)
            if col == 0:
                ax.set_ylabel(enc_name, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Gestalt Segmentation — Image {image_id}\n"
        f"Green=GT boundary, Red=predicted foreground",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    save_path = out_dir / f"gestalt_vis_{image_id}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize gestalt segmentation per level and encoder"
    )
    parser.add_argument(
        "--encoders", nargs="+", default=["clip", "mae", "dino"],
    )
    parser.add_argument("--image-idx", type=int, default=0)
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

    print(f"Loading dataset: {args.dataset} ({args.image_type})...")
    dataset = get_dataset(args.dataset, root=args.data_root, image_type=args.image_type)
    print(f"  {len(dataset)} images")

    idx = args.image_idx
    if idx >= len(dataset):
        print(f"  ERROR: image-idx {idx} out of range (max {len(dataset)-1})")
        return

    sample = dataset[idx]
    print(f"  Selected: idx={idx}, id={sample['image_id']}")

    visualize_gestalt(
        args.encoders, dataset, idx,
        Path(args.out_dir), device=args.device, seed=args.seed,
    )


if __name__ == "__main__":
    main()
