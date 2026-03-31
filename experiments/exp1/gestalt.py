"""Gestalt completion: 2-means segmentation on patch features → IoU."""

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from models.encoder import BaseEncoder

from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image
from src.utils import extract_patch_features, get_encoder_geometry


def _segment_iou(
    encoder: BaseEncoder, masked_pil: Image.Image, gt_fg: np.ndarray,
    return_mask: bool = False,
) -> float | tuple[float, ...]:
    """Run 2-means on pixel-level features (upsampled from patches), compute IoU vs GT foreground mask.

    Returns:
        If return_mask=False: (iou, silhouette)
        If return_mask=True:  (iou, silhouette, pred_mask)
    """
    import torch.nn.functional as F

    patch_feats = extract_patch_features(encoder, masked_pil)  # [N, D]
    N, D = patch_feats.shape
    gh = gw = int(round(N ** 0.5))
    if gh * gw != N:
        for h in range(int(N ** 0.5), 0, -1):
            if N % h == 0:
                gh, gw = h, N // h
                break

    H, W = gt_fg.shape

    # Upsample patch features to 28×28 for speed
    upH, upW = 28, 28
    feat_grid = patch_feats.float().reshape(1, gh, gw, D).permute(0, 3, 1, 2)  # [1, D, gh, gw]
    feat_up = F.interpolate(feat_grid, size=(upH, upW), mode="bilinear", align_corners=False)  # [1, D, upH, upW]
    feat_up = feat_up[0].permute(1, 2, 0).numpy()  # [upH, upW, D]
    up_feats = feat_up.reshape(upH * upW, D)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=2)
    labels = kmeans.fit_predict(up_feats)
    sil_score = float(silhouette_score(up_feats, labels))
    label_up = labels.reshape(upH, upW)

    # Nearest-neighbor upsample labels to GT resolution for IoU
    label_img = np.array(
        Image.fromarray(label_up.astype(np.uint8)).resize((W, H), Image.NEAREST)
    )

    best_iou = 0.0
    best_pred = None
    for fg_label in [0, 1]:
        pred_fg = (label_img == fg_label).astype(np.float32)
        intersection = (pred_fg * gt_fg).sum()
        union = ((pred_fg + gt_fg) > 0).sum()
        iou = intersection / (union + 1e-8)
        if iou > best_iou:
            best_iou = iou
            best_pred = pred_fg
    if return_mask:
        return float(best_iou), sil_score, best_pred
    return float(best_iou), sil_score


@torch.no_grad()
def evaluate_gestalt(
    encoder: BaseEncoder,
    dataset,
    seed: int = 42,
    max_images: int | None = None,
    num_choices: int = 5,
    num_runs: int = 3,
) -> dict:
    """Evaluate gestalt completion: mean IoU and silhouette score per masking level.

    Returns:
        {level: {"mean": ..., "std": ..., "silhouette_mean": ..., "silhouette_std": ...}}
    """
    levels = get_mask_levels()
    ious = {L: [] for L in levels}
    sils = {L: [] for L in levels}
    img_size, patch_size = get_encoder_geometry(encoder)

    n = min(len(dataset), max_images) if max_images else len(dataset)
    for L in levels:
        for run in range(num_runs):
            seed_run = seed + run
            for i in range(n):
                sample = dataset[i]
                pil = sample["image_pil"]
                seg_mask = sample["seg_mask"]
                gt_fg = (seg_mask > 0).astype(np.float32)

                masked = mask_pil_image(pil, L, seg_mask, seed=seed_run, idx=i,
                                        patch_size=patch_size, target_size=img_size)
                iou, sil = _segment_iou(encoder, masked, gt_fg)
                ious[L].append(iou)
                sils[L].append(sil)

        vals = f"L{L}: IoU={np.mean(ious[L]):.3f}±{np.std(ious[L]):.3f}"
        print(f"    gestalt {vals} ({num_runs} runs)")

    return {
        L: {
            "mean": float(np.mean(ious[L])),
            "std": float(np.std(ious[L])),
            "silhouette_mean": float(np.mean(sils[L])),
            "silhouette_std": float(np.std(sils[L])),
        }
        for L in levels
    }
