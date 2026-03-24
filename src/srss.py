#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : srss.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-24-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.

"""SRSS: Semantic-Region Self-Similarity.

Measures whether encoder patch features preserve spatial structure by
comparing cosine similarity of (anchor, positive) vs (anchor, negative)
pairs relative to the ground-truth foreground mask.

  SSM(L) = E_anchor [ cos(f_anchor, f_pos) − cos(f_anchor, f_neg) ]

- Anchor: patches revealed at level 1 (always foreground, ~8% visibility).
- Positive: foreground patches within Manhattan distance ≤ r_near of anchor.
- Negative: background patches at Manhattan distance ≥ r_far from anchor.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

from models.encoder import BaseEncoder

from .masking import (
    get_mask_levels,
    get_visibility_ratio,
    mask_pil_image,
    _find_foreground_patches,
)
from .utils import extract_patch_features, get_encoder_geometry, get_patch_grid_size


def _get_anchor_patches(
    fg_patches: list[tuple[int, int]],
    seed: int,
    idx: int,
) -> list[tuple[int, int]]:
    """Return the patches revealed at level 1 — used as anchors.

    Replicates the same RNG logic as mask_pil_image so anchor set is
    exactly the visible set at L=1.
    """
    shuffled = list(fg_patches)
    rng = random.Random(seed + idx)
    rng.shuffle(shuffled)
    vis_ratio = get_visibility_ratio(1)
    num_visible = max(1, int(vis_ratio * len(shuffled)))
    return shuffled[:num_visible]


@torch.no_grad()
def evaluate_srss(
    encoder: BaseEncoder,
    dataset,
    seed: int = 42,
    max_images: int | None = None,
    num_runs: int = 3,
    r_near: int = 2,
    r_far: int = 4,
) -> dict[int, dict[str, float]]:
    """Evaluate Semantic-Region Self-Similarity per masking level.

    Args:
        encoder: Vision encoder to evaluate.
        dataset: Dataset with image_pil and seg_mask fields.
        seed: Base random seed.
        max_images: Cap on number of images.
        num_runs: Runs with different seeds for variance estimation.
        r_near: Max Manhattan distance for positive pairs.
        r_far: Min Manhattan distance for negative pairs.

    Returns:
        {level: {"mean": float, "std": float}}
    """
    levels = get_mask_levels()
    img_size, patch_size = get_encoder_geometry(encoder)
    gh, gw = get_patch_grid_size(encoder)
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Pre-compute per-image foreground/background patches and anchors per run
    # (these depend only on seg_mask + seed, not on level)
    image_info: list[dict] = []
    for i in range(n):
        sample = dataset[i]
        pil = sample["image_pil"]
        seg_mask = sample["seg_mask"]
        img_np = np.array(pil.resize((img_size, img_size)))
        fg_patches = _find_foreground_patches(img_np, seg_mask, patch_size, img_size)
        fg_set = set(fg_patches)
        bg_patches = [
            (r, c) for r in range(gh) for c in range(gw)
            if (r, c) not in fg_set
        ]
        image_info.append({
            "pil": pil, "seg_mask": seg_mask,
            "fg_patches": fg_patches, "bg_patches": bg_patches,
        })

    ssm_by_level: dict[int, dict[str, float]] = {}

    for L in levels:
        all_ssm: list[float] = []

        for run in range(num_runs):
            seed_run = seed + run

            for i, info in enumerate(image_info):
                fg_patches = info["fg_patches"]
                bg_patches = info["bg_patches"]

                if len(fg_patches) < 2 or not bg_patches:
                    continue

                anchors = _get_anchor_patches(fg_patches, seed_run, i)
                fg_set = set(fg_patches)

                # Extract patch features from masked image at level L
                masked = mask_pil_image(
                    info["pil"], L, info["seg_mask"],
                    seed=seed_run, idx=i,
                    patch_size=patch_size, target_size=img_size,
                )
                patch_feats = extract_patch_features(encoder, masked)  # [gh*gw, D]
                patch_feats = F.normalize(patch_feats.float(), dim=-1)
                feat_grid = patch_feats.reshape(gh, gw, -1)

                for ar, ac in anchors:
                    anchor_feat = feat_grid[ar, ac]  # [D]

                    # Positive: fg patches within Manhattan ≤ r_near (excl. self)
                    pos_feats = [
                        feat_grid[fr, fc]
                        for fr, fc in fg_patches
                        if (fr, fc) != (ar, ac)
                        and abs(fr - ar) + abs(fc - ac) <= r_near
                    ]

                    # Negative: bg patches at Manhattan ≥ r_far
                    neg_feats = [
                        feat_grid[br, bc]
                        for br, bc in bg_patches
                        if abs(br - ar) + abs(bc - ac) >= r_far
                    ]

                    if not pos_feats or not neg_feats:
                        continue

                    cos_pos = (anchor_feat @ torch.stack(pos_feats).T).mean()
                    cos_neg = (anchor_feat @ torch.stack(neg_feats).T).mean()
                    all_ssm.append(float(cos_pos - cos_neg))

        vals = np.array(all_ssm) if all_ssm else np.array([0.0])
        ssm_by_level[L] = {"mean": float(vals.mean()), "std": float(vals.std())}

        vis = get_visibility_ratio(L)
        print(f"    srss [L={L}, vis={vis:.3f}] "
              f"SSM={ssm_by_level[L]['mean']:.4f}±{ssm_by_level[L]['std']:.4f} "
              f"({len(all_ssm)} pairs, {num_runs} runs)")

    return ssm_by_level
