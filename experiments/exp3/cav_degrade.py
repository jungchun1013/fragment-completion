#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cav_degrade.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Degradation curves: CAV ablation effect across masking levels.

Step 3 of the concept-reliance pipeline (after cav_train + cav_ablate).

Tests the core hypothesis: under occlusion, the encoder's reliance on
different concept directions changes. If so, ``double_diff`` curves for
fragile (semantic) concepts should grow toward degraded levels, while
robust (low-level) concepts stay flat.

Pipeline per masking level L (1..8):
    1. Build a cached tensor of all masked images at level L (one shot,
       reused across all CAV ablations to amortize the masking cost).
    2. Compute baseline embeddings + per-category prototypes at level L.
    3. For each trained CAV, project the CAV out of CLS at its best layer
       and re-run forward. Measure pos_drop / neg_drop / double_diff.

Output JSON keys ``results[dim|bin] = [{level, vis, pos_drop, ...}, ...]``
so plotting can group curves per concept across visibility.

Usage:
    uv run python -m experiments.exp3.cav_degrade
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image

from experiments.exp3.cav_train import load_clip, load_concept_labels
from experiments.exp3.cav_ablate import (
    build_prototypes,
    make_cav_hook,
    per_image_correct,
)


# ---------------------------------------------------------------------------
# Cached masked input
# ---------------------------------------------------------------------------


def build_masked_tensor(
    dataset, preprocess, level: int, seed: int, device: str,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """Build [N, 3, 224, 224] tensor of all images masked at *level*.

    Kept on GPU so per-CAV forwards don't pay the H→D transfer cost again.
    """
    n = len(dataset)
    tensors: list[torch.Tensor] = []
    image_ids: list[str] = []
    cat_ids: list[int] = []
    for i in range(n):
        sample = dataset[i]
        masked = mask_pil_image(
            sample["image_pil"], level, sample["seg_mask"],
            seed=seed, idx=i,
        )
        tensors.append(preprocess(masked))
        image_ids.append(sample["image_id"])
        cat_ids.append(int(sample["scene_id"]))
    return torch.stack(tensors).to(device), image_ids, cat_ids


@torch.no_grad()
def encode_cached(
    model: nn.Module,
    cached_input: torch.Tensor,
    batch_size: int,
    hook=None,
    hook_target=None,
) -> torch.Tensor:
    """Run CLIP forward on cached GPU tensor, returning L2-normalized embeds.

    Optional forward hook on ``hook_target`` is installed for the duration
    of the run, then removed.
    """
    handle = None
    if hook is not None and hook_target is not None:
        handle = hook_target.register_forward_hook(hook)

    embeds: list[torch.Tensor] = []
    try:
        for start in range(0, cached_input.shape[0], batch_size):
            end = min(start + batch_size, cached_input.shape[0])
            feats = model.encode_image(cached_input[start:end]).float().cpu()
            embeds.append(F.normalize(feats, dim=-1))
    finally:
        if handle is not None:
            handle.remove()
    return torch.cat(embeds, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_degradation(
    cavs_path: Path,
    out_path: Path,
    dataset_name: str,
    batch_size: int,
    seed: int,
    device: str,
) -> None:
    payload = torch.load(cavs_path, weights_only=False)
    cavs = payload["cavs"]
    model_key = "L-14" if payload["tag"] == "L14" else "B-16"
    print(f"=== CAV degradation curves: {payload['model']} on {dataset_name} ===")
    print(f"  cavs: {cavs_path} ({len(cavs)} CAVs)")

    # 1. Load model + dataset + concept labels
    print("\n[1/3] Loading model, dataset, concept labels")
    model, preprocess, cfg = load_clip(model_key, device)
    dataset = get_dataset(dataset_name)
    image_bins, _ = load_concept_labels(
        Path("data/coco_subset_56/concept_labels.json"),
        Path("data/coco_subset_56/concept_clusters.json"),
    )
    num_cats = dataset.num_scenes
    print(f"  {len(dataset)} images, {num_cats} categories")

    # 2. Per-level loop
    levels = get_mask_levels()  # [1, 2, ..., 8]
    print(f"\n[2/3] Running {len(levels)} levels × {len(cavs)} CAVs "
          f"= {len(levels) * (len(cavs) + 1)} forward passes")

    # results[dim|bin] = list of per-level dicts
    per_cav_curves: dict[str, list[dict]] = {
        f"{d}|{b}": [] for (d, b) in cavs.keys()
    }
    level_summaries: list[dict] = []

    t_start = time.time()
    for li, level in enumerate(levels):
        vis = get_visibility_ratio(level)
        print(f"\n--- Level {level}/{levels[-1]}  vis={vis:.3f} "
              f"[{li + 1}/{len(levels)}] ---")
        t_lvl = time.time()

        # 2a. Build cached masked tensor for this level (on GPU)
        cached, image_ids, cat_ids = build_masked_tensor(
            dataset, preprocess, level, seed, device,
        )
        print(f"  cached input: {tuple(cached.shape)}  "
              f"({(time.time() - t_lvl):.1f}s)")

        # 2b. Baseline embeddings + prototypes for this level
        baseline_embeds = encode_cached(model, cached, batch_size)
        prototypes = build_prototypes(baseline_embeds, cat_ids, num_cats)
        baseline_correct = per_image_correct(
            baseline_embeds, cat_ids, prototypes,
        )
        baseline_acc = float(baseline_correct.mean())
        print(f"  baseline acc: {baseline_acc:.3f}")
        level_summaries.append({
            "level": level, "vis": vis, "baseline_acc": baseline_acc,
        })

        # 2c. Per-CAV ablation
        for ki, (key, cav_data) in enumerate(cavs.items(), 1):
            dim, bin_name = key
            best_layer = int(cav_data["test_acc"].argmax())
            w = cav_data["w"][best_layer].to(device)
            target_block = model.visual.transformer.resblocks[best_layer]
            hook = make_cav_hook(w)

            ablated_embeds = encode_cached(
                model, cached, batch_size,
                hook=hook, hook_target=target_block,
            )
            ablated_correct = per_image_correct(
                ablated_embeds, cat_ids, prototypes,
            )

            pos_mask = torch.tensor([
                image_bins.get(iid, {}).get(dim) == bin_name
                for iid in image_ids
            ])
            neg_mask = torch.tensor([
                (b := image_bins.get(iid, {}).get(dim)) is not None
                and b != bin_name
                for iid in image_ids
            ])
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            b_pos = baseline_correct[pos_mask].mean().item()
            b_neg = baseline_correct[neg_mask].mean().item()
            a_pos = ablated_correct[pos_mask].mean().item()
            a_neg = ablated_correct[neg_mask].mean().item()
            per_cav_curves[f"{dim}|{bin_name}"].append({
                "level": level,
                "vis": vis,
                "best_layer": best_layer,
                "baseline_pos_acc": b_pos,
                "baseline_neg_acc": b_neg,
                "ablated_pos_acc": a_pos,
                "ablated_neg_acc": a_neg,
                "pos_drop": b_pos - a_pos,
                "neg_drop": b_neg - a_neg,
                "double_diff": (b_pos - a_pos) - (b_neg - a_neg),
                "n_pos": int(pos_mask.sum()),
                "n_neg": int(neg_mask.sum()),
            })

        # Free cached input before next level
        del cached
        if device == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t_lvl
        total_elapsed = (time.time() - t_start) / 60
        eta = total_elapsed * (len(levels) - li - 1) / (li + 1)
        print(f"  level done in {elapsed:.0f}s "
              f"(total {total_elapsed:.1f}min, eta {eta:.1f}min)")

    # 3. Save
    out = {
        "model": payload["model"],
        "tag": payload["tag"],
        "dataset": dataset_name,
        "levels": level_summaries,
        "results": per_cav_curves,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[3/3] Saved degradation curves to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CAV degradation sweep across all 8 masking levels.",
    )
    parser.add_argument(
        "--cavs", type=Path,
        default=Path("data/coco_subset_56/cavs/clip_L14.pt"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/exp3/cav_degrade_clip_L14.json"),
    )
    parser.add_argument("--dataset", default="coco_subset_56")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_degradation(
        cavs_path=args.cavs,
        out_path=args.out,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
