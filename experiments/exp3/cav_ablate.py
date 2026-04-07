#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cav_ablate.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Concept-influence ranking via CAV ablation @ full visibility.

Step 2 of the concept-reliance pipeline (after cav_train.py).

For each trained CAV (concept_bin), project the CAV direction out of the
CLS token at its best layer, run the rest of the model, and measure how
much category classification accuracy drops on:
  - positive instances (this concept applies)  : pos_drop
  - negative instances (different concept bin) : neg_drop

The double-difference ``pos_drop - neg_drop`` isolates the
concept-specific contribution: if it's large and positive, the model
genuinely uses that concept direction to recognize those category instances.

Categorization is C-way over the dataset's category labels (e.g., 56 cats
in coco_subset_56), with prototypes built from the BASELINE (un-ablated)
embeddings — frozen across ablations to give a clean reference signal.

Usage:
    uv run python -m experiments.exp3.cav_ablate
    uv run python -m experiments.exp3.cav_ablate --model L-14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataset

from experiments.exp3.cav_train import (
    MODEL_CONFIGS,
    load_clip,
    load_concept_labels,
)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_dataset(
    model: nn.Module,
    dataset,
    preprocess,
    batch_size: int,
    device: str,
    hook=None,
    hook_target=None,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """Encode all images, optionally with a forward hook installed.

    Args:
        model: OpenCLIP model.
        dataset: dataset with __getitem__ returning {image_pil, image_id, scene_id}.
        preprocess: preprocessing transform.
        batch_size: batch size.
        device: cuda/cpu.
        hook: optional ``(mod, inp, out) -> tensor`` to install on ``hook_target``.
        hook_target: ``nn.Module`` on which to install the hook.

    Returns:
        (embeds ``[N, D_proj]`` L2-normalized cpu, image_ids, cat_ids)
    """
    handle = None
    if hook is not None and hook_target is not None:
        handle = hook_target.register_forward_hook(hook)

    embeds: list[torch.Tensor] = []
    image_ids: list[str] = []
    cat_ids: list[int] = []

    try:
        n = len(dataset)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = []
            for i in range(start, end):
                sample = dataset[i]
                batch_imgs.append(preprocess(sample["image_pil"]))
                image_ids.append(sample["image_id"])
                cat_ids.append(int(sample["scene_id"]))
            x = torch.stack(batch_imgs).to(device)
            feats = model.encode_image(x).float().cpu()
            feats = F.normalize(feats, dim=-1)
            embeds.append(feats)
    finally:
        if handle is not None:
            handle.remove()

    return torch.cat(embeds, dim=0), image_ids, cat_ids


def make_cav_hook(cav: torch.Tensor):
    """Create a hook that projects out a CAV direction from the CLS token.

    The hook subtracts the CAV component of the CLS token in the residual
    stream after the targeted block. The CAV is L2-normalized once at hook
    construction so the projection is just an inner product.
    """
    cav_unit = F.normalize(cav.float(), dim=0)  # [D]

    def hook(_mod, _inp, out):
        # out: [B, T, D] (resblock residual stream)
        cls = out[:, 0, :].float()
        coef = cls @ cav_unit  # [B]
        out[:, 0, :] = (cls - coef.unsqueeze(-1) * cav_unit).to(out.dtype)
        return out

    return hook


# ---------------------------------------------------------------------------
# Categorization metrics
# ---------------------------------------------------------------------------


def build_prototypes(
    embeds: torch.Tensor, cat_ids: list[int], num_cats: int,
) -> torch.Tensor:
    """Per-category mean embedding (L2-normalized). Shape ``[C, D]``."""
    cat_t = torch.tensor(cat_ids)
    D = embeds.shape[1]
    proto = torch.zeros(num_cats, D)
    counts = torch.zeros(num_cats)
    for c in range(num_cats):
        mask = cat_t == c
        if mask.any():
            proto[c] = embeds[mask].mean(dim=0)
            counts[c] = mask.sum()
    proto = F.normalize(proto, dim=-1)
    return proto


def per_image_correct(
    embeds: torch.Tensor, cat_ids: list[int], prototypes: torch.Tensor,
) -> torch.Tensor:
    """Per-image 1/0 indicating whether C-way argmax matches GT category."""
    sims = embeds @ prototypes.T  # [N, C]
    pred = sims.argmax(dim=1)
    target = torch.tensor(cat_ids)
    return (pred == target).float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_ablation(
    cavs_path: Path,
    out_path: Path,
    dataset_name: str,
    batch_size: int,
    device: str,
) -> None:
    payload = torch.load(cavs_path, weights_only=False)
    cavs = payload["cavs"]
    model_key = "L-14" if payload["tag"] == "L14" else "B-16"
    print(f"=== CAV ablation: {payload['model']} on {dataset_name} ===")
    print(f"  cavs: {cavs_path}")
    print(f"  {len(cavs)} CAVs to evaluate")

    # 1. Load model + dataset
    print("\n[1/4] Loading CLIP and dataset")
    model, preprocess, cfg = load_clip(model_key, device)
    dataset = get_dataset(dataset_name)
    num_cats = dataset.num_scenes
    print(f"  {len(dataset)} images, {num_cats} categories")

    # 2. Load concept labels (need image-id → bin mapping for pos/neg split)
    print("\n[2/4] Loading concept labels")
    image_bins, _ = load_concept_labels(
        Path("data/coco_subset_56/concept_labels.json"),
        Path("data/coco_subset_56/concept_clusters.json"),
    )

    # 3. Baseline embeddings + prototypes (frozen)
    print("\n[3/4] Computing baseline embeddings and prototypes")
    baseline_embeds, image_ids, cat_ids = encode_dataset(
        model, dataset, preprocess, batch_size, device,
    )
    prototypes = build_prototypes(baseline_embeds, cat_ids, num_cats)
    baseline_correct = per_image_correct(baseline_embeds, cat_ids, prototypes)
    print(f"  baseline overall acc: {baseline_correct.mean().item():.3f}")
    print(f"  baseline embeds shape: {tuple(baseline_embeds.shape)}")

    # 4. Ablate per CAV at its best layer
    print(f"\n[4/4] Running ablation forward passes ({len(cavs)} runs)")
    results: dict[tuple[str, str], dict] = {}
    for k, (key, cav_data) in enumerate(cavs.items(), 1):
        dim, bin_name = key
        test_acc_curve = cav_data["test_acc"]
        best_layer = int(test_acc_curve.argmax())
        w = cav_data["w"][best_layer]  # [D]

        # Hook the residual stream after best_layer's resblock
        target_block = model.visual.transformer.resblocks[best_layer]
        hook = make_cav_hook(w.to(device))

        ablated_embeds, _, _ = encode_dataset(
            model, dataset, preprocess, batch_size, device,
            hook=hook, hook_target=target_block,
        )
        ablated_correct = per_image_correct(ablated_embeds, cat_ids, prototypes)

        # Split images by concept membership for this dim
        pos_mask = torch.tensor([
            image_bins.get(iid, {}).get(dim) == bin_name for iid in image_ids
        ])
        neg_mask = torch.tensor([
            (b := image_bins.get(iid, {}).get(dim)) is not None and b != bin_name
            for iid in image_ids
        ])

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            print(f"  [{k}/{len(cavs)}] [{dim}/{bin_name}] SKIP — empty pos or neg")
            continue

        b_pos = baseline_correct[pos_mask].mean().item()
        b_neg = baseline_correct[neg_mask].mean().item()
        a_pos = ablated_correct[pos_mask].mean().item()
        a_neg = ablated_correct[neg_mask].mean().item()
        pos_drop = b_pos - a_pos
        neg_drop = b_neg - a_neg
        double_diff = pos_drop - neg_drop

        results[key] = {
            "best_layer": best_layer,
            "best_test_acc": float(test_acc_curve[best_layer]),
            "baseline_pos_acc": b_pos,
            "baseline_neg_acc": b_neg,
            "ablated_pos_acc": a_pos,
            "ablated_neg_acc": a_neg,
            "pos_drop": pos_drop,
            "neg_drop": neg_drop,
            "double_diff": double_diff,
            "n_pos": int(pos_mask.sum()),
            "n_neg": int(neg_mask.sum()),
        }
        print(f"  [{k:2d}/{len(cavs)}] [{dim}/{bin_name}] L{best_layer:2d}  "
              f"pos_drop={pos_drop:+.3f}  neg_drop={neg_drop:+.3f}  "
              f"Δ={double_diff:+.3f}")

    # 5. Save
    out = {
        "model": payload["model"],
        "tag": payload["tag"],
        "dataset": dataset_name,
        "baseline_overall_acc": float(baseline_correct.mean()),
        "results": {f"{d}|{b}": v for (d, b), v in results.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(results)} ablation results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CAV ablation @ full visibility, rank concept influence.",
    )
    parser.add_argument(
        "--cavs", type=Path,
        default=Path("data/coco_subset_56/cavs/clip_L14.pt"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/exp3/cav_ablation_full_vis_clip_L14.json"),
    )
    parser.add_argument("--dataset", default="coco_subset_56")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_ablation(
        cavs_path=args.cavs,
        out_path=args.out,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
