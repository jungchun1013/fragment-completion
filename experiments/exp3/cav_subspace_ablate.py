#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cav_subspace_ablate.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Concept-subspace ablation across visibility (rank-k erasure).

Single-direction CAV ablation (cav_degrade.py) had near-zero effect because
distributed encoding spreads concept info across many directions. This
script ablates the FULL subspace spanned by all CAVs of a dimension at
a chosen layer, projecting out the entire concept-related rank-k subspace
in one forward pass.

For each dim ∈ {color, material, function}:
    1. Stack all CAV weights at the chosen ablation layer:  W ∈ R^{k×D}
    2. Orthonormalize via QR → Q ∈ R^{k×D} (rows are orthonormal basis)
    3. Hook layer L: cls' = cls - Q^T (Q cls)
    4. Run categorization → measure acc drop vs unablated baseline

Plus a combined "all dims" condition (rank up to ~45) as the ceiling for
how much overall acc our trained concepts explain.

Usage:
    uv run python -m experiments.exp3.cav_subspace_ablate
    uv run python -m experiments.exp3.cav_subspace_ablate --layer 23
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio

from experiments.exp3.cav_train import load_clip
from experiments.exp3.cav_ablate import build_prototypes, per_image_correct
from experiments.exp3.cav_degrade import build_masked_tensor, encode_cached


DIM_ORDER = ("color", "material", "function")


# ---------------------------------------------------------------------------
# Subspace construction
# ---------------------------------------------------------------------------


def build_subspace(
    cavs: dict, dim: str, layer: int,
) -> torch.Tensor:
    """Stack all CAVs of *dim* at *layer*, return orthonormal basis [k, D].

    Uses QR; the resulting rows of Q span the same subspace as the original
    CAV weight rows but are orthonormal, so projection is just ``Q^T (Q x)``.
    """
    rows: list[torch.Tensor] = []
    for (d, _bin), v in cavs.items():
        if d != dim:
            continue
        rows.append(v["w"][layer].float())
    if not rows:
        return torch.zeros(0, 0)
    W = torch.stack(rows, dim=0)  # [k, D]
    # QR on W^T → W^T = Q' R, where Q' is [D, k] with orthonormal cols
    Q_t, _ = torch.linalg.qr(W.T, mode="reduced")
    return Q_t.T  # [k, D] orthonormal rows


def build_random_subspace(
    rank: int, dim: int, rng: np.random.RandomState,
) -> torch.Tensor:
    """Sample a random rank-k orthonormal subspace in R^dim.

    Uses Gaussian initialization + QR. Match the rank of the concept
    subspace so the comparison controls for "you removed k directions".
    """
    if rank <= 0:
        return torch.zeros(0, dim)
    G = torch.from_numpy(rng.randn(rank, dim).astype(np.float32))
    Q_t, _ = torch.linalg.qr(G.T, mode="reduced")
    return Q_t.T


def make_subspace_hook(Q: torch.Tensor):
    """Hook that projects out a rank-k subspace from the CLS token.

    Args:
        Q: ``[k, D]`` orthonormal rows on the same device as the model.
    """
    def hook(_mod, _inp, out):
        cls = out[:, 0, :].float()  # [B, D]
        coeffs = cls @ Q.T          # [B, k]
        proj = coeffs @ Q           # [B, D]
        out[:, 0, :] = (cls - proj).to(out.dtype)
        return out

    return hook


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_subspace_sweep(
    cavs_path: Path,
    out_path: Path,
    dataset_name: str,
    ablate_layer: int,
    batch_size: int,
    seed: int,
    device: str,
) -> None:
    payload = torch.load(cavs_path, weights_only=False)
    cavs = payload["cavs"]
    model_key = "L-14" if payload["tag"] == "L14" else "B-16"
    print(f"=== Subspace ablation: {payload['model']} on {dataset_name} ===")
    print(f"  ablation layer: L{ablate_layer}")

    # Build per-dim subspaces (rank = number of bins per dim)
    print("\n[1/4] Building per-dim subspaces")
    subspaces: dict[str, torch.Tensor] = {}
    rank_for_random: dict[str, int] = {}
    for dim in DIM_ORDER:
        Q = build_subspace(cavs, dim, ablate_layer)
        subspaces[dim] = Q
        rank_for_random[dim] = int(Q.shape[0])
        print(f"  {dim}: rank={Q.shape[0]} ({Q.shape[0]} CAVs)")

    # Combined "all dims" subspace — concatenate then re-orthonormalize
    combined = torch.cat(
        [subspaces[d] for d in DIM_ORDER if subspaces[d].numel() > 0],
        dim=0,
    )
    if combined.numel() > 0:
        Q_comb_t, _ = torch.linalg.qr(combined.T, mode="reduced")
        subspaces["all"] = Q_comb_t.T
        rank_for_random["all"] = int(subspaces["all"].shape[0])
        print(f"  all: rank={subspaces['all'].shape[0]} (re-orthonormalized)")

    # Random control subspaces — match each concept rank exactly.
    # Sampling once here so the same random subspace is used at every level.
    rng = np.random.RandomState(seed)
    D = next(iter(cavs.values()))["w"].shape[1]
    for cond, k in rank_for_random.items():
        subspaces[f"random_{cond}"] = build_random_subspace(k, D, rng)
        print(f"  random_{cond}: rank={k} (Gaussian + QR control)")

    # Move all subspaces to device
    for k in list(subspaces.keys()):
        subspaces[k] = subspaces[k].to(device)

    # 2. Load model + dataset
    print("\n[2/4] Loading CLIP and dataset")
    model, preprocess, _ = load_clip(model_key, device)
    dataset = get_dataset(dataset_name)
    num_cats = dataset.num_scenes
    print(f"  {len(dataset)} images, {num_cats} categories")

    target_block = model.visual.transformer.resblocks[ablate_layer]

    # 3. Per-level sweep
    levels = get_mask_levels()
    print(f"\n[3/4] Running {len(levels)} levels × "
          f"{1 + len(subspaces)} conditions")

    results = {
        "levels": [],
        "baseline_acc": {},                 # level → float
        "ablated_acc": defaultdict(dict),   # cond → level → float
    }

    t_start = time.time()
    for li, level in enumerate(levels):
        vis = get_visibility_ratio(level)
        print(f"\n--- Level {level}/{levels[-1]}  vis={vis:.3f} ---")
        t_lvl = time.time()

        cached, image_ids, cat_ids = build_masked_tensor(
            dataset, preprocess, level, seed, device,
        )

        # Baseline (no hook)
        baseline_embeds = encode_cached(model, cached, batch_size)
        prototypes = build_prototypes(baseline_embeds, cat_ids, num_cats)
        baseline_correct = per_image_correct(
            baseline_embeds, cat_ids, prototypes,
        )
        baseline_acc = float(baseline_correct.mean())
        results["baseline_acc"][level] = baseline_acc
        print(f"  baseline: {baseline_acc:.3f}")

        # Per-condition ablation: concept first, then matched random control
        cond_order: list[str] = []
        for c in [*DIM_ORDER, "all"]:
            cond_order.append(c)
            cond_order.append(f"random_{c}")

        for cond in cond_order:
            Q = subspaces[cond]
            if Q.numel() == 0:
                continue
            hook = make_subspace_hook(Q)
            ablated_embeds = encode_cached(
                model, cached, batch_size,
                hook=hook, hook_target=target_block,
            )
            ablated_correct = per_image_correct(
                ablated_embeds, cat_ids, prototypes,
            )
            ablated_acc = float(ablated_correct.mean())
            results["ablated_acc"][cond][level] = ablated_acc
            drop = baseline_acc - ablated_acc
            tag = f"  ablate({cond:14s}, rank={Q.shape[0]:2d})"
            print(f"{tag}: {ablated_acc:.3f}  drop={drop:+.4f}")

        results["levels"].append({"level": level, "vis": vis})

        del cached
        if device == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.time() - t_lvl
        total = (time.time() - t_start) / 60
        print(f"  ({elapsed:.0f}s, total {total:.1f}min)")

    # 4. Save
    out = {
        "model": payload["model"],
        "tag": payload["tag"],
        "dataset": dataset_name,
        "ablate_layer": ablate_layer,
        "subspace_ranks": {k: int(v.shape[0]) for k, v in subspaces.items()},
        "levels": results["levels"],
        "baseline_acc": results["baseline_acc"],
        "ablated_acc": {k: dict(v) for k, v in results["ablated_acc"].items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[4/4] Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subspace ablation (rank-k erasure) across visibility.",
    )
    parser.add_argument(
        "--cavs", type=Path,
        default=Path("data/coco_subset_56/cavs/clip_L14.pt"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/exp3/cav_subspace_ablate_clip_L14.json"),
    )
    parser.add_argument("--dataset", default="coco_subset_56")
    parser.add_argument("--layer", type=int, default=23,
                        help="Transformer block index (0-based) to ablate at.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_subspace_sweep(
        cavs_path=args.cavs,
        out_path=args.out,
        dataset_name=args.dataset,
        ablate_layer=args.layer,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
