#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cav_subspace_retrieval.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Concept-subspace ablation: image vs text retrieval drop comparison.

Hypothesis: if function (semantic) is encoded in CLIP's later layers, then
ablating the function subspace should hurt TEXT retrieval (semantic matching
against category names) more than IMAGE retrieval (visual similarity to
per-category image prototypes).

Same setup as cav_subspace_ablate.py, but at each level + condition we
report TWO category-classification metrics:
    image_r1: query @ image_prototypes  (per-category mean of base embeds)
    text_r1:  query @ text_prototypes   (encode_text("an image of {cat}"))

Asymmetry between drop_image and drop_text under function ablation tells us
whether the ablated subspace is "semantic" vs "visual".

Usage:
    uv run python -m experiments.exp3.cav_subspace_retrieval
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio

from experiments.exp3.cav_train import load_clip
from experiments.exp3.cav_ablate import build_prototypes, per_image_correct
from experiments.exp3.cav_degrade import build_masked_tensor, encode_cached
from experiments.exp3.cav_subspace_ablate import (
    DIM_ORDER,
    build_subspace,
    build_random_subspace,
    make_subspace_hook,
)


# ---------------------------------------------------------------------------
# Text prototypes
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_text_prototypes(
    model, arch: str, category_names: list[str], device: str,
    template: str = "an image of {label}",
) -> torch.Tensor:
    """Encode each category name into a text prototype.

    Returns:
        ``[C, D_proj]`` L2-normalized text embeddings on CPU.
    """
    tokenizer = open_clip.get_tokenizer(arch)
    prompts = [template.format(label=c) for c in category_names]
    tokens = tokenizer(prompts).to(device)
    feats = model.encode_text(tokens).float().cpu()
    return F.normalize(feats, dim=-1)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_subspace_retrieval(
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
    print(f"=== Subspace retrieval (image vs text) "
          f"on {dataset_name} @ L{ablate_layer} ===")

    # 1. Subspaces per dim + matched random controls
    print("\n[1/5] Building per-dim subspaces (concept + matched random)")
    subspaces: dict[str, torch.Tensor] = {}
    rank_map: dict[str, int] = {}
    for dim in DIM_ORDER:
        Q = build_subspace(cavs, dim, ablate_layer)
        subspaces[dim] = Q
        rank_map[dim] = int(Q.shape[0])
        print(f"  {dim}: rank={Q.shape[0]}")

    combined = torch.cat(
        [subspaces[d] for d in DIM_ORDER if subspaces[d].numel() > 0], dim=0,
    )
    if combined.numel() > 0:
        Q_t, _ = torch.linalg.qr(combined.T, mode="reduced")
        subspaces["all"] = Q_t.T
        rank_map["all"] = int(subspaces["all"].shape[0])
        print(f"  all: rank={subspaces['all'].shape[0]}")

    rng = np.random.RandomState(seed)
    D = next(iter(cavs.values()))["w"].shape[1]
    for cond, k in list(rank_map.items()):
        subspaces[f"random_{cond}"] = build_random_subspace(k, D, rng)

    for k in list(subspaces.keys()):
        subspaces[k] = subspaces[k].to(device)

    # 2. Model + dataset
    print("\n[2/5] Loading CLIP and dataset")
    model, preprocess, cfg = load_clip(model_key, device)
    dataset = get_dataset(dataset_name)
    num_cats = dataset.num_scenes
    category_names = list(dataset.scene_labels)
    print(f"  {len(dataset)} images, {num_cats} categories")

    target_block = model.visual.transformer.resblocks[ablate_layer]

    # 3. Text prototypes (independent of vis level)
    print("\n[3/5] Building text prototypes")
    text_proto = build_text_prototypes(model, cfg["arch"], category_names, device)
    print(f"  text_proto: {tuple(text_proto.shape)}")

    # 4. Per-level sweep
    levels = get_mask_levels()
    print(f"\n[4/5] Running {len(levels)} levels × "
          f"{1 + len(subspaces)} conditions")

    results = {
        "levels": [],
        "baseline": defaultdict(dict),     # metric -> {level: float}
        "ablated": defaultdict(lambda: defaultdict(dict)),  # cond -> metric -> {level}
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
        img_proto = build_prototypes(baseline_embeds, cat_ids, num_cats)
        b_image = float(per_image_correct(
            baseline_embeds, cat_ids, img_proto,
        ).mean())
        b_text = float(per_image_correct(
            baseline_embeds, cat_ids, text_proto,
        ).mean())
        results["baseline"]["image_r1"][level] = b_image
        results["baseline"]["text_r1"][level] = b_text
        print(f"  baseline:           image_r1={b_image:.3f}  text_r1={b_text:.3f}")

        # Per-condition ablation
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
            a_image = float(per_image_correct(
                ablated_embeds, cat_ids, img_proto,
            ).mean())
            a_text = float(per_image_correct(
                ablated_embeds, cat_ids, text_proto,
            ).mean())
            results["ablated"][cond]["image_r1"][level] = a_image
            results["ablated"][cond]["text_r1"][level] = a_text

            d_img = b_image - a_image
            d_txt = b_text - a_text
            print(f"  {cond:14s} k={Q.shape[0]:2d}: "
                  f"img={a_image:.3f} (Δ={d_img:+.4f})  "
                  f"txt={a_text:.3f} (Δ={d_txt:+.4f})")

        results["levels"].append({"level": level, "vis": vis})
        del cached
        if device == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.time() - t_lvl
        total = (time.time() - t_start) / 60
        print(f"  ({elapsed:.0f}s, total {total:.1f}min)")

    # 5. Save
    out = {
        "model": payload["model"],
        "tag": payload["tag"],
        "dataset": dataset_name,
        "ablate_layer": ablate_layer,
        "subspace_ranks": rank_map,
        "levels": results["levels"],
        "baseline": {k: dict(v) for k, v in results["baseline"].items()},
        "ablated": {
            cond: {m: dict(v) for m, v in metrics.items()}
            for cond, metrics in results["ablated"].items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[5/5] Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subspace ablation: image vs text retrieval drop.",
    )
    parser.add_argument(
        "--cavs", type=Path,
        default=Path("data/coco_subset_56/cavs/clip_L14.pt"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/exp3/cav_subspace_retrieval_clip_L14.json"),
    )
    parser.add_argument("--dataset", default="coco_subset_56")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_subspace_retrieval(
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
