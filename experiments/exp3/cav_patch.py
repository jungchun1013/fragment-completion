#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cav_patch.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Concept-conditioned denoising activation patching on CLIP-L-14.

NEGATIVE FINDING — kept for documentation
=========================================
This script gave a flat / uninformative result: per-dim recovery curves
for color / material / function are nearly identical, with all recovery
concentrated at L_patch = 23 (the very last block). Patching CLS-only at
any earlier L_patch barely helps because subsequent attention re-mixes
corrupted patch tokens back into CLS, diluting the clean signal.

Conclusion: residual stream CLS-only patching is too coarse to localize
where each concept enters the model. The CAV best-layer pattern (color
early, function late) is *correlational* (linear separability), not
*causal* (where computation happens). To do real layer localization,
need component-level patching (replace ``block.attn`` or ``block.mlp``
output, not the full residual or CLS).

Why we keep this file: it documents the bottleneck (residual stream is a
single channel with no recovery pathway) and provides a reference point
for future component-patching follow-ups.

Tests "where does each concept dimension live in the residual stream"
via causal intervention rather than CAV training:

  For each masking level L_mask in {1..8}:
      For each L_patch in {0..23}:
          Hook block L_patch, replace its CLS output with the cached
          clean (full-vis) CLS at the same layer.
          Run forward → measure 56-way categorization correctness
          against FROZEN full-vis prototypes.

Per-image correctness arrays are saved raw (no aggregation), so any
post-hoc concept split (color/material/function bin) and any recovery
metric can be re-derived without re-running the sweep.

LAYER INDEXING — important off-by-one note
==========================================
``clean_cls[i]`` is the residual stream CLS captured *after*
``model.visual.transformer.resblocks[i]`` (= input to ``resblocks[i+1]``).
The forward hook on ``resblocks[L_patch]`` rewrites the *output* of that
block, so we patch in ``clean_cls[L_patch]``.

Sanity check: at L_mask = 8 (full vis), L_patch = 23 should give recovery
≈ 1.0 because we replace the very last block's CLS with itself modulo
``ln_post + proj`` numerics.

PROTOTYPES
==========
Frozen full-vis (clean) prototypes — NOT per-level. Per-level prototypes
drift with the corruption and turn the recovery metric circular. Frozen
prototypes ask the cleaner question: "did patching restore alignment to
the *original* semantic space?" This deliberately differs from
``cav_degrade.py`` which uses per-level prototypes.

Usage:
    uv run python -m experiments.exp3.cav_patch
    uv run python -m experiments.exp3.cav_patch --batch-size 128
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio

from experiments.exp3.cav_train import (
    MODEL_CONFIGS,
    extract_block_cls,
    load_clip,
    load_concept_labels,
)
from experiments.exp3.cav_ablate import build_prototypes, per_image_correct
from experiments.exp3.cav_degrade import build_masked_tensor, encode_cached


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_block_cls_batched(
    model: nn.Module,
    cached_input: torch.Tensor,
    num_layers: int,
    batch_size: int,
) -> torch.Tensor:
    """Run the model in batches, capture per-block CLS via hooks.

    ``cav_train.extract_block_cls`` already does this for one batch and
    moves CLS to CPU inside the hook closure. Wrap in a batched loop so
    the cached input doesn't have to fit a single forward pass.

    Returns:
        ``[num_layers, N, D]`` on CPU (float32).
    """
    n = cached_input.shape[0]
    chunks: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        acts = extract_block_cls(model, cached_input[start:end], num_layers)
        chunks.append(acts)  # [L, b, D]
    return torch.cat(chunks, dim=1)  # [L, N, D]


@torch.no_grad()
def encode_with_per_batch_hook(
    model: nn.Module,
    cached_input: torch.Tensor,
    batch_size: int,
    target_block: nn.Module,
    clean_cls_at_layer: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with a fresh per-batch hook patching CLS.

    The patched CLS slice depends on the current batch's global indices,
    so we install/remove the hook each batch instead of once-per-forward.
    Each batch closes over its own bound slice (no shared mutable state).

    Args:
        model: OpenCLIP model.
        cached_input: ``[N, 3, 224, 224]`` already on device.
        batch_size: Forward batch size.
        target_block: ``model.visual.transformer.resblocks[L_patch]``.
        clean_cls_at_layer: ``[N, D]`` clean CLS to patch in (CPU).

    Returns:
        ``[N, D_proj]`` L2-normalized embeddings on CPU.
    """
    embeds: list[torch.Tensor] = []
    n = cached_input.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        clean_slice = clean_cls_at_layer[start:end].to(
            cached_input.device, non_blocking=True,
        )

        def hook(_mod, _inp, out, src=clean_slice):
            # out: [B, T, D]; replace CLS only.
            out[:, 0, :] = src.to(out.dtype)
            return out

        handle = target_block.register_forward_hook(hook)
        try:
            feats = model.encode_image(cached_input[start:end]).float().cpu()
        finally:
            handle.remove()
        embeds.append(F.normalize(feats, dim=-1))
    return torch.cat(embeds, dim=0)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run_patching(
    out_path: Path,
    dataset_name: str,
    model_key: str,
    labels_path: Path,
    clusters_path: Path,
    batch_size: int,
    seed: int,
    device: str,
) -> None:
    cfg = MODEL_CONFIGS[model_key]
    print(f"=== Activation patching: CLIP {cfg['arch']} on {dataset_name} ===")
    print(f"  output: {out_path}")

    # 1. Load model + dataset + concept labels
    print("\n[1/5] Loading model, dataset, concept labels")
    model, preprocess, _ = load_clip(model_key, device)
    dataset = get_dataset(dataset_name)
    image_bins, _ = load_concept_labels(labels_path, clusters_path)
    num_cats = dataset.num_scenes
    print(f"  {len(dataset)} images, {num_cats} categories")

    # 2. Pre-flight: cache clean per-block CLS + frozen prototypes
    print("\n[2/5] Caching clean per-block CLS at full visibility")
    t0 = time.time()
    clean_cached, image_ids, cat_ids = build_masked_tensor(
        dataset, preprocess, level=8, seed=seed, device=device,
    )
    clean_cls = extract_block_cls_batched(
        model, clean_cached, cfg["num_layers"], batch_size,
    )  # [24, N, 1024] on CPU
    print(f"  clean_cls shape: {tuple(clean_cls.shape)}  "
          f"({(time.time() - t0):.1f}s)")

    print("\n[3/5] Computing clean embeddings + frozen prototypes")
    clean_embeds = encode_cached(model, clean_cached, batch_size)
    prototypes_frozen = build_prototypes(clean_embeds, cat_ids, num_cats)
    correct_clean = per_image_correct(clean_embeds, cat_ids, prototypes_frozen)
    clean_acc = float(correct_clean.mean())
    print(f"  prototypes: {tuple(prototypes_frozen.shape)}")
    print(f"  clean acc:  {clean_acc:.3f}")

    del clean_cached
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Per-level patch sweep
    levels = get_mask_levels()  # [1..8]
    print(f"\n[4/5] Patch sweep: {len(levels)} levels × "
          f"{cfg['num_layers']} layers = "
          f"{len(levels) * (cfg['num_layers'] + 1)} forward passes")

    correct_baseline_per_level: dict[int, list[int]] = {}
    correct_patched_per_level: dict[int, list[list[int]]] = {}
    vis_per_level: dict[int, float] = {}

    t_start = time.time()
    for li, level in enumerate(levels):
        vis = get_visibility_ratio(level)
        vis_per_level[level] = vis
        print(f"\n--- Level {level}/{levels[-1]}  vis={vis:.3f} "
              f"[{li + 1}/{len(levels)}] ---")
        t_lvl = time.time()

        cached, ids_check, _ = build_masked_tensor(
            dataset, preprocess, level, seed, device,
        )
        # Indexing consistency: clean cache and corrupted cache must use the
        # same dataset order so clean_cls slices line up with batches.
        assert ids_check == image_ids, (
            f"image order mismatch at level {level}"
        )

        # Baseline (no hook) at this level
        base_embeds = encode_cached(model, cached, batch_size)
        correct_baseline = per_image_correct(
            base_embeds, cat_ids, prototypes_frozen,
        )
        baseline_acc = float(correct_baseline.mean())
        correct_baseline_per_level[level] = correct_baseline.int().tolist()
        print(f"  baseline acc: {baseline_acc:.3f}")

        # Patch sweep over L_patch
        correct_patched = torch.zeros(
            cfg["num_layers"], len(image_ids), dtype=torch.int8,
        )
        for L_patch in range(cfg["num_layers"]):
            target_block = model.visual.transformer.resblocks[L_patch]
            ablated_embeds = encode_with_per_batch_hook(
                model, cached, batch_size,
                target_block=target_block,
                clean_cls_at_layer=clean_cls[L_patch],
            )
            corr = per_image_correct(
                ablated_embeds, cat_ids, prototypes_frozen,
            )
            correct_patched[L_patch] = corr.to(torch.int8)
        # Compact print: baseline + patched accuracies at L_patch ∈ {0, 11, 23}
        for Lp in (0, 11, 23):
            ac = float(correct_patched[Lp].float().mean())
            print(f"  patched L_patch={Lp:2d}: {ac:.3f}")

        correct_patched_per_level[level] = correct_patched.tolist()

        del cached
        if device == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t_lvl
        total = (time.time() - t_start) / 60
        eta = total * (len(levels) - li - 1) / (li + 1)
        print(f"  level done in {elapsed:.0f}s "
              f"(total {total:.1f}min, eta {eta:.1f}min)")

    # 4. Sanity check: L_mask=8, L_patch=23 should match clean acc
    print("\n[5/5] Sanity checks")
    final_patched = torch.tensor(correct_patched_per_level[8])  # [24, N]
    last_layer_acc = float(final_patched[-1].float().mean())
    diff = abs(last_layer_acc - clean_acc)
    print(f"  L_mask=8 L_patch=23 acc: {last_layer_acc:.4f}  "
          f"(clean {clean_acc:.4f}, diff {diff:.4f})")
    assert diff < 0.02, (
        f"Sanity check failed: patching the last block at full vis should "
        f"reproduce clean acc, but got {last_layer_acc:.4f} vs {clean_acc:.4f}. "
        f"Off-by-one in layer indexing?"
    )
    print("  ✓ last-layer patching at full vis ≈ clean acc")

    # 5. Save raw per-image arrays
    payload = {
        "model": cfg["arch"],
        "tag": cfg["tag"],
        "dataset": dataset_name,
        "num_layers": cfg["num_layers"],
        "image_ids": image_ids,
        "cat_ids": cat_ids,
        "concept_bins": image_bins,
        "correct_clean": correct_clean.int().tolist(),
        "correct_baseline_per_level": correct_baseline_per_level,
        "correct_patched_per_level": correct_patched_per_level,
        "vis_per_level": vis_per_level,
        "clean_acc": clean_acc,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nSaved patching results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concept-conditioned activation patching on CLIP-L-14.",
    )
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), default="L-14")
    parser.add_argument("--dataset", default="coco_subset_56")
    parser.add_argument(
        "--labels", type=Path,
        default=Path("data/coco_subset_56/concept_labels.json"),
    )
    parser.add_argument(
        "--clusters", type=Path,
        default=Path("data/coco_subset_56/concept_clusters.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .pt path (default: results/exp3/cav_patch_clip_{tag}.pt)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    out_path = args.out or Path(f"results/exp3/cav_patch_clip_{cfg['tag']}.pt")

    run_patching(
        out_path=out_path,
        dataset_name=args.dataset,
        model_key=args.model,
        labels_path=args.labels,
        clusters_path=args.clusters,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
