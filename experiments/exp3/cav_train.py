#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cav_train.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Train Concept Activation Vectors (CAVs) per (concept_bin, encoder_layer).

Tests the hypothesis: encoder representations split into discrete concept
subspaces (color, material, function); under degradation, the encoder's
reliance on different concepts shifts. CAV training is the first step —
it identifies which directions in activation space encode each concept,
so later steps can ablate those directions and measure what breaks.

Pipeline:
    1. Load coco_subset_56 (1680 images) + cluster labels.
    2. Extract per-block CLS activations on complete images (one forward
       pass, hooks on every transformer block).
    3. For each (dim, cluster_bin) with n_pos >= MIN_POS:
         positive = images whose label maps to this bin
         negative = random sample from same dim but different bin
                    (so the CAV picks up dim-relevant variation, not
                    arbitrary photo statistics)
       balanced 80/20 train/test, then logistic regression per layer.
    4. Save weights, biases, accuracies into a single .pt file.

Usage:
    uv run python -m experiments.exp3.cav_train
    uv run python -m experiments.exp3.cav_train --model L-14 --min-pos 30
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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.dataset import get_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "B-16": {"arch": "ViT-B-16", "num_layers": 12, "internal_dim": 768,
             "tag": "B16"},
    "L-14": {"arch": "ViT-L-14", "num_layers": 24, "internal_dim": 1024,
             "tag": "L14"},
}

DEFAULT_MODEL = "L-14"
DEFAULT_DATASET = "coco_subset_56"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MIN_POS = 30   # drop bins with fewer positive images
DEFAULT_SEED = 42

# Skip these bins entirely — they're noise/unknown labels, not real concepts.
SKIP_BINS: dict[str, set[str]] = {
    "color": set(),
    "material": {"unclear"},
    "function": {"washing"},  # contains 'unknown'/'unclear' dump
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_concept_labels(
    labels_path: Path,
    clusters_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Build per-image concept-bin assignments from clusters.

    Returns:
        image_bins: {image_id: {dim: bin_name}}. Skips entries where the
            raw value isn't in the cluster mapping (salience='none').
        dim_bins: {dim: [bin_name, ...]} unique bin names per dim.
    """
    with open(labels_path) as f:
        labels = json.load(f)
    with open(clusters_path) as f:
        clusters = json.load(f)

    image_bins: dict[str, dict[str, str]] = defaultdict(dict)
    dim_bins: dict[str, set[str]] = defaultdict(set)

    for entry in labels:
        img_id = entry["image_id"]
        for dim in ("color", "material", "function"):
            if dim not in clusters:
                continue
            raw = entry[dim]["value"]
            r2c = clusters[dim]["raw_to_cluster"]
            if raw not in r2c:
                continue
            bin_name = r2c[raw]
            image_bins[img_id][dim] = bin_name
            dim_bins[dim].add(bin_name)

    return dict(image_bins), {d: sorted(bs) for d, bs in dim_bins.items()}


def load_clip(model_key: str, device: str):
    """Load OpenCLIP model + preprocess for the chosen architecture."""
    cfg = MODEL_CONFIGS[model_key]
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["arch"], pretrained="openai",
    )
    return model.to(device).eval(), preprocess, cfg


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_block_cls(
    model: nn.Module,
    images: torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    """Hook every transformer block, return per-layer CLS tokens.

    Args:
        model: Full OpenCLIP model.
        images: ``[B, 3, 224, 224]`` preprocessed.
        num_layers: Expected number of resblocks (sanity check).

    Returns:
        ``[L, B, D]`` activations (CLS only, residual stream after each block).
    """
    captured: list[torch.Tensor] = []
    handles = []
    for block in model.visual.transformer.resblocks:
        def hook(_mod, _inp, out, buf=captured):
            buf.append(out[:, 0, :].detach().float().cpu())
        handles.append(block.register_forward_hook(hook))

    try:
        model.encode_image(images)
    finally:
        for h in handles:
            h.remove()

    assert len(captured) == num_layers, (
        f"Expected {num_layers} blocks, got {len(captured)}"
    )
    return torch.stack(captured, dim=0)  # [L, B, D]


@torch.no_grad()
def extract_all_activations(
    model: nn.Module,
    dataset,
    preprocess,
    num_layers: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, list[str]]:
    """Run all dataset images through CLIP, gather [L, N, D] CLS activations.

    Returns:
        (activations ``[L, N, D]``, image_ids ``[N]``)
    """
    n = len(dataset)
    image_ids: list[str] = []
    chunks: list[torch.Tensor] = []  # each [L, b, D]

    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = []
        for i in range(start, end):
            sample = dataset[i]
            batch_imgs.append(preprocess(sample["image_pil"]))
            image_ids.append(sample["image_id"])
        x = torch.stack(batch_imgs).to(device)
        acts = extract_block_cls(model, x, num_layers)  # [L, b, D]
        chunks.append(acts)
        if (end // batch_size) % 5 == 0 or end == n:
            elapsed = time.time() - t0
            print(f"  {end}/{n} images  ({elapsed:.1f}s)")

    activations = torch.cat(chunks, dim=1)  # [L, N, D]
    return activations, image_ids


# ---------------------------------------------------------------------------
# CAV training
# ---------------------------------------------------------------------------


def build_pos_neg_splits(
    image_ids: list[str],
    image_bins: dict[str, dict[str, str]],
    dim: str,
    target_bin: str,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Build balanced positive/negative index sets for one bin.

    Negatives are sampled from images that HAVE a label for this dim but
    map to a different bin — this isolates dim-specific variation rather
    than picking up "is labeled at all" artifacts.

    Returns:
        (pos_idx, neg_idx) — both length N_pos.
    """
    pos_idx, neg_pool = [], []
    for i, img_id in enumerate(image_ids):
        b = image_bins.get(img_id, {}).get(dim)
        if b is None:
            continue
        if b == target_bin:
            pos_idx.append(i)
        else:
            neg_pool.append(i)

    pos_idx = np.array(pos_idx)
    neg_pool = np.array(neg_pool)

    if len(pos_idx) == 0 or len(neg_pool) == 0:
        return pos_idx, np.array([], dtype=int)

    n = min(len(pos_idx), len(neg_pool))
    pos_idx = rng.choice(pos_idx, size=n, replace=False) if len(pos_idx) > n else pos_idx
    neg_idx = rng.choice(neg_pool, size=n, replace=False)
    return pos_idx, neg_idx


def train_cav_per_layer(
    activations: torch.Tensor,   # [L, N, D]
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    seed: int,
) -> dict:
    """Fit one logistic regression per layer for a single concept bin.

    Returns:
        {
          "w":         [L, D]  weight vectors (CAV directions),
          "b":         [L]     intercepts,
          "train_acc": [L],
          "test_acc":  [L],
        }
    """
    L, _, D = activations.shape
    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

    train_idx, test_idx, y_train, y_test = train_test_split(
        np.arange(len(idx)), y, test_size=0.2, stratify=y, random_state=seed,
    )

    w = np.zeros((L, D), dtype=np.float32)
    b = np.zeros(L, dtype=np.float32)
    train_acc = np.zeros(L, dtype=np.float32)
    test_acc = np.zeros(L, dtype=np.float32)

    for layer in range(L):
        x = activations[layer, idx].numpy()  # [2n, D]
        x_train = x[train_idx]
        x_test = x[test_idx]
        clf = LogisticRegression(
            C=1.0, max_iter=1000, solver="liblinear", random_state=seed,
        )
        clf.fit(x_train, y_train)
        w[layer] = clf.coef_[0].astype(np.float32)
        b[layer] = float(clf.intercept_[0])
        train_acc[layer] = clf.score(x_train, y_train)
        test_acc[layer] = clf.score(x_test, y_test)

    return {
        "w": torch.from_numpy(w),
        "b": torch.from_numpy(b),
        "train_acc": torch.from_numpy(train_acc),
        "test_acc": torch.from_numpy(test_acc),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train per-layer CAVs on CLIP for color/material/function.",
    )
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--labels", type=Path,
        default=Path("data/coco_subset_56/concept_labels.json"),
    )
    parser.add_argument(
        "--clusters", type=Path,
        default=Path("data/coco_subset_56/concept_clusters.json"),
    )
    parser.add_argument("--min-pos", type=int, default=DEFAULT_MIN_POS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .pt path (default: data/coco_subset_56/cavs/clip_{tag}.pt)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    out_path = args.out or Path(f"data/coco_subset_56/cavs/clip_{cfg['tag']}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== CAV training: CLIP {cfg['arch']} on {args.dataset} ===")
    print(f"  device: {args.device}")
    print(f"  min positives per bin: {args.min_pos}")
    print(f"  output: {out_path}")

    # 1. Load concept labels
    print("\n[1/4] Loading concept labels and clusters")
    image_bins, dim_bins = load_concept_labels(args.labels, args.clusters)
    print(f"  {len(image_bins)} labeled images")
    for dim, bins in dim_bins.items():
        print(f"  {dim}: {len(bins)} bins")

    # 2. Load model + dataset
    print(f"\n[2/4] Loading CLIP {cfg['arch']}")
    model, preprocess, cfg = load_clip(args.model, args.device)
    dataset = get_dataset(args.dataset)
    print(f"  dataset: {len(dataset)} images")

    # 3. Extract activations
    print(f"\n[3/4] Extracting per-block CLS activations "
          f"({cfg['num_layers']} layers, {cfg['internal_dim']}-dim)")
    activations, image_ids = extract_all_activations(
        model, dataset, preprocess,
        num_layers=cfg["num_layers"],
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"  activations shape: {tuple(activations.shape)}")

    # Free GPU memory before training
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # 4. Train CAVs
    print(f"\n[4/4] Training CAVs (logistic regression per layer)")
    rng = np.random.RandomState(args.seed)
    cavs: dict[tuple[str, str], dict] = {}
    skipped: list[tuple[str, str, int]] = []

    for dim, bins in dim_bins.items():
        for bin_name in bins:
            if bin_name in SKIP_BINS.get(dim, set()):
                continue
            pos_idx, neg_idx = build_pos_neg_splits(
                image_ids, image_bins, dim, bin_name, rng,
            )
            if len(pos_idx) < args.min_pos:
                skipped.append((dim, bin_name, len(pos_idx)))
                continue

            result = train_cav_per_layer(
                activations, pos_idx, neg_idx, seed=args.seed,
            )
            result["n_pos"] = int(len(pos_idx))
            result["n_neg"] = int(len(neg_idx))
            cavs[(dim, bin_name)] = result
            best = float(result["test_acc"].max())
            best_layer = int(result["test_acc"].argmax())
            print(f"  [{dim}/{bin_name}] n={len(pos_idx)}  "
                  f"best test_acc={best:.3f} @ layer {best_layer}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} bins with n_pos < {args.min_pos}:")
        for dim, b, n in skipped:
            print(f"    [{dim}/{b}] n={n}")

    # 5. Save
    payload = {
        "model": cfg["arch"],
        "tag": cfg["tag"],
        "num_layers": cfg["num_layers"],
        "internal_dim": cfg["internal_dim"],
        "dataset": args.dataset,
        "min_pos": args.min_pos,
        "seed": args.seed,
        "cavs": cavs,
    }
    torch.save(payload, out_path)
    print(f"\nSaved {len(cavs)} CAVs to {out_path}")


if __name__ == "__main__":
    main()
