#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_masking_examples.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Generate masking example images for visual inspection.

Run after any masking algorithm change to verify output looks correct.
Saves to results/examples/{method}/.

Usage:
    uv run pytest tests/test_masking_examples.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.dataset import get_dataset
from src.masking import get_mask_levels, mask_pil_image, mask_pil_image_saliency

EXAMPLES_DIR = Path("results/examples")
PATCH_SIZE = 16
TARGET_SIZE = 518

# Images to generate: {name: (category, nth)} — nth=1 means first of that category
EXAMPLE_IMAGES = {
    "cat": ("cat", 1),
    "cat2": ("cat", 2),
    "cat3": ("cat", 3),
    "cat4": ("cat", 4),
    "car": ("car", 1),
    "chair": ("chair", 1),
    "dog": ("dog", 1),
    "horse": ("horse", 1),
}

# cat2 gets all 8 levels; others get a single representative level
CAT2_LEVELS = list(range(1, 9))  # L1–L8
OTHER_LEVEL = 4


def _get_sample(
    dataset: object, category: str, nth: int,
) -> tuple[int, dict]:
    """Return (dataset_index, sample) for the nth image of a category."""
    count = 0
    for i in range(len(dataset)):
        if dataset.samples[i]["scene_label"] == category:
            count += 1
            if count == nth:
                return i, dataset[i]
    raise RuntimeError(f"{category} #{nth} not found in dataset")


@pytest.fixture(scope="module")
def coco_dataset() -> object:
    """Load COCO subset dataset (skip if not prepared)."""
    try:
        return get_dataset("coco_subset")
    except FileNotFoundError:
        pytest.skip("COCO subset not prepared — run: uv run python data/prepare_coco.py")


def test_generate_masking_examples(coco_dataset: object) -> None:
    """Generate fragment examples for random masking."""
    method_dir = EXAMPLES_DIR / "random"
    method_dir.mkdir(parents=True, exist_ok=True)

    for name, (category, nth) in EXAMPLE_IMAGES.items():
        _, sample = _get_sample(coco_dataset, category, nth)
        pil = sample["image_pil"]

        # Save full image
        resized = pil.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        resized.save(EXAMPLES_DIR / f"{name}_full.png")

        seg = sample["seg_mask"]

        # Determine levels for this image
        levels = CAT2_LEVELS if name == "cat2" else [OTHER_LEVEL]

        for level in levels:
            masked = mask_pil_image(
                pil, level, seg,
                patch_size=PATCH_SIZE,
                target_size=TARGET_SIZE,
                seed=42, idx=0,
            )
            out_path = method_dir / f"{name}_L{level}.png"
            masked.save(out_path)
            print(f"  Saved: {out_path}")

            arr = np.array(masked)
            assert arr.shape == (TARGET_SIZE, TARGET_SIZE, 3)

    # Verify monotonicity for cat2: more white at lower levels
    white_counts = []
    for level in CAT2_LEVELS:
        arr = np.array(Image.open(method_dir / f"cat2_L{level}.png"))
        white_counts.append(np.all(arr == 255, axis=-1).sum())
    for i in range(len(white_counts) - 1):
        assert white_counts[i] >= white_counts[i + 1], (
            f"L{CAT2_LEVELS[i]} should have >= white pixels than L{CAT2_LEVELS[i+1]}"
        )

    print(f"\n  Random masking examples saved to {method_dir}/")


def test_generate_saliency_examples(coco_dataset: object) -> None:
    """Generate saliency-masked examples (if saliency maps exist)."""
    sal_dir = Path("results/exp2/saliency_masking/saliency")
    sal_path = sal_dir / "dinov2_saliency.pt"

    if not sal_path.exists():
        pytest.skip("DINOv2 saliency not computed — run saliency subcommand first")

    import torch
    from src.saliency import resample_saliency

    sal_data = torch.load(sal_path, weights_only=True)
    sal_maps = sal_data["saliency"]
    sal_indices = sal_data["indices"]

    method_dir = EXAMPLES_DIR / "dinov2_salient"
    method_dir.mkdir(parents=True, exist_ok=True)
    grid_size = TARGET_SIZE // PATCH_SIZE

    for name, (category, nth) in EXAMPLE_IMAGES.items():
        idx, sample = _get_sample(coco_dataset, category, nth)
        pil = sample["image_pil"]

        if idx not in sal_indices:
            print(f"  [skip] {name} (idx={idx}) not in saliency indices")
            continue
        sal_pos = sal_indices.index(idx)

        sal_np = resample_saliency(
            sal_maps[sal_pos].unsqueeze(0), grid_size, grid_size,
        ).squeeze(0).numpy()

        levels = CAT2_LEVELS if name == "cat2" else [OTHER_LEVEL]

        seg = sample["seg_mask"]

        for level in levels:
            masked = mask_pil_image_saliency(
                pil, level, seg, sal_np,
                salient_first=True,
                patch_size=PATCH_SIZE,
                target_size=TARGET_SIZE,
            )
            out_path = method_dir / f"{name}_L{level}.png"
            masked.save(out_path)
            print(f"  Saved: {out_path}")

            arr = np.array(masked)
            assert arr.shape == (TARGET_SIZE, TARGET_SIZE, 3)

    # Verify monotonicity for cat2
    white_counts = []
    for level in CAT2_LEVELS:
        path = method_dir / f"cat2_L{level}.png"
        if path.exists():
            arr = np.array(Image.open(path))
            white_counts.append(np.all(arr == 255, axis=-1).sum())
    for i in range(len(white_counts) - 1):
        assert white_counts[i] >= white_counts[i + 1], (
            f"cat2 L{CAT2_LEVELS[i]} should have >= white pixels than L{CAT2_LEVELS[i+1]}"
        )

    print(f"\n  DINOv2 saliency examples saved to {method_dir}/")
