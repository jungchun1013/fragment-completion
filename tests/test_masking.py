#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_masking.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Unit tests for src/masking.py."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image


def test_get_mask_levels():
    """Should return [1, 2, ..., 8]."""
    levels = get_mask_levels()
    assert levels == [1, 2, 3, 4, 5, 6, 7, 8]


def test_get_visibility_ratio_boundaries():
    """L=1 ~ 0.082, L=8 = 1.0."""
    assert get_visibility_ratio(8) == 1.0
    assert abs(get_visibility_ratio(1) - 0.7 ** 7) < 1e-9
    # Monotonically increasing
    for L in range(1, 8):
        assert get_visibility_ratio(L) < get_visibility_ratio(L + 1)


def test_mask_pil_image_deterministic(tiny_dataset):
    """Same seed + idx should produce identical masked images."""
    sample = tiny_dataset[0]
    pil, seg = sample["image_pil"], sample["seg_mask"]

    m1 = mask_pil_image(pil, level=3, seg_mask=seg, seed=42, idx=0)
    m2 = mask_pil_image(pil, level=3, seg_mask=seg, seed=42, idx=0)
    assert np.array_equal(np.array(m1), np.array(m2))


def test_mask_pil_image_different_seeds(tiny_dataset):
    """Different seeds should produce different masked images (for L < 8)."""
    sample = tiny_dataset[0]
    pil, seg = sample["image_pil"], sample["seg_mask"]

    m1 = mask_pil_image(pil, level=3, seg_mask=seg, seed=42, idx=0)
    m2 = mask_pil_image(pil, level=3, seg_mask=seg, seed=99, idx=0)
    # Very likely different (not guaranteed but extremely probable)
    assert not np.array_equal(np.array(m1), np.array(m2))


def test_mask_pil_image_level8_is_complete(tiny_dataset):
    """Level 8 should not mask anything — output equals resized input."""
    sample = tiny_dataset[0]
    pil, seg = sample["image_pil"], sample["seg_mask"]

    masked = mask_pil_image(pil, level=8, seg_mask=seg, seed=42, idx=0)
    # Level 8 visibility = 1.0, so all fg patches should be visible
    expected = pil.resize((224, 224), Image.BILINEAR)
    assert np.array_equal(np.array(masked), np.array(expected))


def test_mask_pil_image_output_shape(tiny_dataset):
    """Output should be target_size x target_size RGB."""
    sample = tiny_dataset[0]
    pil, seg = sample["image_pil"], sample["seg_mask"]

    masked = mask_pil_image(pil, level=3, seg_mask=seg, target_size=224)
    arr = np.array(masked)
    assert arr.shape == (224, 224, 3)


def test_mask_pil_image_more_visible_at_higher_level(tiny_dataset):
    """Higher levels should reveal more pixels (fewer white patches)."""
    sample = tiny_dataset[0]
    pil, seg = sample["image_pil"], sample["seg_mask"]

    white_counts = []
    for L in [1, 4, 7]:
        masked = mask_pil_image(pil, level=L, seg_mask=seg, seed=42, idx=0)
        arr = np.array(masked)
        # Count pixels that are pure white (255, 255, 255)
        white = np.all(arr == 255, axis=-1).sum()
        white_counts.append(white)

    # L=1 should have more white than L=4, which has more than L=7
    assert white_counts[0] >= white_counts[1] >= white_counts[2]
