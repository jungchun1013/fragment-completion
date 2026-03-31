#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_utils.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Unit tests for src/utils.py helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils import (
    extract_std,
    extract_val,
    fix_json_keys,
    get_foreground_mask,
    get_object_patch_indices,
    save_results,
)


def test_fix_json_keys_converts_int_strings():
    """String keys that look like ints should be converted."""
    d = {"1": {"mean": 0.5}, "2": {"mean": 0.6}}
    result = fix_json_keys(d)
    assert result == {1: {"mean": 0.5}, 2: {"mean": 0.6}}


def test_fix_json_keys_preserves_non_int_keys():
    """String keys that are not ints should be preserved recursively."""
    d = {"encoder": {"1": 0.5, "2": 0.6}}
    result = fix_json_keys(d)
    assert "encoder" in result
    # Inner dict should have int keys
    inner = result["encoder"]
    assert inner == {1: 0.5, 2: 0.6}


def test_extract_val_from_dict():
    """Should extract 'mean' from {mean, std} dict."""
    assert extract_val({"mean": 0.42, "std": 0.01}) == 0.42


def test_extract_val_scalar():
    """Should return scalar directly."""
    assert extract_val(0.42) == 0.42


def test_extract_std_from_dict():
    """Should extract 'std' from {mean, std} dict."""
    assert extract_std({"mean": 0.42, "std": 0.01}) == 0.01


def test_extract_std_missing():
    """Should return 0.0 for scalar or dict without 'std'."""
    assert extract_std(0.42) == 0.0
    assert extract_std({"mean": 0.42}) == 0.0


def test_get_foreground_mask():
    """White background → 0, non-white → 1."""
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    # Place a dark square
    img[2:5, 2:5] = 100
    mask = get_foreground_mask(img)
    assert mask.shape == (10, 10)
    assert mask[3, 3] == 1.0  # foreground
    assert mask[0, 0] == 0.0  # background


def test_get_object_patch_indices():
    """Patches overlapping with non-white pixels should be detected."""
    img = np.ones((8, 8, 3), dtype=np.uint8) * 255
    img[0:4, 0:4] = 100  # top-left quadrant is foreground
    # 2x2 grid of 4x4 patches
    indices = get_object_patch_indices(img, gh=2, gw=2)
    # Only patches touching top-left quadrant
    assert 0 in indices  # top-left patch
    assert len(indices) >= 1


def test_save_results_roundtrip(tmp_path: Path):
    """Save and reload should produce equivalent data."""
    data = {
        "encoders": {
            "CLIP": {
                "original": {
                    "gestalt_iou": {
                        1: {"mean": np.float64(0.5), "std": np.float64(0.1)},
                    }
                }
            }
        }
    }
    path = tmp_path / "results.json"
    save_results(data, path)

    with open(path) as f:
        loaded = json.load(f)

    # Keys become strings in JSON
    assert loaded["encoders"]["CLIP"]["original"]["gestalt_iou"]["1"]["mean"] == 0.5
