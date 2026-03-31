#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_exp1.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Smoke tests for exp1 task functions: verify output schema."""

from __future__ import annotations

import pytest

from src.masking import get_mask_levels


@pytest.fixture
def _patch_extract(monkeypatch):
    """Patch extract_patch_features to return random tensor for MockEncoder.

    gestalt.py calls extract_patch_features which relies on encoder internals.
    We mock it to return a [196, 64] tensor (14x14 grid, dim=64).
    """
    import torch
    import src.utils as utils_mod

    def _fake_extract(encoder, pil):
        return torch.randn(196, 64)

    monkeypatch.setattr(utils_mod, "extract_patch_features", _fake_extract)

    def _fake_geometry(encoder):
        return (224, 16)

    monkeypatch.setattr(utils_mod, "get_encoder_geometry", _fake_geometry)


class TestGestalt:
    """Tests for experiments.exp1.gestalt.evaluate_gestalt."""

    def test_output_schema(self, mock_encoder, tiny_dataset, _patch_extract):
        """Output should be {level: {mean, std, silhouette_mean, silhouette_std}}."""
        from experiments.exp1.gestalt import evaluate_gestalt

        result = evaluate_gestalt(
            mock_encoder, tiny_dataset, seed=42, max_images=2, num_runs=1,
        )
        levels = get_mask_levels()
        for L in levels:
            assert L in result, f"Missing level {L}"
            assert "mean" in result[L]
            assert "std" in result[L]
            assert "silhouette_mean" in result[L]
            assert "silhouette_std" in result[L]

    def test_values_are_finite(self, mock_encoder, tiny_dataset, _patch_extract):
        """All metric values should be finite floats."""
        from experiments.exp1.gestalt import evaluate_gestalt

        result = evaluate_gestalt(
            mock_encoder, tiny_dataset, seed=42, max_images=2, num_runs=1,
        )
        for L, metrics in result.items():
            for key, val in metrics.items():
                assert isinstance(val, float), f"L={L} {key} not float"
                assert not (val != val), f"L={L} {key} is NaN"  # NaN check


class TestMnemonic:
    """Tests for experiments.exp1.mnemonic.evaluate_mnemonic."""

    def test_output_schema(self, mock_encoder, tiny_dataset, monkeypatch):
        """Output should have similarity and retrieval sub-dicts."""
        import torch
        import src.utils as utils_mod

        # Mock embed_pil to avoid processor incompatibility
        def _fake_embed(encoder, pil, transform):
            return torch.randn(encoder.feature_dim)

        monkeypatch.setattr(utils_mod, "embed_pil", _fake_embed)

        from experiments.exp1.mnemonic import evaluate_mnemonic

        result = evaluate_mnemonic(
            mock_encoder, tiny_dataset, seed=42, max_images=2,
            num_choices=2, num_runs=1,
        )
        assert "similarity" in result
        assert "retrieval" in result
        levels = get_mask_levels()
        for L in levels:
            assert L in result["similarity"]
            assert "mean" in result["similarity"][L]
            assert "std" in result["similarity"][L]
            assert L in result["retrieval"]
            assert "mean" in result["retrieval"][L]
            assert "std" in result["retrieval"][L]
