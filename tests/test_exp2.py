#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_exp2.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Unit tests for exp2 shared utility functions in src/utils.py."""

from __future__ import annotations

import torch

from src.utils import compute_category_accuracy, compute_retrieval_metrics


class TestComputeRetrievalMetrics:
    """Tests for compute_retrieval_metrics."""

    def test_perfect_retrieval(self):
        """When query == gallery, R@1 should be 1.0."""
        N, D = 10, 32
        embeds = torch.randn(N, D)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        gt_indices = torch.arange(N, dtype=torch.long)

        result = compute_retrieval_metrics(embeds, embeds, gt_indices)
        assert result["recall_at_1"] == 1.0
        assert result["recall_at_5"] == 1.0
        assert result["mrr"] == 1.0

    def test_output_keys(self):
        """Should return recall_at_1, recall_at_5, mrr."""
        N, D = 5, 16
        query = torch.randn(N, D)
        gallery = torch.randn(N, D)
        gt_indices = torch.arange(N, dtype=torch.long)

        result = compute_retrieval_metrics(query, gallery, gt_indices)
        assert "recall_at_1" in result
        assert "recall_at_5" in result
        assert "mrr" in result

    def test_values_in_range(self):
        """All metrics should be in [0, 1]."""
        N, D = 20, 32
        query = torch.randn(N, D)
        gallery = torch.randn(N, D)
        gt_indices = torch.arange(N, dtype=torch.long)

        result = compute_retrieval_metrics(query, gallery, gt_indices)
        for key in ("recall_at_1", "recall_at_5", "mrr"):
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"


class TestComputeCategoryAccuracy:
    """Tests for compute_category_accuracy."""

    def test_perfect_categorization(self):
        """When query matches exactly, accuracy should be 1.0."""
        N, D, C = 10, 32, 3
        cat_embeds = torch.randn(C, D)
        cat_embeds = cat_embeds / cat_embeds.norm(dim=-1, keepdim=True)

        # Build query as exact copies of category embeddings
        gt_cat_ids = [i % C for i in range(N)]
        query = torch.stack([cat_embeds[cid] for cid in gt_cat_ids])

        acc = compute_category_accuracy(query, cat_embeds, gt_cat_ids)
        assert acc == 1.0

    def test_output_is_float(self):
        """Should return a float in [0, 1]."""
        N, D, C = 5, 16, 2
        query = torch.randn(N, D)
        cat_embeds = torch.randn(C, D)
        gt_cat_ids = [i % C for i in range(N)]

        acc = compute_category_accuracy(query, cat_embeds, gt_cat_ids)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
