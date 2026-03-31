#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : conftest.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Shared pytest fixtures: mock encoder and tiny synthetic dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from models.encoder import BaseEncoder


# ---------------------------------------------------------------------------
# Mock encoder (no GPU required)
# ---------------------------------------------------------------------------

class _DummyModel(nn.Module):
    """Minimal model that returns random features."""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(3, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D]
        return torch.randn(x.shape[0], self.feature_dim)


class MockEncoder(BaseEncoder):
    """Encoder that returns random features of a fixed dimension.

    No pretrained weights, no GPU. Suitable for testing output structure.
    """

    def __init__(self, feature_dim: int = 64, device: str = "cpu"):
        super().__init__(device=device)
        self._feature_dim = feature_dim

    @property
    def name(self) -> str:
        return "MockEncoder"

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def load_model(self) -> tuple[nn.Module, object]:
        model = _DummyModel(self._feature_dim)
        # Minimal processor: just resize + to tensor
        from torchvision import transforms

        processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return model, processor


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Return a MockEncoder on CPU with dim=64."""
    enc = MockEncoder(feature_dim=64, device="cpu")
    _ = enc.model  # trigger lazy load
    return enc


# ---------------------------------------------------------------------------
# Tiny synthetic dataset
# ---------------------------------------------------------------------------

class TinyDataset:
    """Minimal dataset with synthetic images for unit tests.

    4 images, 2 scenes, white background with a colored square.
    """

    def __init__(self, num_images: int = 4, image_size: int = 100):
        self._items: list[dict] = []
        self._scene_labels = ["scene_A", "scene_B"]

        rng = np.random.RandomState(42)
        for i in range(num_images):
            # White background
            img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
            # Place a colored square in the center (foreground)
            c = image_size // 4
            color = rng.randint(50, 200, size=3).tolist()
            img[c:3 * c, c:3 * c] = color

            # Segmentation mask: 1 where foreground
            seg = np.zeros((image_size, image_size), dtype=np.uint8)
            seg[c:3 * c, c:3 * c] = 1

            scene_id = i % len(self._scene_labels)
            self._items.append({
                "image_pil": Image.fromarray(img),
                "seg_mask": seg,
                "scene_label": self._scene_labels[scene_id],
                "scene_id": scene_id,
                "image_id": f"img_{i:03d}",
            })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]

    @property
    def num_scenes(self) -> int:
        return len(self._scene_labels)

    @property
    def scene_labels(self) -> list[str]:
        return self._scene_labels


@pytest.fixture
def tiny_dataset() -> TinyDataset:
    """Return a TinyDataset with 4 synthetic images."""
    return TinyDataset(num_images=4, image_size=100)
