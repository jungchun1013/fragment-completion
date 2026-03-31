#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : experiment_config.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-30-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Centralized experiment configuration.

All controlled variables, model configs, and default parameters in one place.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Controlled variables
# ---------------------------------------------------------------------------

EXPERIMENT_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "num_runs": 3,
    "num_choices": 5,
    "exemplar_k": 10,
}

MASKING_CONFIG: dict[str, Any] = {
    "num_levels": 8,
    "visibility_base": 0.7,
    "default_patch_size": 16,
    "mask_fill_color": 255,
}

SRSS_CONFIG: dict[str, int] = {
    "r_near": 2,
    "r_far": 4,
}

# ---------------------------------------------------------------------------
# Image normalization
# ---------------------------------------------------------------------------

IMAGENET_NORM: dict[str, list[float]] = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# ---------------------------------------------------------------------------
# Model configurations (independent variable)
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "clip_B16": {
        "arch": "ViT-B-16",
        "num_layers": 12,
        "internal_dim": 768,
        "proj_dim": 512,
        "img_size": 224,
        "tag": "B16",
        "label": "CLIP ViT-B-16",
    },
    "clip_L14": {
        "arch": "ViT-L-14",
        "num_layers": 24,
        "internal_dim": 1024,
        "proj_dim": 768,
        "img_size": 224,
        "tag": "clip_L14",
        "label": "CLIP ViT-L-14",
    },
    "dinov2": {
        "arch": "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
        "num_layers": 24,
        "internal_dim": 1024,
        "proj_dim": 2048,
        "img_size": 518,
        "tag": "dinov2",
        "label": "DINOv2+dino.txt",
    },
}

# ---------------------------------------------------------------------------
# Gestalt / segmentation constants
# ---------------------------------------------------------------------------

WHITE_THRESHOLD: int = 250
NUM_KMEANS_CLUSTERS: int = 2
GESTALT_UPSAMPLE_SIZE: int = 28


# ---------------------------------------------------------------------------
# Experiment settings I/O
# ---------------------------------------------------------------------------


def save_experiment_settings(
    args: object,
    results_dir: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save CLI args + config snapshot to results_dir/settings.json.

    Args:
        args: argparse.Namespace from CLI.
        results_dir: Directory to save settings.json in.
        extra: Additional key-value pairs to include.
    """
    def _convert(obj: object) -> object:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 6)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj))

    settings: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "args": {k: v for k, v in vars(args).items() if v is not None},
        "controlled_variables": {
            "experiment_defaults": EXPERIMENT_DEFAULTS,
            "masking_config": MASKING_CONFIG,
        },
    }
    if extra:
        settings.update(extra)

    # Add model config if args has model info
    model_key = getattr(args, "model", None)
    if model_key and model_key in MODEL_CONFIGS:
        settings["model_config"] = MODEL_CONFIGS[model_key]

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "settings.json"
    with open(path, "w") as f:
        json.dump(settings, f, indent=2, default=_convert)
    print(f"  Saved: {path}")
