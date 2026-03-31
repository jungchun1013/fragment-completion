# CLAUDE.md - Fragment Completion Experiments

## Project
Fragment completion experiements on vision encoders
Root: /nfs/turbo/coe-chaijy/jungchun/vault/a-architecture/sandbox/fragment-completion/

## Stack
Python 3.x, PyTorch, NumPy, Hydra, uv, Slurm

## Key Paths
  fragment-completion/
  ├── models/                 # encoder registry (unchanged)
  │   ├── encoder.py          # BaseEncoder ABC
  │   ├── processor.py        # ImageProcessor, to_transform
  │   ├── registry.py         # get_encoder(), register()
  │   └── encoders/           # clip.py, dino.py, dinov2.py, ...
  ├── src/                    # shared infrastructure
  │   ├── config.py           # encoder metadata, plot style, result paths
  │   ├── dataset.py          # FragmentV2, ADE20K, COCOSubset loaders
  │   ├── masking.py          # progressive patch-level masking
  │   ├── utils.py            # embedding, plotting, retrieval metrics
  │   ├── experiment_config.py # MODEL_CONFIGS, save_experiment_settings
  │   └── metrics/srss.py     # SRSS metric
  ├── experiments/
  │   ├── exp1/               # Exp 1: all-encoder fragment completion
  │   │   ├── run.py          # entry point: uv run python -m experiments.exp1.run
  │   │   ├── gestalt.py      # gestalt completion task
  │   │   ├── mnemonic.py     # mnemonic completion task
  │   │   ├── semantic.py     # semantic completion task
  │   │   ├── similarity.py   # similarity analysis task
  │   │   └── plot.py         # standalone plotting from results.json
  │   └── exp2/               # Exp 2: CLIP + DINOv2 interpretability
  │       ├── clip_interp.py  # CLIP mechanistic interpretability
  │       ├── dinov2_interp.py # DINOv2 mechanistic interpretability
  │       ├── ground_retrieval.py # ground-truth retrieval experiments
  │       └── plot.py         # regenerate plots from JSON
  ├── analysis/               # cross-experiment visualization
  ├── tests/                  # pytest unit tests
  ├── results/
  │   ├── exp1/               # all-encoder results + plots
  │   └── exp2/               # interpretability results
  ├── data/                   # datasets
  └── scripts/                # shell scripts

## Datasets
- **Fragment V2** (primary): 260 white-background object images, 3 variants each (original, gray, lined)
  - Path: `data/fragment_v2/`
  - Metadata: `metadata.json` (image IDs, names, categories)
- **ADE20K** (secondary): 109 validation images, ~40 scene classes
  - Path: `/nfs/turbo/coe-chaijy/jungchun/vault/a-MI/p-visual-grounding/vit-object-binding/libs/ADE20K/dataset/ADE20K_2021_17_01`
- Access via `src/dataset.py`: `get_dataset(name, root, image_type)`

## Run Experiments
```bash
# Exp 1: all encoders, fragment completion
uv run python -m experiments.exp1.run --encoders clip mae dino ijepa vit_sup
uv run python -m experiments.exp1.run --max-images 5 --tasks gestalt
uv run python -m experiments.exp1.plot --results results/exp1/results.json

# Exp 2: CLIP / DINOv2 interpretability
uv run python -m experiments.exp2.ground_retrieval --model clip retrieve
uv run python -m experiments.exp2.clip_interp zeroshot --max-images 10
uv run python -m experiments.exp2.dinov2_interp all

# Tests
uv run pytest tests/ -v
```

## File Header (new files)
```python
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : <filename>.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : <MM-DD-YYYY>
#
# This file is part of *.
# Distributed under terms of the MIT license.
```

## Code Style (I'm learning good habits)
When modifying code, always:
0. Follow PEP 8 - help me maintain clean code
1. Add type hints - helps me catch bugs early
2. Add Google-style docstrings - forces me to understand what I wrote
3. Flag functions >50 lines - I tend to write spaghetti
4. Flag vague names - I default to `x`, `tmp`, `data`
5. Keep everythin minimal and clear. Don't add abstractions I didn't ask for. One file if possible.

## Visualization Style

### Color
1 dimension (e.g., diff models) = Tab10
2 dimension (e.g., diff models * 3 type) = Tab20c
heatmap -> Blues in Seaborn

### Plot
PLOT_STYLE = {
    "linewidth": 2,
    "std_alpha": 0.2,
    "tick_width": 2,
    "tick_labelsize": 14,
    "label_fontsize": 16,
    "legend_fontsize": 14,
    "legend_loc": "outside lower center",  # legend always below the plot
    "subplot_title_fontsize": 18,
    "suptitle_fontsize": 20,
    "marker": "o",
    "markersize": 6,
    "dpi": 150,
    "subplot_size": (6.4, 4.8),  # (w, h) desired axes area per subplot
}

## Git (I often forget to commit)
After changes, remind me:
```
git add <files>
git commit -m "<TYPE>: <description>"
```
Types of message: ADD, REMOVE, UPDATE, REFACTOR, FIX, EXP, DATA, DOCS, STYLE

## Ignore
- `results/`, `__pycache__/`, `*.csv`, `*.png`. `*.jpg`

# Others
Every time you response, You have to call my name Yore