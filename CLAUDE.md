# CLAUDE.md - Fragment Completion Experiments

## Project
Fragment completion experiements on vision encoders
Root: /nfs/turbo/coe-chaijy/jungchun/vault/a-architecture/sandbox/fragment-completion/

## Stack
Python 3.x, PyTorch, NumPy, Hydra, uv, Slurm

## Key Paths
  fragment-completion/          
  ├── run.py              # main entry point
  ├── configs/            # Hydra configs (don't modify without permission)
  ├── experiments/        # unit experiments and utils
  ├── models/             # models (ignore other subfolders)
  ├── scripts/            # scripts for running experiments and testing
  ├── results/            # output results
  │   ├── results.json        # aggregated metrics
  │   ├── all_encoders/       # multi-encoder comparisons
  │   ├── encoders/           # per-encoder results (clip, dino_v1, dinov2, mae, ...)
  │   ├── image_types/        # by variant (original, gray, lined)
  │   ├── visualization/      # generated plots
  │   └── deprecated/         # legacy experiment runs
  └── analysis/           # plotting, analysis, visualization

## Datasets
- **Fragment V2** (primary): 260 white-background object images, 3 variants each (original, gray, lined)
  - Path: `data/fragment_v2/`
  - Metadata: `metadata.json` (image IDs, names, categories)
- **ADE20K** (secondary): 109 validation images, ~40 scene classes
  - Path: `/nfs/turbo/coe-chaijy/jungchun/vault/a-MI/p-visual-grounding/vit-object-binding/libs/ADE20K/dataset/ADE20K_2021_17_01`
- Access via `src/dataset.py`: `get_dataset(name, root, image_type)`

## Run Experiments
```bash
uv run python run.py <hydra overrides>
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