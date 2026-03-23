# Fragment Completion

Evaluate vision encoders on fragment completion tasks with progressive masking.

Given an image fragmented into patches, how well can different encoders:
- **Gestalt**: Segment foreground from background (2-means on patch features, measured by IoU)
- **Mnemonic**: Recognize the same image (cosine similarity + retrieval accuracy)
- **Semantic**: Classify the image category (prototype classification + CLIP zero-shot)
- **Similarity**: Distinguish target vs. all (mnemonic) and same-category vs. all (semantic)

## Setup

```bash
cd fragment-completion
source .venv/bin/activate   # or: uv sync
```

## Quick Start

```bash
bash scripts/run_quick.sh   # 1 encoder, 2 images — sanity check
```

## Usage

### `run.py` — Run experiments

```bash
# All tasks, all image types
uv run python run.py --encoders clip mae dino --image-type original gray lined

# Specific tasks only
uv run python run.py --encoders clip --tasks gestalt mnemonic --max-images 10

# All options
uv run python run.py --help
```

Produces `results/results.json` (unified) plus per-encoder and per-image-type breakdowns.

### `plot.py` — Generate plots from results

```bash
uv run python plot.py all --results results/results.json           # everything
uv run python plot.py combined --results results/results.json       # all encoders combined
uv run python plot.py by-encoder --results results/results.json     # per-encoder, lines=image types
uv run python plot.py similarity-diff --results results/results.json  # target-all differences
```

Also accepts a directory path for legacy results: `--results results/`

### `visualize.py` — Single-instance visualizations

```bash
uv run python visualize.py gestalt --image-idx 0 --encoders clip mae dino
uv run python visualize.py embedding --image-idx 0 --encoders clip mae dino
uv run python visualize.py all --image-idx 0
```

## Scripts

| Script | Description |
|---|---|
| `scripts/run_all.sh` | Full pipeline: all encoders x all image types |
| `scripts/run_quick.sh` | Quick test: 1 encoder, 2 images |
| `scripts/plot_all.sh` | Re-generate all plots (no GPU) |
| `scripts/visualize_samples.sh [N]` | Visualize first N images (default: 5) |

## Results Structure

```
results/
├── results.json              # Unified: all encoders x image types x metrics
├── image_types/{type}/       # Per-image-type plots + results.json
├── encoders/{encoder}/       # Per-encoder plots + similarity_analysis.json
├── all_encoders/             # Combined plots + similarity diff
└── visualizations/           # Gestalt grids + embedding trajectories
```

## Supported Encoders

| Registry Key | Display Name | Pre-training |
|---|---|---|
| `clip` | CLIP | Image-text contrastive |
| `dino` | DINO-v1 | Self-distillation |
| `dinov2` | DINOv2 | Self-distillation v2 |
| `mae` | MAE | Masked image modeling |
| `mae_ft` | MAE-FT | MAE fine-tuned |
| `ijepa` | I-JEPA | Predictive (latent) |
| `vit_sup` | ViT-supervised | Supervised classification |
| `siglip` | SigLIP | Sigmoid contrastive |
| `simclr` | SimCLR | Instance discrimination |
| `resnet` | ResNet-50 | Supervised CNN baseline |
| `nepa` | NEPA | Predictive |
| `llava` | LLaVA | Vision-language |
| `qwen2vl` | Qwen2-VL | Vision-language |

## Adding a New Encoder

1. Create `models/your_encoder.py` with a class extending `BaseEncoder`
2. Decorate with `@register("your_key")`
3. Add entry to `src/config.py` `ENCODER_META`
4. Add to `models/registry.py` `_LAZY_MODULES`
