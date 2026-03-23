#!/bin/bash
# Quick sanity check: 1 encoder, 2 images, 1 image type.
set -e
cd "$(dirname "$0")/.."

uv run python run.py \
    --encoders clip \
    --image-type original \
    --max-images 2 \
    --out-dir results/quick

uv run python plot.py all \
    --results results/quick/results.json \
    --out-dir results/quick

echo "Quick test done. Results in results/quick/"
