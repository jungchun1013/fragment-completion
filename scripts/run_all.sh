#!/bin/bash
# Full pipeline: run all encoders, all image types, all tasks.
set -e
cd "$(dirname "$0")/.."

ENCODERS="clip mae dino dinov2 ijepa vit_sup"
IMAGE_TYPES="original gray lined"

uv run python run.py \
    --encoders $ENCODERS \
    --image-type $IMAGE_TYPES \
    --out-dir results

uv run python plot.py all \
    --results results/results.json \
    --out-dir results

echo "Done. Results in results/"
