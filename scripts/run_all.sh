#!/bin/bash
# Exp 1: run all encoders, all image types, all tasks.
set -e
cd "$(dirname "$0")/.."

ENCODERS="clip mae dino dinov2 ijepa vit_sup"
IMAGE_TYPES="original gray lined"

uv run python -m experiments.exp1.run \
    --encoders $ENCODERS \
    --image-type $IMAGE_TYPES \
    --out-dir results/exp1 \
    --plot

echo "Done. Results in results/exp1/"
