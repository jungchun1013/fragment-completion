#!/bin/bash
# Visualize gestalt + embedding for first N images.
set -e
cd "$(dirname "$0")/.."

N=${1:-5}
ENCODERS="clip mae dino dinov2 ijepa vit_sup"

for i in $(seq 0 $((N-1))); do
    echo "=== Image idx=$i ==="
    uv run python visualize.py all --image-idx "$i" --encoders $ENCODERS
done

echo "Visualizations in results/visualizations/"
