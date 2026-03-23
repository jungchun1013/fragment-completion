#!/bin/bash
# Re-generate all plots from existing results.json (no GPU needed).
set -e
cd "$(dirname "$0")/.."

RESULTS=${1:-results/results.json}
OUT=${2:-results}

uv run python plot.py all \
    --results "$RESULTS" \
    --out-dir "$OUT"

echo "All plots generated in $OUT/"
