"""DEPRECATED: Use experiments.exp1.run instead.

Usage:
    uv run python -m experiments.exp1.run
    uv run python -m experiments.exp1.run --max-images 5
    uv run python -m experiments.exp1.run --encoders clip mae dino
"""

import sys

print(
    "DEPRECATED: run.py has moved.\n"
    "  Use: uv run python -m experiments.exp1.run [args]\n"
    "  See: uv run python -m experiments.exp1.run --help",
    file=sys.stderr,
)
sys.exit(1)
