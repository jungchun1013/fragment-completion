"""Plot similarity difference (target - all) across fragmentation levels.

Generates one plot per image_type (original, gray, lined), each showing
all encoders. Two subplots: mnemonic diff and semantic diff.

Usage:
    uv run python plot_similarity_diff.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/encoders")
OUT_DIR = Path("results/similarity_diff")
IMAGE_TYPES = ["original", "gray", "lined"]
LEVELS = list(range(1, 9))

# Visibility schedule for x-tick labels
def vis_ratio(L):
    return 0.7 ** (8 - L)

# Load all encoder data
encoder_data = {}
for enc_dir in sorted(RESULTS_DIR.iterdir()):
    json_path = enc_dir / "similarity_analysis.json"
    if not json_path.exists():
        continue
    with open(json_path) as f:
        encoder_data[enc_dir.name] = json.load(f)

print(f"Loaded {len(encoder_data)} encoders: {list(encoder_data.keys())}")

# Nicer display names
DISPLAY_NAMES = {
    "clip": "CLIP",
    "dino_v1": "DINO v1",
    "dinov2": "DINOv2",
    "mae": "MAE",
    "i_jepa": "I-JEPA",
    "vit_supervised": "ViT-sup",
}

# Colors per encoder (consistent across plots)
COLORS = {
    "clip": "#1f77b4",
    "dino_v1": "#ff7f0e",
    "dinov2": "#2ca02c",
    "mae": "#d62728",
    "i_jepa": "#9467bd",
    "vit_supervised": "#8c564b",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

for img_type in IMAGE_TYPES:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for enc_key, data in encoder_data.items():
        if img_type not in data:
            continue
        d = data[img_type]
        label = DISPLAY_NAMES.get(enc_key, enc_key)
        color = COLORS.get(enc_key, None)

        # Mnemonic diff: target - all
        mn_target = [d["mnemonic_target"][str(L)]["mean"] for L in LEVELS]
        mn_all = [d["mnemonic_all"][str(L)]["mean"] for L in LEVELS]
        mn_diff = [t - a for t, a in zip(mn_target, mn_all)]

        # Propagate std: sqrt(std_t^2 + std_a^2)
        mn_target_std = [d["mnemonic_target"][str(L)]["std"] for L in LEVELS]
        mn_all_std = [d["mnemonic_all"][str(L)]["std"] for L in LEVELS]
        mn_diff_std = [np.sqrt(st**2 + sa**2) for st, sa in zip(mn_target_std, mn_all_std)]

        ax1.plot(LEVELS, mn_diff, "o-", label=label, color=color, linewidth=2, markersize=5)
        ax1.fill_between(LEVELS,
                         [m - s for m, s in zip(mn_diff, mn_diff_std)],
                         [m + s for m, s in zip(mn_diff, mn_diff_std)],
                         alpha=0.12, color=color)

        # Semantic diff: same_cat - all_cat
        sem_same = [d["semantic_same_cat"][str(L)]["mean"] for L in LEVELS]
        sem_all = [d["semantic_all_cat"][str(L)]["mean"] for L in LEVELS]
        sem_diff = [t - a for t, a in zip(sem_same, sem_all)]

        sem_same_std = [d["semantic_same_cat"][str(L)]["std"] for L in LEVELS]
        sem_all_std = [d["semantic_all_cat"][str(L)]["std"] for L in LEVELS]
        sem_diff_std = [np.sqrt(st**2 + sa**2) for st, sa in zip(sem_same_std, sem_all_std)]

        ax2.plot(LEVELS, sem_diff, "o-", label=label, color=color, linewidth=2, markersize=5)
        ax2.fill_between(LEVELS,
                         [m - s for m, s in zip(sem_diff, sem_diff_std)],
                         [m + s for m, s in zip(sem_diff, sem_diff_std)],
                         alpha=0.12, color=color)

    # Format axes
    xtick_labels = [f"L{L}\n({vis_ratio(L):.0%})" for L in LEVELS]
    for ax, title in [(ax1, "Mnemonic (target - all)"),
                      (ax2, "Semantic (same_cat - all_cat)")]:
        ax.set_xlabel("Fragmentation Level", fontsize=12)
        ax.set_ylabel("Similarity Difference", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(LEVELS)
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle(f"Similarity Difference — {img_type}", fontsize=15, fontweight="bold")
    fig.tight_layout()

    save_path = OUT_DIR / f"similarity_diff_{img_type}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

print("\nDone!")
