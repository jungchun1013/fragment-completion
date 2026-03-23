"""Generate gestalt visualizations for images 001-005.

Usage:
    cd fragment-completion
    uv run python run_gestalt_vis.py
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import torch

from models.registry import get_encoder
from src.dataset import get_dataset
from src.gestalt import visualize_gestalt_single

OUT_DIR = "output/completion"
ENCODER_NAMES = ["clip", "mae", "dino", "dinov2", "ijepa", "vit_sup"]
IMAGE_INDICES = list(range(5))  # 0-4 → image IDs 001-005

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load dataset
dataset = get_dataset("fragment_v2")
print(f"Dataset: {len(dataset)} images")

# Load all encoders once
encoders = {}
for enc_name in ENCODER_NAMES:
    try:
        enc = get_encoder(enc_name, device=device)
        _ = enc.model  # trigger lazy load
        encoders[enc.name] = enc
        print(f"  Loaded {enc.name}")
    except Exception as e:
        print(f"  [SKIP] {enc_name}: {e}")

print(f"\n{len(encoders)} encoders ready: {list(encoders.keys())}\n")

# Generate visualizations
for idx in IMAGE_INDICES:
    sample = dataset[idx]
    image_id = sample["image_id"]
    save_path = f"{OUT_DIR}/gestalt_vis_{image_id}.png"
    print(f"Image idx={idx}, id={image_id} → {save_path}")
    visualize_gestalt_single(
        encoders, dataset, image_idx=idx, save_path=save_path,
    )

print("\nDone!")
