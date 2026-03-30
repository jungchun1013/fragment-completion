#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : prepare_coco.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-30-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Prepare COCO val2017 subset for fragment completion experiments.

Downloads COCO val2017 images + annotations if not present, filters for
single-dominant-object images, selects 20 diverse categories x 50 images,
and saves a reusable metadata JSON + segmentation masks.

Usage:
    uv run python data/prepare_coco.py [--data-root data/coco_subset]

Output:
    data/coco_subset/
        metadata.json       # {images: [...], categories: [...], filter_cfg: {...}}
        images/             # symlinks or copies of selected COCO images
        masks/              # per-image binary foreground masks (PNG)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COCO_IMAGE_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

# Candidate categories ordered by preference (pick first 20 that pass)
TARGET_CATEGORIES = [
    # animals
    "dog", "cat", "bird", "horse", "elephant", "bear", "zebra", "giraffe",
    "cow", "sheep",
    # vehicles
    "car", "bus", "airplane", "bicycle", "boat", "truck", "motorcycle", "train",
    # furniture
    "chair", "couch", "bed", "dining table",
    # food
    "banana", "pizza", "cake", "apple", "orange", "broccoli", "donut",
    # misc
    "laptop", "clock", "umbrella", "surfboard", "skateboard", "toilet",
    "teddy bear", "suitcase", "kite", "snowboard", "frisbee",
]

MAX_CATEGORIES = 20
IMAGES_PER_CAT = 30
MIN_AREA_RATIO = 0.08    # dominant object must cover >8% of image
DOMINANCE_RATIO = 1.5    # largest object area > 1.5x second largest


def _download_if_needed(url: str, dest: Path) -> Path:
    """Download and extract a zip if the target doesn't exist."""
    zip_name = url.split("/")[-1]
    zip_path = dest.parent / zip_name

    if not zip_path.exists():
        print(f"  Downloading {zip_name}...")
        urllib.request.urlretrieve(url, zip_path)

    # Extract
    print(f"  Extracting {zip_name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest.parent)

    return dest


def _ensure_coco(raw_root: Path) -> tuple[Path, Path]:
    """Ensure COCO val2017 images and annotations exist.

    Returns:
        (images_dir, annotation_file)
    """
    images_dir = raw_root / "val2017"
    ann_file = raw_root / "annotations" / "instances_val2017.json"

    if not images_dir.exists():
        _download_if_needed(COCO_IMAGE_URL, images_dir)

    if not ann_file.exists():
        _download_if_needed(COCO_ANN_URL, raw_root / "annotations")

    return images_dir, ann_file


def _filter_single_dominant(
    coco: COCO,
    cat_id: int,
    img_ids: list[int],
    min_area_ratio: float,
    dominance_ratio: float,
) -> list[dict]:
    """Filter images with a single dominant object of the given category.

    Returns list of {image_id, ann_id, area_ratio} for qualifying images.
    """
    results: list[dict] = []

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_area = img_info["width"] * img_info["height"]

        all_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if not all_anns:
            continue

        # Sort all annotations by area descending
        all_anns.sort(key=lambda a: a["area"], reverse=True)

        # Find annotations of the target category
        cat_anns = [a for a in all_anns if a["category_id"] == cat_id]
        if not cat_anns:
            continue

        # Largest object in this category
        largest_cat_ann = cat_anns[0]
        largest_area = largest_cat_ann["area"]

        # Check min area ratio
        if largest_area / img_area < min_area_ratio:
            continue

        # Check dominance: largest object overall must be this annotation,
        # AND it must be > dominance_ratio * second largest overall
        if all_anns[0]["id"] != largest_cat_ann["id"]:
            continue  # another category has a bigger object

        if len(all_anns) >= 2:
            second_area = all_anns[1]["area"]
            if second_area > 0 and largest_area / second_area < dominance_ratio:
                continue

        results.append({
            "image_id": img_id,
            "ann_id": largest_cat_ann["id"],
            "area_ratio": round(largest_area / img_area, 4),
        })

    return results


def prepare(
    data_root: Path,
    raw_root: Path | None = None,
    seed: int = 42,
) -> None:
    """Main preparation pipeline."""
    raw_root = raw_root or (data_root.parent / "coco_raw")
    raw_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    # 1. Ensure raw COCO data
    print("=== Ensuring COCO val2017 data ===")
    images_dir, ann_file = _ensure_coco(raw_root)

    # 2. Load COCO API
    print("=== Loading COCO annotations ===")
    coco = COCO(str(ann_file))

    # Build category name -> id mapping
    all_cats = coco.loadCats(coco.getCatIds())
    name_to_catid = {c["name"]: c["id"] for c in all_cats}

    # 3. Filter and select
    print("=== Filtering images ===")
    rng = np.random.RandomState(seed)
    selected_categories: list[str] = []
    all_selected: list[dict] = []

    for cat_name in TARGET_CATEGORIES:
        if len(selected_categories) >= MAX_CATEGORIES:
            break

        if cat_name not in name_to_catid:
            print(f"  WARNING: '{cat_name}' not found in COCO, skipping")
            continue

        cat_id = name_to_catid[cat_name]
        img_ids = coco.getImgIds(catIds=cat_id)

        candidates = _filter_single_dominant(
            coco, cat_id, img_ids, MIN_AREA_RATIO, DOMINANCE_RATIO,
        )

        if len(candidates) < IMAGES_PER_CAT:
            print(f"  SKIP: '{cat_name}' has only {len(candidates)} "
                  f"qualifying images (need {IMAGES_PER_CAT})")
            continue

        # Random sample
        chosen_idx = rng.choice(len(candidates), size=IMAGES_PER_CAT,
                                replace=False)
        chosen = [candidates[i] for i in chosen_idx]

        selected_categories.append(cat_name)
        for item in chosen:
            item["category"] = cat_name
            item["category_id"] = cat_id
        all_selected.extend(chosen)
        print(f"  {cat_name}: {len(candidates)} candidates -> "
              f"selected {IMAGES_PER_CAT}")

    print(f"\n  Total: {len(selected_categories)} categories, "
          f"{len(all_selected)} images")

    # 4. Copy images and generate masks
    print("\n=== Saving images and masks ===")
    out_images = data_root / "images"
    out_masks = data_root / "masks"
    out_images.mkdir(exist_ok=True)
    out_masks.mkdir(exist_ok=True)

    metadata_images: list[dict] = []
    for i, item in enumerate(all_selected):
        img_info = coco.loadImgs(item["image_id"])[0]
        src_path = images_dir / img_info["file_name"]

        # Copy/symlink image
        dst_path = out_images / img_info["file_name"]
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        # Generate binary mask from the dominant annotation
        ann = coco.loadAnns(item["ann_id"])[0]
        mask = coco.annToMask(ann)  # [H, W] binary
        mask_path = out_masks / f"{img_info['file_name'].replace('.jpg', '.png')}"
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

        # Infer object name from COCO (category name is the best we have)
        metadata_images.append({
            "id": img_info["file_name"].replace(".jpg", ""),
            "file_name": img_info["file_name"],
            "name": item["category"],  # COCO doesn't have instance names
            "category": item["category"],
            "coco_image_id": item["image_id"],
            "coco_ann_id": item["ann_id"],
            "area_ratio": item["area_ratio"],
            "width": img_info["width"],
            "height": img_info["height"],
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_selected)} images processed")

    # 5. Save metadata
    metadata = {
        "categories": selected_categories,
        "images": metadata_images,
        "filter_cfg": {
            "min_area_ratio": MIN_AREA_RATIO,
            "dominance_ratio": DOMINANCE_RATIO,
            "images_per_cat": IMAGES_PER_CAT,
            "seed": seed,
        },
    }
    meta_path = data_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Done ===")
    print(f"  Categories: {len(selected_categories)}")
    print(f"  Images: {len(metadata_images)}")
    print(f"  Saved: {data_root}/")
    print(f"  Metadata: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare COCO val2017 subset")
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/coco_subset"))
    parser.add_argument("--raw-root", type=Path, default=None,
                        help="Where to download/cache raw COCO data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare(args.data_root, args.raw_root, args.seed)


if __name__ == "__main__":
    main()
