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
single-dominant-object images, crops objects using bbox annotations, and
saves reusable cropped images + segmentation masks.

Design:
  - Crop objects to remove scene context — encoder must rely on object
    visual information alone.
  - 4 supercategories x 5 categories x 30 images = 600 images.
  - Filter: object must be the largest annotation AND bbox >= 96x96.

Usage:
    uv run python data/prepare_coco.py [--data-root data/coco_subset]

Output:
    data/coco_subset/
        metadata.json
        images/             # bbox-cropped object images (PNG, white bg outside mask)
        masks/              # binary segmentation masks matching crops
"""

from __future__ import annotations

import argparse
import json
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

# 4 Rosch-style natural superordinate categories x 5 basic-level categories
TARGET_SUPERCATEGORIES: dict[str, list[str]] = {
    "animal":    ["cat", "dog", "horse", "elephant", "cow"],
    "vehicle":   ["bus", "car", "motorcycle", "truck", "train"],
    "furniture": ["chair", "couch", "bed", "dining table", "toilet"],
    "food":      ["pizza", "cake", "sandwich", "banana", "broccoli"],
}

CATS_PER_SUPERCAT = 5     # 4 supercategories x 5 = 20 categories
IMAGES_PER_CAT = 30
MIN_BBOX_SIDE = 96        # minimum bbox width AND height in pixels
BBOX_PAD_RATIO = 0.1      # 10% padding around bbox crop


def _download_if_needed(url: str, dest: Path) -> Path:
    """Download and extract a zip if the target doesn't exist."""
    zip_name = url.split("/")[-1]
    zip_path = dest.parent / zip_name

    if not zip_path.exists():
        print(f"  Downloading {zip_name}...")
        urllib.request.urlretrieve(url, zip_path)

    print(f"  Extracting {zip_name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest.parent)

    return dest


def _ensure_coco(raw_root: Path) -> tuple[Path, Path]:
    """Ensure COCO val2017 images and annotations exist."""
    images_dir = raw_root / "val2017"
    ann_file = raw_root / "annotations" / "instances_val2017.json"

    if not images_dir.exists():
        _download_if_needed(COCO_IMAGE_URL, images_dir)
    if not ann_file.exists():
        _download_if_needed(COCO_ANN_URL, raw_root / "annotations")

    return images_dir, ann_file


def _filter_bbox(
    coco: COCO,
    cat_id: int,
    img_ids: list[int],
    min_side: int,
) -> list[dict]:
    """Filter images where target category has bbox >= min_side.

    No dominance requirement — we crop to bbox so scene context is removed.
    Picks the largest annotation of the target category per image.

    Returns:
        List of {image_id, ann_id, bbox, area_ratio}.
    """
    results: list[dict] = []

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_area = img_info["width"] * img_info["height"]

        all_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        cat_anns = [a for a in all_anns if a["category_id"] == cat_id]
        if not cat_anns:
            continue

        # Pick largest annotation of this category
        cat_anns.sort(key=lambda a: a["area"], reverse=True)
        ann = cat_anns[0]
        bbox = ann["bbox"]  # [x, y, w, h]

        if bbox[2] < min_side or bbox[3] < min_side:
            continue

        results.append({
            "image_id": img_id,
            "ann_id": ann["id"],
            "bbox": bbox,
            "area_ratio": round(ann["area"] / img_area, 4),
        })

    return results


def _crop_with_padding(
    img: Image.Image,
    mask_np: np.ndarray,
    bbox: list[float],
    pad_ratio: float,
) -> tuple[Image.Image, np.ndarray]:
    """Crop image and mask using bbox with padding. Fill outside mask with white.

    Args:
        img: Full PIL image.
        mask_np: Binary mask [H, W] (0 or 1).
        bbox: [x, y, w, h] COCO format.
        pad_ratio: Fraction of bbox size to add as padding.

    Returns:
        (cropped_image, cropped_mask) — white background outside object mask.
    """
    W, H = img.size
    x, y, w, h = bbox

    # Add padding
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(W, int(x + w) + pad_x)
    y2 = min(H, int(y + h) + pad_y)

    # Crop
    img_crop = img.crop((x1, y1, x2, y2))
    mask_crop = mask_np[y1:y2, x1:x2]

    # Apply white background outside mask
    img_np = np.array(img_crop)
    img_np[mask_crop == 0] = 255
    img_crop = Image.fromarray(img_np)

    return img_crop, mask_crop


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

    all_cats = coco.loadCats(coco.getCatIds())
    name_to_catid = {c["name"]: c["id"] for c in all_cats}
    name_to_supercat = {c["name"]: c["supercategory"] for c in all_cats}

    # 3. Filter and select — evenly across supercategories
    print("=== Filtering images (bbox >= 96x96, largest object) ===")
    rng = np.random.RandomState(seed)
    selected_categories: list[str] = []
    selected_supercats: list[str] = []
    cat_to_supercat: dict[str, str] = {}
    all_selected: list[dict] = []

    for supercat, candidates_list in TARGET_SUPERCATEGORIES.items():
        print(f"\n  [{supercat}] (target: {CATS_PER_SUPERCAT} categories)")
        picked = 0
        for cat_name in candidates_list:
            if picked >= CATS_PER_SUPERCAT:
                break

            if cat_name not in name_to_catid:
                print(f"    WARNING: '{cat_name}' not in COCO, skipping")
                continue

            cat_id = name_to_catid[cat_name]
            img_ids = coco.getImgIds(catIds=cat_id)

            candidates = _filter_bbox(
                coco, cat_id, img_ids, MIN_BBOX_SIDE,
            )

            if len(candidates) < IMAGES_PER_CAT:
                print(f"    SKIP: '{cat_name}' has only {len(candidates)} "
                      f"qualifying (need {IMAGES_PER_CAT})")
                continue

            chosen_idx = rng.choice(len(candidates), size=IMAGES_PER_CAT,
                                    replace=False)
            chosen = [candidates[i] for i in chosen_idx]

            selected_categories.append(cat_name)
            if supercat not in selected_supercats:
                selected_supercats.append(supercat)
            cat_to_supercat[cat_name] = supercat

            for item in chosen:
                item["category"] = cat_name
                item["category_id"] = cat_id
                item["supercategory"] = supercat
            all_selected.extend(chosen)
            picked += 1
            print(f"    {cat_name}: {len(candidates)} candidates -> "
                  f"selected {IMAGES_PER_CAT}")

        if picked < CATS_PER_SUPERCAT:
            print(f"    WARNING: only {picked}/{CATS_PER_SUPERCAT} categories "
                  f"for [{supercat}]")

    print(f"\n  Total: {len(selected_supercats)} supercategories, "
          f"{len(selected_categories)} categories, "
          f"{len(all_selected)} images")

    # 4. Crop objects and save
    print("\n=== Cropping objects and saving ===")
    out_images = data_root / "images"
    out_masks = data_root / "masks"
    out_images.mkdir(exist_ok=True)
    out_masks.mkdir(exist_ok=True)

    metadata_images: list[dict] = []
    for i, item in enumerate(all_selected):
        img_info = coco.loadImgs(item["image_id"])[0]
        src_path = images_dir / img_info["file_name"]

        # Load image and mask
        img = Image.open(src_path).convert("RGB")
        ann = coco.loadAnns(item["ann_id"])[0]
        mask = coco.annToMask(ann)  # [H, W] binary

        # Crop with padding, white background outside mask
        img_crop, mask_crop = _crop_with_padding(
            img, mask, item["bbox"], BBOX_PAD_RATIO,
        )

        # Save cropped image and mask
        stem = img_info["file_name"].replace(".jpg", "")
        img_path = out_images / f"{stem}.png"
        mask_path = out_masks / f"{stem}.png"
        img_crop.save(img_path)
        Image.fromarray((mask_crop * 255).astype(np.uint8)).save(mask_path)

        cw, ch = img_crop.size
        metadata_images.append({
            "id": stem,
            "file_name": f"{stem}.png",
            "name": item["category"],
            "category": item["category"],
            "supercategory": item["supercategory"],
            "coco_image_id": item["image_id"],
            "coco_ann_id": item["ann_id"],
            "bbox": item["bbox"],
            "area_ratio": item["area_ratio"],
            "crop_width": cw,
            "crop_height": ch,
            "orig_width": img_info["width"],
            "orig_height": img_info["height"],
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_selected)} images processed")

    # 5. Save metadata
    metadata = {
        "supercategories": selected_supercats,
        "categories": selected_categories,
        "cat_to_supercat": cat_to_supercat,
        "images": metadata_images,
        "filter_cfg": {
            "min_bbox_side": MIN_BBOX_SIDE,
            "bbox_pad_ratio": BBOX_PAD_RATIO,
            "cats_per_supercat": CATS_PER_SUPERCAT,
            "images_per_cat": IMAGES_PER_CAT,
            "seed": seed,
        },
    }
    meta_path = data_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Done ===")
    print(f"  Supercategories: {selected_supercats}")
    print(f"  Categories: {len(selected_categories)}")
    print(f"  Images: {len(metadata_images)}")
    print(f"  Saved: {data_root}/")


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
