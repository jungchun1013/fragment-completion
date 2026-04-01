#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : saliency_masking.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Saliency-based masking experiment.

3 masking conditions × 2 encoders × 8 levels × 600 images.
Conditions: random, DINOv2 salient-first, CLIP salient-first.
Salient-first = remove most important patches first.

Subcommands:
  saliency    Precompute and save saliency maps for DINOv2 and CLIP
  evaluate    Run all readout tasks under each condition
  plot        Generate comparison plots
  all         Run everything

Usage:
    uv run python -m experiments.exp2.saliency_masking saliency --max-images 5
    uv run python -m experiments.exp2.saliency_masking evaluate
    uv run python -m experiments.exp2.saliency_masking plot
    uv run python -m experiments.exp2.saliency_masking all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.dataset import get_dataset
from src.masking import (
    get_mask_levels,
    get_visibility_ratio,
    mask_pil_image,
    mask_pil_image_saliency,
)
from src.saliency import clip_gradcam, dinov2_saliency, resample_saliency
from src.utils import (
    compute_category_accuracy,
    compute_exemplar_accuracy,
    compute_retrieval_metrics,
    save_json,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/exp2/saliency_masking")

# Target 4 supercategories × 5 basic each
TARGET_SUPERCATS = ["animal", "vehicle", "furniture", "food"]
TARGET_CATEGORIES = {
    "animal": ["cat", "dog", "horse", "elephant", "cow"],
    "vehicle": ["bus", "car", "motorcycle", "truck", "train"],
    "furniture": ["chair", "couch", "bed", "dining table", "toilet"],
    "food": ["pizza", "cake", "sandwich", "banana", "broccoli"],
}

# Model configs
MODELS = {
    "clip": {
        "img_size": 224,
        "patch_size": 14,
        "grid_size": 16,  # 224 / 14 = 16
        "label": "CLIP ViT-L-14",
    },
    "dinov2": {
        "img_size": 518,
        "patch_size": 14,
        "grid_size": 37,
        "label": "DINOv2+dino.txt",
    },
}

CONDITIONS = ["dinov2_salient", "clip_salient"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_clip(device: str = "cuda") -> tuple[nn.Module, object, object]:
    """Load CLIP ViT-L-14. Returns (model, tokenizer, transform)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai",
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, tokenizer, preprocess


def _load_dinov2(device: str = "cuda") -> tuple[nn.Module, object, object]:
    """Load DINOv2+dino.txt. Returns (model, tokenizer, transform)."""
    sys.path.insert(0, str(
        Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main",
    ))
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
        trust_repo=True,
    )
    model = model.to(device).eval()
    from dinov2.hub.dinotxt import get_tokenizer  # type: ignore
    tokenizer = get_tokenizer()
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, tokenizer, transform


@torch.no_grad()
def _encode_image(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 16,
) -> torch.Tensor:
    """Encode images → [N, D], L2-normalized, CPU."""
    device = next(model.parameters()).device
    parts: list[torch.Tensor] = []
    for start in range(0, imgs.shape[0], batch_size):
        batch = imgs[start:start + batch_size].to(device)
        emb = model.encode_image(batch)
        parts.append(emb.cpu())
    return F.normalize(torch.cat(parts, dim=0).float(), dim=-1)


@torch.no_grad()
def _encode_text(
    model: nn.Module, tokenizer: object, labels: list[str],
    device: str = "cuda",
) -> torch.Tensor:
    """Encode text labels → [C, D], L2-normalized, CPU."""
    prompts = [f"an image of {lab}" for lab in labels]
    if hasattr(tokenizer, "tokenize"):
        tokens = tokenizer.tokenize(prompts).to(device)
    else:
        tokens = tokenizer(prompts).to(device)
    feats = model.encode_text(tokens)
    return F.normalize(feats.float().cpu(), dim=-1)


# ---------------------------------------------------------------------------
# Dataset filtering
# ---------------------------------------------------------------------------

def _filter_dataset(dataset: object, max_images: int | None = None) -> list[int]:
    """Return indices for the 4×5×30=600 target subset."""
    all_cats = set()
    for cats in TARGET_CATEGORIES.values():
        all_cats.update(cats)

    indices = []
    for i in range(len(dataset)):
        s = dataset.samples[i]
        if s["scene_label"] in all_cats:
            indices.append(i)

    if max_images and len(indices) > max_images:
        indices = indices[:max_images]

    return indices


# ---------------------------------------------------------------------------
# Subcommand: saliency
# ---------------------------------------------------------------------------

def run_saliency(
    dataset_name: str = "coco_subset",
    data_root: str | None = None,
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Precompute and save saliency maps for DINOv2 and CLIP."""
    out = RESULTS_DIR / "saliency"
    out.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(dataset_name, root=data_root)
    indices = _filter_dataset(dataset, max_images)
    n = len(indices)
    print(f"  Saliency: {n} images from {len(TARGET_SUPERCATS)} supercategories")

    # --- DINOv2 saliency ---
    print("\n  Computing DINOv2 saliency (register token attention)...")
    model_d, _, transform_d = _load_dinov2(device)
    imgs_d = torch.stack([transform_d(dataset[i]["image_pil"]) for i in indices])
    sal_d = dinov2_saliency(model_d, imgs_d, batch_size=8)
    torch.save({"saliency": sal_d, "indices": indices}, out / "dinov2_saliency.pt")
    print(f"  Saved: {out}/dinov2_saliency.pt  shape={sal_d.shape}")
    del model_d, imgs_d
    torch.cuda.empty_cache()

    # --- CLIP saliency (basic-level text prompt) ---
    print("\n  Computing CLIP saliency (GradCAM on basic-level text)...")
    model_c, tokenizer_c, transform_c = _load_clip(device)

    # Per-image text embedding based on category label
    labels = [dataset[i]["scene_label"] for i in indices]
    unique_labels = sorted(set(labels))
    label_embeds = _encode_text(model_c, tokenizer_c, unique_labels, device)
    label_to_embed = {lab: label_embeds[j] for j, lab in enumerate(unique_labels)}
    text_embeds = torch.stack([label_to_embed[lab] for lab in labels])

    imgs_c = torch.stack([transform_c(dataset[i]["image_pil"]) for i in indices])
    sal_c = clip_gradcam(model_c, imgs_c, text_embeds, batch_size=8)
    torch.save({"saliency": sal_c, "indices": indices}, out / "clip_saliency.pt")
    print(f"  Saved: {out}/clip_saliency.pt  shape={sal_c.shape}")
    del model_c, imgs_c
    torch.cuda.empty_cache()

    # Save index mapping
    save_json(
        {"indices": indices, "labels": labels, "n": n},
        out / "metadata.json",
    )
    print(f"\n  Saliency maps saved to {out}/")


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluate(
    dataset_name: str = "coco_subset",
    data_root: str | None = None,
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Evaluate 3 conditions × 2 encoders × 8 levels."""
    sal_dir = RESULTS_DIR / "saliency"
    if not sal_dir.exists():
        print("  [error] Run 'saliency' subcommand first.")
        return

    # Load saliency maps
    sal_d_data = torch.load(sal_dir / "dinov2_saliency.pt", weights_only=True)
    sal_c_data = torch.load(sal_dir / "clip_saliency.pt", weights_only=True)
    sal_dinov2 = sal_d_data["saliency"]  # [N, 37, 37]
    sal_clip = sal_c_data["saliency"]    # [N, 14, 14]
    indices = sal_d_data["indices"]

    dataset = get_dataset(dataset_name, root=data_root)
    n = len(indices)
    levels = get_mask_levels()

    # Supercategory info
    supercat_labels = dataset.supercategory_labels
    S = len(supercat_labels)
    cat_ids = [dataset[i]["scene_id"] for i in indices]
    supercat_ids = [dataset.samples[i]["supercat_id"] for i in indices]

    # Category info
    all_cats = set()
    for cats in TARGET_CATEGORIES.values():
        all_cats.update(cats)
    active_cat_ids = sorted(set(cat_ids))
    C = len(active_cat_ids)

    print(f"  Evaluate: {n} images, {C} categories, {S} supercategories")
    print(f"  Conditions: {CONDITIONS}")

    for enc_name in ["dinov2", "clip"]:
        cfg = MODELS[enc_name]
        print(f"\n{'=' * 60}")
        print(f"  ENCODER: {cfg['label']}")
        print(f"{'=' * 60}")

        if enc_name == "clip":
            model, tokenizer, transform = _load_clip(device)
        else:
            model, tokenizer, transform = _load_dinov2(device)

        # Precompute galleries (complete images)
        print("  Computing galleries...")
        complete_imgs = torch.stack([
            transform(dataset[i]["image_pil"]) for i in indices
        ])
        image_gallery = _encode_image(model, complete_imgs)  # [N, D]
        gt_instance = torch.arange(n, dtype=torch.long)

        # Text galleries
        cat_labels = [dataset.scene_labels[cid] for cid in active_cat_ids]
        cat_gallery = _encode_text(model, tokenizer, cat_labels, device)
        supercat_gallery = _encode_text(model, tokenizer, supercat_labels, device)

        # Instance text
        instance_labels = [dataset[i]["scene_label"] for i in indices]
        instance_gallery = _encode_text(model, tokenizer, instance_labels, device)

        # Image prototype (per basic-level category)
        D = image_gallery.shape[1]
        img_proto = torch.zeros(C, D)
        proto_count = torch.zeros(C)
        cat_id_remap = {cid: j for j, cid in enumerate(active_cat_ids)}
        remapped_cat_ids = [cat_id_remap[cid] for cid in cat_ids]
        for i in range(n):
            img_proto[remapped_cat_ids[i]] += image_gallery[i]
            proto_count[remapped_cat_ids[i]] += 1
        proto_count = proto_count.clamp(min=1)
        img_proto = F.normalize(img_proto / proto_count.unsqueeze(1), dim=-1)

        # Concept mean prototype (per supercategory, mean of basic-level text)
        concept_proto = torch.zeros(S, D)
        concept_count = torch.zeros(S)
        cat_to_supercat = {}
        for i in range(n):
            cat_to_supercat[cat_ids[i]] = supercat_ids[i]
        for j, cid in enumerate(active_cat_ids):
            if cid in cat_to_supercat:
                sid = cat_to_supercat[cid]
                concept_proto[sid] += cat_gallery[j]
                concept_count[sid] += 1
        concept_count = concept_count.clamp(min=1)
        concept_proto = F.normalize(concept_proto / concept_count.unsqueeze(1), dim=-1)

        for condition in CONDITIONS:
            print(f"\n  --- Condition: {condition} ---")
            results: dict[str, dict] = {}

            for L in levels:
                vis = get_visibility_ratio(L)

                # Generate masked images
                masked_imgs: list[torch.Tensor] = []
                for idx_pos, ds_idx in enumerate(indices):
                    sample = dataset[ds_idx]
                    pil = sample["image_pil"]
                    seg = sample["seg_mask"]

                    if condition == "random":
                        masked = mask_pil_image(
                            pil, L, seg, seed=seed, idx=idx_pos,
                            patch_size=cfg["patch_size"],
                            target_size=cfg["img_size"],
                        )
                    else:
                        # Get saliency for this image, resample if cross-model
                        if condition == "dinov2_salient":
                            sal = sal_dinov2[idx_pos]  # [37, 37]
                        else:
                            sal = sal_clip[idx_pos]    # [14, 14]

                        # Resample to encoder's native grid if needed
                        sal_np = resample_saliency(
                            sal.unsqueeze(0),
                            cfg["grid_size"], cfg["grid_size"],
                        ).squeeze(0).numpy()

                        masked = mask_pil_image_saliency(
                            pil, L, seg, sal_np,
                            salient_first=True,
                            patch_size=cfg["patch_size"],
                            target_size=cfg["img_size"],
                        )

                    masked_imgs.append(transform(masked))

                query = _encode_image(model, torch.stack(masked_imgs))

                # Readout tasks
                img_metrics = compute_retrieval_metrics(query, image_gallery, gt_instance)
                txt_metrics = compute_retrieval_metrics(query, instance_gallery, gt_instance)
                exemplar_acc = compute_exemplar_accuracy(
                    query, image_gallery, remapped_cat_ids, remapped_cat_ids, k=10,
                )
                img_proto_acc = compute_category_accuracy(query, img_proto, remapped_cat_ids)
                concept_proto_acc = compute_category_accuracy(query, concept_proto, supercat_ids)
                cat_concept_acc = compute_category_accuracy(query, supercat_gallery, supercat_ids)

                results[str(L)] = {
                    "image_r1": img_metrics["recall_at_1"],
                    "image_r5": img_metrics["recall_at_5"],
                    "image_mrr": img_metrics["mrr"],
                    "instance_r1": txt_metrics["recall_at_1"],
                    "exemplar_acc": exemplar_acc,
                    "img_proto_acc": img_proto_acc,
                    "concept_proto_acc": concept_proto_acc,
                    "category_acc": cat_concept_acc,
                }

                print(f"    L={L} vis={vis:.3f}  "
                      f"img_r1={img_metrics['recall_at_1']:.4f}  "
                      f"txt_r1={txt_metrics['recall_at_1']:.4f}  "
                      f"exemplar={exemplar_acc:.4f}  "
                      f"img_proto={img_proto_acc:.4f}  "
                      f"concept_proto={concept_proto_acc:.4f}  "
                      f"cat={cat_concept_acc:.4f}")

            # Save
            out_dir = RESULTS_DIR / condition / enc_name
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json(results, out_dir / "results.json")

        del model
        torch.cuda.empty_cache()

    print(f"\n  All results saved to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Subcommand: plot
# ---------------------------------------------------------------------------

def run_plot() -> None:
    """Generate one task-comparison plot per (condition, encoder) pair."""
    from .plot import _plot_tasks, RETRIEVAL_TASKS
    import json

    # Load all results
    all_data: dict[str, dict[str, dict]] = {}  # {condition: {encoder: data}}
    for condition in CONDITIONS:
        all_data[condition] = {}
        for enc_name in ["dinov2", "clip"]:
            path = RESULTS_DIR / condition / enc_name / "results.json"
            if path.exists():
                with open(path) as f:
                    all_data[condition][enc_name] = json.load(f)

    if not any(all_data[c] for c in CONDITIONS):
        print("  [skip] No results found. Run 'evaluate' first.")
        return

    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:
        for enc_name, enc_data in all_data[condition].items():
            save_path = plot_dir / f"{condition}_{enc_name}.png"
            title = f"{MODELS[enc_name]['label']} — {condition}"
            _plot_tasks(enc_data, RETRIEVAL_TASKS, save_path, title)

    print(f"  Plots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for saliency-based masking experiment."""
    parser = argparse.ArgumentParser(
        description="Saliency-based masking experiment",
    )
    parser.add_argument("--dataset", default="coco_subset",
                        choices=["coco_subset"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--results-dir", default=None,
                        help="Override results directory")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("saliency")
    sub.add_parser("evaluate")
    sub.add_parser("plot")
    sub.add_parser("all")

    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    common = dict(
        dataset_name=args.dataset, data_root=args.data_root,
        seed=args.seed, device=args.device, max_images=args.max_images,
    )

    cmd = args.command
    if cmd in ("saliency", "all"):
        print("\n=== Saliency Map Generation ===")
        run_saliency(**common)

    if cmd in ("evaluate", "all"):
        print("\n=== Evaluation ===")
        run_evaluate(**common)

    if cmd in ("plot", "all"):
        print("\n=== Plotting ===")
        run_plot()

    print("\nDone!")


if __name__ == "__main__":
    main()
