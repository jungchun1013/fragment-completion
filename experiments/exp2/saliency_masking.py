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

# 4 supercategories × 5 basic-level = 20 categories
TARGET_SUPERCATS = ["animal", "vehicle", "furniture", "food"]
TARGET_CATEGORIES = {
    "animal":    ["cat", "dog", "horse", "elephant", "cow"],
    "vehicle":   ["bus", "car", "motorcycle", "truck", "train"],
    "furniture": ["chair", "couch", "bed", "dining table", "toilet"],
    "food":      ["pizza", "cake", "sandwich", "banana", "broccoli"],
}

# Model configs
MODELS = {
    "clip": {
        "img_size": 224,
        "patch_size": 16,
        "grid_size": 14,  # 224 / 16 = 14
        "label": "CLIP ViT-L-14",
    },
    "dinov2": {
        "img_size": 518,
        "patch_size": 16,
        "grid_size": 32,  # 518 / 16 = 32 (truncated)
        "label": "DINOv2+dino.txt",
    },
}

ALL_CONDITIONS = ["random", "dinov2_salient", "clip_salient"]
ALL_ENCODERS = ["dinov2", "clip"]


# ---------------------------------------------------------------------------
# Model loading (cached to avoid reloading between saliency → evaluate)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple] = {}


def _load_clip(device: str = "cuda") -> tuple[nn.Module, object, object]:
    """Load CLIP ViT-L-14. Returns (model, tokenizer, transform)."""
    if "clip" in _MODEL_CACHE:
        return _MODEL_CACHE["clip"]
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai",
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    result = (model, tokenizer, preprocess)
    _MODEL_CACHE["clip"] = result
    return result


def _load_dinov2(device: str = "cuda") -> tuple[nn.Module, object, object]:
    """Load DINOv2+dino.txt. Returns (model, tokenizer, transform)."""
    if "dinov2" in _MODEL_CACHE:
        return _MODEL_CACHE["dinov2"]
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
    result = (model, tokenizer, transform)
    _MODEL_CACHE["dinov2"] = result
    return result


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
    conditions: list[str] | None = None,
) -> None:
    """Precompute and save saliency maps for DINOv2 and/or CLIP."""
    conditions = conditions or ALL_CONDITIONS
    out = RESULTS_DIR / "saliency"
    out.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(dataset_name, root=data_root)
    indices = _filter_dataset(dataset, max_images)
    n = len(indices)
    print(f"  Saliency: {n} images from {len(TARGET_SUPERCATS)} supercategories")

    labels = [dataset[i]["scene_label"] for i in indices]

    if "dinov2_salient" in conditions:
        # --- DINOv2 saliency ---
        print("\n  Computing DINOv2 saliency (register token attention)...")
        model_d, _, transform_d = _load_dinov2(device)
        imgs_d = torch.stack([transform_d(dataset[i]["image_pil"]) for i in indices])
        sal_d = dinov2_saliency(model_d, imgs_d, batch_size=8)
        torch.save({"saliency": sal_d, "indices": indices}, out / "dinov2_saliency.pt")
        print(f"  Saved: {out}/dinov2_saliency.pt  shape={sal_d.shape}")
        del imgs_d

    if "clip_salient" in conditions:
        # --- CLIP saliency (basic-level text prompt) ---
        print("\n  Computing CLIP saliency (GradCAM on basic-level text)...")
        model_c, tokenizer_c, transform_c = _load_clip(device)

        # Per-image text embedding based on category label
        unique_labels = sorted(set(labels))
        label_embeds = _encode_text(model_c, tokenizer_c, unique_labels, device)
        label_to_embed = {lab: label_embeds[j] for j, lab in enumerate(unique_labels)}
        text_embeds = torch.stack([label_to_embed[lab] for lab in labels])

        imgs_c = torch.stack([transform_c(dataset[i]["image_pil"]) for i in indices])
        sal_c = clip_gradcam(model_c, imgs_c, text_embeds, batch_size=8)
        torch.save({"saliency": sal_c, "indices": indices}, out / "clip_saliency.pt")
        print(f"  Saved: {out}/clip_saliency.pt  shape={sal_c.shape}")
        del imgs_c

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
    conditions: list[str] | None = None,
    encoders: list[str] | None = None,
) -> None:
    """Evaluate conditions × encoders × 8 levels."""
    conditions = conditions or ALL_CONDITIONS
    encoders = encoders or ALL_ENCODERS

    dataset = get_dataset(dataset_name, root=data_root)
    indices = _filter_dataset(dataset, max_images)

    # Load saliency maps (only those needed)
    sal_dinov2 = sal_clip = None
    sal_dir = RESULTS_DIR / "saliency"
    if "dinov2_salient" in conditions:
        if not (sal_dir / "dinov2_saliency.pt").exists():
            print("  [error] Run 'saliency' subcommand first.")
            return
        sal_d_data = torch.load(sal_dir / "dinov2_saliency.pt", weights_only=True)
        sal_dinov2 = sal_d_data["saliency"]  # [N, 37, 37]
    if "clip_salient" in conditions:
        if not (sal_dir / "clip_saliency.pt").exists():
            print("  [error] Run 'saliency' subcommand first.")
            return
        sal_c_data = torch.load(sal_dir / "clip_saliency.pt", weights_only=True)
        sal_clip = sal_c_data["saliency"]    # [N, 14, 14]

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
    print(f"  Conditions: {conditions}")
    print(f"  Encoders: {encoders}")

    for enc_name in encoders:
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

        # Image prototype (per basic-level category, for proto retrieval)
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

        # Image mean prototype (per supercategory, sample 2 images per category)
        rng_proto = np.random.RandomState(seed)
        img_supercat_proto = torch.zeros(S, D)
        for cid in active_cat_ids:
            # Indices within this category
            cat_indices = [i for i in range(n) if cat_ids[i] == cid]
            sampled = rng_proto.choice(cat_indices, size=min(2, len(cat_indices)), replace=False)
            sid = cat_to_supercat[cid]
            for si in sampled:
                img_supercat_proto[sid] += image_gallery[si]
        img_supercat_proto = F.normalize(img_supercat_proto, dim=-1)

        for condition in conditions:
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
                txt_metrics = compute_retrieval_metrics(query, cat_gallery, gt_proto)
                gt_proto = torch.tensor(remapped_cat_ids, dtype=torch.long)
                proto_metrics = compute_retrieval_metrics(query, img_proto, gt_proto)
                exemplar_acc = compute_exemplar_accuracy(
                    query, image_gallery, supercat_ids, supercat_ids, k=10,
                )
                concept_proto_acc = compute_category_accuracy(query, concept_proto, supercat_ids)
                img_proto_acc = compute_category_accuracy(query, img_supercat_proto, supercat_ids)
                cat_concept_acc = compute_category_accuracy(query, supercat_gallery, supercat_ids)

                results[str(L)] = {
                    "image_r1": img_metrics["recall_at_1"],
                    "image_r5": img_metrics["recall_at_5"],
                    "image_mrr": img_metrics["mrr"],
                    "instance_r1": txt_metrics["recall_at_1"],
                    "instance_r5": txt_metrics["recall_at_5"],
                    "instance_mrr": txt_metrics["mrr"],
                    "proto_r1": proto_metrics["recall_at_1"],
                    "proto_r5": proto_metrics["recall_at_5"],
                    "proto_mrr": proto_metrics["mrr"],
                    "exemplar_acc": exemplar_acc,
                    "concept_proto_acc": concept_proto_acc,
                    "img_proto_acc": img_proto_acc,
                    "category_acc": cat_concept_acc,
                }

                print(f"    L={L} vis={vis:.3f}  "
                      f"img_r1={img_metrics['recall_at_1']:.4f}  "
                      f"proto_r1={proto_metrics['recall_at_1']:.4f}  "
                      f"txt_r1={txt_metrics['recall_at_1']:.4f}  "
                      f"exemplar={exemplar_acc:.4f}  "
                      f"img_proto={img_proto_acc:.4f}  "
                      f"concept_proto={concept_proto_acc:.4f}  "
                      f"cat={cat_concept_acc:.4f}")

            # Save
            out_dir = RESULTS_DIR / "retrieval"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json(results, out_dir / "results_original.json")

        pass  # model cached for reuse

    print(f"\n  All results saved to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Subcommand: plot
# ---------------------------------------------------------------------------

def run_plot() -> None:
    """Generate per-encoder comparison plots across conditions.

    For each encoder, produces one figure with 4 subplots (one per task),
    each subplot comparing all available conditions (random, dinov2_salient, ...).
    """
    import json
    import matplotlib.pyplot as plt
    from src.config import PLOT_STYLE as PS
    from src.masking import get_mask_levels as _levels, get_visibility_ratio as _vis
    _TAB20C = plt.cm.tab20c.colors

    # Tasks to plot: (json_key, display_label, color)
    TASKS = [
        ("image_r1",   "Image Retrieval",      _TAB20C[8]),   # green
        ("proto_r1",   "Prototype Retrieval",   _TAB20C[10]),  # light green
        ("instance_r1", "Text Retrieval",       "#17becf"),    # cyan
    ]

    CONDITION_STYLES = {
        "random":         {"color": "#888888", "ls": "--",  "label": "Random"},
        "dinov2_salient":  {"color": "#1f77b4", "ls": "-",  "label": "DINOv2 Salient"},
        "clip_salient":    {"color": "#d62728", "ls": "-",  "label": "CLIP Salient"},
    }

    # Load all results
    all_data: dict[str, dict[str, dict]] = {}
    for condition in ALL_CONDITIONS:
        all_data[condition] = {}
        for enc_name in ALL_ENCODERS:
            path = RESULTS_DIR / condition / enc_name / "results.json"
            if path.exists():
                with open(path) as f:
                    all_data[condition][enc_name] = json.load(f)

    if not any(all_data[c] for c in ALL_CONDITIONS):
        print("  [skip] No results found. Run 'evaluate' first.")
        return

    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    levels = _levels()
    vis = [_vis(L) for L in levels]

    for enc_name in ALL_ENCODERS:
        # Check if any condition has data for this encoder
        available = {c: all_data[c][enc_name]
                     for c in ALL_CONDITIONS if enc_name in all_data.get(c, {})}
        if not available:
            continue

        ncols = len(TASKS)
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 4.5))
        if ncols == 1:
            axes = [axes]

        all_handles, all_labels = [], []

        for ax, (key, title, task_color) in zip(axes, TASKS):
            for cond, data in available.items():
                style = CONDITION_STYLES.get(cond, {"color": "black", "ls": "-", "label": cond})
                # Skip if key not in data
                if key not in data["1"]:
                    continue
                vals = [data[str(L)][key] for L in levels]
                h, = ax.plot(
                    vis, vals,
                    marker=PS["marker"], markersize=PS["markersize"],
                    linewidth=PS["linewidth"],
                    color=style["color"], linestyle=style["ls"],
                    label=style["label"],
                )
                if style["label"] not in all_labels:
                    all_handles.append(h)
                    all_labels.append(style["label"])

            ax.set_title(title, fontsize=PS["subplot_title_fontsize"])
            ax.set_xlabel("Visibility", fontsize=PS["label_fontsize"])
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=PS["tick_labelsize"], width=PS["tick_width"])
            for spine in ax.spines.values():
                spine.set_linewidth(PS["tick_width"])

        axes[0].set_ylabel("Accuracy / R@1", fontsize=PS["label_fontsize"])

        enc_label = MODELS[enc_name]["label"]
        fig.suptitle(enc_label, fontsize=PS["suptitle_fontsize"], fontweight="bold", y=1.05)
        fig.legend(
            handles=all_handles, labels=all_labels,
            loc="lower center", bbox_to_anchor=(0.5, -0.08),
            ncol=len(all_handles), fontsize=PS["legend_fontsize"], frameon=True,
        )

        save_path = plot_dir / f"retrieval_original_{enc_name}.png"
        fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

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
    parser.add_argument("--encoders", nargs="+", default=None,
                        choices=ALL_ENCODERS,
                        help="Encoders to evaluate (default: all)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=ALL_CONDITIONS,
                        help="Saliency conditions to run (default: all)")

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
        run_saliency(**common, conditions=args.conditions)

    if cmd in ("evaluate", "all"):
        print("\n=== Evaluation ===")
        run_evaluate(**common, conditions=args.conditions, encoders=args.encoders)

    if cmd in ("plot", "all"):
        print("\n=== Plotting ===")
        run_plot()

    print("\nDone!")


if __name__ == "__main__":
    main()
