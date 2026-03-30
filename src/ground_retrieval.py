#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ground_retrieval.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-30-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Ground-truth retrieval + mechanistic interpretability experiments.

Unified experiment for CLIP ViT-L-14 and DINOv2+dino.txt with full retrieval
(rank among all 260 instances) as the downstream task.

Subcommands:
  retrieve      3 retrieval tasks across 8 masking levels
  probe         Linear probe per layer x masking level
  patch         Attn activation patching (STR/SIP x CLS/patch x noise/denoise)
  logit-lens    Intermediate CLS -> text space (CLIP only)
  all           Run all analyses

Usage:
    uv run python -m src.ground_retrieval --model clip retrieve --max-images 10
    uv run python -m src.ground_retrieval --model dinov2 patch
    uv run python -m src.ground_retrieval --model clip all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "clip": {
        "num_layers": 24, "internal_dim": 1024, "proj_dim": 768,
        "img_size": 224, "tag": "clip_L14", "label": "CLIP ViT-L-14",
    },
    "dinov2": {
        "num_layers": 24, "internal_dim": 1024, "proj_dim": 2048,
        "img_size": 518, "tag": "dinov2", "label": "DINOv2+dino.txt",
    },
}

_CFG: dict = MODEL_CONFIGS["clip"]

# Module-level model cache (loaded once per run)
_MODEL = None
_TOKENIZER = None
_TRANSFORM = None


def _num_layers() -> int:
    return _CFG["num_layers"]


def _internal_dim() -> int:
    return _CFG["internal_dim"]


def _proj_dim() -> int:
    return _CFG["proj_dim"]


def _results_dir() -> Path:
    return Path(f"results/ground_retrieval/{_CFG['tag']}")


def _model_label() -> str:
    return _CFG["label"]


def _is_clip() -> bool:
    return _CFG["tag"] == "clip_L14"


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------


@torch.no_grad()
def _load_model(device: str = "cuda") -> tuple[nn.Module, object, object]:
    """Load model + tokenizer + transform based on _CFG.

    Returns:
        (model, tokenizer, transform).
        For CLIP, tokenizer is the OpenCLIP tokenizer function.
        For DINOv2, tokenizer is the dino.txt tokenizer object.
    """
    global _MODEL, _TOKENIZER, _TRANSFORM
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _TRANSFORM

    if _is_clip():
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai",
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        _MODEL, _TOKENIZER, _TRANSFORM = model, tokenizer, preprocess
    else:
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
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        _MODEL, _TOKENIZER, _TRANSFORM = model, tokenizer, preprocess

    return _MODEL, _TOKENIZER, _TRANSFORM


def _get_blocks(model: nn.Module) -> nn.ModuleList:
    """Return transformer blocks list."""
    if _is_clip():
        return model.visual.transformer.resblocks
    return model.visual_model.backbone.model.blocks


@torch.no_grad()
def _encode_image(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 16,
) -> torch.Tensor:
    """Encode images -> [N, proj_dim], L2-normalized, CPU."""
    parts: list[torch.Tensor] = []
    for start in range(0, imgs.shape[0], batch_size):
        batch = imgs[start:start + batch_size]
        emb = model.encode_image(batch)
        parts.append(emb.cpu())
    feats = torch.cat(parts, dim=0)
    return F.normalize(feats.float(), dim=-1)


@torch.no_grad()
def _encode_text(
    model: nn.Module,
    tokenizer: object,
    labels: list[str],
    device: str = "cuda",
    template: str = "an image of {label}",
) -> torch.Tensor:
    """Encode text labels -> [C, proj_dim], L2-normalized, CPU."""
    prompts = [template.format(label=lab) for lab in labels]
    if _is_clip():
        tokens = tokenizer(prompts).to(device)
        feats = model.encode_text(tokens)
    else:
        tokens = tokenizer.tokenize(prompts).to(device)
        feats = model.encode_text(tokens)
    return F.normalize(feats.float().cpu(), dim=-1)


@torch.no_grad()
def _project_cls(
    model: nn.Module, cls_tokens: torch.Tensor,
) -> torch.Tensor:
    """Project intermediate CLS [B, internal_dim] -> [B, proj_dim].

    CLIP only: applies visual.ln_post + visual.proj.
    """
    visual = model.visual
    x = visual.ln_post(cls_tokens)
    x = x @ visual.proj
    return F.normalize(x.float(), dim=-1)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_json(data: dict, path: Path) -> None:
    """Save dict as JSON, converting numpy types."""
    def _convert(obj: object) -> object:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 6)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_convert)
    print(f"  Saved: {path}")


def prepare_masked_batch(
    dataset: object,
    transform: object,
    level: int,
    seed: int,
    max_images: int | None = None,
    device: str = "cuda",
) -> tuple[torch.Tensor, list[int]]:
    """Preprocess all images at a masking level.

    Returns:
        (tensor [N, 3, H, H], cat_ids)
    """
    n = min(len(dataset), max_images) if max_images else len(dataset)
    tensors: list[torch.Tensor] = []
    cat_ids: list[int] = []
    target_size = _CFG["img_size"] if not _is_clip() else None
    for i in range(n):
        sample = dataset[i]
        kw: dict = dict(seed=seed, idx=i)
        if target_size:
            kw["target_size"] = target_size
        masked = mask_pil_image(sample["image_pil"], level, sample["seg_mask"], **kw)
        tensors.append(transform(masked))
        cat_ids.append(sample["scene_id"])
    return torch.stack(tensors).to(device), cat_ids


# ---------------------------------------------------------------------------
# Retrieval metrics (model-agnostic)
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    query: torch.Tensor,
    gallery: torch.Tensor,
    gt_indices: torch.Tensor,
) -> dict[str, float]:
    """Compute R@1, R@5, MRR from cosine similarity.

    Args:
        query: [N, D] L2-normalized.
        gallery: [G, D] L2-normalized.
        gt_indices: [N] long — index of ground-truth match in gallery.

    Returns:
        {recall_at_1, recall_at_5, mrr}
    """
    sims = query @ gallery.T  # [N, G]
    N = sims.shape[0]
    ranks = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        gt = gt_indices[i].item()
        rank = (sims[i] > sims[i, gt]).sum().item() + 1
        ranks[i] = rank

    r1 = float((ranks <= 1).float().mean().item())
    r5 = float((ranks <= 5).float().mean().item())
    mrr = float((1.0 / ranks.float()).mean().item())
    return {"recall_at_1": r1, "recall_at_5": r5, "mrr": mrr}


def compute_category_accuracy(
    query: torch.Tensor,
    cat_embeds: torch.Tensor,
    gt_cat_ids: list[int],
) -> float:
    """Top-1 category classification accuracy."""
    sims = query @ cat_embeds.T  # [N, C]
    preds = sims.argmax(dim=1)
    gt = torch.tensor(gt_cat_ids, dtype=torch.long)
    return float((preds == gt).float().mean().item())


# ---------------------------------------------------------------------------
# Hook-based extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_all_cls(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 16,
) -> list[torch.Tensor]:
    """CLS token from each of 24 layers via hooks.

    Returns:
        List of 24 tensors [N, internal_dim] on CPU.
    """
    blocks = _get_blocks(model)
    N = imgs.shape[0]
    chunks: list[list[torch.Tensor]] = [[] for _ in range(_num_layers())]

    for start in range(0, N, batch_size):
        batch = imgs[start:start + batch_size]
        buf: list[torch.Tensor] = []
        handles: list[torch.utils.hooks.RemovableHook] = []

        for block in blocks:
            def _hook(mod: nn.Module, inp: object, out: object, b: list = buf) -> None:
                o = out[0] if isinstance(out, tuple) else out
                b.append(o[:, 0, :].cpu())
            handles.append(block.register_forward_hook(_hook))

        try:
            model.encode_image(batch)
        finally:
            for h in handles:
                h.remove()

        for k in range(_num_layers()):
            chunks[k].append(buf[k])

    return [torch.cat(c, dim=0) for c in chunks]


@torch.no_grad()
def _extract_attn_acts(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 32,
) -> list[torch.Tensor]:
    """Extract attn activations from all 24 blocks.

    Returns:
        List of 24 tensors [N, T, D] on CPU.
    """
    blocks = _get_blocks(model)
    N = imgs.shape[0]
    chunks: list[list[torch.Tensor]] = [[] for _ in range(_num_layers())]

    for start in range(0, N, batch_size):
        batch = imgs[start:start + batch_size]
        buf: list[torch.Tensor] = []
        handles: list = []

        for block in blocks:
            def _hook(mod: nn.Module, inp: object, out: object, b: list = buf) -> None:
                o = out[0] if isinstance(out, tuple) else out
                b.append(o.cpu())
            handles.append(block.attn.register_forward_hook(_hook))

        try:
            model.encode_image(batch)
        finally:
            for h in handles:
                h.remove()

        for k in range(_num_layers()):
            chunks[k].append(buf[k])

    return [torch.cat(c, dim=0) for c in chunks]


def _make_attn_hook(src: torch.Tensor, token_mode: str) -> callable:
    """Hook that replaces CLS or patch tokens in attn output.

    Args:
        src: source activations [B, T, D] to patch in (may be on CPU).
        token_mode: "cls" or "patch".
    """
    def hook_fn(mod: nn.Module, inp: object, output: object) -> object:
        if isinstance(output, tuple):
            tensor = output[0].clone()
        else:
            tensor = output.clone()

        src_dev = src.to(tensor.device)
        if token_mode == "cls":
            tensor[:, 0, :] = src_dev[:, 0, :]
        else:
            tensor[:, 1:, :] = src_dev[:, 1:, :]

        if isinstance(output, tuple):
            return (tensor, *output[1:])
        return tensor

    return hook_fn


# ===================================================================
# 1. Retrieval
# ===================================================================


@torch.no_grad()
def run_retrieval(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Run 3 retrieval tasks across 8 masking levels.

    Tasks:
        1. Image retrieval: query=masked, gallery=complete images
        2. Instance text retrieval: query=masked, gallery=instance text embeds
        3. Category text retrieval: query=masked, gallery=category text embeds
    """
    out = _results_dir() / "retrieval"
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, transform = _load_model(device)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    # Precompute galleries
    # 1. Complete image embeddings
    print("  Computing complete image gallery...")
    complete_tensors: list[torch.Tensor] = []
    for i in range(n):
        sample = dataset[i]
        complete_tensors.append(transform(sample["image_pil"]))
    complete_imgs = torch.stack(complete_tensors).to(device)
    image_gallery = _encode_image(model, complete_imgs)  # [N, D]

    # 2. Instance text embeddings
    instance_names = [
        dataset.samples[i].get("object_name", dataset.samples[i].get("scene_label", ""))
        for i in range(n)
    ]
    instance_gallery = _encode_text(model, tokenizer, instance_names, device)  # [N, D]

    # 3. Category text embeddings
    cat_gallery = _encode_text(model, tokenizer, dataset.scene_labels, device)  # [C, D]

    # GT indices
    gt_instance = torch.arange(n, dtype=torch.long)  # each image matches itself
    cat_ids = [dataset[i]["scene_id"] for i in range(n)]

    print(f"  retrieval: {n} images, {len(dataset.scene_labels)} categories")
    results: dict[str, dict] = {}

    for L in levels:
        vis = get_visibility_ratio(L)
        imgs, _ = prepare_masked_batch(dataset, transform, L, seed, max_images, device)
        query = _encode_image(model, imgs)  # [N, D]

        # Task 1: Image retrieval
        img_metrics = compute_retrieval_metrics(query, image_gallery, gt_instance)

        # Task 2: Instance text retrieval
        txt_metrics = compute_retrieval_metrics(query, instance_gallery, gt_instance)

        # Task 3: Category text retrieval
        cat_acc = compute_category_accuracy(query, cat_gallery, cat_ids)

        results[str(L)] = {
            "image_r1": img_metrics["recall_at_1"],
            "image_r5": img_metrics["recall_at_5"],
            "image_mrr": img_metrics["mrr"],
            "instance_r1": txt_metrics["recall_at_1"],
            "instance_r5": txt_metrics["recall_at_5"],
            "instance_mrr": txt_metrics["mrr"],
            "category_acc": cat_acc,
        }
        print(f"    L={L} vis={vis:.3f}  "
              f"img_r1={img_metrics['recall_at_1']:.4f}  "
              f"txt_r1={txt_metrics['recall_at_1']:.4f}  "
              f"cat={cat_acc:.4f}")

    _save_json(results, out / f"results_{image_type}.json")
    _plot_retrieval(results, image_type, out)


def _plot_retrieval(results: dict, image_type: str, out: Path) -> None:
    """Single plot with 3 retrieval curves using Tab20c lightness shades."""
    levels = list(range(1, 9))
    vis_x = [get_visibility_ratio(l) for l in levels]

    # Tab20c: groups of 4 shades per hue. Use first group (blue) indices 0,1,2
    tab20c = plt.colormaps.get_cmap("tab20c")
    metrics = [
        ("image_r1", "Image Retrieval R@1", tab20c(0)),       # darkest
        ("instance_r1", "Instance Text R@1", tab20c(1)),      # medium
        ("category_acc", "Category Text Acc", tab20c(2)),     # lightest
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in metrics:
        vals = [results[str(l)][key] for l in levels]
        ax.plot(vis_x, vals, marker="o", linewidth=2, color=color,
                label=label, markersize=6)

    ax.set_xlabel("Visibility Ratio", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title(f"{_model_label()} — Retrieval ({image_type})",
                 fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / f"retrieval_{image_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 2. Probing
# ===================================================================


@torch.no_grad()
def run_probing(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Two probing analyses: category linear probe + instance retrieval R@1.

    For each (layer, masking level):
      1. Category probe: LogisticRegression on raw CLS -> predict scene_id.
      2. Instance retrieval probe: project CLS via output head -> cosine sim
         against instance text embeddings -> R@1.

    Output: two 24x8 heatmaps.
    """
    out = _results_dir() / "probing"
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, transform = _load_model(device)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    # Instance text gallery for retrieval probe
    instance_names = [
        dataset.samples[i].get("object_name", dataset.samples[i].get("scene_label", ""))
        for i in range(n)
    ]
    instance_gallery = _encode_text(model, tokenizer, instance_names, device)
    gt_instance = torch.arange(n, dtype=torch.long)

    cat_acc_matrix = np.zeros((_num_layers(), len(levels)))
    inst_r1_matrix = np.zeros((_num_layers(), len(levels)))

    for li, L in enumerate(levels):
        vis = get_visibility_ratio(L)
        print(f"  probing: L={L} (vis={vis:.3f})")

        imgs, cat_ids = prepare_masked_batch(
            dataset, transform, L, seed, max_images, device,
        )
        all_cls = extract_all_cls(model, imgs)  # 24 x [N, 1024]

        y = np.array(cat_ids)
        unique, counts = np.unique(y, return_counts=True)
        n_splits = min(5, counts.min()) if counts.min() >= 2 else 2

        for k in range(_num_layers()):
            cls_k = all_cls[k]  # [N, 1024]

            # Category probe
            X = cls_k.numpy()
            if n_splits >= 2:
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed)
                scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
                cat_acc_matrix[k, li] = scores.mean()

            # Instance retrieval probe (CLIP only — needs projection)
            if _is_clip():
                proj = _project_cls(model, cls_k.to(device)).cpu()  # [N, 768]
                metrics = compute_retrieval_metrics(proj, instance_gallery,
                                                    gt_instance)
                inst_r1_matrix[k, li] = metrics["recall_at_1"]

        best_cat = int(cat_acc_matrix[:, li].argmax()) + 1
        print(f"    best category probe: layer {best_cat} "
              f"acc={cat_acc_matrix[:, li].max():.4f}")
        if _is_clip():
            best_inst = int(inst_r1_matrix[:, li].argmax()) + 1
            print(f"    best instance R@1:   layer {best_inst} "
                  f"r1={inst_r1_matrix[:, li].max():.4f}")

    save_data: dict = {
        "category_acc": cat_acc_matrix.tolist(),
        "layers": list(range(1, _num_layers() + 1)),
        "levels": list(range(1, 9)),
    }
    if _is_clip():
        save_data["instance_r1"] = inst_r1_matrix.tolist()

    _save_json(save_data, out / f"results_{image_type}.json")

    # Plot category probe heatmap
    _plot_probe_heatmap(
        cat_acc_matrix, f"{_model_label()} Category Probe ({image_type})",
        out / f"category_probe_{image_type}.png",
    )

    # Plot instance retrieval probe heatmap (CLIP only)
    if _is_clip():
        _plot_probe_heatmap(
            inst_r1_matrix,
            f"{_model_label()} Instance Retrieval R@1 Probe ({image_type})",
            out / f"instance_probe_{image_type}.png",
        )


def _plot_probe_heatmap(
    matrix: np.ndarray, title: str, path: Path,
) -> None:
    """Plot a layers x levels heatmap."""
    levels = list(range(1, 9))
    fig, ax = plt.subplots(figsize=(8, 8))
    vis_labels = [f"L{l}\n{get_visibility_ratio(l):.0%}" for l in levels]
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(vis_labels, fontsize=7)
    ax.set_yticks(range(_num_layers()))
    ax.set_yticklabels([f"L{i + 1}" for i in range(_num_layers())], fontsize=6)
    ax.set_xlabel("Masking Level")
    ax.set_ylabel("Transformer Layer")
    ax.set_title(title, fontsize=12)
    for i in range(_num_layers()):
        for j in range(len(levels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=4,
                    color="white" if matrix[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 3. Activation Patching
# ===================================================================


def _instance_retrieval_r1(
    embeds: torch.Tensor,
    instance_gallery: torch.Tensor,
) -> float:
    """Full-gallery instance text retrieval R@1 (deterministic).

    Args:
        embeds: [N, D] image embeddings (will be normalized).
        instance_gallery: [N, D] instance text embeddings (pre-normalized).

    Returns:
        R@1 accuracy.
    """
    embeds_cpu = F.normalize(embeds.float().cpu(), dim=-1)
    N = embeds_cpu.shape[0]
    gt = torch.arange(N, dtype=torch.long)
    return compute_retrieval_metrics(embeds_cpu, instance_gallery, gt)["recall_at_1"]


@torch.no_grad()
def run_activation_patching(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Attn activation patching: STR/SIP x CLS/patch, noising + denoising.

    Metric: instance text retrieval R@1 (all N candidates, deterministic).
    """
    out = _results_dir() / "patching"
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, transform = _load_model(device)
    blocks = _get_blocks(model)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    # Instance text gallery
    instance_names = [
        dataset.samples[i].get("object_name", dataset.samples[i].get("scene_label", ""))
        for i in range(n)
    ]
    cat_ids = [dataset[i]["scene_id"] for i in range(n)]
    instance_gallery = _encode_text(model, tokenizer, instance_names, device)

    # Partner indices
    rng_partner = np.random.RandomState(seed)
    same_cat_idx = np.zeros(n, dtype=int)
    diff_cat_idx = np.zeros(n, dtype=int)
    for i in range(n):
        same_cands = [j for j in range(n) if cat_ids[j] == cat_ids[i] and j != i]
        same_cat_idx[i] = rng_partner.choice(same_cands) if same_cands else i
        diff_cands = [j for j in range(n) if cat_ids[j] != cat_ids[i]]
        diff_cat_idx[i] = rng_partner.choice(diff_cands)

    # Conditions: (key, label, partner_idx, token_mode, direction)
    CONDITIONS: list[tuple] = []
    for corr_name, partner_idx in [("STR", same_cat_idx), ("SIP", diff_cat_idx)]:
        for tmode in ("cls", "patch"):
            tag = f"{corr_name.lower()}_attn_{tmode}"
            label = f"{corr_name} attn {tmode}"
            CONDITIONS.append((tag, label, partner_idx, tmode, "noise"))
            CONDITIONS.append(
                (f"dn_{tag}", f"DN-{label}", partner_idx, tmode, "denoise"),
            )

    all_results: dict[str, dict[int, list[float]]] = {c[0]: {} for c in CONDITIONS}
    baselines: dict[int, float] = {}

    for L in levels:
        vis = get_visibility_ratio(L)
        imgs, _ = prepare_masked_batch(
            dataset, transform, L, seed, max_images, device,
        )

        # Cache attn activations
        attn_acts = _extract_attn_acts(model, imgs)

        baseline_r1 = _instance_retrieval_r1(
            _encode_image(model, imgs), instance_gallery,
        )
        baselines[L] = baseline_r1
        print(f"\n  L={L} (vis={vis:.3f})  baseline R@1={baseline_r1:.4f}")

        # Denoising baselines
        dn_baselines: dict[str, float] = {}
        for corr_name, partner_idx in [("STR", same_cat_idx), ("SIP", diff_cat_idx)]:
            partner_imgs = imgs[partner_idx]
            dn_baselines[corr_name] = _instance_retrieval_r1(
                _encode_image(model, partner_imgs), instance_gallery,
            )

        for cond_key, cond_label, partner_idx, tmode, direction in CONDITIONS:
            if direction == "noise":
                run_imgs = imgs
                ref_r1 = baseline_r1
            else:
                run_imgs = imgs[partner_idx]
                corr_name = cond_label.split("-")[1].split(" ")[0]
                ref_r1 = dn_baselines[corr_name]

            deltas: list[float] = []
            for layer_i in range(_num_layers()):
                if direction == "noise":
                    src = attn_acts[layer_i][partner_idx]
                else:
                    src = attn_acts[layer_i]

                handle = blocks[layer_i].attn.register_forward_hook(
                    _make_attn_hook(src, tmode),
                )
                try:
                    patched_embed = _encode_image(model, run_imgs)
                finally:
                    handle.remove()

                r1 = _instance_retrieval_r1(patched_embed, instance_gallery)
                deltas.append(r1 - ref_r1)

            all_results[cond_key][L] = deltas
            if direction == "noise":
                worst_i = int(np.argmin(deltas)) + 1
                tag_str = f"worst=layer {worst_i:2d}"
            else:
                best_i = int(np.argmax(deltas)) + 1
                tag_str = f"best=layer  {best_i:2d}"
            print(f"    {cond_label:22s}  {tag_str}"
                  f"  range=[{min(deltas):+.4f}, {max(deltas):+.4f}]")

    _save_json({"baselines": baselines, **all_results},
               out / f"results_{image_type}.json")
    _plot_patching_heatmap(all_results, levels, image_type, out)


def _plot_patching_heatmap(
    all_results: dict, levels: list[int], image_type: str, out: Path,
) -> None:
    """2x4 heatmap grid for activation patching results."""
    PLOT_GRID: list[list[tuple[str, str]]] = []
    for direction, dir_label in [("noise", "Noising"), ("denoise", "Denoising")]:
        row: list[tuple[str, str]] = []
        for corr_name in ("STR", "SIP"):
            for tmode in ("cls", "patch"):
                prefix = "" if direction == "noise" else "dn_"
                key = f"{prefix}{corr_name.lower()}_attn_{tmode}"
                title = f"{dir_label}: {corr_name} {tmode}"
                row.append((key, title))
        PLOT_GRID.append(row)

    vis_labels = [f"{L}\n{get_visibility_ratio(L):.0%}" for L in levels]
    layer_labels = [f"L{i + 1}" for i in range(_num_layers())]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for row_i, row in enumerate(PLOT_GRID):
        for col_i, (cond_key, title) in enumerate(row):
            ax = axes[row_i, col_i]
            matrix = np.array([all_results[cond_key][L] for L in levels]).T
            vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
            im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(vis_labels, fontsize=6)
            ax.set_yticks(range(_num_layers()))
            ax.set_yticklabels(layer_labels, fontsize=5)
            ax.set_xlabel("Masking Level", fontsize=7)
            ax.set_ylabel("Layer", fontsize=7)
            ax.set_title(title, fontsize=8, fontweight="bold")
            for i in range(_num_layers()):
                for j in range(len(levels)):
                    ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center",
                            va="center", fontsize=4,
                            color="white" if abs(matrix[i, j]) > vmax * 0.6
                            else "black")
            fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(
        f"{_model_label()} Attn Patching ({image_type}) — Δ Instance R@1",
        fontweight="bold", fontsize=14,
    )
    fig.tight_layout()
    path = out / f"patching_heatmap_{image_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 4. Logit Lens (CLIP only)
# ===================================================================


@torch.no_grad()
def run_logit_lens(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    mask_level: int = 8,
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Project intermediate CLS tokens into text embedding space.

    Track how correct-instance rank and R@1 evolve layer by layer.
    CLIP only — DINOv2 projection head requires CLS+patch-avg concatenation.
    """
    if not _is_clip():
        print("  logit-lens: skipped (CLIP only — DINOv2 projection head "
              "concatenates CLS+patch-avg, not applicable to single CLS)")
        return

    out = _results_dir() / "logit_lens"
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, transform = _load_model(device)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Instance text gallery (all 260)
    instance_names = [
        dataset.samples[i].get("object_name", dataset.samples[i].get("scene_label", ""))
        for i in range(n)
    ]
    instance_gallery = _encode_text(model, tokenizer, instance_names, device)
    gt_instance = torch.arange(n, dtype=torch.long)

    imgs, cat_ids = prepare_masked_batch(
        dataset, transform, mask_level, seed, max_images, device,
    )
    all_cls = extract_all_cls(model, imgs)  # 24 x [N, 1024]

    rank_per_layer = np.zeros(_num_layers())
    r1_per_layer = np.zeros(_num_layers())
    mrr_per_layer = np.zeros(_num_layers())

    for k in range(_num_layers()):
        projected = _project_cls(model, all_cls[k].to(device)).cpu()  # [N, 768]
        metrics = compute_retrieval_metrics(projected, instance_gallery, gt_instance)
        r1_per_layer[k] = metrics["recall_at_1"]
        mrr_per_layer[k] = metrics["mrr"]

        # Mean rank of correct instance
        sims = projected @ instance_gallery.T  # [N, N]
        ranks = []
        for i in range(n):
            rank = (sims[i] > sims[i, i]).sum().item() + 1
            ranks.append(rank)
        rank_per_layer[k] = np.mean(ranks)

        print(f"    layer {k + 1:2d}  R@1={r1_per_layer[k]:.4f}  "
              f"MRR={mrr_per_layer[k]:.4f}  mean_rank={rank_per_layer[k]:.1f}")

    _save_json(
        {"rank_per_layer": rank_per_layer.tolist(),
         "r1_per_layer": r1_per_layer.tolist(),
         "mrr_per_layer": mrr_per_layer.tolist(),
         "mask_level": mask_level},
        out / f"results_L{mask_level}.json",
    )

    _plot_logit_lens(rank_per_layer, r1_per_layer, mrr_per_layer,
                     mask_level, out)


def _plot_logit_lens(
    rank_per_layer: np.ndarray,
    r1_per_layer: np.ndarray,
    mrr_per_layer: np.ndarray,
    mask_level: int,
    out: Path,
) -> None:
    """2-panel plot: mean rank + R@1/MRR vs layer."""
    layers = list(range(1, _num_layers() + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(layers, rank_per_layer, marker="o", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Mean Rank of Correct Instance", fontsize=14)
    ax.set_title("Correct Instance Rank vs Layer", fontsize=16)
    ax.set_xticks(layers[::2])
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers, r1_per_layer, marker="o", color="#2ecc71", linewidth=2,
            label="R@1")
    ax.plot(layers, mrr_per_layer, marker="s", color="#3498db", linewidth=2,
            label="MRR")
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Instance Retrieval Metrics vs Layer", fontsize=16)
    ax.set_xticks(layers[::2])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{_model_label()} Logit Lens (L={mask_level})",
                 fontweight="bold", fontsize=18)
    fig.tight_layout()
    path = out / f"logit_lens_L{mask_level}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    """Entry point for ground-truth retrieval experiments."""
    parser = argparse.ArgumentParser(
        description="Ground-truth retrieval + mechanistic interpretability",
    )
    parser.add_argument("--model", default="clip",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model: clip (ViT-L-14) or dinov2 (dino.txt)")
    parser.add_argument("--dataset", default="fragment_v2",
                        choices=["fragment_v2", "ade20k", "coco_subset"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-type", default="original",
                        choices=["original", "gray", "lined"])
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("retrieve")

    sub.add_parser("probe")

    sub.add_parser("patch")

    p = sub.add_parser("logit-lens")
    p.add_argument("--mask-level", type=int, default=8)

    sub.add_parser("all")

    args = parser.parse_args()

    # Set active model config
    global _CFG
    _CFG = MODEL_CONFIGS[args.model]
    print(f"  Model: {_model_label()} ({_num_layers()} layers, "
          f"dim={_internal_dim()}, proj={_proj_dim()})")
    print(f"  Results: {_results_dir()}/")

    common = dict(
        dataset_name=args.dataset, data_root=args.data_root,
        image_type=args.image_type, seed=args.seed, device=args.device,
        max_images=args.max_images,
    )

    cmd = args.command

    if cmd in ("retrieve", "all"):
        print("\n=== Retrieval ===")
        run_retrieval(**common)

    if cmd in ("probe", "all"):
        print("\n=== Probing ===")
        run_probing(**common)

    if cmd in ("patch", "all"):
        print("\n=== Activation Patching ===")
        run_activation_patching(**common)

    if cmd in ("logit-lens", "all"):
        print("\n=== Logit Lens ===")
        kw = {**common}
        if cmd == "logit-lens":
            kw["mask_level"] = args.mask_level
        run_logit_lens(**kw)

    print("\nDone!")


if __name__ == "__main__":
    main()
