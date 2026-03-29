#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : clip_interp.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-29-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""CLIP mechanistic interpretability experiments.

Standalone analyses on CLIP ViT-B-16 with fragment-completion masking:
  zeroshot      Zero-shot K-choice classification
  visualize     PCA + t-SNE of intermediate CLS tokens
  probe         Linear probing per layer × masking level
  patch         Activation patching (denoising)
  logit-lens    Intermediate layers → text embedding space
  transcoder    Sparse autoencoder on CLS features
  all           Run all analyses

Usage:
    uv run python -m src.clip_interp zeroshot --max-images 5
    uv run python -m src.clip_interp all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score

from models.processor import to_transform
from models.registry import get_encoder
from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image

NUM_LAYERS = 12
INTERNAL_DIM = 768
PROJ_DIM = 512
RESULTS_DIR = Path("results/clip_interp")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_all_cls(
    visual: nn.Module, images: torch.Tensor,
) -> list[torch.Tensor]:
    """Single forward pass; return CLS token from each of 12 resblocks.

    Args:
        visual: ``model.visual`` from OpenCLIP ViT-B-16.
        images: Preprocessed tensor ``[B, 3, 224, 224]``.

    Returns:
        List of 12 tensors, each ``[B, 768]``.
    """
    x = visual.conv1(images)  # [B, 768, 14, 14]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, 196, 768]
    cls_tok = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(
        x.shape[0], -1, -1,
    )
    x = torch.cat([cls_tok, x], dim=1)  # [B, 197, 768]
    x = x + visual.positional_embedding.unsqueeze(0)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # [T, B, D] — OpenCLIP sequence-first

    intermediates: list[torch.Tensor] = []
    for block in visual.transformer.resblocks:
        x = block(x)
        intermediates.append(x[0].clone())  # CLS at seq index 0 → [B, D]

    return intermediates


@torch.no_grad()
def extract_all_seq(
    visual: nn.Module, images: torch.Tensor,
) -> list[torch.Tensor]:
    """Single forward pass; return full sequence from each of 12 resblocks.

    Returns:
        List of 12 tensors, each ``[T, B, D]``.
    """
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls_tok = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(
        x.shape[0], -1, -1,
    )
    x = torch.cat([cls_tok, x], dim=1)
    x = x + visual.positional_embedding.unsqueeze(0)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)

    intermediates: list[torch.Tensor] = []
    for block in visual.transformer.resblocks:
        x = block(x)
        intermediates.append(x.clone())

    return intermediates


def project_to_clip_space(
    visual: nn.Module, cls_tokens: torch.Tensor,
) -> torch.Tensor:
    """Apply ``ln_post`` and ``proj`` to map 768-dim CLS → 512-dim CLIP space.

    Args:
        visual: ``model.visual``.
        cls_tokens: ``[B, 768]``.

    Returns:
        ``[B, 512]`` L2-normalized embeddings.
    """
    x = visual.ln_post(cls_tokens)
    x = x @ visual.proj  # [B, 512]
    return F.normalize(x.float(), dim=-1)


def get_text_embeddings(
    model: nn.Module,
    labels: list[str],
    device: str = "cuda",
    template: str = "an image of {label}",
) -> torch.Tensor:
    """Encode category labels into CLIP text embeddings.

    Returns:
        ``[C, 512]`` L2-normalized text embeddings on CPU.
    """
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    prompts = [template.format(label=lab) for lab in labels]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    return F.normalize(feats.float().cpu(), dim=-1)


def prepare_masked_batch(
    dataset,
    transform,
    level: int,
    seed: int,
    max_images: int | None = None,
    device: str = "cuda",
) -> tuple[torch.Tensor, list[int]]:
    """Embed all images at a given masking level.

    Returns:
        (images_tensor ``[N, 3, 224, 224]``, cat_ids ``[N]``)
    """
    n = min(len(dataset), max_images) if max_images else len(dataset)
    tensors, cat_ids = [], []
    for i in range(n):
        sample = dataset[i]
        masked = mask_pil_image(
            sample["image_pil"], level, sample["seg_mask"], seed=seed, idx=i,
        )
        tensors.append(transform(masked))
        cat_ids.append(sample["scene_id"])
    return torch.stack(tensors).to(device), cat_ids


def _save_json(data: dict, path: Path) -> None:
    """Save dict as JSON, converting numpy types."""

    def _convert(obj):
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


# ===================================================================
# 1. Zero-Shot Classification
# ===================================================================


@torch.no_grad()
def run_zeroshot(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    num_choices: int = 5,
    num_runs: int = 3,
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Zero-shot K-choice classification with 'an image of {label}' template."""
    out = RESULTS_DIR / "zeroshot"
    out.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder("clip", device=device)
    _ = encoder.model
    model = encoder.model
    transform = to_transform(encoder.processor)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    text_feats = get_text_embeddings(model, dataset.scene_labels, device)
    active = list(range(dataset.num_scenes))
    K = min(num_choices, len(active))

    print(f"  zeroshot: {n} images, {dataset.num_scenes} categories, K={K}")

    results: dict[str, dict] = {}
    for L in levels:
        imgs, cat_ids = prepare_masked_batch(
            dataset, transform, L, seed, max_images, device,
        )
        embeds = F.normalize(model.encode_image(imgs).float().cpu(), dim=-1)

        run_accs = []
        for run in range(num_runs):
            rng = np.random.RandomState(seed + run)
            correct = 0
            for i in range(n):
                true_cat = cat_ids[i]
                distractors = [c for c in active if c != true_cat]
                chosen = rng.choice(distractors, size=K - 1, replace=False).tolist()
                candidates = [true_cat] + chosen
                sims = embeds[i] @ text_feats[candidates].T
                correct += int(sims.argmax().item() == 0)
            run_accs.append(correct / n)

        arr = np.array(run_accs)
        vis = get_visibility_ratio(L)
        results[str(L)] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"    L={L} vis={vis:.3f}  acc={arr.mean():.4f}±{arr.std():.4f}")

    _save_json(results, out / "results.json")

    # Plot
    lvls = [int(k) for k in results]
    vis_x = [get_visibility_ratio(l) for l in lvls]
    means = [results[str(l)]["mean"] for l in lvls]
    stds = [results[str(l)]["std"] for l in lvls]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(vis_x, means, yerr=stds, marker="o", capsize=3)
    ax.set_xlabel("Visibility Ratio")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"CLIP Zero-Shot (K={K}, template='an image of ...')")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_visibility.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / 'accuracy_vs_visibility.png'}")


# ===================================================================
# 2. Visualization (PCA + t-SNE)
# ===================================================================


@torch.no_grad()
def run_visualization(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    layer: int | None = None,
    image_idx: int = 0,
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """PCA + t-SNE of intermediate CLIP CLS tokens.

    Mode A: Layer trajectory — one image across masking levels, CLS through layers.
    Mode B: Category clustering — all images at a specific layer, colored by category.
    """
    out = RESULTS_DIR / "visualization"
    out.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder("clip", device=device)
    visual = encoder.model.visual
    transform = to_transform(encoder.processor)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()
    sample = dataset[image_idx]
    image_id = sample["image_id"]

    # --- Mode A: Layer trajectory for one image across masking levels ---
    print(f"  visualization: layer trajectory for image {image_id}")
    all_cls: list[list[torch.Tensor]] = []  # [n_levels][12] each [768]
    for L in levels:
        masked = mask_pil_image(
            sample["image_pil"], L, sample["seg_mask"], seed=seed, idx=image_idx,
        )
        img_t = transform(masked).unsqueeze(0).to(device)
        layer_cls = extract_all_cls(visual, img_t)
        all_cls.append([c[0].cpu() for c in layer_cls])

    # Stack: [n_levels * 12, 768]
    stacked = torch.stack([c for level_cls in all_cls for c in level_cls]).numpy()
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(stacked)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cmap = plt.cm.viridis
    for idx_l, L in enumerate(levels):
        ax = axes[idx_l // 4, idx_l % 4]
        start = idx_l * NUM_LAYERS
        pts = coords[start:start + NUM_LAYERS]
        colors = [cmap(i / (NUM_LAYERS - 1)) for i in range(NUM_LAYERS)]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=60, edgecolors="k", linewidths=0.5)
        ax.plot(pts[:, 0], pts[:, 1], c="gray", alpha=0.4, linewidth=1)
        for i in range(NUM_LAYERS):
            ax.annotate(str(i + 1), (pts[i, 0], pts[i, 1]), fontsize=6, ha="center")
        vis = get_visibility_ratio(L)
        ax.set_title(f"L{L} (vis={vis:.0%})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Layer Trajectory (PCA) — Image {image_id}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = out / f"layer_trajectory_{image_id}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Mode B: Category clustering at specific layers ---
    target_layers = [layer] if layer is not None else [5, 8, 11]
    print(f"  visualization: category clustering at layers {target_layers}")

    for tgt_layer in target_layers:
        # Collect CLS at this layer for all images at L=8 (complete)
        cls_all = []
        cat_ids = []
        for i in range(n):
            s = dataset[i]
            img_t = transform(s["image_pil"]).unsqueeze(0).to(device)
            layer_cls = extract_all_cls(visual, img_t)
            cls_all.append(layer_cls[tgt_layer][0].cpu())
            cat_ids.append(s["scene_id"])

        cls_np = torch.stack(cls_all).numpy()
        cat_arr = np.array(cat_ids)
        perplexity = min(30, len(cls_np) - 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax_idx, (method_name, reducer) in enumerate([
            ("PCA", PCA(n_components=2, random_state=seed)),
            ("t-SNE", TSNE(n_components=2, random_state=seed, perplexity=perplexity)),
        ]):
            coords2 = reducer.fit_transform(cls_np)
            ax = axes[ax_idx]
            scatter = ax.scatter(
                coords2[:, 0], coords2[:, 1],
                c=cat_arr, cmap="tab20", s=25, alpha=0.7,
            )
            ax.set_title(f"{method_name} — Layer {tgt_layer + 1}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(
            f"Category Clusters (L=8, layer {tgt_layer + 1}/{NUM_LAYERS})",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        path = out / f"category_clusters_layer{tgt_layer + 1}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ===================================================================
# 3. Probing
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
    """Linear probing on intermediate layers to predict category.

    Output: 12×8 heatmap of cross-validated accuracy.
    """
    out = RESULTS_DIR / "probing"
    out.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder("clip", device=device)
    visual = encoder.model.visual
    transform = to_transform(encoder.processor)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    acc_matrix = np.zeros((NUM_LAYERS, len(levels)))

    for li, L in enumerate(levels):
        print(f"  probing: level L={L} (vis={get_visibility_ratio(L):.3f})")
        # Collect CLS from all layers for all images
        all_layer_feats: list[list[torch.Tensor]] = [[] for _ in range(NUM_LAYERS)]
        cat_ids: list[int] = []

        for i in range(n):
            sample = dataset[i]
            masked = mask_pil_image(
                sample["image_pil"], L, sample["seg_mask"], seed=seed, idx=i,
            )
            img_t = transform(masked).unsqueeze(0).to(device)
            layer_cls = extract_all_cls(visual, img_t)
            for k in range(NUM_LAYERS):
                all_layer_feats[k].append(layer_cls[k][0].cpu())
            cat_ids.append(sample["scene_id"])

        y = np.array(cat_ids)
        # Need at least 2 samples per class for StratifiedKFold
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        n_splits = min(5, min_count) if min_count >= 2 else 2

        for k in range(NUM_LAYERS):
            X = torch.stack(all_layer_feats[k]).numpy()
            if n_splits < 2:
                acc_matrix[k, li] = 0.0
                continue
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
            acc_matrix[k, li] = scores.mean()
            print(f"    layer {k + 1:2d}  acc={scores.mean():.4f}")

    _save_json(
        {"accuracy": acc_matrix.tolist(), "layers": list(range(1, 13)),
         "levels": levels},
        out / "results.json",
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    vis_labels = [f"L{l}\n{get_visibility_ratio(l):.0%}" for l in levels]
    im = ax.imshow(acc_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(vis_labels, fontsize=8)
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"Layer {i + 1}" for i in range(NUM_LAYERS)], fontsize=8)
    ax.set_xlabel("Masking Level")
    ax.set_ylabel("Transformer Layer")
    ax.set_title("Linear Probe Accuracy (CLIP ViT-B-16)")
    for i in range(NUM_LAYERS):
        for j in range(len(levels)):
            ax.text(j, i, f"{acc_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if acc_matrix[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    path = out / "probe_accuracy_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 4. Activation Patching
# ===================================================================


@torch.no_grad()
def run_activation_patching(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    mask_level: int = 1,
    num_choices: int = 5,
    num_runs: int = 3,
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Denoising activation patching: identify critical layers.

    Patch clean (L=8) activations into masked (L=mask_level) forward pass,
    one layer at a time. Measure zero-shot accuracy recovery.
    """
    out = RESULTS_DIR / "patching"
    out.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder("clip", device=device)
    model = encoder.model
    visual = model.visual
    transform = to_transform(encoder.processor)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)

    text_feats = get_text_embeddings(model, dataset.scene_labels, device)
    active = list(range(dataset.num_scenes))
    K = min(num_choices, len(active))

    # Prepare clean and masked image batches
    clean_imgs, cat_ids = prepare_masked_batch(
        dataset, transform, 8, seed, max_images, device,
    )
    masked_imgs, _ = prepare_masked_batch(
        dataset, transform, mask_level, seed, max_images, device,
    )

    # Cache clean activations (full sequence per layer)
    clean_seqs = extract_all_seq(visual, clean_imgs)  # 12 × [T, B, D]

    def _zeroshot_acc(embeds: torch.Tensor) -> float:
        """K-choice accuracy averaged over runs."""
        embeds_cpu = F.normalize(embeds.float().cpu(), dim=-1)
        run_accs = []
        for run in range(num_runs):
            rng = np.random.RandomState(seed + run)
            correct = 0
            for i in range(n):
                true_cat = cat_ids[i]
                distractors = [c for c in active if c != true_cat]
                chosen = rng.choice(distractors, size=K - 1, replace=False).tolist()
                candidates = [true_cat] + chosen
                sims = embeds_cpu[i] @ text_feats[candidates].T
                correct += int(sims.argmax().item() == 0)
            run_accs.append(correct / n)
        return float(np.mean(run_accs))

    # Baseline: masked forward (no patching)
    baseline_embed = model.encode_image(masked_imgs)
    baseline_acc = _zeroshot_acc(baseline_embed)
    print(f"  patching: baseline (L={mask_level}) acc={baseline_acc:.4f}")

    # Clean reference
    clean_embed = model.encode_image(clean_imgs)
    clean_acc = _zeroshot_acc(clean_embed)
    print(f"  patching: clean (L=8) acc={clean_acc:.4f}")

    # Patch each layer
    deltas: list[float] = []
    patched_accs: list[float] = []
    for patch_layer in range(NUM_LAYERS):
        # Forward with hook that replaces one layer's output
        def _make_hook(stored_seq: torch.Tensor):
            def hook_fn(mod, inp, output):
                return stored_seq
            return hook_fn

        resblock = visual.transformer.resblocks[patch_layer]
        handle = resblock.register_forward_hook(
            _make_hook(clean_seqs[patch_layer]),
        )
        try:
            patched_embed = model.encode_image(masked_imgs)
        finally:
            handle.remove()

        acc = _zeroshot_acc(patched_embed)
        delta = acc - baseline_acc
        deltas.append(delta)
        patched_accs.append(acc)
        print(f"    patch layer {patch_layer + 1:2d}  acc={acc:.4f}  Δ={delta:+.4f}")

    _save_json(
        {"baseline_acc": baseline_acc, "clean_acc": clean_acc,
         "mask_level": mask_level, "deltas": deltas,
         "patched_accs": patched_accs},
        out / f"results_L{mask_level}.json",
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, NUM_LAYERS + 1)
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in deltas]
    ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Δ Accuracy (patched − baseline)")
    ax.set_title(
        f"Activation Patching (denoising, L={mask_level}→L=8)\n"
        f"baseline={baseline_acc:.3f}, clean={clean_acc:.3f}",
    )
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / f"patching_delta_L{mask_level}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 5. Logit Lens
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

    Track how correct-category similarity and rank evolve layer by layer.
    """
    out = RESULTS_DIR / "logit_lens"
    out.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder("clip", device=device)
    model = encoder.model
    visual = model.visual
    transform = to_transform(encoder.processor)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)

    text_feats = get_text_embeddings(model, dataset.scene_labels, device)
    C = text_feats.shape[0]

    imgs, cat_ids = prepare_masked_batch(
        dataset, transform, mask_level, seed, max_images, device,
    )
    layer_cls_list = extract_all_cls(visual, imgs)  # 12 × [N, 768]

    # Per-layer: compute similarities and ranks
    sim_matrix = np.zeros((NUM_LAYERS, C))  # mean sim per category
    rank_per_layer = np.zeros(NUM_LAYERS)

    for k in range(NUM_LAYERS):
        projected = project_to_clip_space(visual, layer_cls_list[k])  # [N, 512]
        sims = projected.cpu() @ text_feats.T  # [N, C]

        # Average similarity per category (across all images)
        sim_matrix[k] = sims.mean(dim=0).numpy()

        # Rank of correct category for each image
        ranks = []
        for i in range(n):
            true_cat = cat_ids[i]
            row = sims[i]
            rank = (row > row[true_cat]).sum().item() + 1
            ranks.append(rank)
        rank_per_layer[k] = np.mean(ranks)
        print(f"    layer {k + 1:2d}  mean_rank={rank_per_layer[k]:.1f}"
              f"  correct_sim={np.mean([sims[i, cat_ids[i]].item() for i in range(n)]):.4f}")

    _save_json(
        {"sim_matrix": sim_matrix.tolist(),
         "rank_per_layer": rank_per_layer.tolist(),
         "mask_level": mask_level,
         "categories": dataset.scene_labels},
        out / f"results_L{mask_level}.json",
    )

    # Heatmap: layer × category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(sim_matrix, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"L{i + 1}" for i in range(NUM_LAYERS)], fontsize=7)
    ax.set_xlabel("Category")
    ax.set_ylabel("Layer")
    ax.set_title(f"Cosine Similarity (mask L={mask_level})")
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    ax.plot(range(1, NUM_LAYERS + 1), rank_per_layer, marker="o", color="#e74c3c")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Rank of Correct Category")
    ax.set_title("Correct Category Rank vs Layer")
    ax.set_xticks(range(1, NUM_LAYERS + 1))
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Logit Lens — CLIP ViT-B-16 (L={mask_level})", fontweight="bold")
    fig.tight_layout()
    path = out / f"logit_lens_L{mask_level}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 6. Transcoder (Sparse Autoencoder)
# ===================================================================


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder: disentangle superposed features.

    Architecture: Linear(D → H) + ReLU → Linear(H → D).
    Loss: MSE + λ·L1(latent).
    """

    def __init__(self, input_dim: int = INTERNAL_DIM, hidden_dim: int = 3072):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent_activations)."""
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


def run_transcoder(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    layer: int = 6,
    hidden_dim: int = 3072,
    sparsity_lambda: float = 1e-3,
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Train a sparse autoencoder on CLS tokens from a specific layer.

    After training, identify top-k features per category.
    """
    out = RESULTS_DIR / "transcoder"
    out.mkdir(parents=True, exist_ok=True)

    encoder_clip = get_encoder("clip", device=device)
    visual = encoder_clip.model.visual
    transform = to_transform(encoder_clip.processor)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Extract CLS tokens at target layer
    print(f"  transcoder: extracting CLS tokens from layer {layer + 1}")
    cls_all, cat_ids = [], []
    with torch.no_grad():
        for i in range(n):
            s = dataset[i]
            img_t = transform(s["image_pil"]).unsqueeze(0).to(device)
            layer_cls = extract_all_cls(visual, img_t)
            cls_all.append(layer_cls[layer][0].cpu())
            cat_ids.append(s["scene_id"])

    X = torch.stack(cls_all)  # [N, 768]
    y = np.array(cat_ids)
    C = dataset.num_scenes

    # Train SAE
    sae = SparseAutoencoder(INTERNAL_DIM, hidden_dim).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    X_dev = X.to(device)

    losses = []
    print(f"  transcoder: training SAE ({INTERNAL_DIM}→{hidden_dim}) for {num_epochs} epochs")
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch = X_dev[perm[start:start + batch_size]]
            x_hat, z = sae(batch)
            mse = F.mse_loss(x_hat, batch)
            l1 = z.abs().mean()
            loss = mse + sparsity_lambda * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= n
        losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1:3d}  loss={epoch_loss:.6f}")

    # Analyze: per-category mean latent activations
    sae.eval()
    with torch.no_grad():
        _, Z = sae(X_dev)  # [N, H]
    Z_np = Z.cpu().numpy()

    cat_mean = np.zeros((C, hidden_dim))
    for c in range(C):
        mask = y == c
        if mask.sum() > 0:
            cat_mean[c] = Z_np[mask].mean(axis=0)

    # Top-k features per category
    top_k = 10
    top_features: dict[str, list[int]] = {}
    for c in range(C):
        label = dataset.scene_labels[c]
        top_idx = np.argsort(cat_mean[c])[::-1][:top_k].tolist()
        top_features[label] = top_idx

    _save_json(
        {"layer": layer, "hidden_dim": hidden_dim, "losses": losses,
         "top_features": top_features,
         "sparsity": float(np.mean(Z_np > 0))},
        out / "results.json",
    )

    # Plot training loss
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, num_epochs + 1), losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"SAE Training (layer {layer + 1}, λ={sparsity_lambda})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Heatmap: category × top features
    # Show top 30 most variable features across categories
    feat_var = cat_mean.var(axis=0)
    top_feats = np.argsort(feat_var)[::-1][:30]
    sub = cat_mean[:, top_feats]

    fig, ax = plt.subplots(figsize=(12, max(4, C * 0.25)))
    im = ax.imshow(sub, aspect="auto", cmap="viridis")
    ax.set_xlabel("Feature Index (top-30 by variance)")
    ax.set_ylabel("Category")
    ax.set_yticks(range(C))
    ax.set_yticklabels(dataset.scene_labels, fontsize=6)
    ax.set_title(f"SAE Feature Activations by Category (layer {layer + 1})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = out / "feature_category_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    """Entry point for CLIP mechanistic interpretability experiments."""
    parser = argparse.ArgumentParser(
        description="CLIP mechanistic interpretability experiments",
    )
    parser.add_argument(
        "--dataset", default="fragment_v2", choices=["fragment_v2", "ade20k"],
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--image-type", default="original", choices=["original", "gray", "lined"],
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("zeroshot")
    p.add_argument("--num-choices", type=int, default=5)
    p.add_argument("--num-runs", type=int, default=3)

    p = sub.add_parser("visualize")
    p.add_argument("--layer", type=int, default=None,
                    help="Layer index 0-11 (default: show 5, 8, 11)")
    p.add_argument("--image-idx", type=int, default=0)

    sub.add_parser("probe")

    p = sub.add_parser("patch")
    p.add_argument("--mask-level", type=int, default=1)
    p.add_argument("--num-choices", type=int, default=5)
    p.add_argument("--num-runs", type=int, default=3)

    p = sub.add_parser("logit-lens")
    p.add_argument("--mask-level", type=int, default=8)

    p = sub.add_parser("transcoder")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=3072)
    p.add_argument("--sparsity-lambda", type=float, default=1e-3)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    sub.add_parser("all")

    args = parser.parse_args()
    common = dict(
        dataset_name=args.dataset, data_root=args.data_root,
        image_type=args.image_type, seed=args.seed, device=args.device,
        max_images=args.max_images,
    )

    cmd = args.command

    if cmd in ("zeroshot", "all"):
        print("\n=== Zero-Shot Classification ===")
        kw = {**common}
        if cmd == "zeroshot":
            kw.update(num_choices=args.num_choices, num_runs=args.num_runs)
        run_zeroshot(**kw)

    if cmd in ("visualize", "all"):
        print("\n=== Visualization ===")
        kw = {**common}
        if cmd == "visualize":
            kw.update(layer=args.layer, image_idx=args.image_idx)
        run_visualization(**kw)

    if cmd in ("probe", "all"):
        print("\n=== Linear Probing ===")
        run_probing(**common)

    if cmd in ("patch", "all"):
        print("\n=== Activation Patching ===")
        kw = {**common}
        if cmd == "patch":
            kw.update(mask_level=args.mask_level, num_choices=args.num_choices,
                      num_runs=args.num_runs)
        run_activation_patching(**kw)

    if cmd in ("logit-lens", "all"):
        print("\n=== Logit Lens ===")
        kw = {**common}
        if cmd == "logit-lens":
            kw.update(mask_level=args.mask_level)
        run_logit_lens(**kw)

    if cmd in ("transcoder", "all"):
        print("\n=== Transcoder (SAE) ===")
        kw = {**common}
        if cmd == "transcoder":
            kw.update(layer=args.layer, hidden_dim=args.hidden_dim,
                      sparsity_lambda=args.sparsity_lambda,
                      num_epochs=args.num_epochs, batch_size=args.batch_size,
                      lr=args.lr)
        run_transcoder(**kw)

    print("\nDone!")


if __name__ == "__main__":
    main()
