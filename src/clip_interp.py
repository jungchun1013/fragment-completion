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

Standalone analyses on CLIP ViT-B-16 or ViT-L-14 with fragment-completion masking:
  zeroshot      Zero-shot K-choice classification
  visualize     PCA fragment trajectory + category clustering
  probe         Linear probing per layer × masking level
  patch         Activation patching on attn (noising + denoising)
  logit-lens    Intermediate layers → text embedding space
  transcoder    Sparse autoencoder on CLS features
  all           Run all analyses

Usage:
    uv run python -m src.clip_interp zeroshot --max-images 5
    uv run python -m src.clip_interp --model L-14 patch
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

from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "B-16": {"arch": "ViT-B-16", "num_layers": 12, "internal_dim": 768,
             "proj_dim": 512, "tag": "B16"},
    "L-14": {"arch": "ViT-L-14", "num_layers": 24, "internal_dim": 1024,
             "proj_dim": 768, "tag": "L14"},
}

# Active config — set by main() before any run_* function
_CFG: dict = MODEL_CONFIGS["B-16"]


def _num_layers() -> int:
    return _CFG["num_layers"]


def _internal_dim() -> int:
    return _CFG["internal_dim"]


def _proj_dim() -> int:
    return _CFG["proj_dim"]


def _results_dir() -> Path:
    return Path(f"results/clip_interp_{_CFG['tag']}")


def _model_label() -> str:
    return f"CLIP {_CFG['arch']}"


@torch.no_grad()
def _load_clip(device: str = "cuda"):
    """Load OpenCLIP model + transform for the active config.

    Returns:
        (model, transform)
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        _CFG["arch"], pretrained="openai",
    )
    model = model.to(device).eval()
    return model, preprocess


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_all_cls(
    model: nn.Module, images: torch.Tensor,
) -> list[torch.Tensor]:
    """Hook-based extraction of CLS tokens from all 12 resblocks.

    Uses ``model.encode_image`` so the internal format is always correct.

    Args:
        model: Full OpenCLIP model (not just ``model.visual``).
        images: Preprocessed tensor ``[B, 3, 224, 224]``.

    Returns:
        List of 12 tensors, each ``[B, 768]``.
    """
    intermediates: list[torch.Tensor] = []
    handles = []
    for block in model.visual.transformer.resblocks:
        def hook_fn(mod, inp, out, buf=intermediates):
            # Resblocks output [B, T, D]; CLS at position 0
            buf.append(out[:, 0, :].clone())
        handles.append(block.register_forward_hook(hook_fn))

    try:
        model.encode_image(images)
    finally:
        for h in handles:
            h.remove()

    return intermediates


@torch.no_grad()
def extract_all_seq(
    model: nn.Module, images: torch.Tensor,
) -> list[torch.Tensor]:
    """Hook-based extraction of full sequences from all 12 resblocks.

    Returns:
        List of 12 tensors, each ``[B, T, D]``.
    """
    intermediates: list[torch.Tensor] = []
    handles = []
    for block in model.visual.transformer.resblocks:
        def hook_fn(mod, inp, out, buf=intermediates):
            buf.append(out.clone())
        handles.append(block.register_forward_hook(hook_fn))

    try:
        model.encode_image(images)
    finally:
        for h in handles:
            h.remove()

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
    tokenizer = open_clip.get_tokenizer(_CFG["arch"])
    prompts = [template.format(label=lab) for lab in labels]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    return F.normalize(feats.float().cpu(), dim=-1)


def get_text_prototypes(
    model: nn.Module,
    dataset,
    device: str = "cuda",
    template: str = "an image of {label}",
    max_images: int | None = None,
) -> torch.Tensor:
    """Build per-category text prototypes by averaging instance text embeddings.

    For each image, encode ``template.format(label=object_name)`` and average
    across images within the same category.

    Returns:
        ``[C, 512]`` L2-normalized text prototype embeddings on CPU.
    """
    tokenizer = open_clip.get_tokenizer(_CFG["arch"])
    n = min(len(dataset), max_images) if max_images else len(dataset)
    C = dataset.num_scenes

    proto_sum = torch.zeros(C, _proj_dim())
    proto_count = torch.zeros(C)

    prompts = []
    cat_ids = []
    for i in range(n):
        s = dataset.samples[i]
        name = s.get("object_name", s.get("scene_label", ""))
        prompts.append(template.format(label=name))
        cat_ids.append(s["scene_id"])

    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens).float().cpu()  # [N, 512]

    for i in range(n):
        cid = cat_ids[i]
        proto_sum[cid] += feats[i]
        proto_count[cid] += 1

    proto_count = proto_count.clamp(min=1)
    prototypes = proto_sum / proto_count.unsqueeze(1)
    return F.normalize(prototypes, dim=-1)


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
    """Instance-level K-choice zero-shot classification."""
    out = _results_dir() / "zeroshot"
    out.mkdir(parents=True, exist_ok=True)

    model, transform = _load_clip(device)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    instance_names = [
        dataset.samples[i].get("object_name", dataset.samples[i].get("scene_label", ""))
        for i in range(n)
    ]
    text_feats = get_text_embeddings(model, instance_names, device)  # [N, D]
    K = min(num_choices, n)

    print(f"  zeroshot: {n} images (instance-level), K={K}")

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
                distractors = [j for j in range(n) if j != i]
                chosen = rng.choice(distractors, size=min(K - 1, len(distractors)),
                                    replace=False).tolist()
                candidates = [i] + chosen
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
    ax.set_title(f"{_model_label()} Instance Zero-Shot (K={K})")
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

    Mode A — Layer trajectory (single plot):
      All masking levels on one plot. Nodes colored by layer (1-12),
      lines colored by visibility level. Image 001 complete embedding
      and text embeddings (target + distractors) are marked with labels.

    Mode B — Category clustering:
      All images at a specific layer, colored by category with legend.
    """
    out = _results_dir() / "visualization"
    out.mkdir(parents=True, exist_ok=True)

    model, transform = _load_clip(device)
    visual = model.visual
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()
    sample = dataset[image_idx]
    image_id = sample["image_id"]
    target_cat = sample["scene_id"]
    target_label = sample["scene_label"]

    # ── Text embeddings ──
    # Category-level: "an image of {category}"
    text_cat_512 = get_text_embeddings(model, dataset.scene_labels, device)
    # Instance-level prototype: average of "an image of {instance_name}" per category
    text_proto_512 = get_text_prototypes(model, dataset, device, max_images=max_images)
    text_np_cat = text_cat_512.numpy()    # [C, 512]
    text_np_proto = text_proto_512.numpy()  # [C, 512]

    # ── Mode A: Fragment trajectory — final embedding across masking levels ──
    print(f"  visualization: fragment trajectory for image {image_id}")

    # Target image: final embedding at each masking level L1-L8
    traj_embeds: list[np.ndarray] = []  # [8, 512]
    for L in levels:
        masked = mask_pil_image(
            sample["image_pil"], L, sample["seg_mask"], seed=seed, idx=image_idx,
        )
        img_t = transform(masked).unsqueeze(0).to(device)
        embed = F.normalize(
            model.encode_image(img_t).float().cpu(), dim=-1,
        )[0].numpy()
        traj_embeds.append(embed)
    traj_np = np.stack(traj_embeds)  # [8, 512]

    # Distractor images: final embedding at L=8 (complete)
    distractor_embeds: list[np.ndarray] = []
    distractor_cats: list[int] = []
    for i in range(n):
        s = dataset[i]
        img_t = transform(s["image_pil"]).unsqueeze(0).to(device)
        embed = F.normalize(
            model.encode_image(img_t).float().cpu(), dim=-1,
        )[0].numpy()
        distractor_embeds.append(embed)
        distractor_cats.append(s["scene_id"])
    distractor_np = np.stack(distractor_embeds)  # [N, 512]

    # PCA fitted on full images (L=8) only, then project everything else
    C = text_np_cat.shape[0]
    n_lev = len(levels)
    n_dist = distractor_np.shape[0]

    # Center: use full-image mean as the origin
    img_mean = distractor_np.mean(axis=0, keepdims=True)
    text_mean = np.concatenate([text_np_cat, text_np_proto]).mean(
        axis=0, keepdims=True,
    )

    pca = PCA(n_components=2, random_state=seed)
    dist_centered = distractor_np - img_mean
    pca.fit(dist_centered)  # fit on full images only
    var_exp = pca.explained_variance_ratio_

    # Transform everything into the same PCA space
    coords_dist = pca.transform(dist_centered)             # [N, 2]
    coords_traj = pca.transform(traj_np - img_mean)        # [8, 2]
    coords_tcat = pca.transform(text_np_cat - text_mean)   # [C, 2]
    coords_tproto = pca.transform(text_np_proto - text_mean)  # [C, 2]

    vis_cmap = plt.cm.plasma
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(10, 8))

    # All full images (background)
    for di in range(n_dist):
        dx, dy = coords_dist[di]
        is_same_cat = (distractor_cats[di] == target_cat)
        if di == image_idx:
            # Target image at L=8 is part of trajectory, mark separately
            ax.scatter([dx], [dy], c="red", s=60, marker="o",
                       edgecolors="k", linewidths=1.0, zorder=4)
        elif is_same_cat:
            ax.scatter([dx], [dy], c="steelblue", s=25, alpha=0.6,
                       edgecolors="navy", linewidths=0.3, zorder=2)
        else:
            ax.scatter([dx], [dy], c="lightgray", s=15, alpha=0.4, zorder=1)

    # Fragment trajectory line L1→L8
    ax.plot(coords_traj[:, 0], coords_traj[:, 1],
            c="red", alpha=0.4, linewidth=2, linestyle="--", zorder=3)

    # Fragment trajectory nodes colored by visibility
    for idx_l, L in enumerate(levels):
        vis = get_visibility_ratio(L)
        color = vis_cmap(idx_l / (len(levels) - 1))
        size = 120 if L == levels[-1] else 60
        marker = "*" if L == levels[-1] else "o"
        ax.scatter(coords_traj[idx_l, 0], coords_traj[idx_l, 1],
                   c=[color], s=size, marker=marker,
                   edgecolors="k", linewidths=0.8, zorder=5)
        ax.annotate(f"L{L}", (coords_traj[idx_l, 0], coords_traj[idx_l, 1]),
                    fontsize=7, fontweight="bold",
                    textcoords="offset points", xytext=(6, 6))

    # Text embeddings: category (◆) and prototype (▲)
    for ti in range(C):
        is_target = (ti == target_cat)
        label_text = dataset.scene_labels[ti]
        cat_color = "darkgreen" if is_target else "gray"
        prefix = "[T] " if is_target else ""

        tx, ty = coords_tcat[ti]
        ax.scatter([tx], [ty], c=cat_color, s=70 if is_target else 30,
                   marker="D", edgecolors="k", linewidths=0.5, zorder=4)
        ax.annotate(f"{prefix}{label_text}", (tx, ty), fontsize=6,
                    color=cat_color,
                    textcoords="offset points", xytext=(5, -8))

        px, py = coords_tproto[ti]
        ax.scatter([px], [py], c=cat_color, s=70 if is_target else 30,
                   marker="^", edgecolors="k", linewidths=0.5, zorder=4)
        ax.annotate(f"{prefix}{label_text} (proto)", (px, py), fontsize=5,
                    color=cat_color, style="italic",
                    textcoords="offset points", xytext=(5, 5))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"Fragment Trajectory — Image {image_id} ({target_label})\n"
        f"PCA (var: {var_exp[0]:.1%}, {var_exp[1]:.1%})",
        fontsize=11, fontweight="bold",
    )

    # Legend
    vis_handles = [
        mlines.Line2D([], [], marker="o" if i < len(levels) - 1 else "*",
                      color="w",
                      markerfacecolor=vis_cmap(i / (len(levels) - 1)),
                      markeredgecolor="k",
                      markersize=8 if i == len(levels) - 1 else 6,
                      label=f"L{L} ({get_visibility_ratio(L):.0%})")
        for i, L in enumerate(levels)
    ]
    img_handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="steelblue",
                      markeredgecolor="navy", markersize=5,
                      label="img same-cat (L=8)"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="lightgray",
                      markersize=5, label="img other-cat (L=8)"),
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor="darkgreen",
                      markeredgecolor="k", markersize=6,
                      label="text category"),
        mlines.Line2D([], [], marker="^", color="w", markerfacecolor="darkgreen",
                      markeredgecolor="k", markersize=6,
                      label="text prototype"),
    ]
    leg1 = ax.legend(handles=vis_handles, loc="upper left", fontsize=6,
                     title="Masking Level", title_fontsize=7, framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=img_handles, loc="upper right", fontsize=6,
              title="Markers", title_fontsize=7, framealpha=0.8)

    fig.tight_layout()
    path = out / f"fragment_trajectory_{image_id}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Mode B: Category clustering at specific layers (with legend) ──
    target_layers = [layer] if layer is not None else [5, 8, 11]
    print(f"  visualization: category clustering at layers {target_layers}")

    for tgt_layer in target_layers:
        cls_all = []
        cat_ids: list[int] = []
        for i in range(n):
            s = dataset[i]
            img_t = transform(s["image_pil"]).unsqueeze(0).to(device)
            layer_cls = extract_all_cls(model, img_t)
            # Project to 512-dim for comparability with text
            cls_512 = project_to_clip_space(visual, layer_cls[tgt_layer])
            cls_all.append(cls_512[0].cpu())
            cat_ids.append(s["scene_id"])

        # Joint projection with both text types (center each modality)
        cls_np = torch.stack(cls_all).numpy()  # [N, 512]
        cls_centered = cls_np - cls_np.mean(axis=0, keepdims=True)
        txt_all = np.concatenate([text_np_cat, text_np_proto], axis=0)
        txt_centered = txt_all - txt_all.mean(axis=0, keepdims=True)
        joint = np.concatenate([cls_centered, txt_centered], axis=0)
        cat_arr = np.array(cat_ids)
        n_imgs = cls_np.shape[0]
        perplexity = min(30, len(joint) - 1)

        num_cats = dataset.num_scenes
        cat_cmap = plt.colormaps.get_cmap("tab20").resampled(num_cats)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, (method_name, reducer) in enumerate([
            ("PCA", PCA(n_components=2, random_state=seed)),
            ("t-SNE", TSNE(n_components=2, random_state=seed,
                           perplexity=perplexity)),
        ]):
            coords2 = reducer.fit_transform(joint)
            img_coords = coords2[:n_imgs]
            tcat_coords = coords2[n_imgs:n_imgs + C]
            tproto_coords = coords2[n_imgs + C:]
            ax = axes[ax_idx]

            # Plot images colored by category
            for c in range(num_cats):
                mask = cat_arr == c
                if not mask.any():
                    continue
                ax.scatter(
                    img_coords[mask, 0], img_coords[mask, 1],
                    c=[cat_cmap(c)], s=25, alpha=0.7,
                    label=dataset.scene_labels[c],
                )

            # Plot category-level text (◆) and prototype text (▲)
            for ti in range(C):
                cc = cat_cmap(ti)
                # Category text
                tx, ty = tcat_coords[ti]
                ax.scatter([tx], [ty], c=[cc], s=70, marker="D",
                           edgecolors="k", linewidths=0.6, zorder=4)
                ax.annotate(dataset.scene_labels[ti], (tx, ty), fontsize=5,
                            textcoords="offset points", xytext=(4, -6),
                            color="black", fontweight="bold")
                # Prototype text
                px, py = tproto_coords[ti]
                ax.scatter([px], [py], c=[cc], s=70, marker="^",
                           edgecolors="k", linewidths=0.6, zorder=4)
                ax.annotate(f"{dataset.scene_labels[ti]} (p)", (px, py),
                            fontsize=4, style="italic",
                            textcoords="offset points", xytext=(4, 5),
                            color="black")

            # Highlight image 001
            if image_idx < n_imgs:
                ix, iy = img_coords[image_idx]
                ax.scatter([ix], [iy], c="none", s=120, marker="o",
                           edgecolors="red", linewidths=2, zorder=5)
                ax.annotate(f"img {image_id}", (ix, iy), fontsize=6,
                            color="red", fontweight="bold",
                            textcoords="offset points", xytext=(6, 6))

            ax.set_title(f"{method_name} — Layer {tgt_layer + 1}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right center", fontsize=7,
                   title="Category", title_fontsize=8, framealpha=0.9)
        fig.suptitle(
            f"Category Clusters (L=8, layer {tgt_layer + 1}/{_num_layers()})"
            f" — ◆ category text, ▲ instance prototype",
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
    out = _results_dir() / "probing"
    out.mkdir(parents=True, exist_ok=True)

    model, transform = _load_clip(device)
    visual = model.visual
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    acc_matrix = np.zeros((_num_layers(), len(levels)))

    for li, L in enumerate(levels):
        print(f"  probing: level L={L} (vis={get_visibility_ratio(L):.3f})")
        # Collect CLS from all layers for all images
        all_layer_feats: list[list[torch.Tensor]] = [[] for _ in range(_num_layers())]
        cat_ids: list[int] = []

        for i in range(n):
            sample = dataset[i]
            masked = mask_pil_image(
                sample["image_pil"], L, sample["seg_mask"], seed=seed, idx=i,
            )
            img_t = transform(masked).unsqueeze(0).to(device)
            layer_cls = extract_all_cls(model, img_t)
            for k in range(_num_layers()):
                all_layer_feats[k].append(layer_cls[k][0].cpu())
            cat_ids.append(sample["scene_id"])

        y = np.array(cat_ids)
        # Need at least 2 samples per class for StratifiedKFold
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        n_splits = min(5, min_count) if min_count >= 2 else 2

        for k in range(_num_layers()):
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
    ax.set_yticks(range(_num_layers()))
    ax.set_yticklabels([f"Layer {i + 1}" for i in range(_num_layers())], fontsize=8)
    ax.set_xlabel("Masking Level")
    ax.set_ylabel("Transformer Layer")
    ax.set_title(f"Linear Probe Accuracy ({_model_label()})")
    for i in range(_num_layers()):
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
def _extract_component_acts(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 64,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract attn and mlp activations from all 12 resblocks.

    Hooks on ``block.attn`` and ``block.mlp`` capture the output of each
    component (the contribution added to the residual stream).
    Processes in mini-batches to avoid OOM, stores on CPU.

    Returns:
        (attn_acts, mlp_acts): each a list of 12 tensors ``[N, T, D]`` on CPU.
    """
    N = imgs.shape[0]
    # Initialize accumulators: 12 lists, will concat at end
    attn_chunks: list[list[torch.Tensor]] = [[] for _ in range(_num_layers())]
    mlp_chunks: list[list[torch.Tensor]] = [[] for _ in range(_num_layers())]

    for start in range(0, N, batch_size):
        batch = imgs[start:start + batch_size]
        attn_buf: list[torch.Tensor] = []
        mlp_buf: list[torch.Tensor] = []
        handles = []

        for block in model.visual.transformer.resblocks:
            def _attn_hook(mod, inp, out, buf=attn_buf):
                o = out[0] if isinstance(out, tuple) else out
                buf.append(o.cpu())

            def _mlp_hook(mod, inp, out, buf=mlp_buf):
                buf.append(out.cpu())

            handles.append(block.attn.register_forward_hook(_attn_hook))
            handles.append(block.mlp.register_forward_hook(_mlp_hook))

        try:
            model.encode_image(batch)
        finally:
            for h in handles:
                h.remove()

        for k in range(_num_layers()):
            attn_chunks[k].append(attn_buf[k])
            mlp_chunks[k].append(mlp_buf[k])

    attn_acts = [torch.cat(chunks, dim=0) for chunks in attn_chunks]
    mlp_acts = [torch.cat(chunks, dim=0) for chunks in mlp_chunks]
    return attn_acts, mlp_acts


def _make_component_hook(
    src: torch.Tensor, token_mode: str, is_attn: bool,
):
    """Create a hook that replaces CLS or patch tokens in attn/mlp output.

    Args:
        src: source activations ``[B, T, D]`` to patch in (may be on CPU).
        token_mode: ``"cls"`` or ``"patch"``.
        is_attn: True if hooking on ``block.attn`` (returns tuple).
    """

    def hook_fn(mod, inp, output):
        if is_attn:
            tensor = output[0].clone()
        else:
            tensor = output.clone()

        src_dev = src.to(tensor.device)
        if token_mode == "cls":
            tensor[:, 0, :] = src_dev[:, 0, :]
        else:
            tensor[:, 1:, :] = src_dev[:, 1:, :]

        if is_attn:
            return (tensor, *output[1:])
        return tensor

    return hook_fn


@torch.no_grad()
def run_activation_patching(
    dataset_name: str = "fragment_v2",
    data_root: str | None = None,
    image_type: str = "original",
    num_choices: int = 5,
    num_runs: int = 3,
    max_images: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Activation patching on attn & mlp activations (not full residual).

    Patches the output of ``block.attn`` or ``block.mlp`` — the component
    contributions added to the residual stream — not the residual itself.

    Directions:
      - Noising:   run image i → inject partner j's component activation
      - Denoising: run partner j → inject image i's component activation

    Corruption sources (same masking level):
      - STR (same-category partner)
      - SIP (different-category partner)

    Token modes:
      - CLS-only:   replace only the CLS token (idx 0)
      - Patch-only: replace only patch tokens (idx 1:)

    Components: attn, mlp

    Output: 4-row × 4-col heatmap grid (12 layers × 8 masking levels).
    """
    out = _results_dir() / "patching"
    out.mkdir(parents=True, exist_ok=True)

    model, transform = _load_clip(device)
    visual = model.visual
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    # Instance-level text embeddings
    instance_names: list[str] = []
    cat_ids: list[int] = []
    for i in range(n):
        s = dataset.samples[i]
        instance_names.append(s.get("object_name", s.get("scene_label", "")))
        cat_ids.append(s["scene_id"])
    text_feats = get_text_embeddings(model, instance_names, device)
    K = min(num_choices, n)

    # Pre-compute partner indices
    rng_partner = np.random.RandomState(seed)
    same_cat_idx = np.zeros(n, dtype=int)
    diff_cat_idx = np.zeros(n, dtype=int)
    for i in range(n):
        same_cands = [j for j in range(n) if cat_ids[j] == cat_ids[i] and j != i]
        same_cat_idx[i] = rng_partner.choice(same_cands) if same_cands else i
        diff_cands = [j for j in range(n) if cat_ids[j] != cat_ids[i]]
        diff_cat_idx[i] = rng_partner.choice(diff_cands)

    # Conditions: (key, label, partner_idx, token_mode, component, direction)
    #   component: "attn" only (mlp removed — negligible CLS effect)
    #   direction: "noise" or "denoise"
    CONDITIONS = []
    for corr_name, partner_idx in [("STR", same_cat_idx), ("SIP", diff_cat_idx)]:
        for tmode in ("cls", "patch"):
            tag = f"{corr_name.lower()}_attn_{tmode}"
            label = f"{corr_name} attn {tmode}"
            CONDITIONS.append((tag, label, partner_idx, tmode, "attn", "noise"))
            dn_tag = f"dn_{tag}"
            dn_label = f"DN-{corr_name} attn {tmode}"
            CONDITIONS.append((dn_tag, dn_label, partner_idx, tmode, "attn", "denoise"))

    all_results: dict[str, dict[int, list[float]]] = {
        c[0]: {} for c in CONDITIONS
    }
    baselines: dict[int, float] = {}

    for L in levels:
        vis = get_visibility_ratio(L)
        imgs, _ = prepare_masked_batch(
            dataset, transform, L, seed, max_images, device,
        )

        # Cache attn activations for all images at this level
        attn_acts, _ = _extract_component_acts(model, imgs)

        baseline_embed = model.encode_image(imgs)
        baseline_acc = _patching_instance_acc(
            baseline_embed, cat_ids, text_feats, K, n, num_runs, seed,
        )
        baselines[L] = baseline_acc
        print(f"\n  L={L} (vis={vis:.3f})  baseline={baseline_acc:.4f}")

        # Pre-compute denoising baselines per partner type
        dn_baselines: dict[str, float] = {}
        for corr_name, partner_idx in [("STR", same_cat_idx), ("SIP", diff_cat_idx)]:
            partner_imgs = imgs[partner_idx]
            partner_embed = model.encode_image(partner_imgs)
            dn_baselines[corr_name] = _patching_instance_acc(
                partner_embed, cat_ids, text_feats, K, n, num_runs, seed,
            )

        for cond_key, cond_label, partner_idx, tmode, comp, direction in CONDITIONS:
            acts = attn_acts if comp == "attn" else mlp_acts
            is_attn = comp == "attn"

            if direction == "noise":
                run_imgs = imgs
                ref_acc = baseline_acc
            else:
                run_imgs = imgs[partner_idx]
                corr_name = cond_label.split("-")[1].split(" ")[0]  # STR or SIP
                ref_acc = dn_baselines[corr_name]

            deltas: list[float] = []
            for layer_i in range(_num_layers()):
                if direction == "noise":
                    src = acts[layer_i][partner_idx]  # partner's activation
                else:
                    src = acts[layer_i]  # original image's activation

                block = visual.transformer.resblocks[layer_i]
                target_mod = block.attn if is_attn else block.mlp
                handle = target_mod.register_forward_hook(
                    _make_component_hook(src, tmode, is_attn),
                )
                try:
                    patched_embed = model.encode_image(run_imgs)
                finally:
                    handle.remove()

                acc = _patching_instance_acc(
                    patched_embed, cat_ids, text_feats, K, n, num_runs, seed,
                )
                deltas.append(acc - ref_acc)

            all_results[cond_key][L] = deltas
            if direction == "noise":
                best_i = int(np.argmin(deltas)) + 1
                tag = f"worst=layer {best_i:2d}"
            else:
                best_i = int(np.argmax(deltas)) + 1
                tag = f"best=layer  {best_i:2d}"
            print(f"    {cond_label:22s}  {tag}"
                  f"  range=[{min(deltas):+.4f}, {max(deltas):+.4f}]")

    _save_json({"baselines": baselines, **all_results},
               out / f"results_{image_type}.json")

    # Plot: 2 rows (noising, denoising) × 4 cols (STR-cls, STR-patch, SIP-cls, SIP-patch)
    PLOT_GRID = []
    for direction, dir_label in [("noise", "Noising"), ("denoise", "Denoising")]:
        row = []
        for corr_name in ("STR", "SIP"):
            for tmode in ("cls", "patch"):
                prefix = "" if direction == "noise" else "dn_"
                key = f"{prefix}{corr_name.lower()}_attn_{tmode}"
                title = f"{dir_label}: {corr_name} attn {tmode}"
                row.append((key, title))
        PLOT_GRID.append(row)

    n_rows = len(PLOT_GRID)
    n_cols = len(PLOT_GRID[0])
    vis_labels = [f"{L}\n{get_visibility_ratio(L):.0%}" for L in levels]
    layer_labels = [f"L{i + 1}" for i in range(_num_layers())]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
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
            ax.set_yticklabels(layer_labels, fontsize=7)
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
        f"Activation Patching (attn) — {image_type} — Δ Instance Accuracy",
        fontweight="bold", fontsize=14,
    )
    fig.tight_layout()
    path = out / f"patching_heatmap_{image_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _patching_instance_acc(
    embeds: torch.Tensor,
    cat_ids: list[int],
    text_feats: torch.Tensor,
    K: int,
    n: int,
    num_runs: int,
    seed: int,
) -> float:
    """Instance-level K-choice zero-shot accuracy.

    For each image i, pick 1 correct instance text + (K-1) distractor
    instance texts (from other images). Match by cosine similarity.
    """
    embeds_cpu = F.normalize(embeds.float().cpu(), dim=-1)
    run_accs = []
    for run in range(num_runs):
        rng = np.random.RandomState(seed + run)
        correct = 0
        for i in range(n):
            distractors = [j for j in range(n) if j != i]
            chosen = rng.choice(distractors, size=min(K - 1, len(distractors)),
                                replace=False).tolist()
            candidates = [i] + chosen  # index 0 = correct instance
            sims = embeds_cpu[i] @ text_feats[candidates].T
            correct += int(sims.argmax().item() == 0)
        run_accs.append(correct / n)
    return float(np.mean(run_accs))


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
    out = _results_dir() / "logit_lens"
    out.mkdir(parents=True, exist_ok=True)

    model, transform = _load_clip(device)
    visual = model.visual
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)

    text_feats = get_text_embeddings(model, dataset.scene_labels, device)
    C = text_feats.shape[0]

    imgs, cat_ids = prepare_masked_batch(
        dataset, transform, mask_level, seed, max_images, device,
    )
    layer_cls_list = extract_all_cls(model, imgs)  # 12 × [N, 768]

    # Per-layer: compute similarities and ranks
    sim_matrix = np.zeros((_num_layers(), C))  # mean sim per category
    rank_per_layer = np.zeros(_num_layers())

    for k in range(_num_layers()):
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
    ax.set_yticks(range(_num_layers()))
    ax.set_yticklabels([f"L{i + 1}" for i in range(_num_layers())], fontsize=7)
    ax.set_xlabel("Category")
    ax.set_ylabel("Layer")
    ax.set_title(f"Cosine Similarity (mask L={mask_level})")
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    ax.plot(range(1, _num_layers() + 1), rank_per_layer, marker="o", color="#e74c3c")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Rank of Correct Category")
    ax.set_title("Correct Category Rank vs Layer")
    ax.set_xticks(range(1, _num_layers() + 1))
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Logit Lens — {_model_label()} (L={mask_level})", fontweight="bold")
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

    def __init__(self, input_dim: int = _internal_dim(), hidden_dim: int = 3072):
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
    out = _results_dir() / "transcoder"
    out.mkdir(parents=True, exist_ok=True)

    model, transform = _load_clip(device)
    visual = model.visual
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Extract CLS tokens at target layer
    print(f"  transcoder: extracting CLS tokens from layer {layer + 1}")
    cls_all, cat_ids = [], []
    with torch.no_grad():
        for i in range(n):
            s = dataset[i]
            img_t = transform(s["image_pil"]).unsqueeze(0).to(device)
            layer_cls = extract_all_cls(model, img_t)
            cls_all.append(layer_cls[layer][0].cpu())
            cat_ids.append(s["scene_id"])

    X = torch.stack(cls_all)  # [N, 768]
    y = np.array(cat_ids)
    C = dataset.num_scenes

    # Train SAE
    sae = SparseAutoencoder(_internal_dim(), hidden_dim).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    X_dev = X.to(device)

    losses = []
    print(f"  transcoder: training SAE ({_internal_dim()}→{hidden_dim}) for {num_epochs} epochs")
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
    # Shared args via parent parser so they work after the subcommand too
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--dataset", default="fragment_v2", choices=["fragment_v2", "ade20k"],
    )
    shared.add_argument("--data-root", default=None)
    shared.add_argument(
        "--image-type", default="original", choices=["original", "gray", "lined"],
    )
    shared.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    shared.add_argument("--seed", type=int, default=42)
    shared.add_argument("--max-images", type=int, default=None)
    shared.add_argument("--model", default="B-16", choices=list(MODEL_CONFIGS.keys()),
                        help="CLIP model variant: B-16 or L-14")

    parser = argparse.ArgumentParser(
        description="CLIP mechanistic interpretability experiments",
        parents=[shared],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("zeroshot", parents=[shared])
    p.add_argument("--num-choices", type=int, default=5)
    p.add_argument("--num-runs", type=int, default=3)

    p = sub.add_parser("visualize", parents=[shared])
    p.add_argument("--layer", type=int, default=None,
                    help="Layer index 0-11 (default: show 5, 8, 11)")
    p.add_argument("--image-idx", type=int, default=0)

    sub.add_parser("probe", parents=[shared])

    p = sub.add_parser("patch", parents=[shared])
    p.add_argument("--num-choices", type=int, default=5)
    p.add_argument("--num-runs", type=int, default=3)

    p = sub.add_parser("logit-lens", parents=[shared])
    p.add_argument("--mask-level", type=int, default=8)

    p = sub.add_parser("transcoder", parents=[shared])
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=3072)
    p.add_argument("--sparsity-lambda", type=float, default=1e-3)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    sub.add_parser("all", parents=[shared])

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
            kw.update(num_choices=args.num_choices, num_runs=args.num_runs)
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
