#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dinov2_interp.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-30-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""DINOv2+dino.txt mechanistic interpretability experiments.

Standalone analyses on DINOv2 ViT-L/14 + dino.txt text alignment:
  zeroshot      Zero-shot K-choice instance classification
  visualize     PCA fragment trajectory + category clustering
  probe         Linear probing per layer × masking level
  patch         Activation patching on attn (noising + denoising)
  all           Run all analyses

Usage:
    uv run python -m src.dinov2_interp zeroshot --max-images 5
    uv run python -m src.dinov2_interp all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.dataset import get_dataset
from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image

NUM_LAYERS = 24  # DINOv2 ViT-L has 24 blocks
INTERNAL_DIM = 1024
PROJ_DIM = 2048  # encode_image output dim (CLS concat patch-avg)
IMG_SIZE = 518  # DINOv2 ViT-L/14 native resolution
RESULTS_DIR = Path("results/dinov2_interp")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(device: str = "cuda"):
    """Load dino.txt model + tokenizer.

    Returns:
        (model, tokenizer, transform)
    """
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

    # Build transform (DINOv2 uses 518×518 with specific normalization)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, tokenizer, transform


def _get_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the 24 backbone transformer blocks."""
    return model.visual_model.backbone.model.blocks


@torch.no_grad()
def _encode_image_batched(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 16,
) -> torch.Tensor:
    """Batched encode_image to avoid OOM on ViT-L with large inputs."""
    parts = []
    for start in range(0, imgs.shape[0], batch_size):
        parts.append(model.encode_image(imgs[start:start + batch_size]).cpu())
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_json(data: dict, path: Path) -> None:
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


@torch.no_grad()
def get_text_embeddings(
    model: nn.Module,
    tokenizer,
    labels: list[str],
    device: str = "cuda",
    template: str = "an image of {label}",
) -> torch.Tensor:
    """Encode labels into dino.txt text embeddings.

    Returns:
        ``[C, 2048]`` L2-normalized text embeddings on CPU.
    """
    prompts = [template.format(label=lab) for lab in labels]
    tokens = tokenizer.tokenize(prompts).to(device)
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
    """Preprocess all images at a masking level.

    Returns:
        (tensor ``[N, 3, 518, 518]``, cat_ids)
    """
    n = min(len(dataset), max_images) if max_images else len(dataset)
    tensors, cat_ids = [], []
    for i in range(n):
        sample = dataset[i]
        masked = mask_pil_image(
            sample["image_pil"], level, sample["seg_mask"],
            seed=seed, idx=i, target_size=IMG_SIZE,
        )
        tensors.append(transform(masked))
        cat_ids.append(sample["scene_id"])
    return torch.stack(tensors).to(device), cat_ids


@torch.no_grad()
def _extract_attn_acts(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 32,
) -> list[torch.Tensor]:
    """Extract attn activations from all 24 backbone blocks.

    Returns:
        List of 24 tensors ``[N, T, D]`` on CPU.
    """
    blocks = _get_blocks(model)
    N = imgs.shape[0]
    chunks: list[list[torch.Tensor]] = [[] for _ in range(NUM_LAYERS)]

    for start in range(0, N, batch_size):
        batch = imgs[start:start + batch_size]
        buf: list[torch.Tensor] = []
        handles = []

        for block in blocks:
            def _hook(mod, inp, out, b=buf):
                o = out[0] if isinstance(out, tuple) else out
                b.append(o.cpu())
            handles.append(block.attn.register_forward_hook(_hook))

        try:
            model.encode_image(batch)
        finally:
            for h in handles:
                h.remove()

        for k in range(NUM_LAYERS):
            chunks[k].append(buf[k])

    return [torch.cat(c, dim=0) for c in chunks]


def _make_attn_hook(src: torch.Tensor, token_mode: str):
    """Hook that replaces CLS or patch tokens in attn output."""

    def hook_fn(mod, inp, output):
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


def _instance_acc(
    embeds: torch.Tensor,
    cat_ids: list[int],
    text_feats: torch.Tensor,
    K: int,
    n: int,
    num_runs: int,
    seed: int,
) -> float:
    """Instance-level K-choice zero-shot accuracy."""
    embeds_cpu = F.normalize(embeds.float().cpu(), dim=-1)
    run_accs = []
    for run in range(num_runs):
        rng = np.random.RandomState(seed + run)
        correct = 0
        for i in range(n):
            distractors = [j for j in range(n) if j != i]
            chosen = rng.choice(distractors, size=min(K - 1, len(distractors)),
                                replace=False).tolist()
            candidates = [i] + chosen
            sims = embeds_cpu[i] @ text_feats[candidates].T
            correct += int(sims.argmax().item() == 0)
        run_accs.append(correct / n)
    return float(np.mean(run_accs))


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
    out = RESULTS_DIR / "zeroshot"
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, transform = _load_model(device)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    instance_names = [
        dataset.samples[i].get("object_name", dataset.samples[i].get("scene_label", ""))
        for i in range(n)
    ]
    text_feats = get_text_embeddings(model, tokenizer, instance_names, device)
    K = min(num_choices, n)

    print(f"  zeroshot: {n} images, K={K}")
    results: dict[str, dict] = {}
    for L in levels:
        imgs, cat_ids = prepare_masked_batch(
            dataset, transform, L, seed, max_images, device,
        )
        embeds = F.normalize(_encode_image_batched(model, imgs).float(), dim=-1)
        acc = _instance_acc(embeds, cat_ids, text_feats, K, n, num_runs, seed)
        vis = get_visibility_ratio(L)
        results[str(L)] = {"acc": acc}
        print(f"    L={L} vis={vis:.3f}  acc={acc:.4f}")

    _save_json(results, out / f"results_{image_type}.json")

    # Plot
    lvls = list(range(1, 9))
    vis_x = [get_visibility_ratio(l) for l in lvls]
    accs = [results[str(l)]["acc"] for l in lvls]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vis_x, accs, marker="o")
    ax.set_xlabel("Visibility Ratio")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"DINOv2+dino.txt Zero-Shot Instance ({image_type})")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"accuracy_{image_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / f'accuracy_{image_type}.png'}")


# ===================================================================
# 2. Visualization
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
    """Fragment trajectory + category clustering."""
    out = RESULTS_DIR / "visualization"
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, transform = _load_model(device)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()
    sample = dataset[image_idx]
    image_id = sample["image_id"]
    target_cat = sample["scene_id"]
    target_label = sample["scene_label"]

    # Text embeddings
    cat_text = get_text_embeddings(model, tokenizer, dataset.scene_labels, device)
    cat_text_np = cat_text.numpy()
    C = cat_text_np.shape[0]

    # ── Fragment trajectory ──
    print(f"  visualization: fragment trajectory for image {image_id}")

    traj_embeds: list[np.ndarray] = []
    for L in levels:
        masked = mask_pil_image(
            sample["image_pil"], L, sample["seg_mask"],
            seed=seed, idx=image_idx, target_size=IMG_SIZE,
        )
        img_t = transform(masked).unsqueeze(0).to(device)
        embed = F.normalize(model.encode_image(img_t).float().cpu(), dim=-1)[0].numpy()
        traj_embeds.append(embed)
    traj_np = np.stack(traj_embeds)

    dist_embeds, dist_cats = [], []
    for i in range(n):
        s = dataset[i]
        img_t = transform(s["image_pil"]).unsqueeze(0).to(device)
        embed = F.normalize(model.encode_image(img_t).float().cpu(), dim=-1)[0].numpy()
        dist_embeds.append(embed)
        dist_cats.append(s["scene_id"])
    dist_np = np.stack(dist_embeds)

    # PCA fit on full images, project everything
    n_lev = len(levels)
    img_mean = dist_np.mean(axis=0, keepdims=True)
    text_mean = cat_text_np.mean(axis=0, keepdims=True)

    pca = PCA(n_components=2, random_state=seed)
    pca.fit(dist_np - img_mean)
    var_exp = pca.explained_variance_ratio_

    coords_traj = pca.transform(traj_np - img_mean)
    coords_dist = pca.transform(dist_np - img_mean)
    coords_text = pca.transform(cat_text_np - text_mean)

    vis_cmap = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(10, 8))

    for di in range(len(dist_np)):
        dx, dy = coords_dist[di]
        if di == image_idx:
            ax.scatter([dx], [dy], c="red", s=60, marker="o",
                       edgecolors="k", linewidths=1.0, zorder=4)
        elif dist_cats[di] == target_cat:
            ax.scatter([dx], [dy], c="steelblue", s=25, alpha=0.6,
                       edgecolors="navy", linewidths=0.3, zorder=2)
        else:
            ax.scatter([dx], [dy], c="lightgray", s=15, alpha=0.4, zorder=1)

    ax.plot(coords_traj[:, 0], coords_traj[:, 1],
            c="red", alpha=0.4, linewidth=2, linestyle="--", zorder=3)
    for idx_l, L in enumerate(levels):
        color = vis_cmap(idx_l / (len(levels) - 1))
        size = 120 if L == levels[-1] else 60
        marker = "*" if L == levels[-1] else "o"
        ax.scatter(coords_traj[idx_l, 0], coords_traj[idx_l, 1],
                   c=[color], s=size, marker=marker,
                   edgecolors="k", linewidths=0.8, zorder=5)
        ax.annotate(f"L{L}", (coords_traj[idx_l, 0], coords_traj[idx_l, 1]),
                    fontsize=7, fontweight="bold",
                    textcoords="offset points", xytext=(6, 6))

    for ti in range(C):
        is_target = (ti == target_cat)
        tx, ty = coords_text[ti]
        color = "darkgreen" if is_target else "gray"
        ax.scatter([tx], [ty], c=color, s=70 if is_target else 30,
                   marker="D", edgecolors="k", linewidths=0.5, zorder=4)
        prefix = "[T] " if is_target else ""
        ax.annotate(f"{prefix}{dataset.scene_labels[ti]}", (tx, ty),
                    fontsize=6, color=color,
                    textcoords="offset points", xytext=(5, -8))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"DINOv2+dino.txt Fragment Trajectory — Image {image_id} ({target_label})\n"
        f"PCA (var: {var_exp[0]:.1%}, {var_exp[1]:.1%})",
        fontsize=11, fontweight="bold",
    )

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
                      markeredgecolor="navy", markersize=5, label="img same-cat"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="lightgray",
                      markersize=5, label="img other-cat"),
        mlines.Line2D([], [], marker="D", color="w", markerfacecolor="darkgreen",
                      markeredgecolor="k", markersize=6, label="text (target)"),
    ]
    leg1 = ax.legend(handles=vis_handles, loc="upper left", fontsize=6,
                     title="Masking Level", title_fontsize=7, framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=img_handles, loc="upper right", fontsize=6,
              title="Markers", title_fontsize=7, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out / f"fragment_trajectory_{image_id}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / f'fragment_trajectory_{image_id}.png'}")


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
    """Linear probing on 24 backbone layers to predict category."""
    out = RESULTS_DIR / "probing"
    out.mkdir(parents=True, exist_ok=True)

    model, _, transform = _load_model(device)
    blocks = _get_blocks(model)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    acc_matrix = np.zeros((NUM_LAYERS, len(levels)))

    for li, L in enumerate(levels):
        print(f"  probing: level L={L} (vis={get_visibility_ratio(L):.3f})")
        # Extract CLS from all layers via hooks
        all_layer_feats: list[list[torch.Tensor]] = [[] for _ in range(NUM_LAYERS)]
        cat_ids: list[int] = []

        for i in range(n):
            sample = dataset[i]
            masked = mask_pil_image(
                sample["image_pil"], L, sample["seg_mask"],
                seed=seed, idx=i, target_size=IMG_SIZE,
            )
            img_t = transform(masked).unsqueeze(0).to(device)

            cls_buf: list[torch.Tensor] = []
            handles = []
            for block in blocks:
                def _hook(mod, inp, out, buf=cls_buf):
                    o = out[0] if isinstance(out, tuple) else out
                    buf.append(o[:, 0, :].cpu())  # CLS token
                handles.append(block.register_forward_hook(_hook))

            try:
                model.encode_image(img_t)
            finally:
                for h in handles:
                    h.remove()

            for k in range(NUM_LAYERS):
                all_layer_feats[k].append(cls_buf[k][0])
            cat_ids.append(sample["scene_id"])

        y = np.array(cat_ids)
        unique, counts = np.unique(y, return_counts=True)
        n_splits = min(5, counts.min()) if counts.min() >= 2 else 2

        for k in range(NUM_LAYERS):
            X = torch.stack(all_layer_feats[k]).numpy()
            if n_splits < 2:
                acc_matrix[k, li] = 0.0
                continue
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
            acc_matrix[k, li] = scores.mean()

        # Print summary for this level
        best_layer = int(acc_matrix[:, li].argmax()) + 1
        print(f"    best=layer {best_layer} acc={acc_matrix[:, li].max():.4f}")

    _save_json(
        {"accuracy": acc_matrix.tolist(), "layers": list(range(1, NUM_LAYERS + 1)),
         "levels": list(range(1, 9))},
        out / f"results_{image_type}.json",
    )

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    vis_labels = [f"L{l}\n{get_visibility_ratio(l):.0%}" for l in range(1, 9)]
    im = ax.imshow(acc_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(8))
    ax.set_xticklabels(vis_labels, fontsize=7)
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"L{i + 1}" for i in range(NUM_LAYERS)], fontsize=6)
    ax.set_xlabel("Masking Level")
    ax.set_ylabel("Backbone Layer")
    ax.set_title(f"DINOv2 Linear Probe Accuracy ({image_type})")
    for i in range(NUM_LAYERS):
        for j in range(8):
            ax.text(j, i, f"{acc_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=4, color="white" if acc_matrix[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out / f"probe_heatmap_{image_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / f'probe_heatmap_{image_type}.png'}")


# ===================================================================
# 4. Activation Patching
# ===================================================================


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
    """Attn activation patching: STR/SIP × CLS/patch, noising + denoising."""
    out = RESULTS_DIR / "patching"
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, transform = _load_model(device)
    blocks = _get_blocks(model)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()

    instance_names = [
        dataset.samples[i].get("object_name", dataset.samples[i].get("scene_label", ""))
        for i in range(n)
    ]
    text_feats = get_text_embeddings(model, tokenizer, instance_names, device)
    K = min(num_choices, n)

    cat_ids: list[int] = [dataset[i]["scene_id"] for i in range(n)]

    # Partner indices
    rng_partner = np.random.RandomState(seed)
    same_cat_idx = np.zeros(n, dtype=int)
    diff_cat_idx = np.zeros(n, dtype=int)
    for i in range(n):
        same_cands = [j for j in range(n) if cat_ids[j] == cat_ids[i] and j != i]
        same_cat_idx[i] = rng_partner.choice(same_cands) if same_cands else i
        diff_cands = [j for j in range(n) if cat_ids[j] != cat_ids[i]]
        diff_cat_idx[i] = rng_partner.choice(diff_cands)

    # Conditions: attn-only
    CONDITIONS = []
    for corr_name, partner_idx in [("STR", same_cat_idx), ("SIP", diff_cat_idx)]:
        for tmode in ("cls", "patch"):
            tag = f"{corr_name.lower()}_attn_{tmode}"
            label = f"{corr_name} attn {tmode}"
            CONDITIONS.append((tag, label, partner_idx, tmode, "noise"))
            CONDITIONS.append((f"dn_{tag}", f"DN-{label}", partner_idx, tmode, "denoise"))

    all_results: dict[str, dict[int, list[float]]] = {c[0]: {} for c in CONDITIONS}
    baselines: dict[int, float] = {}

    for L in levels:
        vis = get_visibility_ratio(L)
        imgs, _ = prepare_masked_batch(
            dataset, transform, L, seed, max_images, device,
        )
        attn_acts = _extract_attn_acts(model, imgs)

        baseline_embed = _encode_image_batched(model, imgs)
        baseline_acc = _instance_acc(
            baseline_embed, cat_ids, text_feats, K, n, num_runs, seed,
        )
        baselines[L] = baseline_acc
        print(f"\n  L={L} (vis={vis:.3f})  baseline={baseline_acc:.4f}")

        # Denoising baselines
        dn_baselines: dict[str, float] = {}
        for corr_name, partner_idx in [("STR", same_cat_idx), ("SIP", diff_cat_idx)]:
            partner_imgs = imgs[partner_idx]
            partner_embed = _encode_image_batched(model, partner_imgs)
            dn_baselines[corr_name] = _instance_acc(
                partner_embed, cat_ids, text_feats, K, n, num_runs, seed,
            )

        for cond_key, cond_label, partner_idx, tmode, direction in CONDITIONS:
            if direction == "noise":
                run_imgs = imgs
                ref_acc = baseline_acc
            else:
                run_imgs = imgs[partner_idx]
                corr_name = cond_label.split("-")[1].split(" ")[0]
                ref_acc = dn_baselines[corr_name]

            deltas: list[float] = []
            for layer_i in range(NUM_LAYERS):
                if direction == "noise":
                    src = attn_acts[layer_i][partner_idx]
                else:
                    src = attn_acts[layer_i]

                handle = blocks[layer_i].attn.register_forward_hook(
                    _make_attn_hook(src, tmode),
                )
                try:
                    patched_embed = model.encode_image(run_imgs)
                finally:
                    handle.remove()

                acc = _instance_acc(
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

    # Plot: 2×4 heatmap
    PLOT_GRID = []
    for direction, dir_label in [("noise", "Noising"), ("denoise", "Denoising")]:
        row = []
        for corr_name in ("STR", "SIP"):
            for tmode in ("cls", "patch"):
                prefix = "" if direction == "noise" else "dn_"
                key = f"{prefix}{corr_name.lower()}_attn_{tmode}"
                title = f"{dir_label}: {corr_name} {tmode}"
                row.append((key, title))
        PLOT_GRID.append(row)

    vis_labels = [f"{L}\n{get_visibility_ratio(L):.0%}" for L in levels]
    layer_labels = [f"L{i + 1}" for i in range(NUM_LAYERS)]

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
            ax.set_yticks(range(NUM_LAYERS))
            ax.set_yticklabels(layer_labels, fontsize=5)
            ax.set_xlabel("Masking Level", fontsize=7)
            ax.set_ylabel("Layer", fontsize=7)
            ax.set_title(title, fontsize=8, fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(
        f"DINOv2+dino.txt Attn Patching ({image_type}) — Δ Instance Accuracy",
        fontweight="bold", fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out / f"patching_heatmap_{image_type}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / f'patching_heatmap_{image_type}.png'}")


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    """Entry point for DINOv2+dino.txt interpretability experiments."""
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--dataset", default="fragment_v2",
                        choices=["fragment_v2", "ade20k"])
    shared.add_argument("--data-root", default=None)
    shared.add_argument("--image-type", default="original",
                        choices=["original", "gray", "lined"])
    shared.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    shared.add_argument("--seed", type=int, default=42)
    shared.add_argument("--max-images", type=int, default=None)

    parser = argparse.ArgumentParser(
        description="DINOv2+dino.txt mechanistic interpretability",
        parents=[shared],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("zeroshot", parents=[shared])
    p.add_argument("--num-choices", type=int, default=5)
    p.add_argument("--num-runs", type=int, default=3)

    p = sub.add_parser("visualize", parents=[shared])
    p.add_argument("--image-idx", type=int, default=0)
    p.add_argument("--layer", type=int, default=None)

    sub.add_parser("probe", parents=[shared])

    p = sub.add_parser("patch", parents=[shared])
    p.add_argument("--num-choices", type=int, default=5)
    p.add_argument("--num-runs", type=int, default=3)

    sub.add_parser("all", parents=[shared])

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
            kw.update(image_idx=args.image_idx, layer=args.layer)
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

    print("\nDone!")


if __name__ == "__main__":
    main()
