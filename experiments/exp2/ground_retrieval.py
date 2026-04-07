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
    uv run python -m experiments.exp2.ground_retrieval --model clip retrieve --max-images 10
    uv run python -m experiments.exp2.ground_retrieval --model dinov2 patch
    uv run python -m experiments.exp2.ground_retrieval --model clip all
"""

from __future__ import annotations

import argparse
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
from src.experiment_config import MODEL_CONFIGS as _ALL_MODEL_CONFIGS
from src.experiment_config import save_experiment_settings
from src.masking import get_mask_levels, get_visibility_ratio
from src.utils import (
    compute_category_accuracy,
    compute_exemplar_accuracy,
    compute_retrieval_metrics,
    save_json,
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Map CLI --model flag to experiment_config keys
MODEL_CONFIGS: dict[str, dict] = {
    "clip": _ALL_MODEL_CONFIGS["clip_L14"],
    "dinov2": _ALL_MODEL_CONFIGS["dinov2"],
}

_CFG: dict = MODEL_CONFIGS["clip"]
_DATASET_TAG: str = ""  # set by main(); non-empty adds subdir

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


_RESULTS_BASE: Path | None = None  # set by --results-dir; None = use default

# Image-prototype k-sweep (COCO only): k = number of same-category images
# averaged into each per-category prototype. Saturation observed by k≈20
# on COCO; pushing beyond doesn't change the curves.
PROTO_K_EXCL: tuple[int, ...] = (1, 2, 5, 10, 20)
PROTO_K_INCL: tuple[int, ...] = (1, 2, 5, 10, 20)
SWEEP_RESULTS_BASE = Path("results/exp2-sweep")


def _results_dir() -> Path:
    if _RESULTS_BASE is not None:
        return _RESULTS_BASE
    return Path("results/exp2/retrieval_16") / _DATASET_TAG / _CFG["tag"]


def _sweep_retrieval_dir() -> Path:
    """Output dir for the prototype k-sweep retrieval run."""
    return SWEEP_RESULTS_BASE / _DATASET_TAG / _CFG["tag"] / "retrieval"


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
    model: nn.Module, imgs: torch.Tensor, batch_size: int | None = None,
) -> torch.Tensor:
    """Encode images -> [N, proj_dim], L2-normalized, CPU.

    Accepts CPU or GPU tensors. Moves each batch to model device, encodes,
    and stores results on CPU to minimize GPU memory.
    """
    if batch_size is None:
        batch_size = 4 if not _is_clip() else 16
    device = next(model.parameters()).device
    parts: list[torch.Tensor] = []
    for start in range(0, imgs.shape[0], batch_size):
        batch = imgs[start:start + batch_size].to(device)
        emb = model.encode_image(batch)
        parts.append(emb.cpu())
        del batch
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


_save_json = save_json  # alias for backward compat within file


def _build_kchoice_candidates(
    query_idx: int,
    query_cat: int,
    cat_to_indices: dict[int, list[int]],
    unique_cats: list[int],
    rng: np.random.RandomState,
) -> list[int] | None:
    """Build candidate list for C-way K-choice cross-instance retrieval.

    Target is a *different* image from the same category (not the query itself).
    Returns None if the query's category has only one image.
    """
    same_cat = [j for j in cat_to_indices[query_cat] if j != query_idx]
    if not same_cat:
        return None
    target = rng.choice(same_cat)
    candidates = [target]
    for c in unique_cats:
        if c == query_cat:
            continue
        candidates.append(rng.choice(cat_to_indices[c]))
    return candidates


def _build_base_cat_proto_gallery(
    image_gallery: torch.Tensor,
    cat_to_indices: dict[int, list[int]],
    k: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """Build C category prototypes, each = mean of k random same-cat images.

    Query-independent base gallery: each cat's prototype is k random images
    from that category. Used for "other" cats in the per-query rebuild loop
    (the query's own cat gets re-sampled per-query to enforce the
    include/exclude target semantics).

    Args:
        image_gallery: [N, D] L2-normalized complete-image embeddings (CPU).
        cat_to_indices: category id -> list of instance indices in that cat.
        k: number of images to average per category.
        rng: numpy RandomState for reproducible sampling.

    Returns:
        [C, D] L2-normalized category prototype gallery (CPU).
    """
    cat_keys = sorted(cat_to_indices.keys())
    C = len(cat_keys)
    D = image_gallery.shape[1]
    proto = torch.zeros(C, D)
    for c in cat_keys:
        members = rng.choice(
            cat_to_indices[c], size=k, replace=False,
        ).tolist()
        proto[c] = image_gallery[members].mean(dim=0)
    return F.normalize(proto, dim=-1)


def _eval_proto_k_sweep(
    query: torch.Tensor,
    image_gallery: torch.Tensor,
    base_gallery: torch.Tensor,
    cat_to_indices: dict[int, list[int]],
    cat_ids: list[int],
    k: int,
    include_target: bool,
    rng: np.random.RandomState,
) -> dict[str, float]:
    """C-way categorization with per-query rebuild of the query's own cat.

    For each query i in cat c_i:
      - Other cats c ≠ c_i use ``base_gallery[c]`` (k random same-cat images,
        query-independent).
      - The query's own cat c_i is rebuilt per-query:
          * include_target: members = [i] + (k-1) random same-cat-others
          * exclude_target: members = k random same-cat-others (excluding i)
      - Compute sim(query[i], gallery) over C cats and rank c_i.
      - R@1 = top-1 acc, R@5 = top-5 acc (out of C), MRR = 1/(rank+1).

    Args:
        query: [N, D] L2-normalized masked-image query embeddings.
        image_gallery: [N, D] L2-normalized complete-image embeddings.
        base_gallery: [C, D] L2-normalized base category prototypes.
        cat_to_indices: category id -> list of instance indices.
        cat_ids: per-instance category id.
        k: number of images per prototype.
        include_target: whether the query's own image is in its cat prototype.
        rng: numpy RandomState for per-query sampling.

    Returns:
        Dict with recall_at_1, recall_at_5, mrr.
    """
    n = query.shape[0]
    C = base_gallery.shape[0]
    r1 = r5 = mrr_val = 0.0
    for i in range(n):
        c_i = cat_ids[i]
        same_cat_others = [j for j in cat_to_indices[c_i] if j != i]
        if include_target:
            if k == 1:
                members = [i]
            else:
                extra = rng.choice(
                    same_cat_others, size=k - 1, replace=False,
                ).tolist()
                members = [i, *extra]
        else:
            members = rng.choice(
                same_cat_others, size=k, replace=False,
            ).tolist()
        proto_ci = image_gallery[members].mean(dim=0)
        proto_ci = F.normalize(proto_ci, dim=0)

        # Substitute the query's own cat in a local copy.
        local = base_gallery.clone()
        local[c_i] = proto_ci

        sims = query[i] @ local.T  # [C]
        ranked = sims.argsort(descending=True).tolist()
        rank = ranked.index(c_i)
        if rank < 1:
            r1 += 1
        if rank < 5:
            r5 += 1
        mrr_val += 1.0 / (rank + 1)
    return {
        "recall_at_1": r1 / n,
        "recall_at_5": r5 / n,
        "mrr": mrr_val / n,
    }


def _cross_instance_metrics(
    sims: torch.Tensor,
    cat_to_indices: dict[int, list[int]],
    cat_ids: list[int],
    n: int,
) -> dict[str, float]:
    """Compute R@1, R@5, MRR for cross-instance retrieval.

    GT = any same-category image excluding the query itself.
    """
    r1 = r5 = mrr_val = 0.0
    n_eval = 0
    for i in range(n):
        same_cat = {j for j in cat_to_indices[cat_ids[i]] if j != i}
        if not same_cat:
            continue
        row = sims[i].clone()
        row[i] = -float("inf")
        ranked = row.argsort(descending=True)
        for rank, idx in enumerate(ranked.tolist()):
            if idx in same_cat:
                if rank < 1:
                    r1 += 1
                if rank < 5:
                    r5 += 1
                mrr_val += 1.0 / (rank + 1)
                break
        n_eval += 1
    denom = max(n_eval, 1)
    return {
        "recall_at_1": r1 / denom,
        "recall_at_5": r5 / denom,
        "mrr": mrr_val / denom,
    }


def _prepare_masked_batch(
    dataset: object, transform: object, level: int, seed: int,
    max_images: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Generate masked images for a given level and stack as tensor."""
    from src.masking import mask_pil_image
    n = min(len(dataset), max_images) if max_images else len(dataset)
    target_size = _CFG["img_size"]
    patch_size = _CFG.get("patch_size", 16)
    tensors: list[torch.Tensor] = []
    indices: list[int] = []
    for i in range(n):
        sample = dataset[i]
        masked = mask_pil_image(
            sample["image_pil"], level, sample["seg_mask"],
            patch_size=patch_size, target_size=target_size,
            seed=seed, idx=i,
        )
        tensors.append(transform(masked))
        indices.append(i)
    return torch.stack(tensors), indices


# ---------------------------------------------------------------------------
# Hook-based extraction (delegates to utils)
# ---------------------------------------------------------------------------


def _extract_all_cls(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 16,
) -> list[torch.Tensor]:
    """CLS token from each layer via hooks."""
    from src.utils import extract_block_cls
    return extract_block_cls(model, _get_blocks(model), imgs,
                             _num_layers(), batch_size)


def _extract_attn_acts(
    model: nn.Module, imgs: torch.Tensor, batch_size: int = 32,
) -> list[torch.Tensor]:
    """Attn activations from each layer via hooks."""
    from src.utils import extract_block_attn
    return extract_block_attn(model, _get_blocks(model), imgs,
                              _num_layers(), batch_size)


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
    num_runs: int = 3,
) -> None:
    """Run retrieval + categorization tasks across 8 masking levels.

    Tasks:
        1. Image retrieval: C-way concept-constrained K-choice (1 correct +
           1 distractor per other category), averaged over num_runs.
        2. Instance text retrieval: query=masked, gallery=instance text embeds (R@1)
        3. Category text retrieval: query=masked, gallery=category text embeds (acc)
        4. Image prototype: per-category mean of complete image embeds (acc)
        5. Instance text prototype: per-category mean of instance text embeds (acc)
    """
    out = _sweep_retrieval_dir()
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, transform = _load_model(device)
    dataset = get_dataset(dataset_name, root=data_root, image_type=image_type)
    n = min(len(dataset), max_images) if max_images else len(dataset)
    levels = get_mask_levels()
    C = dataset.num_scenes

    # Precompute galleries
    # 1. Complete image embeddings (keep on CPU, _encode_image handles GPU batching)
    print("  Computing complete image gallery...")
    complete_tensors: list[torch.Tensor] = []
    for i in range(n):
        sample = dataset[i]
        complete_tensors.append(transform(sample["image_pil"]))
    complete_imgs = torch.stack(complete_tensors)  # CPU
    image_gallery = _encode_image(model, complete_imgs)  # [N, D]

    # 2. Instance text embeddings
    instance_names = [
        dataset.samples[i].get("object_name", dataset.samples[i].get("scene_label", ""))
        for i in range(n)
    ]
    instance_gallery = _encode_text(model, tokenizer, instance_names, device)  # [N, D]

    # 3. Category text embeddings (= scene_labels for fragment_v2, = categories for COCO)
    cat_gallery = _encode_text(model, tokenizer, dataset.scene_labels, device)  # [C, D]

    # 4. Supercategory text embeddings (COCO only)
    has_supercat = hasattr(dataset, "supercategory_labels") and dataset.num_supercategories > 0
    if has_supercat:
        supercat_gallery = _encode_text(
            model, tokenizer, dataset.supercategory_labels, device,
        )  # [S, D]
        supercat_ids = [dataset.samples[i]["supercat_id"] for i in range(n)]

    # GT indices
    gt_instance = torch.arange(n, dtype=torch.long)  # each image matches itself
    cat_ids = [dataset[i]["scene_id"] for i in range(n)]

    # Category → image indices mapping (for concept-constrained K-choice)
    from collections import defaultdict
    cat_to_indices: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        cat_to_indices[cat_ids[i]].append(i)
    unique_cats = sorted(cat_to_indices.keys())

    # 5. Image prototype: per-category mean of complete image embeddings (for proto retrieval)
    D = image_gallery.shape[1]
    img_proto = torch.zeros(C, D)
    proto_count = torch.zeros(C)
    for i in range(n):
        img_proto[cat_ids[i]] += image_gallery[i]
        proto_count[cat_ids[i]] += 1
    proto_count = proto_count.clamp(min=1)
    img_proto = F.normalize(img_proto / proto_count.unsqueeze(1), dim=-1)

    # 5b. Per-CATEGORY image prototype k-sweep (COCO only). Each k yields
    # a base gallery of C category prototypes built from k random images
    # per category. The query's own cat is rebuilt per-query in
    # _eval_proto_k_sweep to enforce include/exclude target semantics.
    # Base galleries are query-independent and masking-level-independent,
    # so we build them once outside the level loop.
    proto_k_base: dict[int, torch.Tensor] = {}
    if dataset_name != "fragment_v2":
        all_ks = sorted(set(PROTO_K_EXCL) | set(PROTO_K_INCL))
        min_cat = min(len(v) for v in cat_to_indices.values())
        # excl needs k same-cat OTHERS → cat must have ≥ k+1
        # incl needs target + k-1 same-cat OTHERS → cat must have ≥ k
        excl_req = (max(PROTO_K_EXCL) + 1) if PROTO_K_EXCL else 0
        incl_req = max(PROTO_K_INCL) if PROTO_K_INCL else 0
        required = max(excl_req, incl_req)
        if min_cat < required:
            print(f"  [warn] smallest category has {min_cat} images "
                  f"(need >= {required} for the requested k-sweep); "
                  f"skipping k-sweep")
        else:
            rng_proto_base = np.random.RandomState(seed)
            for k in all_ks:
                proto_k_base[k] = _build_base_cat_proto_gallery(
                    image_gallery, cat_to_indices, k, rng=rng_proto_base,
                )
            print(f"  built {len(proto_k_base)} base category proto "
                  f"galleries (k={all_ks}); per-query rebuild for own cat "
                  f"(excl k={list(PROTO_K_EXCL)}, incl k={list(PROTO_K_INCL)})")

    # GT for proto/text retrieval (C-way category)
    gt_cat = torch.tensor(cat_ids, dtype=torch.long)

    # 6. Concept mean prototype: per-supercategory mean of basic-level concept
    #    text embeddings. E.g. Animal prototype = mean("cat","dog","horse",...)
    #    Falls back to per-category mean when no supercategory structure.
    if has_supercat:
        S = dataset.num_supercategories
        # Map each basic-level category to its supercategory
        cat_to_supercat = {}
        for i in range(n):
            cid = cat_ids[i]
            sid = dataset.samples[i]["supercat_id"]
            cat_to_supercat[cid] = sid
        # Mean of basic-level category text embeds within each supercategory
        concept_proto = torch.zeros(S, D)
        concept_count = torch.zeros(S)
        for cid in range(C):
            if cid not in cat_to_supercat:
                continue
            sid = cat_to_supercat[cid]
            concept_proto[sid] += cat_gallery[cid]
            concept_count[sid] += 1
        concept_count = concept_count.clamp(min=1)
        concept_proto = F.normalize(concept_proto / concept_count.unsqueeze(1), dim=-1)

        # 7. Image mean prototype per supercategory (sample 2 images per category)
        rng_proto = np.random.RandomState(seed)
        img_supercat_proto = torch.zeros(S, D)
        for cid in range(C):
            if cid not in cat_to_supercat:
                continue
            cat_indices = [i for i in range(n) if cat_ids[i] == cid]
            sampled = rng_proto.choice(cat_indices, size=min(2, len(cat_indices)), replace=False)
            sid = cat_to_supercat[cid]
            for si in sampled:
                img_supercat_proto[sid] += image_gallery[si]
        img_supercat_proto = F.normalize(img_supercat_proto, dim=-1)
    else:
        # No supercategory: concept mean prototype = per-category text prototype
        concept_proto = F.normalize(
            torch.zeros(C, D).scatter_add_(0,
                torch.tensor(cat_ids).unsqueeze(1).expand(-1, D),
                instance_gallery,
            ) / torch.zeros(C).scatter_add_(0, torch.tensor(cat_ids),
                torch.ones(n)).clamp(min=1).unsqueeze(1),
            dim=-1,
        )
        img_supercat_proto = None

    sc_str = f", {dataset.num_supercategories} supercategories" if has_supercat else ""
    C_active = len(unique_cats)
    print(f"  retrieval: {n} images, {C} categories{sc_str}, "
          f"{C_active}-way K-choice image retrieval ({num_runs} runs)")
    results: dict[str, dict] = {}

    for L in levels:
        vis = get_visibility_ratio(L)
        imgs, _ = _prepare_masked_batch(dataset, transform, L, seed, max_images)
        query = _encode_image(model, imgs)  # [N, D]

        # --- Full N-way retrieval ---
        # fragment_v2: in-instance (GT = self)
        # coco_subset:  cross-instance (GT = same category, excl. self)
        cross_instance = dataset_name != "fragment_v2"

        # Image retrieval: query=masked, gallery=all N complete images
        img_sims = query @ image_gallery.T  # [N, N]
        img_metrics = compute_retrieval_metrics(
            query, image_gallery, gt_instance,
        ) if not cross_instance else _cross_instance_metrics(
            img_sims, cat_to_indices, cat_ids, n,
        )

        # Text retrieval: query=masked, gallery=all N instance text embeds
        txt_sims = query @ instance_gallery.T  # [N, N]
        txt_metrics = compute_retrieval_metrics(
            query, instance_gallery, gt_instance,
        ) if not cross_instance else _cross_instance_metrics(
            txt_sims, cat_to_indices, cat_ids, n,
        )

        level_results = {
            "image_r1": img_metrics["recall_at_1"],
            "image_r5": img_metrics["recall_at_5"],
            "image_mrr": img_metrics["mrr"],
            "text_r1": txt_metrics["recall_at_1"],
            "text_r5": txt_metrics["recall_at_5"],
            "text_mrr": txt_metrics["mrr"],
        }

        # Image-prototype k-sweep: C-way categorization (20 cats for COCO).
        # The query's own cat prototype is rebuilt per-query to enforce
        # include/exclude target semantics; other cats use the base gallery.
        # R@1 = top-1 cat acc, R@5 = top-5 cat acc (out of C), MRR = mean
        # reciprocal rank of the true cat. Each (k, mode) combo gets its
        # own deterministic RNG seeded from `seed` so re-runs match.
        for k in PROTO_K_EXCL:
            metrics = _eval_proto_k_sweep(
                query, image_gallery, proto_k_base[k],
                cat_to_indices, cat_ids, k,
                include_target=False,
                rng=np.random.RandomState(seed + k),
            )
            level_results[f"img_proto_excl_k{k}_r1"] = metrics["recall_at_1"]
            level_results[f"img_proto_excl_k{k}_r5"] = metrics["recall_at_5"]
            level_results[f"img_proto_excl_k{k}_mrr"] = metrics["mrr"]
        for k in PROTO_K_INCL:
            metrics = _eval_proto_k_sweep(
                query, image_gallery, proto_k_base[k],
                cat_to_indices, cat_ids, k,
                include_target=True,
                rng=np.random.RandomState(seed + 1000 + k),
            )
            level_results[f"img_proto_incl_k{k}_r1"] = metrics["recall_at_1"]
            level_results[f"img_proto_incl_k{k}_r5"] = metrics["recall_at_5"]
            level_results[f"img_proto_incl_k{k}_mrr"] = metrics["mrr"]

        # Sanity check (full visibility only): at L=8 the query equals
        # image_i, and incl_k1 builds prototype_{cat(i)} = image_i alone,
        # so sim(query_i, proto_{cat(i)}) = 1.0 — strictly the highest of
        # the C cat prototypes (other prototypes are different images,
        # cosine sim < 1.0). Top-1 acc must be ~1.0.
        if "img_proto_incl_k1_r1" in level_results and vis >= 0.99:
            r1_incl_k1 = level_results["img_proto_incl_k1_r1"]
            assert r1_incl_k1 > 1.0 - 5e-3, (
                f"incl_k1 C-way top-1 acc should be ~1.0 at full visibility "
                f"(query's own cat prototype = self); got {r1_incl_k1:.6f} "
                f"at L={L}"
            )

        results[str(L)] = level_results
        sweep_summary = ""
        if proto_k_base:
            sweep_summary = (
                f"  excl_k1_r1={level_results['img_proto_excl_k1_r1']:.4f}"
                f"  incl_k1_r1={level_results['img_proto_incl_k1_r1']:.4f}"
                f"  excl_k20_r1={level_results['img_proto_excl_k20_r1']:.4f}"
                f"  incl_k20_r1={level_results['img_proto_incl_k20_r1']:.4f}"
            )
        print(f"    L={L} vis={vis:.3f}  "
              f"img_r1={img_metrics['recall_at_1']:.4f}  "
              f"img_r5={img_metrics['recall_at_5']:.4f}  "
              f"img_mrr={img_metrics['mrr']:.4f}  "
              f"txt_r1={txt_metrics['recall_at_1']:.4f}  "
              f"txt_r5={txt_metrics['recall_at_5']:.4f}  "
              f"txt_mrr={txt_metrics['mrr']:.4f}{sweep_summary}")

    _save_json(results, out / f"results_{image_type}.json")

    # Plot via shared plot module
    from .plot import plot_proto_k_sweep, plot_task_comparison
    save_path = out / f"task_comparison_{image_type}.png"
    plot_task_comparison(results, save_path, f"{_model_label()} — {image_type}")

    # k-sweep plot (skipped automatically when no proto keys are present).
    if proto_k_base:
        sweep_base = out / f"proto_k_sweep_{image_type}.png"
        plot_proto_k_sweep(
            results, sweep_base,
            f"{_model_label()} — {image_type} — Image Prototype k-sweep",
        )
        plot_proto_k_sweep(
            results,
            sweep_base.with_name(f"proto_k_sweep_{image_type}+text.png"),
            f"{_model_label()} — {image_type} — Image Prototype k-sweep "
            f"(+ text retrieval)",
            with_text=True,
        )


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

        imgs, cat_ids = _prepare_masked_batch(
            dataset, transform, L, seed, max_images,
        )
        all_cls = _extract_all_cls(model, imgs)  # 24 x [N, 1024]

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
    from src.utils import make_attn_hook
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
        imgs, _ = _prepare_masked_batch(
            dataset, transform, L, seed, max_images,
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
                    make_attn_hook(src, tmode),
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

    imgs, cat_ids = _prepare_masked_batch(
        dataset, transform, mask_level, seed, max_images,
    )
    all_cls = _extract_all_cls(model, imgs)  # 24 x [N, 1024]

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
                        choices=["fragment_v2", "ade20k", "coco_subset",
                                 "coco_subset_56"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-type", default="original",
                        choices=["original", "gray", "lined"])
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--results-dir", default=None,
                        help="Override results directory")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("retrieve")

    sub.add_parser("probe")

    sub.add_parser("patch")

    p = sub.add_parser("logit-lens")
    p.add_argument("--mask-level", type=int, default=8)

    sub.add_parser("all")

    args = parser.parse_args()

    # Set active model config and dataset tag
    global _CFG, _DATASET_TAG, _RESULTS_BASE
    _CFG = MODEL_CONFIGS[args.model]
    if args.results_dir:
        _RESULTS_BASE = Path(args.results_dir)
    _DATASET_TAG = "frag" if args.dataset == "fragment_v2" else args.dataset
    print(f"  Model: {_model_label()} ({_num_layers()} layers, "
          f"dim={_internal_dim()}, proj={_proj_dim()})")
    print(f"  Results: {_results_dir()}/")

    common = dict(
        dataset_name=args.dataset, data_root=args.data_root,
        image_type=args.image_type, seed=args.seed, device=args.device,
        max_images=args.max_images,
    )

    cmd = args.command

    # Save experiment settings
    save_experiment_settings(args, _results_dir())

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
