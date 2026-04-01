#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : saliency.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 03-31-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Per-patch saliency maps for DINOv2 and CLIP.

DINOv2: register token attention (last layer).
CLIP: GradCAM conditioned on text prompt (last layer).

Both return [N, gh, gw] normalized to [0, 1].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DINOv2: register token attention
# ---------------------------------------------------------------------------

@torch.no_grad()
def dinov2_saliency(
    model: nn.Module,
    imgs: torch.Tensor,
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute per-patch saliency from DINOv2 register token attention.

    Uses the last transformer block. Computes raw attention weights from QKV,
    then averages attention from register tokens to patch tokens across all heads.

    Args:
        model: DINOv2 model (dinov2_vitl14_reg4_dinotxt_...).
        imgs: [N, 3, 518, 518] preprocessed image batch (CPU or GPU).
        batch_size: Forward pass batch size.

    Returns:
        [N, gh, gw] saliency tensor, normalized to [0, 1] per image.
    """
    device = next(model.parameters()).device
    blocks = model.visual_model.backbone.model.blocks
    last_block = blocks[-1]
    num_heads = last_block.attn.num_heads

    # Hook to capture QKV before attention
    qkv_store: list[torch.Tensor] = []

    def _qkv_hook(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        qkv_store.append(out.detach().cpu())

    handle = last_block.attn.qkv.register_forward_hook(_qkv_hook)

    all_saliency: list[torch.Tensor] = []
    try:
        for start in range(0, imgs.shape[0], batch_size):
            batch = imgs[start : start + batch_size].to(device)
            qkv_store.clear()

            # Forward through visual backbone only
            model.visual_model.backbone.model(batch)

            qkv = qkv_store[0]  # [B, T, 3 * C]
            B, T, _ = qkv.shape
            C = _ // 3
            head_dim = C // num_heads

            # Reshape to [B, T, 3, num_heads, head_dim]
            qkv = qkv.reshape(B, T, 3, num_heads, head_dim)
            q, k, _ = qkv.unbind(dim=2)  # each [B, T, num_heads, head_dim]

            # Transpose to [B, num_heads, T, head_dim]
            q = q.permute(0, 2, 1, 3).float()
            k = k.permute(0, 2, 1, 3).float()

            # Attention weights: [B, num_heads, T, T]
            scale = math.sqrt(head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn_weights = attn_weights.softmax(dim=-1)

            # Token layout: [CLS, patches..., reg0, reg1, reg2, reg3]
            # Patches: positions 1 to T-4
            # Registers: positions T-4 to T-1
            num_register = 4
            num_patches = T - 1 - num_register  # exclude CLS and registers

            # Register → patch attention: [B, num_heads, 4, num_patches]
            reg_to_patch = attn_weights[:, :, -num_register:, 1 : 1 + num_patches]

            # Average across registers and heads → [B, num_patches]
            saliency = reg_to_patch.mean(dim=(1, 2))

            # Reshape to grid
            gh = gw = int(math.sqrt(num_patches))
            assert gh * gw == num_patches, f"Non-square patch grid: {num_patches}"
            saliency = saliency.reshape(B, gh, gw)

            # Normalize per image to [0, 1]
            s_min = saliency.flatten(1).min(dim=1).values[:, None, None]
            s_max = saliency.flatten(1).max(dim=1).values[:, None, None]
            saliency = (saliency - s_min) / (s_max - s_min + 1e-8)

            all_saliency.append(saliency)
    finally:
        handle.remove()

    return torch.cat(all_saliency, dim=0)  # [N, gh, gw]


# ---------------------------------------------------------------------------
# CLIP: GradCAM
# ---------------------------------------------------------------------------

def clip_gradcam(
    model: nn.Module,
    imgs: torch.Tensor,
    text_embeds: torch.Tensor,
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute per-patch saliency via GradCAM on CLIP.

    Backprops through cosine similarity between image and text embedding.
    Uses gradient w.r.t. last resblock's patch activations.

    Args:
        model: OpenCLIP model (ViT-L-14).
        imgs: [N, 3, 224, 224] preprocessed image batch (CPU).
        text_embeds: [N, D] L2-normalized text embeddings (one per image).
        batch_size: Forward pass batch size.

    Returns:
        [N, gh, gw] saliency tensor, normalized to [0, 1] per image.
    """
    device = next(model.parameters()).device
    last_block = model.visual.transformer.resblocks[-1]

    all_saliency: list[torch.Tensor] = []

    for start in range(0, imgs.shape[0], batch_size):
        batch = imgs[start : start + batch_size].to(device).requires_grad_(False)
        text_batch = text_embeds[start : start + batch_size].to(device)
        B = batch.shape[0]

        # Storage for hooked activations
        activation_store: list[torch.Tensor] = []

        def _fwd_hook(
            mod: nn.Module, inp: tuple, out: torch.Tensor,
        ) -> None:
            activation_store.append(out)

        handle = last_block.register_forward_hook(_fwd_hook)

        try:
            # Forward with gradients
            activation_store.clear()
            image_features = model.encode_image(batch)  # [B, proj_dim]
            image_features = F.normalize(image_features.float(), dim=-1)

            # Cosine similarity per image
            sim = (image_features * text_batch.float()).sum(dim=-1)  # [B]
            loss = sim.sum()

            # Backprop
            model.zero_grad()
            loss.backward()

            # Get activation and its gradient
            act = activation_store[0]  # [B, T, D] — full token sequence
            grad = act.grad  # [B, T, D]

            if grad is None:
                # Activation wasn't a leaf — re-register with retain_grad
                handle.remove()

                activation_store.clear()

                def _fwd_hook_retain(
                    mod: nn.Module, inp: tuple, out: torch.Tensor,
                ) -> None:
                    out.retain_grad()
                    activation_store.append(out)

                handle = last_block.register_forward_hook(_fwd_hook_retain)
                model.zero_grad()
                image_features = model.encode_image(batch)
                image_features = F.normalize(image_features.float(), dim=-1)
                sim = (image_features * text_batch.float()).sum(dim=-1)
                sim.sum().backward()
                act = activation_store[0]
                grad = act.grad

            # GradCAM: patch tokens only (skip CLS at position 0)
            # act: [B, T, D], grad: [B, T, D]
            patch_act = act[:, 1:, :].detach().float()   # [B, num_patches, D]
            patch_grad = grad[:, 1:, :].detach().float()  # [B, num_patches, D]

            # Weighted activation: element-wise product, mean over channels
            cam = (patch_grad * patch_act).mean(dim=-1)  # [B, num_patches]
            cam = F.relu(cam)  # GradCAM uses ReLU

            # Reshape to grid
            num_patches = cam.shape[1]
            gh = gw = int(math.sqrt(num_patches))
            assert gh * gw == num_patches, f"Non-square patch grid: {num_patches}"
            cam = cam.reshape(B, gh, gw)

            # Normalize per image to [0, 1]
            c_min = cam.flatten(1).min(dim=1).values[:, None, None]
            c_max = cam.flatten(1).max(dim=1).values[:, None, None]
            cam = (cam - c_min) / (c_max - c_min + 1e-8)

            all_saliency.append(cam.cpu())
        finally:
            handle.remove()

    return torch.cat(all_saliency, dim=0)  # [N, gh, gw]


# ---------------------------------------------------------------------------
# Utility: resample saliency between grids
# ---------------------------------------------------------------------------

def resample_saliency(
    saliency: torch.Tensor,
    target_gh: int,
    target_gw: int,
) -> torch.Tensor:
    """Resample saliency map to a different patch grid size.

    Args:
        saliency: [N, gh, gw] source saliency.
        target_gh: Target grid height.
        target_gw: Target grid width.

    Returns:
        [N, target_gh, target_gw] resampled saliency.
    """
    if saliency.shape[1] == target_gh and saliency.shape[2] == target_gw:
        return saliency
    # Use bilinear interpolation
    s = saliency.unsqueeze(1).float()  # [N, 1, gh, gw]
    out = F.interpolate(s, size=(target_gh, target_gw), mode="bilinear", align_corners=False)
    return out.squeeze(1)  # [N, target_gh, target_gw]
