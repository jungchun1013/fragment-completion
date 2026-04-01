"""Progressive patch-level masking for fragment completion.

Mirrors the fragment_v2 approach: only patches that contain foreground objects
are candidates for masking/revealing, and visibility follows an exponential
schedule: P = 0.7^(8-L) for levels L=1..8.
"""

import random

import numpy as np
from PIL import Image


# 8 levels: L=1 (most masked) to L=8 (complete)
NUM_LEVELS = 8


def get_mask_levels() -> list[int]:
    """Return level indices 1..8 (1 = most masked, 8 = complete)."""
    return list(range(1, NUM_LEVELS + 1))


def get_visibility_ratio(level: int) -> float:
    """Fraction of foreground patches visible at this level.

    P = 0.7^(8 - L):  L=1 → 0.082, L=4 → 0.240, L=8 → 1.0.
    """
    return 0.7 ** (NUM_LEVELS - level)


def _find_foreground_patches(
    img_np: np.ndarray,
    seg_mask: np.ndarray,
    patch_size: int,
    target_size: int,
) -> list[tuple[int, int]]:
    """Find patches that contain at least one foreground pixel.

    Args:
        img_np: Resized image array [target_size, target_size, 3].
        seg_mask: Original segmentation mask (any resolution).
        patch_size: Patch size in pixels.
        target_size: Image resolution.

    Returns:
        List of (row, col) patch coordinates with foreground content.
    """
    # Resize seg mask to target_size
    seg_resized = np.array(
        Image.fromarray(seg_mask.astype(np.uint16)).resize(
            (target_size, target_size), Image.NEAREST
        )
    )
    grid_size = target_size // patch_size
    fg_patches = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0, y1 = row * patch_size, (row + 1) * patch_size
            x0, x1 = col * patch_size, (col + 1) * patch_size
            patch_seg = seg_resized[y0:y1, x0:x1]
            if np.any(patch_seg > 0):
                fg_patches.append((row, col))
    return fg_patches


def mask_pil_image(
    pil: Image.Image,
    level: int,
    seg_mask: np.ndarray,
    patch_size: int = 16,
    target_size: int = 224,
    seed: int = 42,
    idx: int = 0,
) -> Image.Image:
    """Apply progressive patch-level masking following fragment_v2 logic.

    Only foreground patches are candidates. At level L, P = 0.7^(8-L) fraction
    of foreground patches are revealed; the rest are filled with gray (128,128,128).
    Background patches are always shown.

    Args:
        pil: Input PIL image.
        level: Masking level 1..8 (1 = most masked, 8 = full).
        seg_mask: Segmentation mask (original resolution).
        patch_size: Size of each square patch.
        target_size: Resize image to this before masking.
        seed: Base random seed.
        idx: Per-image index (for per-image determinism).

    Returns:
        Masked PIL image at target_size x target_size.
    """
    pil = pil.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(pil)

    fg_patches = _find_foreground_patches(img_np, seg_mask, patch_size, target_size)

    if not fg_patches or level >= NUM_LEVELS:
        return Image.fromarray(img_np)

    # Shuffle foreground patches deterministically
    rng = random.Random(seed + idx)
    rng.shuffle(fg_patches)

    # Number of foreground patches to reveal
    vis_ratio = get_visibility_ratio(level)
    num_visible = int(vis_ratio * len(fg_patches))
    visible_set = set(fg_patches[:num_visible])

    # Mask non-visible foreground patches
    grid_size = target_size // patch_size
    for row, col in fg_patches:
        if (row, col) not in visible_set:
            y0, y1 = row * patch_size, (row + 1) * patch_size
            x0, x1 = col * patch_size, (col + 1) * patch_size
            img_np[y0:y1, x0:x1] = 255

    return Image.fromarray(img_np)


def mask_pil_image_saliency(
    pil: Image.Image,
    level: int,
    seg_mask: np.ndarray,
    saliency: np.ndarray,
    salient_first: bool = True,
    patch_size: int = 16,
    target_size: int = 224,
) -> Image.Image:
    """Apply saliency-based progressive patch masking.

    Instead of random patch order, patches are sorted by saliency score.
    salient_first=True removes the most salient patches first — at low
    visibility, only non-salient patches remain.

    Args:
        pil: Input PIL image.
        level: Masking level 1..8 (1 = most masked, 8 = full).
        seg_mask: Segmentation mask (original resolution).
        saliency: [gh, gw] per-patch saliency scores (higher = more important).
        salient_first: If True, most salient patches are hidden first.
        patch_size: Size of each square patch.
        target_size: Resize image to this before masking.

    Returns:
        Masked PIL image at target_size × target_size.
    """
    pil = pil.resize((target_size, target_size), Image.BILINEAR)
    img_np = np.array(pil)

    fg_patches = _find_foreground_patches(img_np, seg_mask, patch_size, target_size)

    if not fg_patches or level >= NUM_LEVELS:
        return Image.fromarray(img_np)

    # Sort foreground patches by saliency score
    # salient_first: highest saliency last in the list (revealed last = hidden first)
    fg_with_score = [
        (row, col, float(saliency[row, col])) for row, col in fg_patches
    ]
    # Sort ascending by saliency: least salient first (will be revealed first)
    if salient_first:
        fg_with_score.sort(key=lambda x: x[2])
    else:
        fg_with_score.sort(key=lambda x: -x[2])

    sorted_patches = [(row, col) for row, col, _ in fg_with_score]

    # Number of foreground patches to reveal
    vis_ratio = get_visibility_ratio(level)
    num_visible = int(vis_ratio * len(sorted_patches))
    visible_set = set(sorted_patches[:num_visible])

    # Mask non-visible foreground patches
    for row, col in sorted_patches:
        if (row, col) not in visible_set:
            y0, y1 = row * patch_size, (row + 1) * patch_size
            x0, x1 = col * patch_size, (col + 1) * patch_size
            img_np[y0:y1, x0:x1] = 255

    return Image.fromarray(img_np)
