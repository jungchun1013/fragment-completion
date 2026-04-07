"""Mnemonic completion: embedding similarity & retrieval accuracy.

Two retrieval modes:
  - K-choice: forced choice among K candidates (psychophysics-style)
  - Full-rank: rank among all N images (IR-style, R@1 / R@5 / MRR)
"""

import numpy as np
import torch
import torch.nn.functional as F

from models.encoder import BaseEncoder
from models.processor import get_normalize_transform

from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image, prepare_image
from src.utils import compute_retrieval_metrics, embed_pil, get_encoder_geometry


@torch.no_grad()
def evaluate_mnemonic(
    encoder: BaseEncoder,
    dataset,
    seed: int = 42,
    max_images: int | None = None,
    num_runs: int = 3,
) -> dict[str, dict]:
    """Evaluate mnemonic completion: similarity and full-rank retrieval.

    For each masking level, embed masked images with *num_runs* different
    random mask seeds and compute full-rank retrieval metrics (R@1, R@5, MRR)
    and paired cosine similarity, averaged across runs.

    Args:
        num_runs: Number of runs with different random mask seeds.

    Returns:
        {
            "similarity":    {level: {mean, std}},
            "retrieval_r1":  {level: {mean, std}},
            "retrieval_r5":  {level: {mean, std}},
            "retrieval_mrr": {level: {mean, std}},
        }
    """
    img_size, patch_size = get_encoder_geometry(encoder)
    norm_transform = get_normalize_transform(encoder.processor)
    levels = get_mask_levels()
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Step 1: embed all complete images (level 8 = full)
    # Use the same spatial prep (center-crop + resize) as masking for consistency
    print(f"    mnemonic: embedding {n} complete images...")
    complete_embeds = []
    for i in range(n):
        pil = prepare_image(dataset[i]["image_pil"], img_size)
        complete_embeds.append(embed_pil(encoder, pil, norm_transform))
    complete_mat = torch.stack(complete_embeds)  # [N, D]
    complete_mat = F.normalize(complete_mat, dim=-1)

    gt_indices = torch.arange(n, dtype=torch.long)

    # Step 2: for each level, embed masked images and compute metrics
    similarity: dict = {}
    retrieval_r1: dict = {}
    retrieval_r5: dict = {}
    retrieval_mrr: dict = {}

    for L in levels:
        run_sims: list[torch.Tensor] = []
        run_r1: list[float] = []
        run_r5: list[float] = []
        run_mrr: list[float] = []

        for run in range(num_runs):
            seed_run = seed + run
            masked_embeds = []
            for i in range(n):
                sample = dataset[i]
                masked = mask_pil_image(sample["image_pil"], L, sample["seg_mask"],
                                        seed=seed_run, idx=i,
                                        patch_size=patch_size, target_size=img_size)
                masked_embeds.append(embed_pil(encoder, masked, norm_transform))

            masked_mat = torch.stack(masked_embeds)  # [N, D]
            masked_mat = F.normalize(masked_mat, dim=-1)

            # Cosine similarity (paired)
            cos_sims = (complete_mat * masked_mat).sum(dim=-1)  # [N]
            run_sims.append(cos_sims)

            # Full-rank retrieval
            fr = compute_retrieval_metrics(masked_mat, complete_mat, gt_indices)
            run_r1.append(fr["recall_at_1"])
            run_r5.append(fr["recall_at_5"])
            run_mrr.append(fr["mrr"])

        all_sims = torch.stack(run_sims)  # [num_runs, N]
        similarity[L] = {"mean": float(all_sims.mean()), "std": float(all_sims.std())}
        retrieval_r1[L] = {"mean": float(np.mean(run_r1)), "std": float(np.std(run_r1))}
        retrieval_r5[L] = {"mean": float(np.mean(run_r5)), "std": float(np.std(run_r5))}
        retrieval_mrr[L] = {"mean": float(np.mean(run_mrr)), "std": float(np.std(run_mrr))}

        vis = get_visibility_ratio(L)
        print(f"    mnemonic [L={L}, vis={vis:.3f}] sim={similarity[L]['mean']:.4f} "
              f"R@1={retrieval_r1[L]['mean']:.4f}±{retrieval_r1[L]['std']:.4f} "
              f"R@5={retrieval_r5[L]['mean']:.4f} "
              f"MRR={retrieval_mrr[L]['mean']:.4f}")

    return {
        "similarity": similarity,
        "retrieval_r1": retrieval_r1,
        "retrieval_r5": retrieval_r5,
        "retrieval_mrr": retrieval_mrr,
    }
