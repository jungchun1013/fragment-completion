"""Mnemonic completion: embedding similarity & retrieval accuracy."""

import numpy as np
import torch
import torch.nn.functional as F

from models.encoder import BaseEncoder

from .masking import get_mask_levels, get_visibility_ratio, mask_pil_image
from .utils import embed_pil


@torch.no_grad()
def evaluate_mnemonic(
    encoder: BaseEncoder,
    dataset,
    seed: int = 42,
    max_images: int | None = None,
    num_choices: int = 5,
    num_runs: int = 3,
) -> dict[str, dict[int, float]]:
    """Evaluate mnemonic completion.

    Args:
        num_choices: Number of candidates per retrieval query
                     (1 correct + num_choices-1 distractors).
        num_runs: Number of retrieval runs with different distractor samples
                  to reduce variance. Results are averaged across runs.

    Returns:
        {
            "similarity": {level: mean_cos_sim},
            "retrieval":  {level: top1_accuracy},
        }
    """
    from models.processor import to_transform
    transform = to_transform(encoder.processor)
    levels = get_mask_levels()
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Step 1: embed all complete images (level 8 = full)
    print(f"    mnemonic: embedding {n} complete images...")
    complete_embeds = []
    for i in range(n):
        pil = dataset[i]["image_pil"]
        complete_embeds.append(embed_pil(encoder, pil, transform))
    complete_mat = torch.stack(complete_embeds)  # [N, D]
    complete_mat = F.normalize(complete_mat, dim=-1)

    # Step 2: for each level, embed masked images and compute metrics
    similarity = {}
    retrieval = {}

    for L in levels:
        run_sims: list[torch.Tensor] = []
        run_accs: list[float] = []
        K = min(num_choices, n)

        for run in range(num_runs):
            seed_run = seed + run
            rng = np.random.RandomState(seed_run)
            masked_embeds = []
            for i in range(n):
                sample = dataset[i]
                masked = mask_pil_image(sample["image_pil"], L, sample["seg_mask"],
                                        seed=seed_run, idx=i)
                masked_embeds.append(embed_pil(encoder, masked, transform))

            masked_mat = torch.stack(masked_embeds)  # [N, D]
            masked_mat = F.normalize(masked_mat, dim=-1)

            # Cosine similarity (paired)
            cos_sims = (complete_mat * masked_mat).sum(dim=-1)  # [N]
            run_sims.append(cos_sims)

            # Retrieval: K-choice
            correct_per_image = []
            for i in range(n):
                distractors = list(range(n))
                distractors.remove(i)
                chosen = rng.choice(distractors, size=K - 1, replace=False).tolist()
                candidates = [i] + chosen
                cand_embeds = complete_mat[candidates]  # [K, D]
                sims = (masked_mat[i].unsqueeze(0) @ cand_embeds.T).squeeze(0)  # [K]
                correct_per_image.append(1.0 if sims.argmax().item() == 0 else 0.0)
            run_accs.append(np.mean(correct_per_image))

        all_sims = torch.stack(run_sims)  # [num_runs, N]
        similarity[L] = {"mean": float(all_sims.mean()), "std": float(all_sims.std())}
        run_accs_arr = np.array(run_accs)
        retrieval[L] = {"mean": float(run_accs_arr.mean()), "std": float(run_accs_arr.std())}

        vis = get_visibility_ratio(L)
        print(f"    mnemonic [L={L}, vis={vis:.3f}] sim={similarity[L]['mean']:.4f} "
              f"ret@1={retrieval[L]['mean']:.4f}±{retrieval[L]['std']:.4f} "
              f"(K={K}, {num_runs} runs)")

    return {"similarity": similarity, "retrieval": retrieval}
