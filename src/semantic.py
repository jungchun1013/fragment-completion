"""Semantic completion: category classification accuracy under masking.

Prototype = mean embedding of all complete images in the same category.
At each masking level, classify the masked image by nearest category prototype.
"""

import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from wrappers.encoder import BaseEncoder

from .masking import get_mask_levels, get_visibility_ratio, mask_pil_image


def _embed_pil(encoder: BaseEncoder, pil, transform) -> torch.Tensor:
    """Extract [D] feature vector from a PIL image."""
    img_t = transform(pil).unsqueeze(0).to(encoder.device)
    feat = encoder.extract_features(img_t)  # [1, D]
    return feat[0].cpu()


@torch.no_grad()
def evaluate_semantic(
    encoder: BaseEncoder,
    dataset,
    seed: int = 42,
    max_images: int | None = None,
    num_choices: int = 5,
    num_runs: int = 3,
) -> dict[str, dict[int, float]]:
    """Evaluate semantic completion via category prototype classification.

    1. Embed all complete (L=8) images.
    2. Build per-category prototypes = mean of member embeddings.
    3. At each masking level, K-choice classification among prototypes.
    4. (CLIP only) Also do K-choice zero-shot text matching.

    Args:
        num_choices: Number of candidate categories per query
                     (1 correct + num_choices-1 distractors).
        num_runs: Number of classification runs with different distractor samples
                  to reduce variance. Results are averaged across runs.

    Returns:
        {
            "prototype_acc": {level: accuracy},
            "zeroshot_acc":  {level: accuracy},  # only if CLIP
        }
    """
    from wrappers.processor import to_transform
    transform = to_transform(encoder.processor)
    levels = get_mask_levels()
    n = min(len(dataset), max_images) if max_images else len(dataset)
    num_categories = dataset.num_scenes

    # Step 1: embed all complete images
    print(f"    semantic: embedding {n} complete images for {num_categories} categories...")
    complete_embeds = []
    cat_ids = []
    for i in range(n):
        sample = dataset[i]
        complete_embeds.append(_embed_pil(encoder, sample["image_pil"], transform))
        cat_ids.append(sample["scene_id"])

    complete_mat = torch.stack(complete_embeds)  # [N, D]
    complete_mat = F.normalize(complete_mat, dim=-1)

    # Step 2: build category prototypes (mean of members)
    D = complete_mat.shape[1]
    proto_sum = torch.zeros(num_categories, D)
    proto_count = torch.zeros(num_categories)
    for i in range(n):
        cid = cat_ids[i]
        proto_sum[cid] += complete_mat[i]
        proto_count[cid] += 1

    # Avoid division by zero for categories with no samples in subset
    proto_count = proto_count.clamp(min=1)
    prototypes = F.normalize(proto_sum / proto_count.unsqueeze(1), dim=-1)  # [C, D]

    cat_labels = dataset.scene_labels
    active = [i for i in range(num_categories) if proto_count[i] > 0]
    print(f"    semantic: {len(active)} active categories with prototypes")

    K = min(num_choices, len(active))

    # Step 3: K-choice prototype classification, averaged over num_runs
    proto_acc = {}
    for L in levels:
        # Pre-embed masked images once (deterministic)
        masked_embeds = []
        for i in range(n):
            sample = dataset[i]
            masked = mask_pil_image(
                sample["image_pil"], L, sample["seg_mask"], seed=seed, idx=i
            )
            embed = _embed_pil(encoder, masked, transform)
            masked_embeds.append(F.normalize(embed.unsqueeze(0), dim=-1))

        run_accs = []
        for run in range(num_runs):
            rng = np.random.RandomState(seed + run)
            correct_per_image = []
            for i in range(n):
                true_cat = cat_ids[i]
                distractor_cats = [c for c in active if c != true_cat]
                chosen = rng.choice(distractor_cats, size=K - 1, replace=False).tolist()
                candidates = [true_cat] + chosen
                cand_protos = prototypes[candidates]  # [K, D]
                sims = (masked_embeds[i] @ cand_protos.T).squeeze(0)  # [K]
                correct_per_image.append(1.0 if sims.argmax().item() == 0 else 0.0)
            run_accs.append(np.mean(correct_per_image))

        run_accs = np.array(run_accs)
        proto_acc[L] = {"mean": float(run_accs.mean()), "std": float(run_accs.std())}
        vis = get_visibility_ratio(L)
        print(f"    semantic [L={L}, vis={vis:.3f}] proto_acc={proto_acc[L]['mean']:.4f}"
              f"±{proto_acc[L]['std']:.4f} (K={K}, {num_runs} runs)")

    result = {"prototype_acc": proto_acc}

    # Step 4: CLIP K-choice zero-shot text matching, averaged over num_runs
    if encoder.name == "CLIP":
        print("    semantic: CLIP zero-shot text matching...")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        prompts = [f"a photo of {label}" for label in cat_labels]
        tokens = tokenizer(prompts).to(encoder.device)
        text_feats = encoder.model.encode_text(tokens)
        text_feats = F.normalize(text_feats.float().cpu(), dim=-1)  # [C, D]

        zs_acc = {}
        for L in levels:
            masked_embeds = []
            for i in range(n):
                sample = dataset[i]
                masked = mask_pil_image(
                    sample["image_pil"], L, sample["seg_mask"], seed=seed, idx=i
                )
                embed = _embed_pil(encoder, masked, transform)
                masked_embeds.append(F.normalize(embed.unsqueeze(0), dim=-1))

            run_accs = []
            for run in range(num_runs):
                rng = np.random.RandomState(seed + run)
                correct_per_image = []
                for i in range(n):
                    true_cat = cat_ids[i]
                    distractor_cats = [c for c in active if c != true_cat]
                    chosen = rng.choice(distractor_cats, size=K - 1, replace=False).tolist()
                    candidates = [true_cat] + chosen
                    cand_text = text_feats[candidates]  # [K, D]
                    sims = (masked_embeds[i] @ cand_text.T).squeeze(0)
                    correct_per_image.append(1.0 if sims.argmax().item() == 0 else 0.0)
                run_accs.append(np.mean(correct_per_image))

            run_accs = np.array(run_accs)
            zs_acc[L] = {"mean": float(run_accs.mean()), "std": float(run_accs.std())}
            vis = get_visibility_ratio(L)
            print(f"    semantic [L={L}, vis={vis:.3f}] zeroshot_acc={zs_acc[L]['mean']:.4f}"
                  f"±{zs_acc[L]['std']:.4f} (K={K}, {num_runs} runs)")
        result["zeroshot_acc"] = zs_acc

    return result
