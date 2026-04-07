"""Semantic completion: category classification accuracy under masking.

Prototype = mean embedding of all complete images in the same category.
At each masking level, classify the masked image by nearest category prototype.
"""

import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from models.encoder import BaseEncoder
from models.processor import get_normalize_transform

from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image, prepare_image
from src.utils import embed_pil, get_encoder_geometry


@torch.no_grad()
def evaluate_semantic(
    encoder: BaseEncoder,
    dataset,
    seed: int = 42,
    max_images: int | None = None,
    num_runs: int = 3,
) -> dict[str, dict]:
    """Evaluate semantic completion via full-rank prototype classification.

    1. Embed all complete (L=8) images.
    2. Build per-category prototypes = mean of member embeddings.
    3. At each masking level, classify against ALL category prototypes.
    4. (CLIP only) Also classify against ALL text embeddings (zero-shot).

    Each run uses a different mask seed; accuracy is averaged across runs.

    Returns:
        {
            "prototype_acc": {level: {mean, std}},
            "zeroshot_acc":  {level: {mean, std}},  # only if CLIP
        }
    """
    img_size, patch_size = get_encoder_geometry(encoder)
    norm_transform = get_normalize_transform(encoder.processor)
    levels = get_mask_levels()
    n = min(len(dataset), max_images) if max_images else len(dataset)
    num_categories = dataset.num_scenes

    # Step 1: embed all complete images (same spatial prep as masking)
    print(f"    semantic: embedding {n} complete images for {num_categories} categories...")
    complete_embeds = []
    cat_ids: list[int] = []
    for i in range(n):
        sample = dataset[i]
        pil = prepare_image(sample["image_pil"], img_size)
        complete_embeds.append(embed_pil(encoder, pil, norm_transform))
        cat_ids.append(sample["scene_id"])

    complete_mat = torch.stack(complete_embeds)  # [N, D]
    complete_mat = F.normalize(complete_mat, dim=-1)
    cat_ids_t = torch.tensor(cat_ids, dtype=torch.long)

    # Step 2: build category prototypes (mean of members)
    D = complete_mat.shape[1]
    proto_sum = torch.zeros(num_categories, D)
    proto_count = torch.zeros(num_categories)
    for i in range(n):
        cid = cat_ids[i]
        proto_sum[cid] += complete_mat[i]
        proto_count[cid] += 1

    proto_count = proto_count.clamp(min=1)
    prototypes = F.normalize(proto_sum / proto_count.unsqueeze(1), dim=-1)  # [C, D]

    cat_labels = dataset.scene_labels
    active = [i for i in range(num_categories) if proto_count[i] > 0]
    active_protos = prototypes[active]  # [C_active, D]
    # Map original cat_id → index in active list
    active_idx = {cat: idx for idx, cat in enumerate(active)}
    print(f"    semantic: {len(active)} active categories with prototypes")

    # Step 3: full-rank prototype classification, averaged over num_runs
    proto_acc: dict = {}
    for L in levels:
        run_accs: list[float] = []
        for run in range(num_runs):
            seed_run = seed + run
            masked_embeds = []
            for i in range(n):
                sample = dataset[i]
                masked = mask_pil_image(
                    sample["image_pil"], L, sample["seg_mask"], seed=seed_run, idx=i,
                    patch_size=patch_size, target_size=img_size,
                )
                embed = embed_pil(encoder, masked, norm_transform)
                masked_embeds.append(embed)

            masked_mat = F.normalize(torch.stack(masked_embeds), dim=-1)  # [N, D]
            sims = masked_mat @ active_protos.T  # [N, C_active]
            preds = sims.argmax(dim=1)  # [N]
            gt = torch.tensor([active_idx[c] for c in cat_ids], dtype=torch.long)
            acc = float((preds == gt).float().mean())
            run_accs.append(acc)

        run_accs_arr = np.array(run_accs)
        proto_acc[L] = {"mean": float(run_accs_arr.mean()), "std": float(run_accs_arr.std())}
        vis = get_visibility_ratio(L)
        print(f"    semantic [L={L}, vis={vis:.3f}] proto_acc={proto_acc[L]['mean']:.4f}"
              f"±{proto_acc[L]['std']:.4f} ({num_runs} runs, {len(active)} classes)")

    result: dict = {"prototype_acc": proto_acc}

    # Step 4: CLIP full-rank zero-shot text classification
    if encoder.name == "CLIP":
        print("    semantic: CLIP zero-shot text matching...")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        prompts = [f"a photo of {label}" for label in cat_labels]
        tokens = tokenizer(prompts).to(encoder.device)
        text_feats = encoder.model.encode_text(tokens)
        text_feats = F.normalize(text_feats.float().cpu(), dim=-1)  # [C, D]
        active_text = text_feats[active]  # [C_active, D]

        zs_acc: dict = {}
        for L in levels:
            run_accs: list[float] = []
            for run in range(num_runs):
                seed_run = seed + run
                masked_embeds = []
                for i in range(n):
                    sample = dataset[i]
                    masked = mask_pil_image(
                        sample["image_pil"], L, sample["seg_mask"],
                        seed=seed_run, idx=i,
                        patch_size=patch_size, target_size=img_size,
                    )
                    embed = embed_pil(encoder, masked, norm_transform)
                    masked_embeds.append(embed)

                masked_mat = F.normalize(torch.stack(masked_embeds), dim=-1)
                sims = masked_mat @ active_text.T  # [N, C_active]
                preds = sims.argmax(dim=1)
                gt = torch.tensor([active_idx[c] for c in cat_ids], dtype=torch.long)
                acc = float((preds == gt).float().mean())
                run_accs.append(acc)

            run_accs_arr = np.array(run_accs)
            zs_acc[L] = {"mean": float(run_accs_arr.mean()), "std": float(run_accs_arr.std())}
            vis = get_visibility_ratio(L)
            print(f"    semantic [L={L}, vis={vis:.3f}] zeroshot_acc={zs_acc[L]['mean']:.4f}"
                  f"±{zs_acc[L]['std']:.4f} ({num_runs} runs)")
        result["zeroshot_acc"] = zs_acc

    return result
