"""Similarity analysis: frag->target vs frag->all for mnemonic & semantic.

For each masking level, computes:
  - Mnemonic: cos-sim(frag_i, complete_i) vs cos-sim(frag_i, complete_all)
  - Semantic: cos-sim(frag_i, proto_same_cat) vs cos-sim(frag_i, proto_all_cat)
"""

import numpy as np
import torch
import torch.nn.functional as F

from models.encoder import BaseEncoder
from models.processor import get_normalize_transform

from src.masking import get_mask_levels, get_visibility_ratio, mask_pil_image, prepare_image
from src.utils import embed_pil, get_encoder_geometry


@torch.no_grad()
def compute_similarity_analysis(encoder: BaseEncoder, dataset, seed: int = 42,
                                max_images: int | None = None) -> dict:
    """Compute frag->target and frag->all similarities for mnemonic & semantic.

    Returns dict with keys: mnemonic_target, mnemonic_all,
    semantic_same_cat, semantic_all_cat. Each maps level -> {mean, std}.
    """
    img_size, patch_size = get_encoder_geometry(encoder)
    norm_transform = get_normalize_transform(encoder.processor)
    levels = get_mask_levels()
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Embed all complete images (same spatial prep as masking)
    print(f"    Embedding {n} complete images...")
    complete_embeds = []
    cat_ids = []
    for i in range(n):
        sample = dataset[i]
        pil = prepare_image(sample["image_pil"], img_size)
        complete_embeds.append(embed_pil(encoder, pil, norm_transform))
        cat_ids.append(sample["scene_id"])
    complete_mat = F.normalize(torch.stack(complete_embeds), dim=-1)  # [N, D]

    # Build category prototypes
    num_cats = dataset.num_scenes
    D = complete_mat.shape[1]
    proto_sum = torch.zeros(num_cats, D)
    proto_count = torch.zeros(num_cats)
    for i in range(n):
        proto_sum[cat_ids[i]] += complete_mat[i]
        proto_count[cat_ids[i]] += 1
    proto_count = proto_count.clamp(min=1)
    prototypes = F.normalize(proto_sum / proto_count.unsqueeze(1), dim=-1)  # [C, D]

    mnemonic_target = {}
    mnemonic_all = {}
    semantic_same = {}
    semantic_all = {}

    for L in levels:
        masked_embeds = []
        for i in range(n):
            sample = dataset[i]
            masked = mask_pil_image(sample["image_pil"], L, sample["seg_mask"],
                                    seed=seed, idx=i,
                                    patch_size=patch_size, target_size=img_size)
            masked_embeds.append(embed_pil(encoder, masked, norm_transform))
        masked_mat = F.normalize(torch.stack(masked_embeds), dim=-1)  # [N, D]

        # Mnemonic: frag->target (paired) vs frag->all (average)
        paired_sims = (masked_mat * complete_mat).sum(dim=-1)  # [N]
        all_sims = masked_mat @ complete_mat.T  # [N, N]
        avg_all_sims = all_sims.mean(dim=1)  # [N]

        mnemonic_target[L] = {"mean": float(paired_sims.mean()), "std": float(paired_sims.std())}
        mnemonic_all[L] = {"mean": float(avg_all_sims.mean()), "std": float(avg_all_sims.std())}

        # Semantic: frag->same category prototype vs frag->all prototypes
        same_cat_sims = []
        for i in range(n):
            sim = F.cosine_similarity(masked_mat[i].unsqueeze(0),
                                       prototypes[cat_ids[i]].unsqueeze(0)).item()
            same_cat_sims.append(sim)
        same_cat_arr = np.array(same_cat_sims)

        frag_proto_sims = masked_mat @ prototypes.T  # [N, C]
        avg_proto_sims = frag_proto_sims.mean(dim=1)  # [N]

        semantic_same[L] = {"mean": float(same_cat_arr.mean()), "std": float(same_cat_arr.std())}
        semantic_all[L] = {"mean": float(avg_proto_sims.mean()), "std": float(avg_proto_sims.std())}

        vis = get_visibility_ratio(L)
        print(f"    L={L} vis={vis:.3f} | mnem tgt={mnemonic_target[L]['mean']:.4f} "
              f"all={mnemonic_all[L]['mean']:.4f} | sem same={semantic_same[L]['mean']:.4f} "
              f"all={semantic_all[L]['mean']:.4f}")

    return {
        "mnemonic_target": mnemonic_target,
        "mnemonic_all": mnemonic_all,
        "semantic_same_cat": semantic_same,
        "semantic_all_cat": semantic_all,
    }
