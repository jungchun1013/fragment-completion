import json

import torch
import torch.nn as nn
from torchvision import transforms
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig

from models.registry import register
from wrappers.encoder import BaseEncoder
from wrappers.processor import ImageProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
RESOLUTION = 448  # must be divisible by 14
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2


@register("qwen2vl")
class Qwen2VLEncoder(BaseEncoder):
    name = "Qwen2-VL"
    feature_dim = 1280  # ViT embed_dim (pre-merger)

    def load_model(self):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VisionTransformerPretrainedModel,
        )

        config = AutoConfig.from_pretrained(MODEL_ID)
        vision_model = Qwen2VisionTransformerPretrainedModel(config.vision_config)

        # Download only the shard(s) containing vision weights
        index_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        visual_shards = set()
        for key, filename in index["weight_map"].items():
            if key.startswith("visual."):
                visual_shards.add(filename)

        state_dict = {}
        for shard in visual_shards:
            shard_path = hf_hub_download(MODEL_ID, shard)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith("visual."):
                        state_dict[key[len("visual."):]] = f.get_tensor(key)

        vision_model.load_state_dict(state_dict, strict=True)

        transform = transforms.Compose([
            transforms.Resize(RESOLUTION,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        return vision_model, ImageProcessor(transform)

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        h_patches = H // PATCH_SIZE
        w_patches = W // PATCH_SIZE

        # Duplicate temporally: single image → 2 identical frames
        # [B, C, H, W] → [B, C, 2, H, W]
        x = images.unsqueeze(2).expand(-1, -1, TEMPORAL_PATCH_SIZE, -1, -1)

        # Extract patches: [B, C, 2, h_p*14, w_p*14]
        #   → [B, C, 2, h_p, 14, w_p, 14]
        #   → [B*h_p*w_p, C*2*14*14]
        x = x.reshape(B, C, TEMPORAL_PATCH_SIZE,
                       h_patches, PATCH_SIZE, w_patches, PATCH_SIZE)
        x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        x = x.reshape(-1, C * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)

        grid_thw = torch.tensor(
            [[1, h_patches, w_patches]] * B,
            dtype=torch.long, device=images.device,
        )

        output = self.model(hidden_states=x, grid_thw=grid_thw)

        # Pre-merger features: (total_patches, 1280)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        features = hidden.reshape(B, -1, hidden.shape[-1]).mean(dim=1)
        return features  # [B, 1280]
