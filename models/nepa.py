import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms

from models.registry import register
from wrappers.encoder import BaseEncoder
from wrappers.processor import ImageProcessor

MODEL_ID = "SixAILab/nepa-base-patch14-224"


# ---------------------------------------------------------------------------
# Minimal NEPA model (ViT + RoPE + LayerScale + causal attention)
# ---------------------------------------------------------------------------

def _build_rope_freqs(seq_len: int, dim: int, theta: float = 100.0) -> torch.Tensor:
    """Precompute 2D RoPE frequencies for a square grid of patches."""
    grid_size = int(math.sqrt(seq_len))
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
    ys = torch.arange(grid_size, dtype=torch.float32)
    xs = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)
    # [seq_len, half//2] for each axis
    freqs_y = grid_y.unsqueeze(1) * freqs.unsqueeze(0)
    freqs_x = grid_x.unsqueeze(1) * freqs.unsqueeze(0)
    # Interleave y and x frequencies → [seq_len, half]
    freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1)
    # cos and sin → [seq_len, dim]  (applied to pairs)
    return freqs_2d  # [seq_len, half]


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings. x: [B, heads, N, head_dim]."""
    half = x.shape[-1] // 2
    f = freqs[:x.shape[2], :half]  # [N, half]
    cos_f = f.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, N, half]
    sin_f = f.sin().unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1)


class NEPAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, is_causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_causal = is_causal
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.dense = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if rope_freqs is not None:
            q = _apply_rope(q, rope_freqs)
            k = _apply_rope(k, rope_freqs)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        x = attn.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.dense(x)


class NEPABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int,
                 layerscale_value: float = 1e-5, is_causal: bool = True):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(dim)
        self.attention = NEPAAttention(dim, num_heads, is_causal=is_causal)
        self.layer_scale_1 = nn.Parameter(torch.ones(dim) * layerscale_value)
        self.layernorm_after = nn.LayerNorm(dim)
        self.up_proj = nn.Linear(dim, mlp_dim)
        self.down_proj = nn.Linear(mlp_dim, dim)
        self.layer_scale_2 = nn.Parameter(torch.ones(dim) * layerscale_value)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.layer_scale_1 * self.attention(self.layernorm_before(x), rope_freqs)
        h = self.layernorm_after(x)
        h = F.gelu(self.up_proj(h))
        h = self.down_proj(h)
        x = x + self.layer_scale_2 * h
        return x


class NEPAModel(nn.Module):
    def __init__(self, image_size=224, patch_size=14, num_channels=3,
                 hidden_size=768, num_layers=12, num_heads=12,
                 intermediate_size=3072, layerscale_value=1e-5,
                 is_causal=True, rope_theta=100.0):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.projection = nn.Conv2d(num_channels, hidden_size,
                                    kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            NEPABlock(hidden_size, num_heads, intermediate_size,
                      layerscale_value, is_causal)
            for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(hidden_size)

        # Precompute RoPE frequencies (for patch tokens, excluding CLS)
        head_dim = hidden_size // num_heads
        self.register_buffer(
            "rope_freqs",
            _build_rope_freqs(num_patches, head_dim, theta=rope_theta),
            persistent=False,
        )

        # Store config-like attributes for compatibility
        self.config = type("Config", (), {
            "image_size": image_size,
            "patch_size": patch_size,
            "hidden_size": hidden_size,
        })()

    def forward(self, pixel_values: torch.Tensor) -> object:
        B = pixel_values.shape[0]
        # Patch embedding
        x = self.projection(pixel_values)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, D]

        # RoPE freqs: pad with zeros for CLS token position
        N_patches = x.shape[1] - 1
        cls_pad = torch.zeros(1, self.rope_freqs.shape[1], device=x.device)
        rope = torch.cat([cls_pad, self.rope_freqs[:N_patches]], dim=0)

        for block in self.blocks:
            x = block(x, rope)

        x = self.layernorm(x)

        # Return object with last_hidden_state for HF compatibility
        return type("Output", (), {"last_hidden_state": x})()


def _load_nepa_weights(model: NEPAModel, state_dict: dict) -> None:
    """Map HF checkpoint keys to our model's parameter names."""
    mapping = {}
    for key in state_dict:
        new_key = key
        # Strip model prefix
        new_key = new_key.replace("vit_nepa.", "", 1)
        # Embeddings
        new_key = new_key.replace("embeddings.cls_token", "cls_token")
        new_key = new_key.replace("embeddings.patch_embeddings.projection.", "projection.")
        # Encoder blocks
        new_key = new_key.replace("encoder.layer.", "blocks.")
        # Attention
        new_key = new_key.replace(".attention.attention.query.", ".attention.query.")
        new_key = new_key.replace(".attention.attention.key.", ".attention.key.")
        new_key = new_key.replace(".attention.attention.value.", ".attention.value.")
        new_key = new_key.replace(".attention.output.dense.", ".attention.dense.")
        # MLP
        new_key = new_key.replace(".intermediate.up_proj.", ".up_proj.")
        new_key = new_key.replace(".output.dense.", ".down_proj.")
        # Layer scale
        new_key = new_key.replace(".layer_scale.lambda1", ".layer_scale_1")
        new_key = new_key.replace(".output.layer_scale.lambda1", ".layer_scale_2")
        # Fix: output.layer_scale came before output.dense rename, re-check
        mapping[key] = new_key

    # Second pass: fix layer_scale_2 (the output block's layer_scale)
    # The original key pattern: encoder.layer.X.output.layer_scale.lambda1
    # After first pass it becomes: blocks.X.layer_scale_2 — but we also renamed
    # encoder.layer.X.output.dense → blocks.X.down_proj, so the output.layer_scale
    # may have been partially renamed. Let's be explicit:
    final_mapping = {}
    for orig_key, new_key in mapping.items():
        # Handle the case where "output.layer_scale" wasn't fully caught
        # because "output.dense" replacement happened first
        if "layer_scale.lambda1" in orig_key and "output" in orig_key:
            # This is the MLP layer scale
            base = orig_key.split(".layer_scale.lambda1")[0]
            base = base.replace("vit_nepa.", "").replace("encoder.layer.", "blocks.")
            base = base.replace(".output", "")
            final_mapping[orig_key] = base + ".layer_scale_2"
        elif "layer_scale.lambda1" in orig_key:
            # This is the attention layer scale
            base = orig_key.split(".layer_scale.lambda1")[0]
            base = base.replace("vit_nepa.", "").replace("encoder.layer.", "blocks.")
            final_mapping[orig_key] = base + ".layer_scale_1"
        else:
            final_mapping[orig_key] = new_key

    new_state = {final_mapping[k]: v for k, v in state_dict.items()}
    result = model.load_state_dict(new_state, strict=False)
    if result.unexpected_keys:
        print(f"  NEPA: unexpected keys: {result.unexpected_keys}")
    if result.missing_keys:
        print(f"  NEPA: missing keys: {result.missing_keys}")


@register("nepa")
class NEPAEncoder(BaseEncoder):
    name = "NEPA"
    feature_dim = 768

    def load_model(self):
        # Load config
        cfg_path = hf_hub_download(MODEL_ID, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

        model = NEPAModel(
            image_size=cfg.get("image_size", 224),
            patch_size=cfg.get("patch_size", 14),
            num_channels=cfg.get("num_channels", 3),
            hidden_size=cfg.get("hidden_size", 768),
            num_layers=cfg.get("num_hidden_layers", 12),
            num_heads=cfg.get("num_attention_heads", 12),
            intermediate_size=cfg.get("intermediate_size", 3072),
            layerscale_value=cfg.get("layerscale_value", 1e-5),
            is_causal=cfg.get("is_causal", True),
            rope_theta=cfg.get("rope_theta", 100.0),
        )

        weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
        state_dict = load_file(weights_path)
        _load_nepa_weights(model, state_dict)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        return model, ImageProcessor(transform)

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images)
        # Mean pool all tokens (CLS + patches)
        return output.last_hidden_state.mean(dim=1)
