import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms

from models.registry import register
from models.encoder import BaseEncoder
from models.processor import ImageProcessor

MODEL_ID = "SixAILab/nepa-base-patch14-224"


# ---------------------------------------------------------------------------
# Minimal NEPA model (ViT + RoPE + LayerScale + causal attention)
# vit_nepa is NOT in any released transformers version — custom impl required.
# ---------------------------------------------------------------------------

def _build_rope_freqs(seq_len: int, dim: int, theta: float = 100.0) -> torch.Tensor:
    """Precompute 2D RoPE frequencies for a square grid of patches."""
    grid_size = int(math.sqrt(seq_len))
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
    ys = torch.arange(grid_size, dtype=torch.float32)
    xs = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    freqs_y = grid_y.reshape(-1).unsqueeze(1) * freqs.unsqueeze(0)
    freqs_x = grid_x.reshape(-1).unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([freqs_y, freqs_x], dim=-1)  # [seq_len, half]


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings. x: [B, heads, N, head_dim]."""
    half = x.shape[-1] // 2
    f = freqs[:x.shape[2], :half]
    cos_f = f.cos().unsqueeze(0).unsqueeze(0)
    sin_f = f.sin().unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1)


class NEPAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 is_causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_causal = is_causal
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.dense = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor,
                rope_freqs: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if rope_freqs is not None:
            q = _apply_rope(q, rope_freqs)
            k = _apply_rope(k, rope_freqs)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        return self.dense(attn.permute(0, 2, 1, 3).reshape(B, N, C))


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

    def forward(self, x: torch.Tensor,
                rope_freqs: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.layer_scale_1 * self.attention(self.layernorm_before(x), rope_freqs)
        h = F.gelu(self.up_proj(self.layernorm_after(x)))
        x = x + self.layer_scale_2 * self.down_proj(h)
        return x


class NEPAModel(nn.Module):
    """Minimal NEPA ViT: patch embed + CLS + RoPE + causal attn + LayerScale."""

    def __init__(self, image_size: int = 224, patch_size: int = 14,
                 num_channels: int = 3, hidden_size: int = 768,
                 num_layers: int = 12, num_heads: int = 12,
                 intermediate_size: int = 3072, layerscale_value: float = 1e-5,
                 is_causal: bool = True, rope_theta: float = 100.0):
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

        head_dim = hidden_size // num_heads
        self.register_buffer(
            "rope_freqs",
            _build_rope_freqs(num_patches, head_dim, theta=rope_theta),
            persistent=False,
        )
        self.config = type("Cfg", (), {
            "image_size": image_size, "patch_size": patch_size,
            "hidden_size": hidden_size,
        })()

    def forward(self, pixel_values: torch.Tensor) -> object:
        B = pixel_values.shape[0]
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        N_patches = x.shape[1] - 1
        cls_pad = torch.zeros(1, self.rope_freqs.shape[1], device=x.device)
        rope = torch.cat([cls_pad, self.rope_freqs[:N_patches]], dim=0)

        for block in self.blocks:
            x = block(x, rope)
        x = self.layernorm(x)
        return type("Out", (), {"last_hidden_state": x})()


def _load_nepa_weights(model: NEPAModel, state_dict: dict) -> None:
    """Map HF checkpoint keys -> custom model parameter names."""
    new_state = {}
    for key, val in state_dict.items():
        k = key.replace("vit_nepa.", "", 1)
        k = k.replace("embeddings.cls_token", "cls_token")
        k = k.replace("embeddings.patch_embeddings.projection.", "projection.")
        k = k.replace("encoder.layer.", "blocks.")
        k = k.replace(".attention.attention.query.", ".attention.query.")
        k = k.replace(".attention.attention.key.", ".attention.key.")
        k = k.replace(".attention.attention.value.", ".attention.value.")
        k = k.replace(".attention.output.dense.", ".attention.dense.")
        k = k.replace(".intermediate.up_proj.", ".up_proj.")
        k = k.replace(".output.dense.", ".down_proj.")

        # Layer scales
        if ".output.layer_scale.lambda1" in key:
            base = key.split(".layer_scale.lambda1")[0]
            base = base.replace("vit_nepa.", "").replace("encoder.layer.", "blocks.")
            base = base.replace(".output", "")
            k = base + ".layer_scale_2"
        elif ".layer_scale.lambda1" in key:
            base = key.split(".layer_scale.lambda1")[0]
            base = base.replace("vit_nepa.", "").replace("encoder.layer.", "blocks.")
            k = base + ".layer_scale_1"

        new_state[k] = val

    result = model.load_state_dict(new_state, strict=False)
    if result.unexpected_keys:
        print(f"  NEPA unexpected keys: {result.unexpected_keys}")
    if result.missing_keys:
        print(f"  NEPA missing keys: {result.missing_keys}")


@register("nepa")
class NEPAEncoder(BaseEncoder):
    name = "NEPA"
    feature_dim = 768

    def load_model(self):
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
        _load_nepa_weights(model, load_file(weights_path))
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
        return output.last_hidden_state.mean(dim=1)
