import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

from models.registry import register
from wrappers.encoder import BaseEncoder

MODEL_ID = "openai/clip-vit-large-patch14-336"


@register("llava")
class LLaVAEncoder(BaseEncoder):
    name = "LLaVA"
    feature_dim = 1024

    def load_model(self):
        model = CLIPVisionModel.from_pretrained(MODEL_ID)
        processor = CLIPImageProcessor.from_pretrained(MODEL_ID)
        return model, processor

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images, output_hidden_states=True)
        hidden = output.hidden_states[-2]
        patches = hidden[:, 1:]
        return patches.mean(dim=1)
