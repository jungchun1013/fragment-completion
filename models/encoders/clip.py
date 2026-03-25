import torch
import torch.nn as nn
import open_clip

from models.registry import register
from models.encoder import BaseEncoder
from models.processor import ImageProcessor


@register("clip")
class CLIPEncoder(BaseEncoder):
    name = "CLIP"
    feature_dim = 512

    def load_model(self):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        return model, ImageProcessor(preprocess)

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)
