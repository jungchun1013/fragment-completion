import torch
import torch.nn as nn
from transformers import ViTMAEModel, ViTMAEConfig, AutoImageProcessor

from models.registry import register
from wrappers.encoder import BaseEncoder

MODEL_ID = "facebook/vit-mae-base"


@register("mae")
class MAEEncoder(BaseEncoder):
    name = "MAE"
    feature_dim = 768

    def load_model(self):
        config = ViTMAEConfig.from_pretrained(MODEL_ID)
        config.mask_ratio = 0.0
        model = ViTMAEModel.from_pretrained(MODEL_ID, config=config)
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        return model, processor

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images)
        return output.last_hidden_state[:, 0]  # CLS token
