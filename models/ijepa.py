import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

from models.registry import register
from wrappers.encoder import BaseEncoder

MODEL_ID = "facebook/ijepa_vith14_1k"


@register("ijepa")
class IJEPAEncoder(BaseEncoder):
    name = "I-JEPA"
    feature_dim = 1280

    def load_model(self):
        model = AutoModel.from_pretrained(MODEL_ID)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        return model, processor

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images)
        return output.last_hidden_state.mean(dim=1)
