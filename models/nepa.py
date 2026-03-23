import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import transforms

from models.registry import register
from wrappers.encoder import BaseEncoder
from wrappers.processor import ImageProcessor

MODEL_ID = "SixAILab/nepa-base-patch14-224"


@register("nepa")
class NEPAEncoder(BaseEncoder):
    name = "NEPA"
    feature_dim = 768

    def load_model(self):
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
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
