import torch
import torch.nn as nn
from torchvision import transforms

from models.registry import register
from models.encoder import BaseEncoder
from models.processor import ImageProcessor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@register("dino")
class DINOEncoder(BaseEncoder):
    name = "DINO-v1"
    feature_dim = 768

    def load_model(self):
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return model, ImageProcessor(transform)
