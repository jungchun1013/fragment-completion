import timm
import timm.data
import torch.nn as nn

from models.registry import register
from wrappers.encoder import BaseEncoder
from wrappers.processor import ImageProcessor

MODEL_ID = "vit_base_patch16_224"


@register("vit_sup")
class ViTSupEncoder(BaseEncoder):
    name = "ViT-supervised"
    feature_dim = 768

    def load_model(self):
        model = timm.create_model(MODEL_ID, pretrained=True, num_classes=0)
        data_cfg = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_cfg, is_training=False)
        return model, ImageProcessor(transform)
