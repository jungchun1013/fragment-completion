import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from pathlib import Path

from models.registry import register
from models.encoder import BaseEncoder
from models.processor import ImageProcessor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
SIMCLR_CHECKPOINT = CHECKPOINT_DIR / "simclr-rn50.torch"


@register("simclr")
class SimCLREncoder(BaseEncoder):
    name = "SimCLR"
    feature_dim = 2048

    def load_model(self):
        if not SIMCLR_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"SimCLR checkpoint not found at {SIMCLR_CHECKPOINT}\n"
                "Download (VISSL pre-trained ResNet-50, 800 epochs):\n"
                "  mkdir -p checkpoints\n"
                "  wget -O checkpoints/simclr-rn50.torch \\\n"
                "    https://dl.fbaipublicfiles.com/vissl/model_zoo/"
                "simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/"
                "model_final_checkpoint_phase799.torch"
            )

        # ResNet-50 without classification head
        model = tvm.resnet50(weights=None)
        model.fc = nn.Identity()

        # Load VISSL checkpoint — extract trunk weights
        ckpt = torch.load(str(SIMCLR_CHECKPOINT), map_location="cpu")

        if "classy_state_dict" in ckpt:
            trunk_sd = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]
            cleaned = {k.replace("_feature_blocks.", ""): v
                       for k, v in trunk_sd.items()}
        else:
            raw = ckpt.get("state_dict", ckpt)
            cleaned = {k.replace("module.", "").replace("backbone.", ""): v
                       for k, v in raw.items()}

        msg = model.load_state_dict(cleaned, strict=False)
        if msg.missing_keys:
            print(f"  SimCLR: missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"  SimCLR: unexpected keys: {msg.unexpected_keys}")

        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return model, ImageProcessor(transform)
