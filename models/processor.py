import torch
from PIL import Image
from torchvision import transforms as T


class ImageProcessor:
    """Wraps a torchvision transform to match HuggingFace processor interface."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, return_tensors="pt"):
        if isinstance(images, (list, tuple)):
            pixel_values = torch.stack([self.transform(img) for img in images])
        else:
            pixel_values = self.transform(images).unsqueeze(0)
        return {"pixel_values": pixel_values}


def to_transform(processor):
    """Extract a dataset-compatible transform (PIL -> Tensor) from a processor.

    For ImageProcessor, returns the underlying torchvision transform.
    For HuggingFace processors, returns a wrapper that produces tensors.
    """
    if isinstance(processor, ImageProcessor):
        return processor.transform
    # HF AutoProcessor wraps an image_processor inside
    proc = getattr(processor, "image_processor", processor)
    def _transform(img):
        return proc(img, return_tensors="pt")["pixel_values"].squeeze(0)
    return _transform


def get_normalize_transform(processor) -> T.Compose:
    """Return a ToTensor + Normalize transform (no spatial resize/crop).

    Use this for images that have already been spatially prepared
    (center-cropped and resized to the encoder's target resolution)
    via ``masking.prepare_image`` or ``masking.mask_pil_image``.
    """
    if isinstance(processor, ImageProcessor):
        # Extract Normalize from the existing torchvision pipeline
        for tr in processor.transform.transforms:
            if isinstance(tr, T.Normalize):
                return T.Compose([T.ToTensor(), tr])
        # Fallback: ImageNet defaults
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # HuggingFace processor — extract mean/std from its config
    proc = getattr(processor, "image_processor", processor)
    mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(proc, "image_std", [0.229, 0.224, 0.225])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ])
