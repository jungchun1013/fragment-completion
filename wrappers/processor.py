import torch
from PIL import Image


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
