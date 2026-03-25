from abc import ABC, abstractmethod

from PIL import Image
import torch
import torch.nn as nn


class BaseEncoder(ABC):
    """Unified interface for all vision encoders."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        self._processor = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable encoder name."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of output feature vectors."""

    @property
    def cls_token_position(self) -> int | None:
        """Index of the CLS token in the sequence, or None if no CLS token.

        Most ViTs prepend CLS at position 0.  Override in subclasses
        where the architecture differs (e.g. no CLS token).
        """
        return 0

    @abstractmethod
    def load_model(self) -> tuple[nn.Module, object]:
        """Load and return (pretrained_model, processor)."""

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            model, processor = self.load_model()
            self._model = model.to(self.device).eval()
            self._processor = processor
        return self._model

    @property
    def processor(self):
        """Return the processor (triggers lazy model load if needed)."""
        if self._processor is None:
            _ = self.model
        return self._processor

    def _prepare_images(self, images) -> torch.Tensor:
        """Convert a list of PIL images (or a tensor) to a batched tensor on device."""
        if isinstance(images, torch.Tensor):
            return images.to(self.device)
        if isinstance(images, (list, tuple)) and len(images) and isinstance(images[0], Image.Image):
            out = self.processor(images, return_tensors="pt")
            return out["pixel_values"].to(self.device)
        return images.to(self.device)

    @torch.no_grad()
    def extract_features(self, images) -> torch.Tensor:
        """Extract features from a batch of images. Returns [B, feature_dim]."""
        images = self._prepare_images(images)
        return self._forward(images)

    @torch.no_grad()
    def extract_features_from_layer(
        self, images: torch.Tensor, layer: str, pool: str = "mean",
    ) -> torch.Tensor:
        """Extract activations from a specific named layer using a forward hook.

        Args:
            images: input batch [B, C, H, W]
            layer: dot-separated layer name (e.g. 'blocks.5', 'layer3')
                   Use list_layers() to discover available names.
            pool: pooling strategy for 3D outputs [B, T, D].
                  "mean" — average over tokens expect CLS token (default).
                  "last" — take the last token only.
                  "cls"  — take the CLS token (position from cls_token_position).
                  "flatten" — concatenate all tokens into [B, T*D] (deprecated).

        Returns:
            Activation tensor [B, D].
        """
        images = self._prepare_images(images)
        module = self._get_submodule(layer)
        activation = {}

        def hook_fn(mod, inp, out):
            activation["out"] = out

        handle = module.register_forward_hook(hook_fn)
        try:
            self._forward(images)
        finally:
            handle.remove()

        if "out" not in activation:
            raise RuntimeError(
                f"Layer '{layer}' was not reached during forward pass. "
                f"It may belong to an unused branch (e.g. text encoder). "
                f"Use --list-layers to find valid layers."
            )
        out = activation["out"]
        if isinstance(out, tuple):
            out = out[0]

        # Pool to [B, D] if needed
        if out.dim() == 3:  # [B, tokens, D]
            if pool == "flatten":
                out = out.flatten(1)
            elif pool == "cls":
                pos = self.cls_token_position
                if pos is None:
                    print(f"{self.name} has no CLS token. Use 'last' pooling.")
                    pos = -1
                out = out[:, pos, :]
            elif pool == "last":
                out = out[:, -1, :]
            elif pool == "mean":
                out = out.mean(dim=1)
            else:
                raise ValueError(f"Invalid pooling method: {pool}")
        elif out.dim() == 4:  # [B, C, H, W] — spatial avg pool
            out = out.flatten(2).mean(dim=2)

        return out

    def list_layers(self) -> list[str]:
        """Return all named submodule paths in the model."""
        return [name for name, _ in self.model.named_modules() if name]

    def _get_submodule(self, layer: str) -> nn.Module:
        """Resolve a dot-separated layer name to a submodule."""
        module = self.model
        for attr in layer.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """Default forward pass. Override for encoders with special extraction."""
        return self.model(images)
