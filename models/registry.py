from models.encoder import BaseEncoder

_REGISTRY: dict[str, type[BaseEncoder]] = {}

# Mapping from encoder name → module name (under models/encoders/)
_LAZY_MODULES: dict[str, str] = {
    "clip": "clip",
    "dino": "dino",
    "dinov2": "dinov2",
    "mae": "mae",
    "mae_ft": "mae_ft",
    "nepa": "nepa",
    "resnet": "resnet",
    "siglip": "siglip",
    "simclr": "simclr",
    "ijepa": "ijepa",
    "vit_sup": "vit_sup",
    "llava": "llava",
    "qwen2vl": "qwen2vl",
}


def register(name: str):
    """Decorator to register an encoder class under a string name."""
    def decorator(cls: type[BaseEncoder]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_loaded(name: str) -> None:
    """Lazily import the module that registers the given encoder."""
    if name in _REGISTRY:
        return
    if name not in _LAZY_MODULES:
        return
    import importlib
    importlib.import_module(f"models.encoders.{_LAZY_MODULES[name]}")


def get_encoder(name: str, **kwargs) -> BaseEncoder:
    """Instantiate an encoder by name (lazy-loads its module on first use)."""
    _ensure_loaded(name)
    if name not in _REGISTRY:
        available = ", ".join(sorted(set(list(_REGISTRY) + list(_LAZY_MODULES))))
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_encoders() -> list[str]:
    """Return sorted list of all known encoder names."""
    return sorted(set(list(_REGISTRY) + list(_LAZY_MODULES)))
