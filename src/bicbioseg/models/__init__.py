from importlib import import_module
from .registry import MODEL_SPECS

__all__ = [spec.class_name for spec in MODEL_SPECS.values()]


def __getattr__(name):
    for spec in MODEL_SPECS.values():
        if name == spec.class_name:
            return getattr(import_module(f".{spec.module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
