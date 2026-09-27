from dataclasses import dataclass
from importlib import import_module

from ..exceptions import ModelError


@dataclass(frozen=True)
class ModelSpec:
    module: str
    class_name: str
    channel_arg: str | None
    class_arg: str
    size_arg: str | None = None
    aliases: tuple[str, ...] = ()


MODEL_SPECS = {
    "unet": ModelSpec("unet", "UNet", "n_channels", "n_classes", "image_size", ("u-net",)),
    "attention_unet": ModelSpec("attention_unet", "AttUNet", "n_channels", "n_classes", aliases=("attunet", "attention-u-net")),
    "double_unet": ModelSpec("doubleunet", "DoubleUNet", None, "n_classes", aliases=("doubleunet",)),
    "transunet": ModelSpec("transunet", "TransUNet", "in_channels", "n_classes", "img_dim", ("trans_unet",)),
    "segformer": ModelSpec("segformer", "Segformer", "channels", "num_classes"),
}
MODEL_ALIASES = {alias: name for name, spec in MODEL_SPECS.items() for alias in (name, *spec.aliases)}


def build_model(name, in_channels, num_classes, image_size, model_kwargs):
    if name not in MODEL_SPECS:
        raise ModelError(f"Unknown architecture '{name}'. Available: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]
    if name == "double_unet" and (in_channels != 3 or num_classes != 1):
        raise ModelError("DoubleUNet currently supports only RGB binary segmentation (in_channels=3, num_classes=1).")
    if name == "transunet" and image_size[0] != image_size[1]:
        raise ModelError("TransUNet currently requires a square image_size.")
    contract = {spec.class_arg: num_classes}
    if spec.channel_arg:
        contract[spec.channel_arg] = in_channels
    if spec.size_arg:
        contract[spec.size_arg] = image_size[0] if spec.size_arg == "img_dim" else image_size
    for key, value in contract.items():
        if key in model_kwargs and model_kwargs[key] != value:
            raise ModelError(f"model_kwargs['{key}'] conflicts with the Segmenter configuration; set it on Segmenter instead.")
    kwargs = {"pretrained": False} if name == "double_unet" else {}
    kwargs.update(model_kwargs)
    kwargs.update(contract)
    cls = getattr(import_module(f".{spec.module}", __package__), spec.class_name)
    return cls(**kwargs)
