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
    "cattention_unet": ModelSpec(
        "cattention_unet", "CAttentionUNet", "n_channels", "n_classes", "image_size",
        aliases=("cattunet", "cattention-u-net"),
    ),
    "double_unet": ModelSpec("doubleunet", "DoubleUNet", None, "n_classes", aliases=("doubleunet",)),
    "transunet": ModelSpec("transunet", "TransUNet", "in_channels", "n_classes", "img_dim", ("trans_unet",)),
    "segformer": ModelSpec("segformer", "Segformer", "channels", "num_classes"),
    "deit": ModelSpec("transformer_segmentation", "DeiTSegmenter", "in_channels", "n_classes", aliases=("deit_seg",)),
    "swin_unet": ModelSpec("transformer_segmentation", "SwinUNet", "in_channels", "n_classes", aliases=("swin", "swinunet")),
    "pvt_unet": ModelSpec("transformer_segmentation", "PVTUNet", "in_channels", "n_classes", aliases=("pvt", "pvtv2")),
    "pvtformer_full": ModelSpec("pvtformer_full", "PVTFormerFull", "in_channels", "n_classes", aliases=("pvtformer",)),
    "resunetplusplus_full": ModelSpec("resunetplusplus_full", "ResUNetPlusPlusFull", "in_channels", "n_classes", aliases=("resunetplusplus", "resunet++")),
    "unext_full": ModelSpec("unext_full", "UNeXtFull", "in_channels", "n_classes", aliases=("unext",)),
    "swin_unet_full": ModelSpec("swin_unet_full", "SwinUNetFull", "in_channels", "n_classes"),
}
MODEL_ALIASES = {alias: name for name, spec in MODEL_SPECS.items() for alias in (name, *spec.aliases)}


def model_metadata(name):
    descriptions = {
        "unet": "Configurable convolutional U-Net with skip connections.",
        "cattention_unet": "Configurable CAttention U-Net with CBAM after skip concatenation in each decoder stage.",
        "double_unet": "Cascaded U-Nets with a torchvision encoder and ASPP.",
        "transunet": "CNN/ViT hybrid encoder with a convolutional skip decoder.",
        "segformer": "Hierarchical Mix Transformer with an MLP segmentation decoder.",
        "deit": "DeiT encoder with a convolutional upsampling decoder.",
        "swin_unet": "Swin encoder with a convolutional pyramid decoder.",
        "pvt_unet": "PVTv2 encoder with a convolutional pyramid decoder.",
        "pvtformer_full": "PVTv2 three-scale encoder with residual hierarchical decoding and full-resolution multiscale fusion.",
        "resunetplusplus_full": "SE residual U-Net with decoder attention and bridge/output ASPP.",
        "unext_full": "Convolutional U-Net with shifted token MLP encoder and decoder stages.",
        "swin_unet_full": "Swin encoder and Swin decoder with patch expansion and skip fusion.",
    }
    transformer = name in {"deit", "swin_unet", "pvt_unet", "swin_unet_full", "pvtformer_full"}
    presets = {
        "deit": ("tiny", "small", "base"), "swin_unet": ("tiny", "small", "base"),
        "swin_unet_full": ("tiny", "small", "base"),
        "pvt_unet": ("b0", "b1", "b2", "b3", "b4", "b5"),
        "transunet": ("custom", "resnet50"),
        "pvtformer_full": ("b0", "b1", "b2", "b3", "b4", "b5"),
        "unext_full": ("base", "small"),
    }
    constraints = {
        "unet": "Spatial dimensions must accommodate 2**num_decoder_blocks downsampling.",
        "cattention_unet": "Spatial dimensions >=2**num_decoder_blocks; training BatchNorm needs multiple values per channel.",
        "double_unet": "RGB binary segmentation only; dimensions divisible by 16, >=32 recommended.",
        "transunet": "Dimensions divisible by 16; ResNet encoder accepts 1 or 3 channels.",
        "segformer": "Dimensions >=32 recommended for spatial-reduction attention.",
        "resunetplusplus_full": "Positive spatial dimensions padded to multiples of 8, minimum 16.",
        "unext_full": "Padded to multiples of 32; training requires batch * ceil(H/32) * ceil(W/32) > 1 for BatchNorm.",
    }
    return {
        "name": name, "class_name": MODEL_SPECS[name].class_name,
        "description": descriptions[name], "aliases": list(MODEL_SPECS[name].aliases),
        "presets": list(presets.get(name, ())),
        "preset_argument": "encoder_name" if name == "transunet" else ("variant" if transformer or name == "unext_full" else None),
        "dependencies": ["torch", "timm>=1.0.25,<2"] if transformer else ["torch", "torchvision", "einops"],
        "install_extra": "bicbioseg[transformers]" if transformer else None,
        "supported_devices": ["cpu", "cuda", "mps"],
        "device_notes": "Requires an available PyTorch backend; run validate_setup on the target device.",
        "input_constraints": "1 or 3 channels; positive spatial dimensions are padded internally." if transformer else constraints[name],
        "deep_supervision": name in {"swin_unet_full", "pvtformer_full", "resunetplusplus_full", "unext_full"},
        "pretrained": ("ImageNet encoder only via pretrained=True; decoder starts from scratch." if transformer
                       else "ResNet-50 encoder via encoder_weights; ViT starts from scratch." if name == "transunet"
                       else "Torchvision encoder via pretrained=True." if name == "double_unet" else None),
    }


def build_model(name, in_channels, num_classes, image_size, model_kwargs):
    if name not in MODEL_SPECS:
        raise ModelError(f"Unknown architecture '{name}'. Available: {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]
    if name == "double_unet" and (in_channels != 3 or num_classes != 1):
        raise ModelError("DoubleUNet currently supports only RGB binary segmentation (in_channels=3, num_classes=1).")
    contract = {spec.class_arg: num_classes}
    if spec.channel_arg:
        contract[spec.channel_arg] = in_channels
    if spec.size_arg:
        contract[spec.size_arg] = image_size[0] if spec.size_arg == "img_dim" and image_size[0] == image_size[1] else image_size
    for key, value in contract.items():
        if key in model_kwargs and model_kwargs[key] != value:
            raise ModelError(f"model_kwargs['{key}'] conflicts with the Segmenter configuration; set it on Segmenter instead.")
    kwargs = {"pretrained": False} if name == "double_unet" else {}
    kwargs.update(model_kwargs)
    kwargs.update(contract)
    cls = getattr(import_module(f".{spec.module}", __package__), spec.class_name)
    return cls(**kwargs)
