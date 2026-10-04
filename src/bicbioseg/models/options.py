'''Dependency-free model option discovery and validation.'''

from copy import deepcopy
from math import isfinite
from .registry import MODEL_ALIASES, MODEL_SPECS
from ..exceptions import ModelError


def option(default, kind, description, **constraints):
    return dict(default=default, type=kind, description=description, **constraints)


POS = lambda default, description: option(default, "integer", description, minimum=1)
BOOL = lambda default, description: option(default, "boolean", description)
SEQ = lambda default, length, description: option(default, "sequence", description, length=length, minimum=1)
CHOICE = lambda default, choices, description: option(default, "choice", description, choices=list(choices))
DROP = option(0.1, "number", "Dropout probability.", minimum=0, exclusive_maximum=1)
CNN = dict(
    base_channels=POS(16, "First encoder width."),
    num_decoder_blocks=option(4, "integer", "Encoder and decoder depth.", minimum=1, maximum=8),
    bilinear=BOOL(False, "Use bilinear upsampling instead of transposed convolutions."),
)
PRETRAINED = dict(
    pretrained=BOOL(False, "Initialize encoder with ImageNet weights."),
    freeze_encoder=BOOL(False, "Freeze encoder weights and keep it in evaluation mode."),
    normalize_input=BOOL(True, "Apply ImageNet normalization inside the model."),
)
SWIN = CHOICE("tiny", ("tiny", "small", "base"), "Swin encoder size.")
PVT = CHOICE("b0", tuple(f"b{i}" for i in range(6)), "PVTv2 encoder size.")
DEEP = BOOL(False, "Use two auxiliary prediction heads during training.")
MODEL_OPTIONS = {
    "unet": CNN,
    "cattention_unet": deepcopy(CNN),
    "double_unet": dict(
        backbone=CHOICE("vgg19", ("vgg19", "resnet50", "resnet101"), "First encoder backbone."),
        pretrained=PRETRAINED["pretrained"],
    ),
    "segformer": dict(
        dims=SEQ((32, 64, 160, 256), 4, "Stage widths."),
        heads=SEQ((1, 2, 5, 8), 4, "Stage attention heads."),
        ff_expansion=SEQ((8, 8, 4, 4), 4, "Feed-forward expansion."),
        reduction_ratio=SEQ((8, 4, 2, 1), 4, "Attention spatial reduction."),
        num_layers=SEQ(2, 4, "Layers per stage."),
        decoder_dim=POS(256, "Decoder projection width."),
    ),
    "deit": dict(
        **PRETRAINED,
        variant=CHOICE("tiny", ("tiny", "small", "base"), "DeiT encoder size."),
        distilled=BOOL(False, "Use distilled encoder tokens."),
        decoder_channels=SEQ((128, 64, 32, 16), 4, "Decoder widths, deepest first."),
        dropout=DROP,
    ),
    "swin_unet": dict(
        **PRETRAINED, variant=SWIN, decoder_channels=POS(128, "Pyramid decoder width."), dropout=DROP
    ),
    "pvt_unet": dict(
        **PRETRAINED, variant=PVT, decoder_channels=POS(128, "Pyramid decoder width."), dropout=DROP
    ),
    "swin_unet_full": dict(
        **PRETRAINED,
        variant=SWIN,
        decoder_depths=SEQ((2, 2, 2), 3, "Swin decoder depths, coarse to fine."),
        dropout=DROP,
        deep_supervision=DEEP,
    ),
    "pvtformer_full": dict(
        **PRETRAINED,
        variant={**PVT, "default": "b3"},
        decoder_channels=POS(64, "Residual decoder width."),
        dropout={**DROP, "default": 0.0},
        deep_supervision=DEEP,
    ),
    "resunetplusplus_full": dict(
        base_channels=POS(16, "First encoder width."),
        aspp_rates=SEQ((1, 6, 12, 18), None, "Positive ASPP dilation rates."),
        deep_supervision=DEEP,
    ),
    "unext_full": dict(
        variant=CHOICE("base", ("base", "small"), "Stage width preset."),
        widths={**SEQ(None, 5, "Override all five stage widths."), "nullable": True},
        dropout={**DROP, "default": 0.0},
        deep_supervision=DEEP,
    ),
    "transunet": dict(
        preset=CHOICE("standard", ("lightweight", "standard", "heavy", None), "CNN/transformer size preset."),
        encoder_name=CHOICE("custom", ("custom", "resnet50"), "CNN encoder."),
        encoder_weights=CHOICE(
            None, (None, "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"), "ResNet-50 weights."
        ),
        pretrained_vit=CHOICE(False, (False,), "Pretrained ViT weights are unsupported."),
        **{
            key: {**POS(None, desc), "nullable": True}
            for key, desc in [
                ("out_channels", "CNN/bottleneck width, at least 8."),
                ("embedding_dim", "Transformer width, defaults to 8 times out_channels."),
                ("head_num", "Attention heads."),
                ("mlp_dim", "Feed-forward width."),
                ("block_num", "Transformer depth."),
            ]
        },
        patch_dim=CHOICE(None, (None, 16), "Fixed CNN stride."),
        decoder_channels={**SEQ(None, 4, "Decoder widths, deepest first."), "nullable": True},
        dropout=DROP,
        n_skip=option(3, "integer", "Number of deepest CNN skips.", minimum=0, maximum=3),
        normalize_input={**BOOL(None, "Defaults to True only for ResNet-50."), "nullable": True},
    ),
}


def model_options(architecture):
    """Return an independent option schema without constructing or downloading a model."""
    name = MODEL_ALIASES.get(str(architecture).lower(), str(architecture).lower())
    if name not in MODEL_OPTIONS:
        raise ModelError(f"Unknown architecture '{architecture}'. Available: {sorted(MODEL_OPTIONS)}")
    return deepcopy(MODEL_OPTIONS[name])


def normalize_model_options(
    architecture, kwargs=None, *, in_channels=3, num_classes=1, image_size=(224, 224)
):
    name = MODEL_ALIASES.get(str(architecture).lower(), str(architecture).lower())
    schema = model_options(name)
    result = deepcopy(dict(kwargs or {}))
    spec = MODEL_SPECS[name]
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    if len(size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in size):
        raise ModelError("image_size must be a positive (height, width) pair.")
    for label, value in [("in_channels", in_channels), ("num_classes", num_classes)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ModelError(f"{label} must be a positive integer.")
    if name == "double_unet" and (in_channels != 3 or num_classes != 1):
        raise ModelError("DoubleUNet supports only RGB binary segmentation.")
    contract = {spec.class_arg: num_classes}
    if spec.channel_arg:
        contract[spec.channel_arg] = in_channels
    if spec.size_arg:
        contract[spec.size_arg] = size
    for key, expected in contract.items():
        if key in result:
            actual = result.pop(key)
            if key == spec.size_arg:
                actual = (actual, actual) if isinstance(actual, int) else tuple(actual)
            if actual != expected:
                raise ModelError(
                    f"model_kwargs['{key}'] conflicts with the Segmenter configuration; set it on Segmenter instead."
                )
    unknown = set(result) - schema.keys()
    if unknown:
        raise ModelError(
            f"Unsupported {name} model option(s): {sorted(unknown)}. Use Segmenter.model_options('{name}')."
        )
    for key, value in result.items():
        rule = schema[key]
        kind = rule["type"]
        if value is None and rule.get("nullable"):
            continue
        valid = True
        if kind == "choice":
            valid = any(type(value) is type(c) and value == c for c in rule["choices"])
        elif kind == "boolean":
            valid = isinstance(value, bool)
        elif kind == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif kind == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
        elif kind == "sequence":
            if name == "segformer" and isinstance(value, int) and not isinstance(value, bool):
                value = (value,) * 4
            valid = isinstance(value, (tuple, list)) and bool(value)
            valid = (
                valid
                and (rule["length"] is None or len(value) == rule["length"])
                and all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in value)
            )
            if valid:
                result[key] = tuple(value)
        if valid and kind in ("integer", "number"):
            valid = (
                value >= rule.get("minimum", float("-inf"))
                and value <= rule.get("maximum", float("inf"))
                and value < rule.get("exclusive_maximum", float("inf"))
            )
        if not valid:
            raise ModelError(f"Invalid {name}.{key}={value!r}. Expected {rule}.")
    effective = {k: r["default"] for k, r in schema.items()}
    effective.update(result)
    if name in ("unet", "cattention_unet") and min(size) < 2 ** effective["num_decoder_blocks"]:
        raise ModelError("image_size is too small for num_decoder_blocks.")
    if name == "double_unet" and (in_channels != 3 or num_classes != 1 or any(n % 16 for n in size)):
        raise ModelError("DoubleUNet requires RGB binary segmentation and dimensions divisible by 16.")
    if name in ("deit", "swin_unet", "pvt_unet", "swin_unet_full", "pvtformer_full") and in_channels not in (
        1,
        3,
    ):
        raise ModelError(f"{name} supports only 1 or 3 input channels.")
    if name == "segformer":
        stage = {
            k: ((v,) * 4 if isinstance(v, int) else v) for k, v in effective.items() if k != "decoder_dim"
        }
        if any(d % h for d, h in zip(stage["dims"], stage["heads"])):
            raise ModelError("Each SegFormer dims value must be divisible by heads.")
        h, w = size
        for stride, reduction in zip((4, 2, 2, 2), stage["reduction_ratio"]):
            h, w = (h + stride - 1) // stride, (w + stride - 1) // stride
            if min(h, w) < reduction:
                raise ModelError("SegFormer image_size is too small for reduction_ratio.")
    if name == "transunet":
        presets = {"lightweight": (64, 4), "standard": (128, 4), "heavy": (256, 8)}
        width, heads = presets[effective["preset"] or "standard"]
        width = effective["out_channels"] or width
        heads = effective["head_num"] or heads
        embed = effective["embedding_dim"] or width * 8
        if width < 8 or embed % heads:
            raise ModelError(
                "TransUNet out_channels must be >=8 and embedding_dim must be divisible by head_num."
            )
        if any(n % 16 for n in size):
            raise ModelError("TransUNet dimensions must be divisible by 16.")
        if effective["encoder_name"] == "custom" and effective["encoder_weights"] is not None:
            raise ModelError("encoder_weights requires encoder_name=resnet50.")
        if (effective["encoder_name"] == "resnet50" or effective["normalize_input"]) and in_channels not in (
            1,
            3,
        ):
            raise ModelError("Normalized TransUNet and ResNet-50 accept 1 or 3 channels.")
    return result
