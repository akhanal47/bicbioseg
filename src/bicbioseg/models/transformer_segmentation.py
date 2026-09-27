# timm encoder for transformer-based segmentation models

from torch import nn
from torch.nn import functional as F

from ._common import ImageNetInput, PyramidDecoder, conv_block, positive_int


def _timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError("These transformer models require pip install 'bicbioseg[transformers]'.") from exc
    return timm


class _EncoderSegmentation(nn.Module):
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _configure(self, in_channels, n_classes, pretrained, freeze_encoder, normalize_input, dropout):
        positive_int("n_classes", n_classes)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not isinstance(pretrained, bool):
            raise ValueError("pretrained must be True or False.")
        if not isinstance(freeze_encoder, bool):
            raise ValueError("freeze_encoder must be True or False.")
        self.input_transform = ImageNetInput(in_channels, normalize_input)
        self.freeze_encoder = freeze_encoder

    def _freeze(self):
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()


class DeiTSegmenter(_EncoderSegmentation):
    """DeiT patch tokens decoded by four convolutional upsampling stages.

    Supports tiny/small/base, including distilled encoders. Class/distillation
    tokens participate in attention but are excluded from the segmentation map.
    Inputs are padded to a patch multiple and cropped back to their exact size.
    """

    def __init__(self, in_channels=3, n_classes=1, variant="tiny", pretrained=False,
                 distilled=False, decoder_channels=(128, 64, 32, 16), dropout=0.1,
                 freeze_encoder=False, normalize_input=True):
        super().__init__()
        self._configure(in_channels, n_classes, pretrained, freeze_encoder, normalize_input, dropout)
        if variant not in ("tiny", "small", "base"):
            raise ValueError("DeiT variant must be 'tiny', 'small', or 'base'.")
        if not isinstance(distilled, bool):
            raise ValueError("distilled must be True or False.")
        decoder_channels = tuple(decoder_channels)
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must have four positive widths.")
        for width in decoder_channels:
            positive_int("decoder channel", width)
        name = f"deit_{variant}{'_distilled' if distilled else ''}_patch16_224.fb_in1k"
        self.encoder = _timm().create_model(name, pretrained=pretrained, num_classes=0,
                                          dynamic_img_size=True)
        widths = (self.encoder.num_features, *decoder_channels)
        self.decoder = nn.ModuleList(conv_block(a, b) for a, b in zip(widths, widths[1:]))
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(widths[-1], n_classes, 1))
        self._freeze()

    def forward(self, x):
        size = x.shape[-2:]
        x = self.input_transform(x)
        x = F.pad(x, (0, -size[1] % 16, 0, -size[0] % 16))
        height, width = x.shape[-2] // 16, x.shape[-1] // 16
        tokens = self.encoder.forward_features(x)[:, self.encoder.num_prefix_tokens:]
        x = tokens.transpose(1, 2).reshape(x.shape[0], -1, height, width)
        for block in self.decoder:
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(x)[..., :size[0], :size[1]]


class _PyramidSegmenter(_EncoderSegmentation):
    def _build(self, name, in_channels, n_classes, pretrained, decoder_channels,
               dropout, freeze_encoder, normalize_input, **encoder_kwargs):
        self._configure(in_channels, n_classes, pretrained, freeze_encoder, normalize_input, dropout)
        positive_int("decoder_channels", decoder_channels)
        self.encoder = _timm().create_model(name, pretrained=pretrained, features_only=True,
                                          out_indices=(0, 1, 2, 3), **encoder_kwargs)
        self.decoder = PyramidDecoder(self.encoder.feature_info.channels(), decoder_channels, n_classes, dropout)
        self._freeze()

    def forward(self, x):
        size = x.shape[-2:]
        x = self.input_transform(x)
        # PVT's spatial reduction requires >=32; multiples of 32 also cover
        # Swin patch merging. Padding preserves the original image geometry.
        x = F.pad(x, (0, -size[1] % 32, 0, -size[0] % 32))
        padded_size = x.shape[-2:]
        features = self.encoder(x)
        if self.channels_last:
            features = [feature.permute(0, 3, 1, 2).contiguous() for feature in features]
        return self.decoder(features, padded_size)[..., :size[0], :size[1]]


class SwinUNet(_PyramidSegmenter):

    channels_last = True

    def __init__(self, in_channels=3, n_classes=1, variant="tiny", pretrained=False,
                 decoder_channels=128, dropout=0.1, freeze_encoder=False, normalize_input=True):
        super().__init__()
        if variant not in ("tiny", "small", "base"):
            raise ValueError("Swin variant must be 'tiny', 'small', or 'base'.")
        self._build(f"swin_{variant}_patch4_window7_224.ms_in1k", in_channels, n_classes,
                    pretrained, decoder_channels, dropout, freeze_encoder, normalize_input,
                    strict_img_size=False)


class PVTUNet(_PyramidSegmenter):
    channels_last = False

    def __init__(self, in_channels=3, n_classes=1, variant="b0", pretrained=False,
                 decoder_channels=128, dropout=0.1, freeze_encoder=False, normalize_input=True):
        super().__init__()
        if variant not in ("b0", "b1", "b2", "b3", "b4", "b5"):
            raise ValueError("PVT variant must be one of b0, b1, b2, b3, b4, b5.")
        self._build(f"pvt_v2_{variant}.in1k", in_channels, n_classes, pretrained,
                    decoder_channels, dropout, freeze_encoder, normalize_input)
