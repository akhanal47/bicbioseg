"""DINOv3 features with a bicbioseg convolutional segmentation decoder.

Encoder and weight reference: https://github.com/facebookresearch/dinov3
Pretrained weights retain the DINOv3 license; the decoder is trained from scratch.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ._common import conv_block, positive_int
from .transformer_segmentation import _EncoderSegmentation, _timm


class _DINOv3Decoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels, n_classes, dropout):
        super().__init__()
        width = decoder_channels[0]
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in feature_channels)
        self.fusion = conv_block(len(feature_channels) * width, width)
        self.upsample = nn.ModuleList(
            conv_block(a, b) for a, b in zip((width, *decoder_channels[:-1]), decoder_channels)
        )
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(decoder_channels[-1], n_classes, 1))

    def forward(self, features):
        # Intermediate ViT features share the same patch grid, not a spatial pyramid.
        x = self.fusion(torch.cat([layer(t) for layer, t in zip(self.projections, features)], dim=1))
        for block in self.upsample:
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(x)


class DINOv3Segmenter(_EncoderSegmentation):
    """Small/base DINOv3 encoder with four-depth fusion and trainable upsampling."""

    def __init__(self, in_channels=3, n_classes=1, variant="small", pretrained=False,
                 decoder_channels=(128, 64, 32, 16), dropout=0.1,
                 freeze_encoder=False, normalize_input=True):
        super().__init__()
        self._configure(in_channels, n_classes, pretrained, freeze_encoder, normalize_input, dropout)
        if variant not in ("small", "base"):
            raise ValueError("DINOv3 variant must be 'small' or 'base'.")
        decoder_channels = tuple(decoder_channels)
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four positive widths.")
        for width in decoder_channels:
            positive_int("decoder channel", width)
        self.encoder = _timm().create_model(
            f"vit_{variant}_patch16_dinov3.lvd1689m", pretrained=pretrained,
            features_only=True, out_indices=(2, 5, 8, 11), dynamic_img_size=True,
        )
        self.decoder = _DINOv3Decoder(
            self.encoder.feature_info.channels(), decoder_channels, n_classes, dropout
        )
        self._freeze()

    def forward(self, x):
        size = x.shape[-2:]
        x = self.input_transform(x)
        x = F.pad(x, (0, -size[1] % 16, 0, -size[0] % 16))
        return self.decoder(self.encoder(x))[..., :size[0], :size[1]]
