"""
Configurable U-Net with CBAM on concatenated decoder features.
CBAM follows Woo et al. (2018), https://arxiv.org/abs/1807.06521.
"""

import torch
import torch.nn as nn

from .unet import UNet, Up as UNetUp


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        _positive_integer("in_channels", in_channels)
        _positive_integer("ratio", ratio)
        hidden_channels = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = torch.sigmoid(avg_out + max_out)
        return out * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        out = self.conv1(concat)
        out = self.sigmoid(out)
        return out * x

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class Up(UNetUp):
    """The U-Net decoder with CBAM between concatenation and convolution."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__(in_channels, out_channels, bilinear)
        self.cbam = CBAM(in_channels)

    def _refine_features(self, x):
        return self.cbam(x)


class CAttentionUNet(UNet):
    """U-Net with one CBAM block before each decoder's double convolution.

    Width, depth, channel progression, upsampling, and defaults match UNet.
    CBAM uses reduction ratio 8 and a 7x7 spatial kernel. Returns raw logits
    of shape (N, n_classes, H, W). Both dimensions must be >=2**depth.
    """

    UP_BLOCK = Up

    def __init__(self, n_channels=3, n_classes=1, num_decoder_blocks=4,
                 base_channels=16, bilinear=False, image_size=None):
        for name, value in (("n_channels", n_channels), ("n_classes", n_classes),
                            ("num_decoder_blocks", num_decoder_blocks),
                            ("base_channels", base_channels)):
            _positive_integer(name, value)
        super().__init__(
            n_channels=n_channels, n_classes=n_classes,
            num_decoder_blocks=num_decoder_blocks, base_channels=base_channels,
            bilinear=bilinear, image_size=image_size,
        )

    @staticmethod
    def _minimum_dimension(image_size):
        if isinstance(image_size, int):
            _positive_integer("image_size", image_size)
            return image_size
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise ValueError("image_size must be a positive integer or (height, width) pair")
        for dimension in image_size:
            _positive_integer("image_size dimension", dimension)
        return min(image_size)

    @classmethod
    def calculate_max_decoder_blocks(cls, image_size):
        return min(cls._minimum_dimension(image_size).bit_length() - 1, cls.MAX_DECODER_BLOCKS)

    @classmethod
    def _validate_image_size(cls, image_size, num_decoder_blocks):
        if cls._minimum_dimension(image_size) < 2**num_decoder_blocks:
            raise ValueError(
                f"num_decoder_blocks={num_decoder_blocks} requires image_size >= {2**num_decoder_blocks}. "
                f"For image_size={image_size}, maximum num_decoder_blocks={cls.calculate_max_decoder_blocks(image_size)}"
            )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.n_channels:
            raise ValueError(f"Expected NCHW input with {self.n_channels} channels")
        self._validate_image_size(tuple(x.shape[-2:]), self.num_decoder_blocks)
        return super().forward(x)

    def get_architecture_info(self):
        return {
            **super().get_architecture_info(),
            "attention_parameters": sum(
                p.numel() for m in self.modules() if isinstance(m, CBAM)
                for p in m.parameters()
            ),
        }
