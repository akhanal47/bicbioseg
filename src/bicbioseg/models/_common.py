import torch
from torch import nn
from torch.nn import functional as F


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def spatial_size(value):
    size = (value, value) if isinstance(value, int) else tuple(value)
    if len(size) != 2:
        raise ValueError("image size must contain height and width.")
    for dim in size:
        positive_int("image dimension", dim)
    return size


class ImageNetInput(nn.Module):
    # normalize [0, 1] inputs; repeat grayscale before RGB normalization

    def __init__(self, in_channels, enabled=True):
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError("This encoder supports in_channels=1 or 3.")
        if not isinstance(enabled, bool):
            raise ValueError("normalize_input must be True or False.")
        self.in_channels = in_channels
        self.enabled = enabled
        # Constants are independent of initialization weights, including on restore.
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected NCHW input with {self.in_channels} channels.")
        if self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std if self.enabled else x


def conv_block(in_channels, out_channels):
    # GroupNorm also works with the small batches common in segmentation.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(1, out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(1, out_channels),
        nn.GELU(),
    )


class PyramidDecoder(nn.Module):
    # fuse coarse-to-fine encoder maps with learned lateral projections.

    def __init__(self, feature_channels, decoder_channels, n_classes, dropout):
        super().__init__()
        positive_int("decoder_channels", decoder_channels)
        self.lateral = nn.ModuleList(nn.Conv2d(c, decoder_channels, 1) for c in feature_channels)
        self.refine = nn.ModuleList(conv_block(2 * decoder_channels, decoder_channels)
                                    for _ in feature_channels[:-1])
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(decoder_channels, n_classes, 1))

    def forward(self, features, output_size):
        maps = [layer(feature) for layer, feature in zip(self.lateral, features)]
        x = maps[-1]
        for skip, block in zip(reversed(maps[:-1]), self.refine):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return F.interpolate(self.head(x), size=output_size, mode="bilinear", align_corners=False)
