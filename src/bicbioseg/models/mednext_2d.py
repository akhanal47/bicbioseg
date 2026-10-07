"""MedNeXt v1 in 2D, adapted from MIC-DKFZ/MedNeXt (Apache-2.0).

https://github.com/MIC-DKFZ/MedNeXt
Copyright 2019 Division of Medical Image Computing, DKFZ, Heidelberg, Germany.
See licenses/NOTICE.txt and licenses/Apache-2.0.txt.
Changes: 2D-only modules, configurable width/depth, padding/cropping, and bicbioseg
auxiliary output contract. No nnU-Net pipeline or upstream checkpoint loader.
"""

from torch import nn
from torch.nn import functional as F

from ._common import positive_int


class MedNeXtBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size, resample=None):
        super().__init__()
        self.resample = resample
        conv = nn.ConvTranspose2d if resample == "up" else nn.Conv2d
        stride = 2 if resample else 1
        self.depthwise = conv(in_channels, in_channels, kernel_size, stride=stride,
                              padding=kernel_size // 2, groups=in_channels)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.expand = nn.Conv2d(in_channels, expansion * in_channels, 1)
        self.act = nn.GELU()
        self.compress = nn.Conv2d(expansion * in_channels, out_channels, 1)
        self.residual = conv(in_channels, out_channels, 1, stride=2) if resample else nn.Identity()

    def forward(self, x):
        y = self.compress(self.act(self.expand(self.norm(self.depthwise(x)))))
        residual = self.residual(x)
        if self.resample == "up":
            # Preserve the reference model's asymmetric transposed-convolution alignment.
            y = F.pad(y, (1, 0, 1, 0))
            residual = F.pad(residual, (1, 0, 1, 0))
        return y + residual


class MedNeXt2D(nn.Module):
    """Residual MedNeXt encoder/decoder with optional four-head deep supervision."""

    def __init__(self, in_channels=3, n_classes=1, variant="small", base_channels=32,
                 kernel_size=3, block_counts=(2, 2, 2, 2, 2, 2, 2, 2, 2),
                 deep_supervision=False):
        super().__init__()
        if variant not in ("small", "base"):
            raise ValueError("MedNeXt variant must be 'small' or 'base'.")
        for name, value in (("in_channels", in_channels), ("n_classes", n_classes),
                            ("base_channels", base_channels), ("kernel_size", kernel_size)):
            positive_int(name, value)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        block_counts = tuple(block_counts)
        if len(block_counts) != 9:
            raise ValueError("block_counts must contain nine positive integers.")
        for count in block_counts:
            positive_int("block count", count)
        if not isinstance(deep_supervision, bool):
            raise ValueError("deep_supervision must be boolean.")
        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        expansion = (2,) * 9 if variant == "small" else (2, 3, 4, 4, 4, 4, 4, 3, 2)
        widths = [base_channels * 2 ** i for i in range(5)]
        self.stem = nn.Conv2d(in_channels, widths[0], 1)

        def stage(width, index):
            return nn.Sequential(*[MedNeXtBlock2D(width, width, expansion[index], kernel_size)
                                   for _ in range(block_counts[index])])

        self.encoder = nn.ModuleList(stage(width, i) for i, width in enumerate(widths))
        self.downsample = nn.ModuleList(
            MedNeXtBlock2D(a, b, expansion[i + 1], kernel_size, "down")
            for i, (a, b) in enumerate(zip(widths, widths[1:]))
        )
        self.upsample = nn.ModuleList(
            MedNeXtBlock2D(a, b, expansion[i + 5], kernel_size, "up")
            for i, (a, b) in enumerate(zip(reversed(widths[1:]), reversed(widths[:-1])))
        )
        self.decoder = nn.ModuleList(stage(width, i + 5) for i, width in enumerate(reversed(widths[:-1])))
        self.head = nn.ConvTranspose2d(widths[0], n_classes, 1)
        self.aux_heads = nn.ModuleList(
            nn.ConvTranspose2d(width, n_classes, 1) for width in reversed(widths[1:])
        ) if deep_supervision else nn.ModuleList()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) < 1:
            raise ValueError(f"Expected positive NCHW input with {self.in_channels} channels.")
        size = x.shape[-2:]
        # Keep at least 2x2 values per group at the bottleneck, even for batch size 1.
        padded = tuple(max(32, (s + 15) // 16 * 16) for s in size)
        x = self.stem(F.pad(x, (0, padded[1] - size[1], 0, padded[0] - size[0])))
        skips = []
        for block, down in zip(self.encoder[:-1], self.downsample):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.encoder[-1](x)
        auxiliary = [self.aux_heads[0](x)] if self.deep_supervision and self.training else []
        for i, (up, block, skip) in enumerate(zip(self.upsample, self.decoder, reversed(skips))):
            x = block(up(x) + skip)
            if self.deep_supervision and self.training and i < 3:
                auxiliary.append(self.aux_heads[i + 1](x))
        logits = self.head(x)[..., :size[0], :size[1]]
        if auxiliary:
            return {"logits": logits, "aux_logits": [
                F.interpolate(t, size=padded, mode="bilinear", align_corners=False)[..., :size[0], :size[1]]
                for t in reversed(auxiliary)
            ]}
        return logits
