'''
UNeXt with convolutional stages and spatially shifted token MLPs.
Paper: Valanarasu and Patel https://arxiv.org/abs/2203.04967
'''

import torch
from torch import nn
from torch.nn import functional as F
from ._common import positive_int


class ChannelNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.movedim(1, -1).contiguous()).movedim(-1, 1).contiguous()


class ShiftedMLP(nn.Module):
    def __init__(self, width, shift_size=5, dropout=0.0):
        super().__init__()
        self.shift_size = shift_size
        self.norm = ChannelNorm(width)
        # 1x1 projections are equivalent to token-wise linear layers.
        self.first = nn.Conv2d(width, width, 1)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.last = nn.Conv2d(width, width, 1)
        self.drop = nn.Dropout(dropout)

    def shift(self, x, axis):
        pad = self.shift_size // 2
        h, w = x.shape[-2:]
        padded = F.pad(x, (pad, pad, pad, pad))
        chunks = torch.tensor_split(padded, self.shift_size, dim=1)
        shifted = torch.cat([torch.roll(chunk, index - pad, axis) for index, chunk in enumerate(chunks)], dim=1)
        return shifted[:, :, pad:pad+h, pad:pad+w].contiguous()

    def forward(self, x):
        y = self.first(self.shift(self.norm(x), 2))
        y = self.drop(F.gelu(self.spatial(y)))
        return x + self.drop(self.last(self.shift(y, 3)))


class UNeXtFull(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, variant="base", widths=None,
                 dropout=0.0, deep_supervision=False):
        super().__init__()
        if variant not in {"base", "small"}:
            raise ValueError("UNeXt variant must be 'base' or 'small'.")
        widths = tuple(widths or ((16, 32, 128, 160, 256) if variant == "base" else (8, 16, 32, 64, 128)))
        if len(widths) != 5:
            raise ValueError("widths must contain five positive integers.")
        for name, value in [("in_channels", in_channels), ("n_classes", n_classes), *[("width", w) for w in widths]]:
            positive_int(name, value)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if not isinstance(deep_supervision, bool):
            raise ValueError("deep_supervision must be boolean.")
        self.deep_supervision = deep_supervision
        self.encoder = nn.ModuleList()
        for a, b in zip((in_channels, *widths[:2]), widths[:3]):
            self.encoder.append(nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.MaxPool2d(2), nn.ReLU()))
        self.token_encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b, 3, stride=2, padding=1), ChannelNorm(b), ShiftedMLP(b, dropout=dropout), ChannelNorm(b))
            for a, b in zip(widths[2:4], widths[3:5])])
        self.decoder = nn.ModuleList([nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b))
                                     for a, b in zip(reversed(widths[1:]), reversed(widths[:-1]))])
        self.token_decoder = nn.ModuleList([nn.Sequential(ShiftedMLP(w, dropout=dropout), ChannelNorm(w)) for w in (widths[3], widths[2])])
        self.final = nn.Conv2d(widths[0], widths[0], 3, padding=1)
        self.head = nn.Conv2d(widths[0], n_classes, 1)
        self.aux_heads = nn.ModuleList([nn.Conv2d(w, n_classes, 1) for w in (widths[3], widths[2])]) if deep_supervision else nn.ModuleList()

    def forward(self, x):
        size = x.shape[-2:]
        x = F.pad(x, (0, -size[1] % 32, 0, -size[0] % 32))
        padded_size, skips = x.shape[-2:], []
        for block in [*self.encoder, *self.token_encoder]:
            x = block(x)
            skips.append(x)
        auxiliary = []
        for index, (block, skip) in enumerate(zip(self.decoder, reversed(skips[:-1]))):
            x = F.relu(F.interpolate(block(x), size=skip.shape[-2:], mode="bilinear", align_corners=False)) + skip
            if index < 2:
                x = self.token_decoder[index](x)
                if self.deep_supervision and self.training:
                    auxiliary.append(self.aux_heads[index](x))
        x = F.relu(F.interpolate(self.final(x), size=padded_size, mode="bilinear", align_corners=False))
        logits = self.head(x)[..., :size[0], :size[1]]
        if auxiliary:
            return {"logits": logits, "aux_logits": [F.interpolate(t, size=padded_size, mode="bilinear", align_corners=False)[..., :size[0], :size[1]] for t in auxiliary]}
        return logits
