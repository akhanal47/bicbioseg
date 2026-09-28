'''
PVTFormer: three-scale PVTv2 encoding and residual multiscale decoding.
Paper: Jha et al., ISBI 2024, https://arxiv.org/abs/2401.09630.
'''

import torch
from torch import nn
from torch.nn import functional as F

from ._common import positive_int
from .transformer_segmentation import _EncoderSegmentation, _timm


class ResidualRefinement(nn.Module):
    def __init__(self, incoming, width):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(incoming, width, 3, padding=1), nn.BatchNorm2d(width),
                                  nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width))
        self.skip = nn.Sequential(nn.Conv2d(incoming, width, 1), nn.BatchNorm2d(width))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class PVTFormerFull(_EncoderSegmentation):
    def __init__(self, in_channels=3, n_classes=1, variant="b3", pretrained=False,
                 decoder_channels=64, dropout=0.0, freeze_encoder=False,
                 normalize_input=True, deep_supervision=False):
        super().__init__()
        if variant not in ("b0", "b1", "b2", "b3", "b4", "b5"):
            raise ValueError("PVT variant must be one of b0, b1, b2, b3, b4, b5.")
        positive_int("decoder_channels", decoder_channels)
        self._configure(in_channels, n_classes, pretrained, freeze_encoder, normalize_input, dropout)
        if not isinstance(deep_supervision, bool):
            raise ValueError("deep_supervision must be boolean.")
        self.deep_supervision = deep_supervision
        # The published decoder uses stages 1-3, not the fourth PVT stage.
        self.encoder = _timm().create_model(f"pvt_v2_{variant}.in1k", pretrained=pretrained,
                                            features_only=True, out_indices=(0, 1, 2))
        c = decoder_channels
        self.reductions = nn.ModuleList(nn.Sequential(nn.Conv2d(w, c, 1), nn.BatchNorm2d(c), nn.ReLU())
                                        for w in self.encoder.feature_info.channels())
        self.decoder = nn.ModuleList([ResidualRefinement(c * 2, c) for _ in range(2)])
        self.full_resolution = nn.ModuleList([ResidualRefinement(c, c) for _ in range(4)])
        self.fusion = ResidualRefinement(c * 4, c)
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(c, n_classes, 1))
        self.aux_heads = nn.ModuleList([nn.Conv2d(c, n_classes, 1) for _ in range(2)]) if deep_supervision else nn.ModuleList()
        self._freeze()

    def forward(self, x):
        size = x.shape[-2:]
        x = F.pad(self.input_transform(x), (0, -size[1] % 32, 0, -size[0] % 32))
        padded_size = x.shape[-2:]
        scales = [reduce(feature) for reduce, feature in zip(self.reductions, self.encoder(x))]
        x, intermediate = scales[-1], []
        for skip, block in zip(reversed(scales[:-1]), self.decoder):
            x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True), skip], 1))
            intermediate.append(x)
        maps = [block(F.interpolate(feature, size=padded_size, mode="bilinear", align_corners=True))
                for block, feature in zip(self.full_resolution, [x, *scales])]
        logits = self.head(self.fusion(torch.cat(maps, 1)))[..., :size[0], :size[1]]
        if self.deep_supervision and self.training:
            auxiliary = [F.interpolate(head(feature), size=padded_size, mode="bilinear", align_corners=False)[..., :size[0], :size[1]]
                         for head, feature in zip(self.aux_heads, intermediate)]
            return {"logits": logits, "aux_logits": auxiliary}
        return logits
