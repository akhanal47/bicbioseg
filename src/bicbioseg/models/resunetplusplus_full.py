'''
ResUNet++ with SE residual blocks, attention gates and two ASPP modules.
Paper: Jha et al., https://arxiv.org/abs/1911.07067.
'''

import torch
from torch import nn
from torch.nn import functional as F
from ._common import positive_int


def preactivated_conv(a, b, stride=1):
    return nn.Sequential(nn.BatchNorm2d(a), nn.ReLU(), nn.Conv2d(a, b, 3, stride=stride, padding=1))


class SEResidual(nn.Module):
    def __init__(self, a, b, stride=1, stem=False):
        super().__init__()
        self.body = (nn.Sequential(nn.Conv2d(a, b, 3, stride=stride, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1))
                     if stem else nn.Sequential(preactivated_conv(a, b, stride), preactivated_conv(b, b)))
        self.shortcut = nn.Sequential(nn.Conv2d(a, b, 1, stride=stride), nn.BatchNorm2d(b))
        self.excite = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(b, max(1, b // 8), 1, bias=False),
                                    nn.ReLU(), nn.Conv2d(max(1, b // 8), b, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = self.body(x) + self.shortcut(x)
        return x * self.excite(x)


class ASPP(nn.Module):
    def __init__(self, a, b, rates):
        super().__init__()
        self.branches = nn.ModuleList(nn.Sequential(nn.Conv2d(a, b, 3, padding=r, dilation=r), nn.BatchNorm2d(b)) for r in rates)
        self.project = nn.Conv2d(b, b, 1)

    def forward(self, x):
        return self.project(torch.stack([branch(x) for branch in self.branches]).sum(0))


class AttentionDecoder(nn.Module):
    def __init__(self, skip_channels, incoming, outgoing):
        super().__init__()
        self.skip_gate = preactivated_conv(skip_channels, incoming)
        self.decoder_gate = preactivated_conv(incoming, incoming)
        self.gate = preactivated_conv(incoming, incoming)
        self.refine = SEResidual(skip_channels + incoming, outgoing)

    def forward(self, x, skip):
        gate = F.max_pool2d(self.skip_gate(skip), 2)
        x = x * self.gate(gate + self.decoder_gate(x))
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.refine(torch.cat([x, skip], dim=1))


class ResUNetPlusPlusFull(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, base_channels=16,
                 aspp_rates=(1, 6, 12, 18), deep_supervision=False):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("n_classes", n_classes), ("base_channels", base_channels)):
            positive_int(name, value)
        if not aspp_rates:
            raise ValueError("aspp_rates cannot be empty.")
        for rate in aspp_rates:
            positive_int("ASPP rate", rate)
        if not isinstance(deep_supervision, bool):
            raise ValueError("deep_supervision must be boolean.")
        c = base_channels
        self.deep_supervision = deep_supervision
        self.encoder = nn.ModuleList([SEResidual(in_channels, c, stem=True),
                                      SEResidual(c, 2*c, 2), SEResidual(2*c, 4*c, 2), SEResidual(4*c, 8*c, 2)])
        self.bridge = ASPP(8*c, 16*c, aspp_rates)
        self.decoder = nn.ModuleList([AttentionDecoder(4*c, 16*c, 8*c), AttentionDecoder(2*c, 8*c, 4*c), AttentionDecoder(c, 4*c, 2*c)])
        self.output_aspp = ASPP(2*c, c, aspp_rates)
        self.head = nn.Conv2d(c, n_classes, 1)
        self.aux_heads = nn.ModuleList([nn.Conv2d(w, n_classes, 1) for w in (8*c, 4*c)]) if deep_supervision else nn.ModuleList()

    def forward(self, x):
        size = x.shape[-2:]
        # >=16 leaves enough spatial values for BatchNorm with a single image.
        padded = tuple(max(16, ((v + 7) // 8) * 8) for v in size)
        x = F.pad(x, (0, padded[1] - size[1], 0, padded[0] - size[0]))
        skips, auxiliary = [], []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = self.bridge(x)
        for index, (block, skip) in enumerate(zip(self.decoder, reversed(skips[:-1]))):
            x = block(x, skip)
            if self.deep_supervision and self.training and index < 2:
                auxiliary.append(F.interpolate(self.aux_heads[index](x), size=padded, mode="bilinear", align_corners=False)[..., :size[0], :size[1]])
        logits = self.head(self.output_aspp(x))[..., :size[0], :size[1]]
        return {"logits": logits, "aux_logits": auxiliary} if auxiliary else logits
