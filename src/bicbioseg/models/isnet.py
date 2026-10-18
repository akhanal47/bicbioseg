'''IS-Net DIS segmentation network from Qin et al., ECCV 2022 (Apache-2.0).

Paper: https://github.com/xuebinqin/DIS/blob/main/IS-Net/models/isnet.py
See licenses/NOTICE.txt and licenses/Apache-2.0.txt.

This is not a full implementation but rather a simple one

Includes: shared RSU implementation, input validation, optional input centering, and logits with bicbioseg's auxiliary output contract. 
Parameter names and shapes match upstream ISNetDIS for RGB binary segmentation. 

Not Inlcuded: The separate ground-truth encoder and intermediate feature supervision training procedure are not included.
'''

import torch
from torch import nn
from torch.nn import functional as F

from ._common import positive_int


def _upsample_like(source, target):
    return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class _RSU(nn.Module):
    '''Residual U-block with upstream names for strict checkpoint loading.'''

    def __init__(self, depth, in_ch, mid_ch, out_ch):
        super().__init__()
        self.depth = depth
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        for level in range(1, depth):
            setattr(self, f"rebnconv{level}", REBNCONV(out_ch if level == 1 else mid_ch, mid_ch))
            if level < depth - 1:
                setattr(self, f"pool{level}", nn.MaxPool2d(2, stride=2, ceil_mode=True))
        setattr(self, f"rebnconv{depth}", REBNCONV(mid_ch, mid_ch, dirate=2))
        for level in range(depth - 1, 0, -1):
            setattr(self, f"rebnconv{level}d", REBNCONV(2 * mid_ch, out_ch if level == 1 else mid_ch))

    def forward(self, x):
        residual = self.rebnconvin(x)
        x = residual
        skips = []
        for level in range(1, self.depth):
            x = getattr(self, f"rebnconv{level}")(x)
            skips.append(x)
            if level < self.depth - 1:
                x = getattr(self, f"pool{level}")(x)
        x = getattr(self, f"rebnconv{self.depth}")(x)
        for level in range(self.depth - 1, 0, -1):
            skip = skips[level - 1]
            if level < self.depth - 1:
                x = _upsample_like(x, skip)
            x = getattr(self, f"rebnconv{level}d")(torch.cat((x, skip), dim=1))
        return x + residual


class RSU4F(nn.Module):
    '''Residual U-block using dilated convolutions without pooling.'''

    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(2 * mid_ch, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(2 * mid_ch, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(2 * mid_ch, out_ch)

    def forward(self, x):
        residual = self.rebnconvin(x)
        x1 = self.rebnconv1(residual)
        x2 = self.rebnconv2(x1)
        x3 = self.rebnconv3(x2)
        x4 = self.rebnconv4(x3)
        x3 = self.rebnconv3d(torch.cat((x4, x3), dim=1))
        x2 = self.rebnconv2d(torch.cat((x3, x2), dim=1))
        return self.rebnconv1d(torch.cat((x2, x1), dim=1)) + residual


class ISNetDIS(nn.Module):
    '''Full IS-Net segmentation backbone with optional five-head deep supervision.

    Inputs and outputs use NCHW geometry. Training with deep supervision returns
    ``{"logits": d1, "aux_logits": [d2, d3, d4, d5, d6]}``; otherwise only d1 is
    returned. All predictions are raw logits at the input spatial size.
    ``normalize_input=True`` centers [0, 1] inputs by subtracting 0.5, as required
    by the authors' pretrained DIS weights. No weights are downloaded.
    '''

    def __init__(self, in_channels=3, n_classes=1, deep_supervision=False, normalize_input=False):
        super().__init__()
        positive_int("in_channels", in_channels)
        positive_int("n_classes", n_classes)
        for name, value in (("deep_supervision", deep_supervision), ("normalize_input", normalize_input)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean.")
        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        self.normalize_input = normalize_input

        self.conv_in = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.stage1 = _RSU(7, 64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = _RSU(6, 64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = _RSU(5, 128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = _RSU(4, 256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)

        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = _RSU(4, 1024, 128, 256)
        self.stage3d = _RSU(5, 512, 64, 128)
        self.stage2d = _RSU(6, 256, 32, 64)
        self.stage1d = _RSU(7, 128, 16, 64)
        for level, channels in enumerate((64, 64, 128, 256, 512, 512), start=1):
            setattr(self, f"side{level}", nn.Conv2d(channels, n_classes, 3, padding=1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected NCHW input with {self.in_channels} channels.")
        batch, _, height, width = x.shape
        if min(batch, height, width) < 1:
            raise ValueError("Batch size and spatial dimensions must be positive.")
        if self.training and batch * ((height + 63) // 64) * ((width + 63) // 64) <= 1:
            raise ValueError(
                "IS-Net training BatchNorm requires batch * ceil(H/64) * ceil(W/64) > 1. "
                "Increase the batch size or image size, or use eval() for inference."
            )
        image = x
        x = self.conv_in(x - 0.5 if self.normalize_input else x)
        skips = []
        for level in range(1, 7):
            x = getattr(self, f"stage{level}")(x)
            skips.append(x)
            if level < 6:
                x = getattr(self, f"pool{level}{level + 1}")(x)

        features = [x]
        for level in range(5, 0, -1):
            skip = skips[level - 1]
            x = getattr(self, f"stage{level}d")(torch.cat((_upsample_like(x, skip), skip), dim=1))
            features.append(x)
        logits = _upsample_like(self.side1(x), image)
        if self.training and self.deep_supervision:
            auxiliary = [
                _upsample_like(getattr(self, f"side{level}")(feature), image)
                for level, feature in enumerate(reversed(features[:-1]), start=2)
            ]
            return {"logits": logits, "aux_logits": auxiliary}
        return logits
