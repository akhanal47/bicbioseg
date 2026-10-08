'''
Adapted from MIT Han Lab EfficientViT:
https://github.com/mit-han-lab/efficientvit
Notice and Licenses: licenses/NOTICE.txt and licenses/Apache-2.0.txt.
'''

from torch import nn
from torch.nn import functional as F

from ._common import positive_int
from .transformer_segmentation import _EncoderSegmentation, _timm


def _projection(in_channels, out_channels, activation=False):
    layers = [nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)]
    if activation:
        layers.append(nn.Hardswish())
    return nn.Sequential(*layers)


class _EfficientViTHead(nn.Module):
    def __init__(self, feature_channels, width, depth, n_classes, dropout):
        super().__init__()
        from timm.models.efficientvit_mit import MBConv

        self.projections = nn.ModuleList(_projection(c, width) for c in feature_channels)
        self.blocks = nn.ModuleList(
            MBConv(width, width, expand_ratio=4, act_layer=(nn.Hardswish, nn.Hardswish, None))
            for _ in range(depth)
        )
        self.head = nn.Sequential(
            _projection(width, 4 * width, activation=True),
            nn.Dropout2d(dropout), nn.Conv2d(4 * width, n_classes, 1),
        )

    def forward(self, features):
        # Project and add the 1/8, 1/16 and 1/32 maps on the 1/8 grid.
        size = features[0].shape[-2:]
        maps = [F.interpolate(layer(t), size=size, mode="bilinear", align_corners=False)
                for layer, t in zip(self.projections, features)]
        x = maps[0] + maps[1] + maps[2]
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


class EfficientViTSegmenter(_EncoderSegmentation):
    
    # MIT EfficientViT B0–B3 with the B-series Cityscapes-style segmentation head.

    def __init__(self, in_channels=3, n_classes=1, variant="b0", pretrained=False,
                 decoder_channels=None, decoder_depth=None, dropout=0.0,
                 freeze_encoder=False, normalize_input=True):
        super().__init__()
        self._configure(in_channels, n_classes, pretrained, freeze_encoder, normalize_input, dropout)
        presets = {"b0": (32, 1), "b1": (64, 3), "b2": (96, 3), "b3": (128, 3)}
        if variant not in presets:
            raise ValueError("EfficientViT variant must be 'b0', 'b1', 'b2', or 'b3'.")
        width, depth = presets[variant]
        width = width if decoder_channels is None else positive_int("decoder_channels", decoder_channels)
        depth = depth if decoder_depth is None else positive_int("decoder_depth", decoder_depth)
        self.encoder = _timm().create_model(
            f"efficientvit_{variant}.r224_in1k", pretrained=pretrained,
            features_only=True, out_indices=(1, 2, 3),
        )
        self.decoder = _EfficientViTHead(self.encoder.feature_info.channels(), width, depth, n_classes, dropout)
        self._freeze()

    def forward(self, x):
        size = x.shape[-2:]
        x = self.input_transform(x)
        x = F.pad(x, (0, -size[1] % 32, 0, -size[0] % 32))
        logits = self.decoder(self.encoder(x))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)[..., :size[0], :size[1]]
