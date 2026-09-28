'''
Swin encoder and transformer decoder with learned patch expansion.
Paper: https://arxiv.org/abs/2105.05537.
'''

import torch
from torch import nn
from torch.nn import functional as F

from ._common import positive_int
from .transformer_segmentation import _EncoderSegmentation, _timm


class PatchExpand(nn.Module):
    # project each NHWC token into a scale-by-scale grid of output tokens.

    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.out_channels = out_channels
        self.projection = nn.Linear(in_channels, scale * scale * out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        batch, height, width, _ = x.shape
        s, c = self.scale, self.out_channels
        x = self.projection(x).reshape(batch, height, width, s, s, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height * s, width * s, c)
        return self.norm(x)


class SwinUNetFull(_EncoderSegmentation):
    # swin blocks on both sides of a U-shaped network.
    # ``pretrained`` initializes only the encoder from ImageNet. Decoder blocks,
    # skip projections and patch expansions are trained from scratch. Inputs are padded to multiples of 32 and logits are cropped to the original size.
    # decoder depths are ordered coarse to fine (1/16, 1/8, 1/4 resolution).


    def __init__(self, in_channels=3, n_classes=1, variant="tiny", pretrained=False,
                 decoder_depths=(2, 2, 2), dropout=0.1, freeze_encoder=False,
                 normalize_input=True, deep_supervision=False):
        super().__init__()
        if variant not in ("tiny", "small", "base"):
            raise ValueError("Swin variant must be 'tiny', 'small', or 'base'.")
        decoder_depths = tuple(decoder_depths)
        if len(decoder_depths) != 3:
            raise ValueError("decoder_depths must contain three positive depths.")
        for depth in decoder_depths:
            positive_int("decoder depth", depth)
        self._configure(in_channels, n_classes, pretrained, freeze_encoder, normalize_input, dropout)
        self.encoder = _timm().create_model(
            f"swin_{variant}_patch4_window7_224.ms_in1k", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3), strict_img_size=False,
        )
        from timm.models.swin_transformer import SwinTransformerBlock

        if not isinstance(deep_supervision, bool):
            raise ValueError("deep_supervision must be boolean.")
        self.deep_supervision = deep_supervision
        widths = self.encoder.feature_info.channels()
        self.aux_heads = nn.ModuleList([nn.Conv2d(w, n_classes, 1) for w in (widths[2], widths[1])]) if deep_supervision else nn.ModuleList()
        heads = (4, 8, 16) if variant == "base" else (3, 6, 12)
        self.bottleneck_norm = nn.LayerNorm(widths[-1])
        self.expansions = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index, depth in zip((2, 1, 0), decoder_depths):
            width = widths[index]
            self.expansions.append(PatchExpand(widths[index + 1], width))
            self.skip_fusions.append(nn.Linear(2 * width, width))
            self.decoder.append(nn.Sequential(*[
                SwinTransformerBlock(
                    dim=width, input_resolution=(224 // (4 * 2 ** index),) * 2,
                    num_heads=heads[index], window_size=7,
                    shift_size=0 if block % 2 == 0 else 3,
                    dynamic_mask=True, always_partition=True, proj_drop=dropout,
                ) for block in range(depth)
            ]))
        self.output_norm = nn.LayerNorm(widths[0])
        self.final_expand = PatchExpand(widths[0], widths[0], scale=4)
        self.head = nn.Conv2d(widths[0], n_classes, 1)
        self._freeze()

    def forward(self, x):
        height, width = x.shape[-2:]
        x = self.input_transform(x)
        x = F.pad(x, (0, -width % 32, 0, -height % 32))
        padded_size = x.shape[-2:]
        auxiliary = []
        features = self.encoder(x)
        x = self.bottleneck_norm(features[-1])
        for skip, expand, fuse, blocks in zip(
            reversed(features[:-1]), self.expansions, self.skip_fusions, self.decoder
        ):
            x = blocks(fuse(torch.cat((expand(x), skip), dim=-1)))
            if self.deep_supervision and self.training and len(auxiliary) < 2:
                aux = self.aux_heads[len(auxiliary)](x.permute(0, 3, 1, 2))
                auxiliary.append(F.interpolate(aux, size=padded_size, mode="bilinear", align_corners=False)[..., :height, :width])
        x = self.final_expand(self.output_norm(x)).permute(0, 3, 1, 2).contiguous()
        logits = self.head(x)[..., :height, :width]
        return {"logits": logits, "aux_logits": auxiliary} if auxiliary else logits
