import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from ._common import ImageNetInput, positive_int, spatial_size
from einops import rearrange, repeat

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) / self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, dropout=0.1):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, dropout=0.1):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim, dropout)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12, dropout=0.1):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim, dropout) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1, dropout=0.1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.grid_size = tuple(d // patch_dim for d in spatial_size(img_dim))
        self.num_tokens = self.grid_size[0] * self.grid_size[1]
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num, dropout)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        grid = (x.shape[-2] // self.patch_dim, x.shape[-1] // self.patch_dim)
        position = self.embedding
        if grid != self.grid_size:
            spatial = position[1:].T.reshape(1, -1, *self.grid_size)
            spatial = F.interpolate(spatial, size=grid, mode="bicubic", align_corners=False)
            position = torch.cat((position[:1], spatial.flatten(2).squeeze(0).T), dim=0)
        patches = patches + position

        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = (F.interpolate(x, size=x_concat.shape[-2:], mode="bilinear", align_corners=True)
             if x_concat is not None else self.upsample(x))

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim,
                 embedding_dim=None, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = tuple(d // patch_dim for d in spatial_size(img_dim))
        embedding_dim = embedding_dim or out_channels * 8
        self.vit = ViT(self.vit_img_dim, out_channels * 8, embedding_dim,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False, dropout=dropout)

        self.conv2 = nn.Conv2d(embedding_dim, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels * 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        height, width = x.shape[-2:]
        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=height, y=width)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3


class ResNetEncoder(nn.Module):

    def __init__(self, img_dim, out_channels, embedding_dim, head_num, mlp_dim,
                 block_num, dropout, encoder_weights):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        weights = None if encoder_weights is None else ResNet50_Weights[encoder_weights]
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.vit = ViT(tuple(d // 16 for d in spatial_size(img_dim)), 1024, embedding_dim,
                       head_num, mlp_dim, block_num, 1, classification=False, dropout=dropout)
        self.projection = nn.Sequential(nn.Conv2d(embedding_dim, out_channels * 4, 3, padding=1),
                                        nn.BatchNorm2d(out_channels * 4), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.layer1(self.pool(x1))
        x3 = self.layer2(x2)
        x = self.layer3(x3)
        height, width = x.shape[-2:]
        x = self.vit(x).transpose(1, 2).reshape(x.shape[0], -1, height, width)
        return self.projection(x), x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num, decoder_channels=None, skip_channels=None, n_skip=3):
        super().__init__()
        widths = decoder_channels or (out_channels * 2, out_channels, out_channels // 2, out_channels // 8)
        skips = skip_channels or (out_channels, out_channels * 2, out_channels * 4)
        self.n_skip = n_skip
        skip_widths = [c if i < n_skip else 0 for i, c in enumerate(reversed(skips))]
        self.decoder1 = DecoderBottleneck(out_channels * 4 + skip_widths[0], widths[0])
        self.decoder2 = DecoderBottleneck(widths[0] + skip_widths[1], widths[1])
        self.decoder3 = DecoderBottleneck(widths[1] + skip_widths[2], widths[2])
        self.decoder4 = DecoderBottleneck(widths[2], widths[3])
        self.conv1 = nn.Conv2d(widths[3], class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3 if self.n_skip >= 1 else None)
        x = self.decoder2(x, x2 if self.n_skip >= 2 else None)
        x = self.decoder3(x, x1 if self.n_skip >= 3 else None)
        return self.conv1(self.decoder4(x))


class TransUNet(nn.Module):
    PRESETS = {
        'lightweight': {
            'out_channels': 64,
            'head_num': 4,
            'mlp_dim': 512,
            'block_num': 6,
            'patch_dim': 16
        },
        'standard': {
            'out_channels': 128,
            'head_num': 4,
            'mlp_dim': 512,
            'block_num': 8,
            'patch_dim': 16
        },
        'heavy': {
            'out_channels': 256,
            'head_num': 8,
            'mlp_dim': 1024,
            'block_num': 12,
            'patch_dim': 16
        }
    }

    def __init__(
        self, img_dim=224, in_channels=3, n_classes=1, preset='standard',
        out_channels=None, head_num=None, mlp_dim=None, block_num=None,
        patch_dim=None, pretrained_vit=False, *, encoder_name="custom",
        encoder_weights=None, embedding_dim=None, decoder_channels=None,
        dropout=0.1, n_skip=3, normalize_input=None,
    ):
        """Configurable TransUNet variant returning full-resolution logits.

        ``resnet50`` accepts torchvision ImageNet CNN weights. The transformer
        and decoder always start from scratch; official R50+ViT NPZ files are
        not compatible. ``patch_dim`` describes the fixed CNN stride (16).
        """
        
        super().__init__()
        if pretrained_vit:
            raise NotImplementedError("Pretrained ViT weights are not implemented; use pretrained_vit=False.")
        if preset is not None and preset not in self.PRESETS:
            raise ValueError(f"Unknown TransUNet preset: {preset}")
        defaults = self.PRESETS[preset or "standard"]
        values = dict(out_channels=out_channels, head_num=head_num, mlp_dim=mlp_dim,
                      block_num=block_num, patch_dim=patch_dim)
        for name, value in values.items():
            values[name] = positive_int(name, defaults[name] if value is None else value)
            setattr(self, name, values[name])
        out_channels, head_num, mlp_dim, block_num, patch_dim = values.values()
        embedding_dim = out_channels * 8 if embedding_dim is None else positive_int("embedding_dim", embedding_dim)
        if encoder_name not in ("custom", "resnet50"):
            raise ValueError("encoder_name must be 'custom' or 'resnet50'.")
        if encoder_weights not in (None, "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"):
            raise ValueError("encoder_weights must be None, 'DEFAULT', 'IMAGENET1K_V1', or 'IMAGENET1K_V2'.")
        if encoder_name == "custom" and encoder_weights is not None:
            raise ValueError("encoder_weights requires encoder_name='resnet50'.")
        if patch_dim != 16:
            raise ValueError("TransUNet's encoder has fixed stride 16; patch_dim must be 16.")
        if out_channels < 8 or embedding_dim % head_num:
            raise ValueError("out_channels must be >= 8 and embedding_dim must be divisible by head_num.")
        size = spatial_size(img_dim)
        if any(d % 16 for d in size):
            raise ValueError("img_dim dimensions must be divisible by 16.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if isinstance(n_skip, bool) or not isinstance(n_skip, int) or not 0 <= n_skip <= 3:
            raise ValueError("n_skip must be an integer from 0 to 3.")
        if decoder_channels is not None:
            decoder_channels = tuple(decoder_channels)
            if len(decoder_channels) != 4:
                raise ValueError("decoder_channels must have four positive widths.")
            for width in decoder_channels:
                positive_int("decoder channel", width)
        self.img_dim = img_dim
        self.in_channels = positive_int("in_channels", in_channels)
        self.n_classes = positive_int("n_classes", n_classes)
        self.pretrained_vit = False
        self.encoder_name, self.encoder_weights = encoder_name, encoder_weights
        self.embedding_dim, self.dropout, self.n_skip = embedding_dim, dropout, n_skip
        self.decoder_channels = decoder_channels or (out_channels * 2, out_channels, out_channels // 2, out_channels // 8)
        self.normalize_input = encoder_name == "resnet50" if normalize_input is None else normalize_input
        if self.normalize_input or encoder_name == "resnet50":
            self.input_transform = ImageNetInput(in_channels, self.normalize_input)
        else:
            self.input_transform = nn.Identity()
        if encoder_name == "resnet50":
            self.encoder = ResNetEncoder(size, out_channels, embedding_dim, head_num, mlp_dim,
                                         block_num, dropout, encoder_weights)
            skips = (64, 256, 512)
        else:
            self.encoder = Encoder(size, 3 if self.normalize_input else in_channels, out_channels, head_num, mlp_dim, block_num,
                                   patch_dim, embedding_dim, dropout)
            skips = None
        self.decoder = Decoder(out_channels, n_classes, self.decoder_channels, skips, n_skip)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected NCHW input with {self.in_channels} channels.")
        if any(d < 16 or d % 16 for d in x.shape[-2:]):
            raise ValueError("TransUNet input height and width must be divisible by 16.")
        x, x1, x2, x3 = self.encoder(self.input_transform(x))
        return self.decoder(x, x1, x2, x3)

    def get_architecture_info(self):
        names = ("img_dim", "in_channels", "n_classes", "out_channels", "head_num", "mlp_dim",
                 "block_num", "patch_dim", "pretrained_vit", "encoder_name", "encoder_weights",
                 "embedding_dim", "decoder_channels", "dropout", "n_skip", "normalize_input")
        return {**{name: getattr(self, name) for name in names},
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}
