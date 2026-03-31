import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, resnet50, resnet101, efficientnet_b0

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)

        self.c5 = Conv2D(out_c*5, out_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)

        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)

        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.c5(xc)

        return y


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.a1 = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.a1(x)
        return x


class BackboneEncoder(nn.Module):
    SUPPORTED_BACKBONES = {
        'vgg19': {
            'channels': [64, 128, 256, 512, 512],
            'model_fn': vgg19
        },
        'resnet50': {
            'channels': [64, 256, 512, 1024, 2048],
            'model_fn': resnet50
        },
        'resnet101': {
            'channels': [64, 256, 512, 1024, 2048],
            'model_fn': resnet101
        }
    }

    def __init__(self, backbone='vgg19', pretrained=True):
        super().__init__()
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from: {list(self.SUPPORTED_BACKBONES.keys())}"
            )
        
        self.backbone_name = backbone
        self.channels = self.SUPPORTED_BACKBONES[backbone]['channels']
        model_fn = self.SUPPORTED_BACKBONES[backbone]['model_fn']
        
        if backbone.startswith('vgg'):
            self._init_vgg(model_fn, pretrained)
        elif backbone.startswith('resnet'):
            self._init_resnet(model_fn, pretrained)
        elif backbone.startswith('efficientnet'):
            self._init_efficientnet(model_fn, pretrained)
    
    def _init_vgg(self, model_fn, pretrained):
        network = model_fn(pretrained=pretrained)
        self.x1 = network.features[:4]
        self.x2 = network.features[4:9]
        self.x3 = network.features[9:18]
        self.x4 = network.features[18:27]
        self.x5 = network.features[27:36]
    
    def _init_resnet(self, model_fn, pretrained):
        network = model_fn(pretrained=pretrained)        
        network.conv1.stride = (1, 1)
        self.x1 = nn.Sequential(network.conv1, network.bn1, network.relu)
        self.x2 = nn.Sequential(network.maxpool, network.layer1)
        self.x3 = network.layer2
        self.x4 = network.layer3
        self.x5 = network.layer4
    
    def forward(self, x):
        x0 = x
        x1 = self.x1(x0)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        return x5, [x4, x3, x2, x1]
    
    def get_output_channels(self):
        return self.channels[-1]
    
    def get_skip_channels(self):
        return self.channels[:-1][::-1]  # reverse order for decoder


class decoder1(nn.Module):
    def __init__(self, skip_channels, aspp_out=64):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        # skip_channels = [512, 256, 128, 64] for VGG19
        # skip_channels = [1024, 512, 256, 64] for ResNet50
        self.c1 = conv_block(aspp_out + skip_channels[0], 256)
        self.c2 = conv_block(256 + skip_channels[1], 128)
        self.c3 = conv_block(128 + skip_channels[2], 64)
        self.c4 = conv_block(64 + skip_channels[3], 32)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, s4], axis=1)
        x = self.c4(x)

        return x


class encoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]


class decoder2(nn.Module):
    def __init__(self, skip1_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        # adjustment for diff backbone skip channels
        # skip1_channels = [512, 256, 128, 64] for VGG19
        # skip2_channels = [256, 128, 64, 32] (always from encoder2)
        self.c1 = conv_block(64 + skip1_channels[0] + 256, 256)
        self.c2 = conv_block(256 + skip1_channels[1] + 128, 128)
        self.c3 = conv_block(128 + skip1_channels[2] + 64, 64)
        self.c4 = conv_block(64 + skip1_channels[3] + 32, 32)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x


class DoubleUNet(nn.Module):
    def __init__(self, backbone='vgg19', pretrained=True, n_classes=1):
        super().__init__()
        
        self.backbone_name = backbone
        self.n_classes = n_classes
        
        # encoder 1 with selectable backbone
        self.e1 = BackboneEncoder(backbone=backbone, pretrained=pretrained)
        bottleneck_channels = self.e1.get_output_channels()
        skip_channels = self.e1.get_skip_channels()
        
        self.a1 = ASPP(bottleneck_channels, 64)
        self.d1 = decoder1(skip_channels, aspp_out=64)
        self.y1 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # encoder 2 (fixed architecture)
        self.e2 = encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = decoder2(skip_channels)
        self.y2 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x0 = x
        
        # 1st unet
        x, skip1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        # Multiply input with first output
        input_x = x0 * self.sigmoid(y1)
        
        # 2nd unet
        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)
        
        # final output (element-wise maximum)
        final_output = torch.maximum(y1, y2)
        
        return final_output
    
    def get_architecture_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.backbone_name,
            'n_classes': self.n_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'bottleneck_channels': self.e1.get_output_channels(),
            'skip_channels': self.e1.get_skip_channels()
        }


if __name__ == "__main__":
    batch_size = 2
    
    # test all backbones
    backbones = ['vgg19', 'resnet50', 'resnet101']
    
    for backbone in backbones:
        print(f"\n{'='*50}")
        print(f"Testing {backbone}")
        print(f"{'='*50}")
        
        x = torch.randn((batch_size, 3, 224, 224))
        model = DoubleUNet(backbone=backbone, pretrained=False, n_classes=1)
        output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        info = model.get_architecture_info()
        print(f"\nArchitecture info:")
        for key, value in info.items():
            print(f"  {key}: {value}")