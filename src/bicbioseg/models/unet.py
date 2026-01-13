import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    # convolution -> [BN] -> ReLU) * 2

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    # downscaling with maxpool then double conv

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    # upscaling then double conv

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    MAX_DECODER_BLOCKS = 8
    
    def __init__(
        self, 
        n_channels=3, 
        n_classes=1, 
        num_decoder_blocks=4,
        base_channels=16,
        bilinear=False,
        image_size=None
    ):
        super(UNet, self).__init__()
        
        # decoder block validation
        if num_decoder_blocks < 1:
            raise ValueError(f"num_decoder_blocks must be >= 1, got {num_decoder_blocks}")
        
        if num_decoder_blocks > self.MAX_DECODER_BLOCKS:
            raise ValueError(
                f"num_decoder_blocks cannot exceed {self.MAX_DECODER_BLOCKS}, got {num_decoder_blocks}"
            )
        
        if image_size is not None:
            self._validate_image_size(image_size, num_decoder_blocks)
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_decoder_blocks = num_decoder_blocks
        self.base_channels = base_channels
        self.bilinear = bilinear
        
        # build encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.encoders = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(num_decoder_blocks):
            out_ch = in_ch * 2
            self.encoders.append(Down(in_ch, out_ch))
            in_ch = out_ch
        
        # build decoder
        self.decoders = nn.ModuleList()
        
        # Track encoder output channels for skip connections
        encoder_channels = [base_channels]
        ch = base_channels
        for i in range(num_decoder_blocks):
            ch = ch * 2
            encoder_channels.append(ch)
        
        # Build decoders in reverse
        for i in range(num_decoder_blocks):
            # Current level output channels from encoder (for skip connection)
            skip_channels = encoder_channels[-(i+2)]  # -2 because we skip bottleneck
            
            # Input channels to decoder
            if bilinear:
                # Bilinear: upsampled channels = current channels (no reduction during upsample)
                # Total input = upsampled + skip
                in_ch_decoder = in_ch + skip_channels
            else:
                # Transposed conv: upsampled channels = in_ch // 2
                # Total input = (in_ch // 2) + skip
                in_ch_decoder = (in_ch // 2) + skip_channels
            
            # Output channels
            if i == num_decoder_blocks - 1:
                out_ch = base_channels
            else:
                out_ch = skip_channels
            
            self.decoders.append(Up(in_ch_decoder, out_ch, bilinear))
            in_ch = out_ch
        
        # out conv
        self.outc = OutConv(base_channels, n_classes)
    
    @staticmethod
    def _validate_image_size(image_size, num_decoder_blocks):
        """Validate that image size is sufficient for the number of decoder blocks."""
        min_size = image_size if isinstance(image_size, int) else min(image_size)
        min_required = 2 ** num_decoder_blocks
        
        if min_size < min_required:
            max_blocks = int(math.log2(min_size))
            raise ValueError(
                f"num_decoder_blocks={num_decoder_blocks} requires image_size >= {min_required}. "
                f"For image_size={image_size}, maximum num_decoder_blocks={max_blocks}"
            )
    
    @staticmethod
    def calculate_max_decoder_blocks(image_size):
        min_size = image_size if isinstance(image_size, int) else min(image_size)
        theoretical_max = int(math.log2(min_size))
        return min(theoretical_max, UNet.MAX_DECODER_BLOCKS)
    
    def forward(self, x):
        # skip storage
        skip_connections = []
        
        x = self.inc(x)
        skip_connections.append(x)
        
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        
        # bottleneck is last in skip_connections
        x = skip_connections.pop()
        
        for decoder in self.decoders:
            skip = skip_connections.pop()
            x = decoder(x, skip)
        
        # final out
        output = self.outc(x)
        return output
    
    def get_architecture_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # channel size at each level 
        channels = [self.base_channels]
        current_ch = self.base_channels
        
        for i in range(self.num_decoder_blocks):
            current_ch *= 2
            channels.append(current_ch)
        
        return {
            'num_decoder_blocks': self.num_decoder_blocks,
            'base_channels': self.base_channels,
            'input_channels': self.n_channels,
            'output_classes': self.n_classes,
            'bilinear': self.bilinear,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'channel_progression': channels,
        }

if __name__ == "__main__":
    x = torch.randn((2, 3, 224, 224))
    model = UNet(n_channels=3, n_classes=1, base_channels=16, num_decoder_blocks=6, bilinear=True, image_size=224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    info = model.get_architecture_info()
    print(f"Architecture info:")
    for key, value in info.items():
        print(f"  {key}: {value}")