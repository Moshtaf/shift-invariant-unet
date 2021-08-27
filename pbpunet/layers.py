import torch
import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        
        padding_size = 1
        
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding_size),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling with the MaxPool followed by the double convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, None)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down_lpf(nn.Module):
    """Downsampling with the MaxBlurPool followed by the double convolution"""
    def __init__(self, in_channels, out_channels, lpf_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            antialiased_cnns.BlurPool(in_channels, filt_size=lpf_size, stride=2),
            DoubleConv(in_channels, out_channels, None)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling followed by the double convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Replacing the transposed convolutions in the original U-Net 
        # with bilinear upsampling layers in favor of memory efficiency.
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = abs(x2.size()[2] - x1.size()[2])
        diffX = abs(x2.size()[3] - x1.size()[3])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, input_size):
        output = self.conv(x)

        h, w = x.size()[2:4]
        new_h, new_w = input_size[2:4]
        top = int(round((h - new_h) / 2.))
        left = int(round((w - new_w) / 2.))
        output_cropped = output[:,:,top: top + new_h, left: left + new_w]

        return output_cropped
