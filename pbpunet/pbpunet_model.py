import torch.nn.functional as F
from .layers import *

class PBPUNet(nn.Module):
    def __init__(self, n_channels, n_classes, lpf_size=None):
        super(PBPUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        if (lpf_size == None):
            self.inc = DoubleConv(n_channels, 64, None)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 512)

            self.up1 = Up(1024, 256)
            self.up2 = Up(512, 128)
            self.up3 = Up(256, 64)
            self.up4 = Up(128, 64)
            self.outc = OutConv(64, n_classes)
        else:
            print("Pyramidal BlurPooling: [" + ', '.join(map(str, lpf_size)) + "]")
            self.inc = DoubleConv(n_channels, 64, None)
            self.down1 = Down_lpf(64, 128, lpf_size=lpf_size[0])
            self.down2 = Down_lpf(128, 256, lpf_size=lpf_size[1])
            self.down3 = Down_lpf(256, 512, lpf_size=lpf_size[2])
            self.down4 = Down_lpf(512, 512, lpf_size=lpf_size[3])

            self.up1 = Up(1024, 256)
            self.up2 = Up(512, 128)
            self.up3 = Up(256, 64)
            self.up4 = Up(128, 64)
            self.outc = OutConv(64, n_classes)

    def forward(self, x):
        input_size = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x, input_size)
        return logits
