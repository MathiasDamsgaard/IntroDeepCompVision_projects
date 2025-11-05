import torch
import torch.nn.functional as F
from torch import nn


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(512, 1024, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(1024 + 512, 512, 3, padding=1)  # concatenated channels
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(512 + 256, 256, 3, padding=1)  # concatenated channels
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(256 + 128, 128, 3, padding=1)  # concatenated channels
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128 + 64, 1, 3, padding=1)  # concatenated channels

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        p0 = self.pool0(e0)
        e1 = F.relu(self.enc_conv1(p0))
        p1 = self.pool1(e1)
        e2 = F.relu(self.enc_conv2(p1))
        p2 = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(p2))
        p3 = self.pool3(e3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(p3))

        # decoder with skip connections
        d0 = F.relu(self.dec_conv0(torch.cat([self.upsample0(b), e3], dim=1)))
        d1 = F.relu(self.dec_conv1(torch.cat([self.upsample1(d0), e2], dim=1)))
        d2 = F.relu(self.dec_conv2(torch.cat([self.upsample2(d1), e1], dim=1)))
        return self.dec_conv3(torch.cat([self.upsample3(d2), e0], dim=1))  # no activation
