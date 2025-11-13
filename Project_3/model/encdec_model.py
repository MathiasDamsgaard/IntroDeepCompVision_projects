import torch
from torch import nn
from torch.nn import functional


class EncDec(nn.Module):
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
        self.upsample0 = nn.Upsample(scale_factor=2)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)  # 64 -> 128 (or any input size / 8)
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        e0 = self.pool0(functional.relu(self.enc_conv0(x)))
        e1 = self.pool1(functional.relu(self.enc_conv1(e0)))
        e2 = self.pool2(functional.relu(self.enc_conv2(e1)))
        e3 = self.pool3(functional.relu(self.enc_conv3(e2)))

        # bottleneck
        b = functional.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = functional.relu(self.dec_conv0(self.upsample0(b)))
        d1 = functional.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = functional.relu(self.dec_conv2(self.upsample2(d1)))
        return self.dec_conv3(self.upsample3(d2))  # no activation
