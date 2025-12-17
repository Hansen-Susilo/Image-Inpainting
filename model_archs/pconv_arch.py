import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        with torch.no_grad():
            mask_out = self.mask_conv(mask)
        out = self.input_conv(x * mask)
        eps = 1e-8
        out = out / (mask_out + eps)
        mask_out = torch.clamp(mask_out, 0, 1)
        return out, mask_out

class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pconv = PartialConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        x = self.relu(x)
        return x, mask

class PConvUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = PConvBlock(3, 64)
        self.down2 = PConvBlock(64, 128)
        self.down3 = PConvBlock(128, 256)
        self.bottleneck = PConvBlock(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = PConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = PConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = PConvBlock(128, 64)
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x, mask):
        d1, m1 = self.down1(x, mask)
        d2, m2 = self.down2(F.interpolate(d1, scale_factor=0.5), F.interpolate(m1, scale_factor=0.5))
        d3, m3 = self.down3(F.interpolate(d2, scale_factor=0.5), F.interpolate(m2, scale_factor=0.5))
        bn, bm = self.bottleneck(F.interpolate(d3, scale_factor=0.5), F.interpolate(m3, scale_factor=0.5))
        u3 = self.up3(bn)
        u3, m3u = self.conv3(torch.cat([u3, d3], dim=1), F.interpolate(bm, scale_factor=2))
        u2 = self.up2(u3)
        u2, m2u = self.conv2(torch.cat([u2, d2], dim=1), F.interpolate(m3u, scale_factor=2))
        u1 = self.up1(u2)
        u1, _ = self.conv1(torch.cat([u1, d1], dim=1), F.interpolate(m2u, scale_factor=2))
        return torch.sigmoid(self.final(u1))