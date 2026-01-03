import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)

        # Initialize mask weights to 1 and freeze them
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        # 1. Update mask: if any part of the kernel touches a valid pixel (1), 
        # the output pixel becomes valid (1).
        with torch.no_grad():
            mask_out = self.mask_conv(mask)

        # 2. Calculate scaling factor (sum of all weights / sum of valid weights)
        # This prevents the signal from vanishing in holes.
        kernel_vol = self.mask_conv.weight.shape[2] * self.mask_conv.weight.shape[3]
        mask_ratio = kernel_vol / (mask_out + 1e-8)
        
        # Only scale where the mask is > 0
        mask_ratio = mask_ratio * (mask_out > 0).float()

        # 3. Apply Convolution to masked input
        raw_out = self.input_conv(x * mask)
        
        # 4. Apply scaling and bias
        if self.input_conv.bias is not None:
            bias_view = self.input_conv.bias.view(1, -1, 1, 1)
            out = (raw_out - bias_view) * mask_ratio + bias_view
            # Keep bias term where mask is 0
            out = out * (mask_out > 0).float()
        else:
            out = raw_out * mask_ratio

        # 5. Finalize new mask for next layer
        updated_mask = torch.clamp(mask_out, 0, 1)
        
        return out, updated_mask

# 4. PCONV BLOCK
class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        # Every Conv layer in UNet becomes a PConv layer
        self.pconv1 = PartialConv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.pconv2 = PartialConv2d(out_ch, out_ch, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv1(x, mask)
        x = self.relu(x)
        x, mask = self.pconv2(x, mask)
        x = self.relu(x)
        return x, mask

class PConvUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = PConvBlock(3, 64)
        self.pool = nn.MaxPool2d(2) # Pooling doesn't change mask logic significantly

        self.down2 = PConvBlock(64, 128)
        self.down3 = PConvBlock(128, 256)
        self.bottleneck = PConvBlock(256, 512)

        # Decoder (Using Upsample instead of ConvTranspose to keep mask alignment simple)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # decoder blocks (note the concat channel increase)
        self.dec3 = PConvBlock(512 + 256, 256)
        self.dec2 = PConvBlock(256 + 128, 128)
        self.dec1 = PConvBlock(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x, mask):
        # Encoder
        d1, m1 = self.down1(x, mask)
        p1 = self.pool(d1)
        pm1 = self.pool(m1)

        d2, m2 = self.down2(p1, pm1)
        p2 = self.pool(d2)
        pm2 = self.pool(m2)

        d3, m3 = self.down3(p2, pm2)
        p3 = self.pool(d3)
        pm3 = self.pool(m3)

        bn, bm = self.bottleneck(p3, pm3)

        # Decoder
        u3 = self.up(bn)
        um3 = self.up(bm)
        u3 = torch.cat([u3, d3], dim=1)
        um3 = torch.cat([um3, m3], dim=1) # Note: we take the mask from the encoder too
        # To handle the 2-channel mask after concat, we take the max (if either is 1, it's 1)
        um3 = torch.max(um3[:, :1, :, :], um3[:, 1:, :, :]) 
        
        u3, um3 = self.dec3(u3, um3)

        u2 = self.up(u3)
        um2 = self.up(um3)
        u2 = torch.cat([u2, d2], dim=1)
        um2 = torch.max(um2, m2) # Simplified mask merge
        u2, um2 = self.dec2(u2, um2)

        u1 = self.up(u2)
        um1 = self.up(um2)
        u1 = torch.cat([u1, d1], dim=1)
        um1 = torch.max(um1, m1)
        u1, _ = self.dec1(u1, um1)

        return torch.sigmoid(self.final(u1))
