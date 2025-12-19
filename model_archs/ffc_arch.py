import torch
import torch.nn as nn

class SpectralTransform(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # We process in the spectral domain. 
        # rfft2 produces (h, w//2 + 1) complex numbers.
        # We treat real and imaginary as separate channels (2 * in_ch).
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch * 2, 1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch * 2, out_ch * 2, 1)
        )

    def forward(self, x):
        batch, c, h, w = x.shape
        ffted = torch.fft.rfft2(x, norm='ortho')
        
        # Concatenate real and imaginary parts
        ffted_cat = torch.cat([ffted.real, ffted.imag], dim=1)
        
        # Transform in spectral domain
        out = self.conv(ffted_cat)
        
        # Split back to complex
        real, imag = torch.chunk(out, 2, dim=1)
        ffted_out = torch.complex(real, imag)
        
        # Inverse FFT back to spatial domain
        # Explicitly set output size to match input spatial dimensions
        output = torch.fft.irfft2(ffted_out, s=(h, w), norm='ortho')
        return output

class FFCBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Split input channels for local and global branches
        self.l_in = in_ch // 2
        self.g_in = in_ch - self.l_in
        
        # Split output channels
        self.l_out = out_ch // 2
        self.g_out = out_ch - self.l_out
        
        self.local_branch = nn.Conv2d(self.l_in, self.l_out, 3, padding=1)
        self.global_branch = SpectralTransform(self.g_in, self.g_out)
        
        # Fusion layer expects exactly out_ch (l_out + g_out)
        self.fusion = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x):
        l, g = torch.split(x, [self.l_in, self.g_in], dim=1)
        
        l_out = self.local_branch(l)
        g_out = self.global_branch(g)
        
        # Concatenate along channel dimension
        combined = torch.cat([l_out, g_out], dim=1)
        return self.fusion(combined)


class FFCUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # Downsampling
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        
        # FFC Bottleneck (Crucial for large holes)
        self.bottleneck1 = FFCBlock(128, 256)
        self.bottleneck2 = FFCBlock(256, 128)
        
        # Upsampling
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU())
        
        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        
        x = self.up2(x)
        x = self.up1(x)
        return torch.sigmoid(self.final(x))