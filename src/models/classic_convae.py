"""
Classic Deep Learning Model (Category 1): Deep Convolutional Autoencoder (ConvAE)
Deep Learning AI403 - PUA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class DeepConvAutoencoder(nn.Module):
    """
    Classic Deep Learning Approach: 5-Stage Deep Convolutional Autoencoder.
    Utilizes multi-scale perceptual features and is optimized via combined SSIM + L2 loss.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)          # 224x224
        self.pool1 = nn.MaxPool2d(2, 2)                           # -> 112x112
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)    # 112x112
        self.pool2 = nn.MaxPool2d(2, 2)                           # -> 56x56
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)# 56x56
        self.pool3 = nn.MaxPool2d(2, 2)                           # -> 28x28
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)# 28x28
        self.pool4 = nn.MaxPool2d(2, 2)                           # -> 14x14
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2) # -> 28x28
        self.dec4 = ConvBlock(base_channels * 4, base_channels * 4)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2) # -> 56x56
        self.dec3 = ConvBlock(base_channels * 2, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)     # -> 112x112
        self.dec2 = ConvBlock(base_channels, base_channels)
        
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)         # -> 224x224
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor):
        # Forward pass
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.dec4(self.up4(b))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        out = self.dec1(self.up1(d2))
        return out

    def get_anomaly_map_and_score(self, x: torch.Tensor, gaussian_sigma: float = 4.0):
        """
        Calculates smoothed anomaly map using L1/L2 reconstruction discrepancy and Gaussian post-filtering.
        """
        self.eval()
        with torch.no_grad():
            x_rec = self.forward(x)
            diff = torch.abs(x - x_rec).mean(dim=1) # (B, H, W)
            
            B, H, W = diff.shape
            anomaly_maps = []
            anomaly_scores = []
            
            for b in range(B):
                raw_map = diff[b].cpu().numpy()
                # Apply Gaussian smoothing to reduce high-frequency pixel noise
                smooth_map = gaussian_filter(raw_map, sigma=gaussian_sigma)
                anomaly_maps.append(smooth_map)
                
                # Top 1% pixel mean as image anomaly score
                flat = smooth_map.flatten()
                k = max(1, int(0.01 * len(flat)))
                topk_val = np.partition(flat, -k)[-k:].mean()
                anomaly_scores.append(float(topk_val))
                
            return np.array(anomaly_maps), np.array(anomaly_scores)
