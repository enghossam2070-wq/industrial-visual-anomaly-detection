"""
Baseline Reference Model: Shallow Convolutional Autoencoder
Deep Learning AI403 - PUA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaselineAutoencoder(nn.Module):
    """
    Shallow 3-layer Convolutional Autoencoder serving as the benchmark baseline.
    Trained strictly on defect-free samples; anomaly score is the L2 pixel reconstruction error.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1), # -> 28x28
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1), # -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # -> 224x224
            nn.Tanh() # normalized input matching
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

    def get_anomaly_map_and_score(self, x: torch.Tensor):
        """
        Calculates pixel-level anomaly heatmap and sample-level anomaly score.
        Score = mean of top 1% largest pixel reconstruction residuals.
        """
        self.eval()
        with torch.no_grad():
            x_rec = self.forward(x)
            
            # Pixel-wise squared reconstruction error across RGB channels: (B, H, W)
            diff = (x - x_rec) ** 2
            anomaly_map = diff.mean(dim=1) # (B, H, W)
            
            # Score: top-k or 99th percentile residual error
            B, H, W = anomaly_map.shape
            scores = []
            for b in range(B):
                flat_map = anomaly_map[b].view(-1)
                k = max(1, int(0.01 * H * W))
                top_k, _ = torch.topk(flat_map, k)
                scores.append(top_k.mean().item())
                
            return anomaly_map.cpu().numpy(), np.array(scores)
