"""
Custom Loss Functions for Industrial Anomaly Detection
Deep Learning AI403 - PUA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity Index Measure (SSIM) Loss.
    Ensures spatial structure and texture fidelity are preserved in autoencoder reconstructions.
    """
    def __init__(self, window_size: int = 11, in_channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = in_channels
        self.window = self._create_window(window_size, in_channels)

    def _gaussian(self, window_size: int, sigma: float):
        gauss = torch.Tensor([
            -(x - window_size // 2)**2 / float(2 * sigma**2)
            for x in range(window_size)
        ]).exp()
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channel: int):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return nn.Parameter(window, requires_grad=False)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        p = self.window_size // 2
        mu1 = F.conv2d(img1, self.window, padding=p, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=p, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=p, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=p, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=p, groups=self.channel) - mu1_mu2

        c1 = 0.01**2
        c2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()

class CombinedReconstructionLoss(nn.Module):
    """Combines L2 pixel loss and Structural Similarity (SSIM) loss."""
    def __init__(self, ssim_weight: float = 0.5, l2_weight: float = 0.5):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l2_weight = l2_weight
        self.ssim_loss = SSIMLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_mse = self.mse_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        return self.l2_weight * l_mse + self.ssim_weight * l_ssim

class CosineSimilarityLoss(nn.Module):
    """
    Cosine Distance Loss for Feature Distillation (Reverse Distillation RD4AD).
    Computes 1 - CosineSimilarity across multiscale feature maps.
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_features: list, teacher_features: list):
        total_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Normalize along channel dimension
            s_norm = F.normalize(s_feat, p=2, dim=1)
            t_norm = F.normalize(t_feat, p=2, dim=1)
            # 1 - cos_sim
            cos_dist = 1.0 - (s_norm * t_norm).sum(dim=1).mean()
            total_loss += cos_dist
        return total_loss / len(student_features)
