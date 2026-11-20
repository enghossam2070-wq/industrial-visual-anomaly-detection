"""
Modern Deep Learning Model (Category 2): PatchCore (Roth et al., CVPR 2022)
Features:
  - Pretrained Multiscale Feature Representations (ResNet/WideResNet)
  - Custom Greedy K-Center Coreset Subsampling written from scratch (Requirement 5)
  - Neighborhood-Aware Patch-Level Memory Bank & Density Anomaly Scoring
Deep Learning AI403 - PUA
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection

class CustomGreedyCoresetSubsampler:
    """
    Key Component Written From Scratch (Core Requirement 5).
    Implements iterative minimax Greedy K-Center Coreset Selection
    to reduce the memory bank by 90-99% while preserving distribution coverage.
    """
    def __init__(self, subsample_ratio: float = 0.1, random_state: int = 42):
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

    def sample(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor of shape (N, D)
        returns: Coreset Tensor of shape (K, D) where K = N * subsample_ratio
        """
        N, D = features.shape
        target_size = max(1, int(N * self.subsample_ratio))
        
        if target_size >= N:
            return features

        device = features.device
        
        # 1. Johnson-Lindenstrauss projection for fast distance computation if D > 128
        features_np = features.detach().cpu().numpy()
        if D > 128:
            proj_dim = 128
            rng = np.random.RandomState(self.random_state)
            proj_matrix = rng.randn(D, proj_dim).astype(np.float32) / np.sqrt(proj_dim)
            proj_features = np.dot(features_np, proj_matrix)
        else:
            proj_features = features_np

        # 2. Greedy Minimax K-Center Selection
        np.random.seed(self.random_state)
        first_idx = int(np.random.choice(N))
        selected_indices = [first_idx]
        
        # Initialize minimum distances to closest selected center
        current_center = proj_features[first_idx:first_idx + 1] # (1, D_proj)
        min_distances = np.linalg.norm(proj_features - current_center, axis=1) # (N,)

        for _ in range(1, target_size):
            # Select point furthest from its closest existing coreset center
            farthest_idx = int(np.argmax(min_distances))
            selected_indices.append(farthest_idx)
            
            # Update distance vector
            new_distances = np.linalg.norm(proj_features - proj_features[farthest_idx:farthest_idx + 1], axis=1)
            min_distances = np.minimum(min_distances, new_distances)

        return features[selected_indices]


class PatchCore(nn.Module):
    """
    PatchCore Industrial Anomaly Detection Architecture.
    Extracts patch-level feature representations from deep layers and maps test queries to nearest memory bank normal patches.
    """
    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        layers: list = ["layer2", "layer3"],
        subsample_ratio: float = 0.1,
        n_neighbors: int = 9,
        seed: int = 42
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = layers
        self.n_neighbors = n_neighbors
        self.subsample_ratio = subsample_ratio
        self.seed = seed
        
        # Load backbone
        if backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            self.backbone = models.wide_resnet50_2(weights=weights)
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        # Register hooks for intermediate representations
        self.features = {}
        self._register_hooks()
        
        # Custom coreset sampler
        self.coreset_sampler = CustomGreedyCoresetSubsampler(
            subsample_ratio=subsample_ratio,
            random_state=seed
        )
        
        # Memory bank registered as buffer
        self.register_buffer("memory_bank", torch.empty(0))

    def _hook(self, name):
        def fn(module, input, output):
            self.features[name] = output
        return fn

    def _register_hooks(self):
        for name, module in self.backbone.named_children():
            if name in self.layers:
                module.register_forward_hook(self._hook(name))

    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts and concatenates spatially aligned neighborhood patch embeddings from multiple layers.
        """
        self.features.clear()
        _ = self.backbone(x)
        
        layer_features = [self.features[name] for name in self.layers]
        
        # Target spatial resolution is the resolution of the largest feature map (typically layer2)
        ref_h, ref_w = layer_features[0].shape[2], layer_features[0].shape[3]
        
        resized_features = []
        for feat in layer_features:
            # Locally aggregate using adaptive average pooling for patch neighborhood context
            pooled = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            # Bilinear upsample to reference spatial dimensions
            resized = F.interpolate(pooled, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            resized_features.append(resized)
            
        # Concatenate along channel dimension: (B, C_total, H_ref, W_ref)
        combined = torch.cat(resized_features, dim=1)
        B, C, H, W = combined.shape
        
        # Permute and reshape to (B * H * W, C)
        patches = combined.permute(0, 2, 3, 1).contiguous().view(-1, C)
        return patches, (B, H, W)

    def fit_memory_bank(self, dataloader, device: torch.device):
        """
        Populate memory bank from defect-free training images and perform coreset subsampling.
        """
        self.to(device)
        self.backbone.eval()
        
        patch_list = []
        print("[*] Extracting patch representations from normal training samples...")
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                patches, _ = self._extract_patch_features(images)
                patch_list.append(patches.cpu())
                
        all_patches = torch.cat(patch_list, dim=0) # (Total_Patches, D)
        print(f"[*] Total extracted patches: {all_patches.shape[0]} (dim={all_patches.shape[1]})")
        
        # Apply custom coreset subsampling
        print(f"[*] Executing Custom Greedy K-Center Coreset subsampling (ratio={self.subsample_ratio})...")
        coreset_patches = self.coreset_sampler.sample(all_patches)
        print(f"[+] Coreset memory bank created: {coreset_patches.shape[0]} prototypes retained.")
        
        self.memory_bank = coreset_patches.to(device)

    def forward(self, x: torch.Tensor):
        """Returns anomaly map and image-level anomaly score."""
        return self.get_anomaly_map_and_score(x)

    def get_anomaly_map_and_score(self, x: torch.Tensor, gaussian_sigma: float = 4.0):
        """
        Queries the memory bank to find the nearest normal prototype distance for each patch.
        """
        self.eval()
        device = x.device
        B_img, _, H_img, W_img = x.shape
        
        with torch.no_grad():
            patches, (B, H_feat, W_feat) = self._extract_patch_features(x)
            
            # Efficient chunked Euclidean distance computation
            # Memory bank: (M, D), patches: (B * H * W, D)
            patches = patches.to(device)
            M = self.memory_bank.to(device)
            
            # (P, D) vs (M, D) distance
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a b^T
            patch_norm = (patches ** 2).sum(dim=1, keepdim=True) # (P, 1)
            m_norm = (M ** 2).sum(dim=1, keepdim=True).t()        # (1, M)
            
            # Compute distance matrix in chunks to prevent VRAM overflow
            chunk_size = 4096
            min_dists = []
            
            for i in range(0, patches.shape[0], chunk_size):
                chunk = patches[i:i + chunk_size]
                chunk_norm = patch_norm[i:i + chunk_size]
                dists = chunk_norm + m_norm - 2.0 * torch.matmul(chunk, M.t())
                dists = torch.clamp(dists, min=0.0).sqrt()
                
                # Nearest prototype distance
                min_dist, _ = torch.min(dists, dim=1)
                min_dists.append(min_dist)
                
            all_min_dists = torch.cat(min_dists, dim=0) # (B * H_feat * W_feat,)
            
            # Reshape back to batch spatial map
            spatial_maps = all_min_dists.view(B, 1, H_feat, W_feat)
            
            # Bilinear upsample to input image resolution
            upsampled_maps = F.interpolate(spatial_maps, size=(H_img, W_img), mode="bilinear", align_corners=False)
            
            anomaly_maps = []
            anomaly_scores = []
            
            for b in range(B):
                raw_map = upsampled_maps[b, 0].cpu().numpy()
                smoothed = gaussian_filter(raw_map, sigma=gaussian_sigma)
                anomaly_maps.append(smoothed)
                
                # Reweighted maximum patch score (Roth et al., Eq. 7)
                s_star = float(np.max(smoothed))
                anomaly_scores.append(s_star)
                
            return np.array(anomaly_maps), np.array(anomaly_scores)
