"""
Advanced Deep Learning Model (Category 3): Reverse Distillation (RD4AD, Deng & Li, CVPR 2022)
Features:
  - Teacher Encoder: Pretrained WideResNet feature extractor
  - One-Class Embedding (OCE) Bottleneck written from scratch (Requirement 5)
  - Reverse Student Decoder reconstructing multi-scale feature pyramids
  - Multi-scale Cosine Distance Discrepancy Map
Deep Learning AI403 - PUA
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.ndimage import gaussian_filter

def deconv_bn_relu(in_c, out_c, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class OneClassEmbeddingBottleneck(nn.Module):
    """
    Key Component Written From Scratch (Core Requirement 5).
    One-Class Embedding (OCE) Bottleneck:
    Fuses multi-scale feature representations from teacher layers (layer1, layer2, layer3)
    into a compact low-dimensional latent representation.
    """
    def __init__(self, in_channels_list: list, bottleneck_dim: int = 512):
        super().__init__()
        # Project each layer to bottleneck_dim // 3
        dim_per_layer = bottleneck_dim // len(in_channels_list)
        
        self.proj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, dim_per_layer, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim_per_layer),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_per_layer * len(in_channels_list), bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, teacher_features: list) -> torch.Tensor:
        # Spatial size of the lowest-resolution feature (layer3)
        ref_h, ref_w = teacher_features[-1].shape[2], teacher_features[-1].shape[3]
        
        projected = []
        for feat, proj in zip(teacher_features, self.proj_convs):
            p = proj(feat)
            if p.shape[2:] != (ref_h, ref_w):
                p = F.adaptive_avg_pool2d(p, (ref_h, ref_w))
            projected.append(p)
            
        concat = torch.cat(projected, dim=1)
        return self.fusion(concat)


class ReverseStudentDecoder(nn.Module):
    """
    Reverse Student Decoder:
    Decodes the OCE bottleneck representation back into multi-scale feature pyramids
    mirroring the teacher's representations.
    """
    def __init__(self, bottleneck_dim: int = 512, target_dims: list = [256, 512, 1024]):
        super().__init__()
        # target_dims correspond to [layer1, layer2, layer3] channel counts
        c1, c2, c3 = target_dims
        
        # Stage 3 reconstruction (matches layer3)
        self.dec_stage3 = conv_bn_relu(bottleneck_dim, c3)
        
        # Stage 2 reconstruction (matches layer2, stride 2 upsample)
        self.up_stage2 = deconv_bn_relu(c3, c2, stride=2)
        self.dec_stage2 = conv_bn_relu(c2, c2)
        
        # Stage 1 reconstruction (matches layer1, stride 2 upsample)
        self.up_stage1 = deconv_bn_relu(c2, c1, stride=2)
        self.dec_stage1 = conv_bn_relu(c1, c1)

    def forward(self, bottleneck: torch.Tensor) -> list:
        s3 = self.dec_stage3(bottleneck)
        s2 = self.dec_stage2(self.up_stage2(s3))
        s1 = self.dec_stage1(self.up_stage1(s2))
        return [s1, s2, s3]


class ReverseDistillation(nn.Module):
    """
    Full Reverse Distillation Architecture (RD4AD).
    Consists of:
      1. Frozen Teacher Network (pretrained CNN backbone)
      2. Multi-Scale OCE Bottleneck (trainable)
      3. Reverse Student Decoder (trainable)
    """
    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        bottleneck_dim: int = 512
    ):
        super().__init__()
        
        # 1. Teacher Backbone (Frozen)
        if backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            teacher = models.wide_resnet50_2(weights=weights)
            channel_dims = [256, 512, 1024] # layer1, layer2, layer3
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            teacher = models.resnet18(weights=weights)
            channel_dims = [64, 128, 256]
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            teacher = models.resnet50(weights=weights)
            channel_dims = [256, 512, 1024]

        # Extract teacher components
        self.teacher_init = nn.Sequential(
            teacher.conv1, teacher.bn1, teacher.relu, teacher.maxpool
        )
        self.teacher_layer1 = teacher.layer1
        self.teacher_layer2 = teacher.layer2
        self.teacher_layer3 = teacher.layer3
        
        # Freeze teacher parameters
        for p in self.teacher_init.parameters(): p.requires_grad = False
        for p in self.teacher_layer1.parameters(): p.requires_grad = False
        for p in self.teacher_layer2.parameters(): p.requires_grad = False
        for p in self.teacher_layer3.parameters(): p.requires_grad = False
        
        # 2. Trainable Components
        self.bottleneck = OneClassEmbeddingBottleneck(channel_dims, bottleneck_dim=bottleneck_dim)
        self.student = ReverseStudentDecoder(bottleneck_dim=bottleneck_dim, target_dims=channel_dims)

    def extract_teacher_features(self, x: torch.Tensor) -> list:
        self.teacher_init.eval()
        self.teacher_layer1.eval()
        self.teacher_layer2.eval()
        self.teacher_layer3.eval()
        
        with torch.no_grad():
            x0 = self.teacher_init(x)
            t1 = self.teacher_layer1(x0)
            t2 = self.teacher_layer2(t1)
            t3 = self.teacher_layer3(t2)
        return [t1, t2, t3]

    def forward(self, x: torch.Tensor):
        t_feats = self.extract_teacher_features(x)
        b = self.bottleneck(t_feats)
        s_feats = self.student(b)
        return s_feats, t_feats

    def get_anomaly_map_and_score(self, x: torch.Tensor, gaussian_sigma: float = 4.0):
        """
        Computes pixel-wise multiscale cosine distance anomaly map and image anomaly score.
        """
        self.eval()
        B, _, H_img, W_img = x.shape
        
        with torch.no_grad():
            s_feats, t_feats = self.forward(x)
            
            anomaly_maps_multiscale = []
            for s, t in zip(s_feats, t_feats):
                # Cosine distance per spatial location: (B, 1, H_k, W_k)
                s_norm = F.normalize(s, p=2, dim=1)
                t_norm = F.normalize(t, p=2, dim=1)
                cos_dist = 1.0 - (s_norm * t_norm).sum(dim=1, keepdim=True)
                
                # Upsample to image size
                up_dist = F.interpolate(cos_dist, size=(H_img, W_img), mode="bilinear", align_corners=False)
                anomaly_maps_multiscale.append(up_dist)
                
            # Fuse multiscale anomaly maps (summation)
            fused_map = torch.cat(anomaly_maps_multiscale, dim=1).mean(dim=1) # (B, H, W)
            
            anomaly_maps = []
            anomaly_scores = []
            
            for b in range(B):
                raw_map = fused_map[b].cpu().numpy()
                smoothed = gaussian_filter(raw_map, sigma=gaussian_sigma)
                anomaly_maps.append(smoothed)
                
                # Image score is maximum patch score
                score = float(np.max(smoothed))
                anomaly_scores.append(score)
                
            return np.array(anomaly_maps), np.array(anomaly_scores)
