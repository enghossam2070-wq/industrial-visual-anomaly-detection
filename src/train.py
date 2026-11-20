"""
Unified Training Pipeline for Industrial Anomaly Detection
Supports Baseline AE, Classic ConvAE, Modern PatchCore, and Advanced RD4AD
Deep Learning AI403 - PUA
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, count_parameters, setup_logger
from src.datasets.mvtec_dataset import get_dataloaders
from src.models import (
    BaselineAutoencoder,
    DeepConvAutoencoder,
    PatchCore,
    ReverseDistillation,
    CombinedReconstructionLoss,
    CosineSimilarityLoss
)

def train_autoencoder(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, logger, save_path):
    best_val_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation loss on normal validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        
        if scheduler is not None:
            scheduler.step()
            
        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, save_path)
            logger.info(f"  --> Saved new best checkpoint to {save_path}")
            
    total_time = time.time() - start_time
    logger.info(f"[+] Training finished in {total_time:.2f} seconds.")
    return total_time

def train_reverse_distillation(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, logger, save_path):
    best_val_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.bottleneck.train()
        model.student.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            s_feats, t_feats = model(images)
            loss = criterion(s_feats, t_feats)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.bottleneck.eval()
        model.student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                s_feats, t_feats = model(images)
                loss = criterion(s_feats, t_feats)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        
        if scheduler is not None:
            scheduler.step()
            
        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] - Distill Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, save_path)
            logger.info(f"  --> Saved new best checkpoint to {save_path}")
            
    total_time = time.time() - start_time
    logger.info(f"[+] Reverse Distillation training finished in {total_time:.2f} seconds.")
    return total_time

def run_train(args):
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # CLI args override config
    model_name = args.model or config.get("model", {}).get("name", "patchcore")
    category = args.category or config.get("category", "bottle")
    data_dir = args.data_dir or config.get("data_dir", "./data/mvtec_anomaly_detection")
    epochs = args.epochs or config.get("training", {}).get("epochs", 40)
    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 16)
    lr = args.lr or config.get("training", {}).get("learning_rate", 0.001)
    seed = args.seed if args.seed is not None else config.get("training", {}).get("seed", 42)
    use_augmentation = args.use_augmentation or config.get("training", {}).get("use_augmentation", False)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_dir = f"./experiments/logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(f"train_{model_name}_{seed}", os.path.join(log_dir, f"{model_name}_{category}_seed{seed}.log"))
    logger.info(f"=== Starting Training Run: {model_name} | Category: {category} | Seed: {seed} | Device: {device} ===")

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=data_dir,
        category=category,
        batch_size=batch_size,
        seed=seed,
        use_augmentation=use_augmentation
    )

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"{model_name}_{category}_seed{seed}.pth")

    if model_name == "baseline_ae":
        model = BaselineAutoencoder().to(device)
        param_stats = count_parameters(model)
        logger.info(f"Model parameters: {param_stats['total_parameters']:,} ({param_stats['model_size_mb']:.2f} MB)")
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()
        train_time = train_autoencoder(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, logger, save_path)

    elif model_name == "classic_convae":
        model = DeepConvAutoencoder().to(device)
        param_stats = count_parameters(model)
        logger.info(f"Model parameters: {param_stats['total_parameters']:,} ({param_stats['model_size_mb']:.2f} MB)")
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = CombinedReconstructionLoss(ssim_weight=0.5, l2_weight=0.5)
        train_time = train_autoencoder(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, logger, save_path)

    elif model_name == "patchcore":
        subsample_ratio = config.get("model", {}).get("subsample_ratio", 0.1)
        model = PatchCore(
            backbone_name=config.get("model", {}).get("backbone", "wide_resnet50_2"),
            pretrained=config.get("model", {}).get("pretrained", True),
            subsample_ratio=subsample_ratio,
            seed=seed
        ).to(device)
        param_stats = count_parameters(model)
        logger.info(f"Model parameters: {param_stats['total_parameters']:,} (Backbone feature extractor)")
        t0 = time.time()
        model.fit_memory_bank(train_loader, device=device)
        train_time = time.time() - t0
        logger.info(f"[+] Memory bank constructed and subsampled in {train_time:.2f} seconds.")
        torch.save({
            "memory_bank": model.memory_bank.cpu(),
            "subsample_ratio": subsample_ratio,
            "backbone": model.backbone_name
        }, save_path)
        logger.info(f"[+] Saved memory bank to {save_path}")

    elif model_name in ["reverse_distillation", "advanced_rd4ad"]:
        model = ReverseDistillation(
            backbone_name=config.get("model", {}).get("backbone", "wide_resnet50_2"),
            pretrained=config.get("model", {}).get("pretrained", True),
            bottleneck_dim=config.get("model", {}).get("bottleneck_channels", 512)
        ).to(device)
        param_stats = count_parameters(model)
        logger.info(f"Model parameters: Total={param_stats['total_parameters']:,}, Trainable={param_stats['trainable_parameters']:,}")
        
        trainable_params = list(model.bottleneck.parameters()) + list(model.student.parameters())
        optimizer = Adam(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=40, gamma=0.2)
        criterion = CosineSimilarityLoss()
        train_time = train_reverse_distillation(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, logger, save_path)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return train_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, choices=["baseline_ae", "classic_convae", "patchcore", "reverse_distillation", "advanced_rd4ad"])
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--data_dir", type=str, default="./data/mvtec_anomaly_detection")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_augmentation", action="store_true")
    args = parser.parse_args()

    run_train(args)
