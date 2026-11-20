"""
Utility Functions: Reproducibility, Metrics, Visualization, and Profiling
Deep Learning AI403 - PUA
"""

import os
import time
import random
import logging
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    """Ensure multi-run deterministic reproducibility across PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024) # Float32
    }

def measure_latency(model, sample_input: torch.Tensor, device: torch.device, warmups: int = 10, reps: int = 50) -> float:
    """Benchmark per-image inference latency in milliseconds."""
    model.eval()
    sample_input = sample_input.to(device)
    
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(sample_input)
            
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(reps):
                _ = model(sample_input)
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / reps
        else:
            t0 = time.perf_counter()
            for _ in range(reps):
                _ = model(sample_input)
            t1 = time.perf_counter()
            return ((t1 - t0) / reps) * 1000.0

def overlay_heatmap(image_np: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a continuous JET colormap anomaly map on top of an RGB image.
    image_np: (H, W, 3), range [0, 255] or [0, 1]
    anomaly_map: (H, W), range [0, 1]
    """
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
        
    norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    heatmap = (norm_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return overlay

def plot_inspection_result(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray = None,
    anomaly_score: float = None,
    threshold: float = None,
    save_path: str = None
):
    """Generate high-resolution 4-panel industrial inspection visual figure."""
    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    
    # 1. Original
    axes[0].imshow(image)
    axes[0].set_title("Input Sample")
    axes[0].axis("off")
    
    # 2. Anomaly Heatmap
    im = axes[1].imshow(anomaly_map, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Anomaly Heatmap\nScore: {anomaly_score:.3f}" if anomaly_score is not None else "Anomaly Heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Superimposed Overlay
    overlay = overlay_heatmap(image, anomaly_map)
    status = "DEFECT" if (anomaly_score is not None and threshold is not None and anomaly_score >= threshold) else "NORMAL"
    color = "red" if status == "DEFECT" else "green"
    axes[2].imshow(overlay)
    axes[2].set_title(f"Inspection Result: {status}", color=color, fontweight="bold")
    axes[2].axis("off")
    
    # 4. Ground Truth Mask (if available)
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="gray")
        axes[3].set_title("Ground Truth Mask")
        axes[3].axis("off")
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup dual console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
