"""
Evaluation Metrics for Industrial Anomaly Detection
Implements Image-level AUROC, Pixel-level AUROC, Per-Region Overlap (PRO), and Optimal F1.
Deep Learning AI403 - PUA
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
from scipy.ndimage import label

def compute_image_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Image-level Area Under the ROC Curve (I-AUROC)."""
    if len(np.unique(y_true)) < 2:
        return 1.0 # Trivial case
    return float(roc_auc_score(y_true, y_score))

def compute_pixel_auroc(gt_masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    """
    Computes Pixel-level AUROC (P-AUROC).
    gt_masks: (N, H, W) binary
    anomaly_maps: (N, H, W) continuous float
    """
    y_true = gt_masks.flatten().astype(np.int32)
    y_score = anomaly_maps.flatten().astype(np.float32)
    
    if len(np.unique(y_true)) < 2:
        return 1.0
        
    # Subsample if pixel array is excessively large (>5M pixels) for speed
    if len(y_true) > 5_000_000:
        idx = np.random.choice(len(y_true), size=5_000_000, replace=False)
        y_true = y_true[idx]
        y_score = y_score[idx]
        
    return float(roc_auc_score(y_true, y_score))

def compute_pro_score(gt_masks: np.ndarray, anomaly_maps: np.ndarray, max_fpr: float = 0.3, num_thresholds: int = 200) -> float:
    """
    Computes Per-Region Overlap (PRO) score up to max_fpr (typically 0.3) as defined by Bergmann et al. (CVPR 2019).
    Measures defect overlap per individual connected defect component.
    """
    # Normalize anomaly maps to [0, 1]
    maps_min, maps_max = anomaly_maps.min(), anomaly_maps.max()
    if maps_max > maps_min:
        norm_maps = (anomaly_maps - maps_min) / (maps_max - maps_min)
    else:
        norm_maps = anomaly_maps

    # Find connected components in ground truth masks
    labeled_components = []
    total_components = 0
    for mask in gt_masks:
        lbl, num_comp = label(mask > 0)
        labeled_components.append((lbl, num_comp))
        total_components += num_comp

    if total_components == 0:
        return 1.0

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    pros = []
    fprs = []

    flat_gt = (gt_masks > 0).astype(bool)
    n_normal_pixels = np.sum(~flat_gt)

    if n_normal_pixels == 0:
        return 1.0

    for th in thresholds:
        binary_pred = norm_maps >= th
        
        # 1. False Positive Rate on normal pixels
        fp_pixels = np.sum(binary_pred & (~flat_gt))
        fpr = fp_pixels / n_normal_pixels
        
        if fpr > max_fpr:
            break
            
        fprs.append(fpr)

        # 2. Per-region overlap across all components
        comp_overlaps = []
        for i, (lbl, num_comp) in enumerate(labeled_components):
            pred_slice = binary_pred[i]
            for c in range(1, num_comp + 1):
                comp_mask = (lbl == c)
                intersection = np.sum(pred_slice & comp_mask)
                comp_size = np.sum(comp_mask)
                comp_overlaps.append(intersection / comp_size if comp_size > 0 else 0.0)

        pros.append(np.mean(comp_overlaps) if comp_overlaps else 0.0)

    if len(fprs) < 2:
        return 0.0

    # Sort by FPR
    sorted_idx = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_idx]
    pros = np.array(pros)[sorted_idx]

    # Normalize area under curve by max_fpr
    pro_auc = np.trapz(pros, fprs) / max_fpr
    return float(np.clip(pro_auc, 0.0, 1.0))

def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray):
    """
    Finds optimal decision threshold maximizing F1 score on validation set.
    Returns: (optimal_threshold, best_f1, precision, recall)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 1.0, 1.0, 1.0
        
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # Avoid zero division
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])
    return best_threshold, float(f1_scores[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])
