"""
Shared Unified Evaluation Pipeline for Industrial Anomaly Detection
Evaluates I-AUROC, P-AUROC, PRO score, F1-Score, Latency, and Exports Error Analysis Cases.
Deep Learning AI403 - PUA
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, count_parameters, measure_latency, plot_inspection_result, setup_logger
from src.datasets.mvtec_dataset import get_dataloaders
from src.models import (
    BaselineAutoencoder,
    DeepConvAutoencoder,
    PatchCore,
    ReverseDistillation,
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_score,
    find_optimal_threshold
)

def evaluate_model(
    model,
    model_name: str,
    test_loader,
    val_loader,
    device: torch.device,
    category: str = "bottle",
    save_dir: str = "./experiments/eval_outputs",
    save_visuals: bool = True
):
    os.makedirs(save_dir, exist_ok=True)
    error_analysis_dir = os.path.join(save_dir, "error_analysis", model_name)
    os.makedirs(error_analysis_dir, exist_ok=True)
    
    # 1. Optimal threshold calibration using validation normal samples
    print("[*] Calibrating optimal decision threshold on validation set...")
    val_scores = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            _, scores = model.get_anomaly_map_and_score(imgs)
            val_scores.extend(scores.tolist())
            
    val_scores = np.array(val_scores)
    # 95th percentile of normal validation samples as standard decision boundary
    threshold = float(np.percentile(val_scores, 95))
    print(f"[+] Calibrated Anomaly Threshold: {threshold:.5f} (Val 95th percentile)")

    # 2. Test Set Evaluation
    print(f"[*] Running inference on {len(test_loader.dataset)} test samples...")
    test_labels = []
    test_scores = []
    test_maps = []
    test_masks = []
    defect_types = []
    image_paths = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            masks = batch["mask"].squeeze(1).cpu().numpy() # (B, H, W)
            d_types = batch["defect_type"]
            paths = batch["image_path"]

            maps, scores = model.get_anomaly_map_and_score(imgs)

            test_labels.extend(labels.tolist())
            test_scores.extend(scores.tolist())
            test_maps.extend(maps)
            test_masks.extend(masks)
            defect_types.extend(d_types)
            image_paths.extend(paths)

    test_labels = np.array(test_labels)
    test_scores = np.array(test_scores)
    test_maps = np.array(test_maps)
    test_masks = np.array(test_masks)

    # 3. Calculate Global Metrics
    image_auroc = compute_image_auroc(test_labels, test_scores)
    pixel_auroc = compute_pixel_auroc(test_masks, test_maps)
    pro_score = compute_pro_score(test_masks, test_maps)

    # Threshold classification metrics
    preds = (test_scores >= threshold).astype(int)
    tp = np.sum((preds == 1) & (test_labels == 1))
    fp = np.sum((preds == 1) & (test_labels == 0))
    tn = np.sum((preds == 0) & (test_labels == 0))
    fn = np.sum((preds == 0) & (test_labels == 1))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 4. Measure Inference Latency & Parameter Counts
    sample_tensor = torch.randn(1, 3, 224, 224).to(device)
    latency_ms = measure_latency(model, sample_tensor, device)
    param_info = count_parameters(model)

    # 5. Per-Defect Type Breakdown
    df_results = pd.DataFrame({
        "label": test_labels,
        "score": test_scores,
        "pred": preds,
        "defect_type": defect_types,
        "path": image_paths
    })
    
    per_defect_metrics = {}
    for dtype, group in df_results.groupby("defect_type"):
        if dtype == "good":
            fp_rate = float(np.mean(group["pred"] == 1))
            per_defect_metrics[dtype] = {"sample_count": len(group), "false_positive_rate": fp_rate}
        else:
            recall_defect = float(np.mean(group["pred"] == 1))
            per_defect_metrics[dtype] = {"sample_count": len(group), "detection_rate": recall_defect}

    # 6. Error Analysis: Save False Positives and False Negatives
    false_positives = df_results[(df_results["label"] == 0) & (df_results["pred"] == 1)]
    false_negatives = df_results[(df_results["label"] == 1) & (df_results["pred"] == 0)]
    
    print(f"\n[+] Error Analysis Summary:")
    print(f"    False Positives (normal flagged anomalous): {len(false_positives)}")
    print(f"    False Negatives (defects missed):          {len(false_negatives)}")

    if save_visuals:
        # Save up to 5 error cases
        for idx, row in pd.concat([false_positives.head(3), false_negatives.head(3)]).iterrows():
            i = int(idx)
            err_kind = "FP" if row["label"] == 0 else "FN"
            base_fname = os.path.basename(row["path"]).replace(".png", "")
            save_img_path = os.path.join(error_analysis_dir, f"{err_kind}_{row['defect_type']}_{base_fname}.png")
            
            raw_img = (test_loader.dataset[i]["image"].permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            plot_inspection_result(
                image=raw_img,
                anomaly_map=test_maps[i],
                gt_mask=test_masks[i] if row["label"] == 1 else None,
                anomaly_score=row["score"],
                threshold=threshold,
                save_path=save_img_path
            )

    results = {
        "model_name": model_name,
        "category": category,
        "threshold": threshold,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "pro_score": pro_score,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "latency_ms": latency_ms,
        "parameters": param_info["total_parameters"],
        "model_size_mb": param_info["model_size_mb"],
        "hardware": "NVIDIA GPU" if device.type == "cuda" else "Intel Core CPU",
        "per_defect_breakdown": per_defect_metrics,
        "error_counts": {
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives)
        }
    }

    # Print Formatted Table
    table_data = [
        ["Image-level AUROC", f"{image_auroc * 100:.2f}%"],
        ["Pixel-level AUROC", f"{pixel_auroc * 100:.2f}%"],
        ["PRO Score (Per-Region)", f"{pro_score * 100:.2f}%"],
        ["F1 Score", f"{f1 * 100:.2f}%"],
        ["Precision", f"{precision * 100:.2f}%"],
        ["Recall (Sensitivity)", f"{recall * 100:.2f}%"],
        ["Inference Latency", f"{latency_ms:.2f} ms / image"],
        ["Model Parameters", f"{param_info['total_parameters']:,}"]
    ]
    print("\n" + tabulate(table_data, headers=["Metric", "Result"], tablefmt="fancy_grid"))

    # Save JSON summary
    summary_path = os.path.join(save_dir, f"{model_name}_{category}_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

def load_and_evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_loader, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        category=args.category,
        batch_size=args.batch_size,
        seed=args.seed
    )

    checkpoint_path = args.checkpoint or f"./checkpoints/{args.model}_{args.category}_seed{args.seed}.pth"
    
    if args.model == "baseline_ae":
        model = BaselineAutoencoder().to(device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
    elif args.model == "classic_convae":
        model = DeepConvAutoencoder().to(device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
    elif args.model == "patchcore":
        model = PatchCore(seed=args.seed).to(device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.memory_bank = ckpt["memory_bank"].to(device)
    elif args.model in ["reverse_distillation", "advanced_rd4ad"]:
        model = ReverseDistillation().to(device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return evaluate_model(
        model=model,
        model_name=args.model,
        test_loader=test_loader,
        val_loader=val_loader,
        device=device,
        category=args.category,
        save_dir=args.output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--data_dir", type=str, default="./data/mvtec_anomaly_detection")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./experiments/eval_outputs")
    args = parser.parse_args()

    load_and_evaluate(args)
