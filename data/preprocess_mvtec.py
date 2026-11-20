"""
MVTec AD Dataset Verification, Preprocessing, and Leakage-Free Split Builder
Deep Learning AI403 - PUA
"""

import os
import glob
import json
import random
from PIL import Image
import numpy as np

def verify_and_split(
    category_dir: str,
    output_dir: str,
    val_ratio: float = 0.15,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    
    train_good_dir = os.path.join(category_dir, "train", "good")
    test_dir = os.path.join(category_dir, "test")
    gt_dir = os.path.join(category_dir, "ground_truth")

    if not os.path.exists(train_good_dir):
        raise FileNotFoundError(f"Training path not found: {train_good_dir}")

    # 1. Collect all normal training samples
    train_images = sorted(glob.glob(os.path.join(train_good_dir, "*.png")))
    if len(train_images) == 0:
        raise ValueError(f"No training images found in {train_good_dir}")

    # Leakage-free train / validation split
    shuffled = list(train_images)
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    val_files = sorted(shuffled[:n_val])
    train_files = sorted(shuffled[n_val:])

    # 2. Collect test samples and ground truth masks
    test_records = []
    test_subdirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

    for defect_type in test_subdirs:
        img_paths = sorted(glob.glob(os.path.join(test_dir, defect_type, "*.png")))
        for img_p in img_paths:
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            is_anomaly = (defect_type != "good")
            
            mask_path = None
            if is_anomaly:
                expected_mask = os.path.join(gt_dir, defect_type, f"{base_name}_mask.png")
                if os.path.exists(expected_mask):
                    mask_path = expected_mask

            test_records.append({
                "image_path": os.path.relpath(img_p, start=os.path.dirname(category_dir)),
                "label": 1 if is_anomaly else 0,
                "defect_type": defect_type,
                "mask_path": os.path.relpath(mask_path, start=os.path.dirname(category_dir)) if mask_path else None
            })

    # Prepare split payload
    manifest = {
        "category": os.path.basename(category_dir),
        "val_ratio": val_ratio,
        "seed": seed,
        "train_samples": [os.path.relpath(p, start=os.path.dirname(category_dir)) for p in train_files],
        "val_samples": [os.path.relpath(p, start=os.path.dirname(category_dir)) for p in val_files],
        "test_samples": test_records,
        "summary": {
            "total_train_normal": len(train_files),
            "total_val_normal": len(val_files),
            "total_test": len(test_records),
            "test_good": sum(1 for r in test_records if r["label"] == 0),
            "test_anomalous": sum(1 for r in test_records if r["label"] == 1)
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{os.path.basename(category_dir)}_split_seed{seed}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Dataset verification completed for category: {os.path.basename(category_dir)}")
    print(f"    Train normal: {len(train_files)}")
    print(f"    Val normal:   {len(val_files)}")
    print(f"    Test total:   {len(test_records)} (Good: {manifest['summary']['test_good']}, Defective: {manifest['summary']['test_anomalous']})")
    print(f"[+] Saved deterministic split manifest: {manifest_path}")

    return manifest_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_dir", type=str, default="./data/mvtec_anomaly_detection/bottle")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.path.exists(args.category_dir):
        verify_and_split(args.category_dir, args.output_dir, args.val_ratio, args.seed)
    else:
        print(f"[*] Note: {args.category_dir} does not exist yet. Run download_mvtec.py first.")
