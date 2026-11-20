"""
MVTec Anomaly Detection PyTorch Dataset and DataLoader Factory
Deep Learning AI403 - PUA
"""

import os
import glob
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import get_transforms

class MVTecDataset(Dataset):
    """
    Leakage-free MVTec AD Dataset Loader.
    Provides strict separation between normal training, validation, and ground-truth annotated test sets.
    """
    def __init__(
        self,
        root_dir: str,
        category: str = "bottle",
        split: str = "train",
        img_size: int = 224,
        val_ratio: float = 0.15,
        seed: int = 42,
        use_augmentation: bool = False
    ):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.img_size = img_size
        self.category_dir = os.path.join(root_dir, category)
        
        self.img_transform, self.mask_transform = get_transforms(
            img_size=img_size,
            is_train=(split == "train"),
            use_augmentation=use_augmentation
        )
        
        self.samples = []
        self._load_samples(val_ratio, seed)

    def _load_samples(self, val_ratio: float, seed: int):
        train_good_dir = os.path.join(self.category_dir, "train", "good")
        test_dir = os.path.join(self.category_dir, "test")
        gt_dir = os.path.join(self.category_dir, "ground_truth")

        if not os.path.exists(self.category_dir):
            return

        # 1. Normal training images
        all_train_good = sorted(glob.glob(os.path.join(train_good_dir, "*.png")))
        
        # Deterministic split for validation without leakage
        random.seed(seed)
        shuffled = list(all_train_good)
        random.shuffle(shuffled)
        n_val = int(len(shuffled) * val_ratio)
        
        val_files = sorted(shuffled[:n_val])
        train_files = sorted(shuffled[n_val:])

        if self.split == "train":
            for f in train_files:
                self.samples.append({
                    "image_path": f,
                    "label": 0,
                    "defect_type": "good",
                    "mask_path": None
                })
        elif self.split == "val":
            for f in val_files:
                self.samples.append({
                    "image_path": f,
                    "label": 0,
                    "defect_type": "good",
                    "mask_path": None
                })
        elif self.split == "test":
            test_subdirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
            for defect_type in test_subdirs:
                img_paths = sorted(glob.glob(os.path.join(test_dir, defect_type, "*.png")))
                for img_p in img_paths:
                    is_anomaly = (defect_type != "good")
                    base_name = os.path.splitext(os.path.basename(img_p))[0]
                    mask_p = None
                    if is_anomaly:
                        expected_mask = os.path.join(gt_dir, defect_type, f"{base_name}_mask.png")
                        if os.path.exists(expected_mask):
                            mask_p = expected_mask
                    
                    self.samples.append({
                        "image_path": img_p,
                        "label": 1 if is_anomaly else 0,
                        "defect_type": defect_type,
                        "mask_path": mask_p
                    })
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image_tensor = self.img_transform(image)
        
        # Load mask if available, else zero mask
        if item["mask_path"] is not None and os.path.exists(item["mask_path"]):
            mask = Image.open(item["mask_path"]).convert("L")
            mask_tensor = self.mask_transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "mask": mask_tensor,
            "defect_type": item["defect_type"],
            "image_path": item["image_path"]
        }

def get_dataloaders(
    root_dir: str,
    category: str = "bottle",
    img_size: int = 224,
    batch_size: int = 16,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    use_augmentation: bool = False
):
    """Factory helper to instantiate train, val, and test loaders."""
    train_dataset = MVTecDataset(
        root_dir=root_dir, category=category, split="train",
        img_size=img_size, val_ratio=val_ratio, seed=seed, use_augmentation=use_augmentation
    )
    val_dataset = MVTecDataset(
        root_dir=root_dir, category=category, split="val",
        img_size=img_size, val_ratio=val_ratio, seed=seed
    )
    test_dataset = MVTecDataset(
        root_dir=root_dir, category=category, split="test",
        img_size=img_size, seed=seed
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
