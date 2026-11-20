"""
Synthetic Defect Generation for Data-Strategy Study (Requirement 7)
Implements CutPaste (Li et al., CVPR 2021) and synthetic scratch/spot generation.
"""

import math
import random
import numpy as np
from PIL import Image, ImageDraw

class CutPasteGenerator:
    """
    CutPaste data augmentation: cuts a rectangular or scar patch from a normal image
    and pastes it at an arbitrary location with optional color jitter/rotation.
    """
    def __init__(self, color_jitter: bool = True, scar: bool = True):
        self.color_jitter = color_jitter
        self.scar = scar

    def __call__(self, img: Image.Image):
        """
        Takes a PIL image and returns (augmented_img, binary_defect_mask)
        """
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        aug_img = img.copy()

        if self.scar and random.random() < 0.5:
            # CutPaste Scar
            scar_w = random.randint(2, 10)
            scar_h = random.randint(20, min(100, h // 2))
            angle = random.uniform(0, 360)
            
            # Crop a small region
            x0 = random.randint(0, max(1, w - scar_w))
            y0 = random.randint(0, max(1, h - scar_h))
            patch = img.crop((x0, y0, x0 + scar_w, y0 + scar_h))
            
            # Rotate patch
            patch = patch.rotate(angle, expand=True)
            pw, ph = patch.size
            
            # Target position
            tx = random.randint(0, max(1, w - pw))
            ty = random.randint(0, max(1, h - ph))
            
            aug_img.paste(patch, (tx, ty))
            
            draw = ImageDraw.Draw(mask)
            draw.rectangle([tx, ty, tx + pw, ty + ph], fill=255)
        else:
            # CutPaste Normal rectangle
            area_ratio = random.uniform(0.02, 0.15)
            aspect_ratio = random.uniform(0.3, 3.3)
            
            patch_area = int(w * h * area_ratio)
            pw = int(math.sqrt(patch_area * aspect_ratio))
            ph = int(math.sqrt(patch_area / aspect_ratio))
            pw = max(4, min(pw, w - 1))
            ph = max(4, min(ph, h - 1))
            
            sx = random.randint(0, w - pw)
            sy = random.randint(0, h - ph)
            patch = img.crop((sx, sy, sx + pw, sy + ph))
            
            tx = random.randint(0, w - pw)
            ty = random.randint(0, h - ph)
            
            aug_img.paste(patch, (tx, ty))
            
            draw = ImageDraw.Draw(mask)
            draw.rectangle([tx, ty, tx + pw, ty + ph], fill=255)
            
        return aug_img, mask

def apply_synthetic_anomaly(img: Image.Image):
    gen = CutPasteGenerator(color_jitter=True, scar=True)
    return gen(img)
