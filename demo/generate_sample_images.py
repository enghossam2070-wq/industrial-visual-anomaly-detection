"""
Generates synthetic sample industrial inspection images for instant live demo testing.
Creates normal bottles, bottles with cracks, contaminations, and broken lips.
Deep Learning AI403 - PUA
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_bottle_image(defect_type: str = "normal", size: int = 224) -> Image.Image:
    # 1. Industrial inspection dark background
    bg = Image.new("RGB", (size, size), color=(28, 30, 34))
    draw = ImageDraw.Draw(bg)
    
    # 2. Draw glossy glass bottle silhouette
    cx = size // 2
    # Bottle body
    body_box = [cx - 50, 80, cx + 50, 205]
    draw.rounded_rectangle(body_box, radius=18, fill=(195, 215, 225), outline=(230, 240, 248), width=3)
    
    # Bottle neck
    neck_box = [cx - 20, 35, cx + 20, 85]
    draw.rectangle(neck_box, fill=(185, 208, 220), outline=(230, 240, 248), width=2)
    
    # Bottle cap / lip
    cap_box = [cx - 24, 25, cx + 24, 38]
    draw.rounded_rectangle(cap_box, radius=4, fill=(160, 185, 200), outline=(240, 245, 250), width=2)
    
    # Specular highlights (reflections typical of industrial bottles)
    draw.line([cx - 32, 90, cx - 32, 195], fill=(255, 255, 255), width=3)
    draw.line([cx + 34, 90, cx + 34, 195], fill=(220, 235, 245), width=2)
    draw.line([cx - 12, 40, cx - 12, 80], fill=(255, 255, 255), width=2)
    
    # 3. Add Defect if requested
    if defect_type == "crack":
        # Jagged sharp crack on glass body
        pts = [(cx + 10, 120), (cx + 18, 132), (cx + 12, 145), (cx + 25, 160)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=(40, 45, 50), width=3)
            draw.line([pts[i][0]+1, pts[i][1], pts[i+1][0]+1, pts[i+1][1]], fill=(255, 255, 255), width=1)
            
    elif defect_type == "contamination":
        # Dark fluid foreign particle contamination inside bottle
        spot_box = [cx - 18, 140, cx + 6, 164]
        draw.ellipse(spot_box, fill=(50, 42, 38), outline=(30, 25, 20), width=2)
        # Small splatter
        draw.ellipse([cx + 12, 135, cx + 18, 141], fill=(60, 45, 40))
        draw.ellipse([cx - 12, 170, cx - 7, 175], fill=(50, 40, 35))
        
    elif defect_type == "broken_large":
        # Chipped jagged glass edge at top lip
        draw.polygon([(cx + 10, 24), (cx + 25, 24), (cx + 25, 39), (cx + 14, 34)], fill=(28, 30, 34))
        draw.line([(cx + 10, 24), (cx + 14, 34), (cx + 25, 39)], fill=(255, 255, 255), width=2)

    # Mild gaussian blur to simulate realistic industrial camera optics
    img = bg.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img

def main():
    dest_dir = "./demo/sample_images"
    os.makedirs(dest_dir, exist_ok=True)
    
    samples = [
        ("sample_normal.png", "normal"),
        ("sample_crack.png", "crack"),
        ("sample_contamination.png", "contamination"),
        ("sample_broken_lip.png", "broken_large")
    ]
    
    for fname, dtype in samples:
        path = os.path.join(dest_dir, fname)
        img = create_synthetic_bottle_image(dtype)
        img.save(path)
        print(f"[+] Created demo sample: {path} ({dtype})")

if __name__ == "__main__":
    main()
