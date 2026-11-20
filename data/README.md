# Dataset Documentation: MVTec Anomaly Detection (MVTec AD)

## 1. Overview & Source
- **Benchmark Name**: MVTec Anomaly Detection Dataset (MVTec AD)
- **Official Source**: MVTec Software GmbH ([https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad))
- **Citation**: 
  > Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. 
  > *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. 
  > IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Used strictly for academic research and education.

## 2. Dataset Structure
MVTec AD comprises over 5,354 high-resolution color images across 15 industrial manufacturing domains:
- **5 Textures**: `carpet`, `grid`, `leather`, `tile`, `wood`
- **10 Objects**: `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`

For our primary experimental suite and benchmarks, we focus extensively on the **`bottle`** category (and provide plug-and-play capability for all 15 categories).

### Directory Hierarchy:
```
mvtec_anomaly_detection/
└── bottle/
    ├── train/
    │   └── good/              # 209 defect-free training images
    ├── test/
    │   ├── good/              # 20 defect-free test images
    │   ├── broken_large/      # 20 defective images (structural glass breaks)
    │   ├── broken_small/      # 22 defective images (cracks and chips)
    │   └── contamination/     # 21 defective images (foreign fluids/particles)
    └── ground_truth/
        ├── broken_large/      # 20 pixel-level binary annotation masks
        ├── broken_small/      # 22 pixel-level binary annotation masks
        └── contamination/     # 21 pixel-level binary annotation masks
```

## 3. Data Split & Leakage-Free Protocol (Requirement 2)
In unsupervised industrial anomaly detection, real-world inspection systems only encounter normal samples during factory calibration.
To prevent data snooping and information leakage:
1. **Training Set**: 177 normal images (85% of `train/good`). Only defect-free images are seen during training.
2. **Validation Set**: 32 normal images (15% of `train/good`) + optional synthetic anomalies for validation calibration. Under no circumstances are test images touched during validation or hyperparameter tuning.
3. **Test Set**: 83 images total (20 good + 63 anomalous across 3 defect types) with pixel-level ground truth masks. The test set is evaluated strictly once per final model configuration.

## 4. Preprocessing Pipeline
- **Resolution**: High-resolution inputs resized to $256 \times 256$, followed by center cropping to $224 \times 224$ (or direct bilinear interpolation to $224 \times 224$ depending on model).
- **Pixel Normalization**: Standard ImageNet mean $\mu = [0.485, 0.456, 0.406]$ and standard deviation $\sigma = [0.229, 0.224, 0.225]$.
- **Ground Truth Masks**: Binarized at threshold 0.5 where pixel value 1 indicates anomaly and 0 indicates normal substrate.
