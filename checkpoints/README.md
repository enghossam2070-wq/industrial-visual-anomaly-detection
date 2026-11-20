# Model Checkpoints & Pretrained Weights

This directory stores model weight checkpoints for all trained architectures:
1. `baseline_ae_bottle_seed42.pth`: Baseline Shallow Autoencoder weights
2. `classic_convae_bottle_seed42.pth`: Classic 5-Stage Deep Convolutional Autoencoder weights
3. `patchcore_bottle_seed42.pth`: Modern PatchCore greedy coreset memory bank buffer
4. `advanced_rd4ad_bottle_seed42.pth`: Advanced Reverse Distillation student decoder and OCE weights

## Automated Reproduction
To reproduce and generate fresh checkpoints from scratch, run:
```bash
# Train Baseline AE
python src/train.py --config configs/baseline_ae.yaml --seed 42

# Train Classic ConvAE
python src/train.py --config configs/classic_convae.yaml --seed 42

# Train Modern PatchCore (Extract features & coreset subsample memory bank)
python src/train.py --config configs/modern_patchcore.yaml --seed 42

# Train Advanced Reverse Distillation
python src/train.py --config configs/advanced_reverse_distillation.yaml --seed 42
```

## Shared Drive Download Link
In accordance with PUA AI403 submission specifications (Section 5, item 4):
- **Pretrained Checkpoint Archive**: `https://drive.google.com/drive/folders/PUA_AI403_DL_Checkpoints_TeamX` *(Simulated shared cloud drive link for viva submission)*
