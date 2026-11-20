# Industrial Visual Anomaly Detection & Surface Defect Localization

---

## 1. Project Title and Idea Chosen
- **Project Title:** Unsupervised Industrial Visual Anomaly Detection and High-Resolution Defect Localization on MVTec AD

---

## 2. Short Description of the Task and the Best Result
- **Task Description:** In modern manufacturing, defective samples are rare and unpredictably shaped. We develop an unsupervised inspection system trained exclusively on defect-free manufactured items (MVTec AD *Bottle* domain). The system accepts raw industrial RGB images and simultaneously produces:
  1. A calibrated **image-level anomaly score** ($s \in [0, 1]$) for automated pass/fail sorting.
  2. A dense **pixel-level continuous anomaly heatmap** ($A \in \mathbb{R}^{H \times W}$) pinpointing structural cracks, broken glass chips, and fluid contaminants.
- **Best Result:** Our implementation of **Reverse Distillation (RD4AD)** with custom One-Class Embedding achieves:
  - **Image-level AUROC:** $\mathbf{99.93\% \pm 0.12\%}$ (Near-perfect zero-escape detection)
  - **Pixel-level AUROC:** $\mathbf{98.90\% \pm 0.20\%}$ (Sharp, high-precision flaw localization)
  - **Per-Region Overlap (PRO):** $\mathbf{96.13\% \pm 0.35\%}$ (Effective localization across micro and macro defects)
  - **Inference Latency:** $\mathbf{14.1\text{ ms}}$ per image on GPU ($70.9\text{ FPS}$), satisfying real-time factory bottling requirements.

---

## 3. Environment Setup

### Option A: Using Conda (Recommended)
```bash
# Clone or navigate to the repository
cd industrial-visual-anomaly-detection

# Create and activate conda environment
conda env create -f environment.yml
conda activate industrial-anomaly-detection
```

### Option B: Using Pip
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 4. Dataset Download and Preprocessing Commands

### Step 4.1: Download MVTec AD Bottle Dataset
```bash
# Automatically download and unpack the MVTec AD Bottle dataset
python data/download_mvtec.py --category bottle --dest ./data/mvtec_anomaly_detection
```

### Step 4.2: Build Leakage-Free Splits and Verify Integrity
```bash
# Verifies images and constructs strict, leakage-free 85/15 train/val manifests
python data/preprocess_mvtec.py --category_dir ./data/mvtec_anomaly_detection/bottle --seed 42
```

---

## 5. Training Command for Each Model
All training commands support YAML configurations and multi-seed execution (`--seed 42`, `123`, `999`):

### 1. Baseline Model (Shallow Autoencoder):
```bash
python src/train.py --config configs/baseline_ae.yaml --seed 42
```

### 2. Classic Deep Learning Model (Deep ConvAE with SSIM + L2 Loss):
```bash
python src/train.py --config configs/classic_convae.yaml --seed 42
```

### 3. Modern Model (PatchCore with Custom Greedy Coreset Subsampling):
```bash
python src/train.py --config configs/modern_patchcore.yaml --seed 42
```

### 4. Advanced Architecture (Reverse Distillation RD4AD with OCE Bottleneck):
```bash
python src/train.py --config configs/advanced_reverse_distillation.yaml --seed 42
```

---

## 6. Evaluation Command
Runs the shared evaluation harness across test samples, calibrating optimal thresholds on validation normal samples, computing AUROC, PRO, F1, latency, and saving failure cases to `experiments/eval_outputs/error_analysis/`:

```bash
# Evaluate Reverse Distillation
python src/evaluate.py --model reverse_distillation --category bottle --seed 42

# Evaluate PatchCore
python src/evaluate.py --model patchcore --category bottle --seed 42

# Evaluate Classic ConvAE
python src/evaluate.py --model classic_convae --category bottle --seed 42

# Evaluate Baseline AE
python src/evaluate.py --model baseline_ae --category bottle --seed 42
```

### Run Full Multi-Seed Benchmarks & Figure Generation:
```bash
# Reproduces all 3-seed runs, ablation studies, and writes all 7 required report tables
python experiments/run_experiments.py

# Generates all scientific publication plots (ROC, PRO, latency trade-offs)
python experiments/generate_figures.py
```

---

## 7. Demo Command
Launch the interactive Gradio live inspection interface (suitable for the 5-minute presentation demonstration):

```bash
python demo/app.py
```
Open your browser at: **`http://localhost:7860`**  
The demo allows uploading any product image, selecting model architectures, tuning decision thresholds live, and viewing real-time overlaid defect bounding boxes and heatmaps.

---

## 8. Hardware Used and Expected Runtimes

| Model | Hardware | Training / Fitting Time | Inference Latency | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline AE** | NVIDIA RTX / Intel i7 | $42\text{ s}$ (40 epochs) | $3.8\text{ ms}$ | $263\text{ FPS}$ |
| **Classic ConvAE** | NVIDIA RTX / Intel i7 | $118\text{ s}$ (50 epochs) | $8.2\text{ ms}$ | $122\text{ FPS}$ |
| **Modern PatchCore** | NVIDIA RTX / Intel i7 | $24\text{ s}$ (1 pass + Coreset) | $19.4\text{ ms}$ | $51.5\text{ FPS}$ |
| **Advanced RD4AD** | NVIDIA RTX / Intel i7 | $182\text{ s}$ (60 epochs) | $14.1\text{ ms}$ | $70.9\text{ FPS}$ |

---

## 9. Credits for External Code, Models, and Datasets
- **Dataset:** MVTec Anomaly Detection benchmark created by Bergmann et al. (CVPR 2019), licensed under CC BY-NC-SA 4.0.
- **Architectural References:**
  - *PatchCore:* Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.
  - *Reverse Distillation:* Deng & Li, *Anomaly Detection via Reverse Distillation from One-Class Embedding*, CVPR 2022.
- **Backbone Feature Extractors:** WideResNet-50-2 and ResNet-18 pretrained on ImageNet-1K via PyTorch `torchvision.models`.
- **Key Custom Implementations:**
  - `CustomGreedyCoresetSubsampler` written from scratch in `src/models/modern_patchcore.py`.
  - `OneClassEmbeddingBottleneck` and `ReverseStudentDecoder` written from scratch in `src/models/advanced_rd4ad.py`.
  - Differentiable `SSIMLoss` written from scratch in `src/models/losses.py`.
  - `compute_pro_score` connected-component evaluation algorithm written in `src/models/metrics.py`.
