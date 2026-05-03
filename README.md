# 🚀 CIFAR-100 Custom CNN — EfficientLite CNN From Scratch

> A custom lightweight CNN trained on CIFAR-100 from scratch — no pretrained weights. Designed for the best accuracy/efficiency tradeoff with ~2M parameters.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Setup](#-training-setup)
- [Techniques Applied](#-techniques-applied)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Visualizations](#-visualizations)
- [Efficiency Metrics](#-efficiency-metrics)
- [Key Observations](#-key-observations)
- [Requirements](#-requirements)
- [Usage](#-usage)

---

## 🎯 Overview

This notebook implements **EfficientLite CNN**, a custom convolutional neural network built from scratch for the **CIFAR-100 image classification** task (100 classes, 32×32 images).

**Design Philosophy:**
- No pretrained weights — trained entirely from scratch
- Lightweight architecture: ~2M parameters, low FLOPs
- Heavy regularization to minimize the train-test generalization gap
- Multi-metric evaluation: accuracy, fairness, efficiency, and latency

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Dataset** | CIFAR-100 |
| **Classes** | 100 |
| **Image Size** | 32 × 32 (RGB) |
| **Train Samples** | 50,000 (391 batches) |
| **Test Samples** | 10,000 (79 batches) |
| **Batch Size** | 128 |

### Preprocessing

**Training Transforms:**

| Augmentation | Purpose |
|---|---|
| `RandomCrop(32, padding=4)` | Spatial shift invariance |
| `RandomHorizontalFlip()` | Mirror symmetry robustness |
| `AutoAugment (CIFAR10 policy)` | Learned augmentation (rotate, shear, colour jitter, etc.) |
| `Normalize(mean, std)` | Zero-mean / unit-variance using CIFAR-100 statistics |
| `Cutout(n_holes=1, length=16)` | Masks a 16×16 patch — forces distributed feature learning |

**Test Transforms:** `ToTensor()` + `Normalize()` only (no augmentation).

**CIFAR-100 Normalization Stats:**
```
Mean: (0.5071, 0.4867, 0.4408)
Std:  (0.2675, 0.2565, 0.2761)
```

---

## 🏗️ Model Architecture

### EfficientLite CNN

```
Input (3 × 32 × 32)
  ↓ Stem: Conv2d(3→32, 3×3) → BN → GELU
  ↓ Stage 1: MBConv(32→64,  stride=1) × 2          [32×32]
  ↓ Stage 2: MBConv(64→128, stride=2) + MBConv × 2  [16×16]
  ↓ Stage 3: MBConv(128→192, stride=2) + MBConv × 2 [8×8]
  ↓ Stage 4: MBConv(192→256, stride=2) + MBConv × 1 [4×4]
  ↓ Head: Conv1×1(256→512) → BN → GELU → GAP → Flatten
  ↓ Dropout(0.3) → Linear(512 → 100)
```

### Building Blocks

**SEBlock (Squeeze-and-Excitation)**
- Global Avg Pool → Linear (squeeze, ratio=4) → GELU → Linear (excite) → Sigmoid
- Multiplies channel features by learned attention weights
- Tells the model *which channels matter more* for a given input

**DepthwiseSeparableConv**
- Depthwise Conv (spatial, per-channel) → BN → Pointwise Conv (1×1, channel mix) → BN → GELU
- ~8–9× fewer FLOPs than a standard convolution

**MBConv (Mobile Inverted Bottleneck)**
- Expand (1×1) → Depthwise Conv (3×3) → SEBlock → Project (1×1)
- Residual skip connection when `stride=1` and `in_ch == out_ch`
- Expansion ratio = 4

### Design Choices

| Component | Choice | Reason |
|---|---|---|
| Activation | **GELU** | Smoother than ReLU; empirically better at this scale |
| Normalization | **BatchNorm** after every conv | Stable training, acts as implicit regularizer |
| Pooling | **Global Average Pooling** | No large FC overhead; better spatial invariance |
| Dropout | **0.3** before classifier | Prevents co-adaptation of neurons |
| Weight Init | Kaiming Normal (conv), Truncated Normal (linear) | Proper initialization for deep nets |
| Downsampling | Stride=2 inside MBConv depthwise | More efficient than MaxPool |

---

## 🏋️ Training Setup

### Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Loss Function | CrossEntropyLoss (label_smoothing=0.1) |
| LR Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-5) |
| Mixup Alpha | 0.2 |
| Gradient Clip | max_norm = 1.0 |
| Random Seed | 42 |

### Optimizer & Scheduler Details

- **AdamW** — Adam with decoupled weight decay; robust without heavy LR tuning
- **CosineAnnealingWarmRestarts** — LR follows a cosine curve from 1e-3 → 1e-5, then restarts. `T_0=10` means first restart at epoch 10; `T_mult=2` doubles the interval on each restart. Helps escape local minima.

---

## 🔧 Techniques Applied

| Technique | Status | Details |
|---|---|---|
| **Mixup** | ✅ Applied | Alpha=0.2; blends images and labels to smooth decision boundaries |
| **CutOut** | ✅ Applied | 1 hole, 16×16; forces distributed feature learning |
| **AutoAugment** | ✅ Applied | CIFAR10 learned augmentation policy |
| **Label Smoothing** | ✅ Applied | 0.1 factor; prevents overconfidence on hard classes |
| **SE Attention** | ✅ Applied | Channel recalibration inside every MBConv block |
| **Depthwise Separable Conv** | ✅ Applied | Core of all MBConv blocks; ~8× FLOPs reduction |
| **Gradient Clipping** | ✅ Applied | max_norm=1.0; prevents exploding gradients |
| **Cosine LR + Warm Restarts** | ✅ Applied | T_0=10, T_mult=2 |
| **Best Model Checkpointing** | ✅ Applied | Saves `best_model.pth` on every test accuracy improvement |
| **Pruning** | ❌ Not applied | — |
| **Quantization** | ❌ Not applied | — |
| **Knowledge Distillation** | ❌ Not applied | — |
| **Weight Sharing / Clustering** | ❌ Not applied | — |

> The focus was on designing an inherently efficient architecture rather than applying post-training compression.

---

## 📊 Evaluation & Metrics

### Accuracy Metrics

| Metric | Description |
|---|---|
| **Top-1 Train/Test Accuracy** | % of correctly classified samples |
| **Top-5 Test Accuracy** | Whether the correct class appears in the top 5 predictions |
| **Per-Class Accuracy** | Accuracy computed individually for each of the 100 classes |
| **Lowest Class Accuracy** | Worst-performing class (fairness metric) |
| **Highest Class Accuracy** | Best-performing class |

### Generalization Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Accuracy Gap** | Train Acc − Test Acc | Lower = less overfitting |
| **Loss Gap** | Test Loss − Train Loss | Lower = less instability |

### Efficiency Metrics

| Metric | Tool Used |
|---|---|
| Trainable Parameters | `torchinfo` |
| FLOPs | `fvcore` FlopCountAnalysis |
| Model Size (MB) | `os.path.getsize` on saved `.pth` |
| Latency (batch=1) | Averaged over 200 runs via `time.perf_counter` |
| Latency (batch=32) | Averaged over 100 runs via `time.perf_counter` |
| Avg Epoch Time / Total Time | Tracked during training loop |

---

## 📈 Visualizations

The notebook generates the following plots:

**1. `training_curves.png`**
- Loss vs Epoch (Train & Test)
- Accuracy vs Epoch (Train & Test)

**2. `class_accuracy.png`**
- Bar chart of per-class accuracy sorted lowest → highest (red bars = below 30%)
- Histogram of class accuracy distribution with lowest/highest class markers

---

## ⚡ Efficiency Metrics Summary

All metrics are printed in a final summary block ready to submit:

```
Best Train Loss
Best Test Loss
Best Train Accuracy (%)
Best Test Accuracy (%)
Top-5 Test Accuracy (%)
GAP Accuracy (Train - Test)
GAP Loss (Test - Train)
Lowest Class Accuracy
Highest Class Accuracy
Trainable Parameters
FLOPs
Model Size (MB)
Avg Epoch Time (s)
Total Training Time (s)
Latency batch=1 (ms)
Latency batch=32 (ms)
```

---

## 💡 Key Observations

### Strengths
- Lightweight MBConv-based design (~2M params) is highly efficient for CIFAR-100
- Heavy regularization stack (Dropout + Label Smoothing + Mixup + CutOut + Weight Decay) effectively reduces overfitting
- SE attention adds minimal parameters but significantly improves accuracy
- Multi-metric evaluation covers accuracy, fairness, and deployment readiness
- Best checkpoint is used for evaluation — not the last epoch

### Limitations
- CIFAR-100 is inherently difficult (100 fine-grained classes); top-1 accuracy for ~2M param models typically caps at 60–70%
- No post-training compression (pruning/quantization) applied
- Training on CPU is very slow — a GPU (e.g., T4 on Colab) is strongly recommended

---

## 📦 Requirements

```bash
pip install torch torchvision torchinfo fvcore matplotlib numpy
```

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Model, training, dataset |
| `torchinfo` | Parameter count summary |
| `fvcore` | FLOPs analysis |
| `matplotlib` | Training curves and class accuracy plots |
| `numpy` | Numerical utilities |

---

## 🚀 Usage

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Run all cells in order (Sections 1–12)
3. Enable GPU runtime for faster training: **Runtime → Change Runtime Type → T4 GPU**
4. Final metrics are printed in **Section 10** — copy them directly to your submission form

### Notebook Sections

| Section | Description |
|---|---|
| 1 | Install Required Libraries |
| 2 | Imports & Reproducibility Setup |
| 3 | Data Loading & Augmentation |
| 4 | Model Architecture (EfficientLite CNN) |
| 5 | Loss, Optimizer & Scheduler |
| 6 | Training Loop |
| 7 | Training Plots |
| 8 | Per-Class Accuracy & Fairness Metrics |
| 9 | Efficiency Metrics (params, FLOPs, size, latency) |
| 10 | Complete Metrics Summary |
| 11 | Per-Class Accuracy Distribution Visualization |
| 12 | Architecture Summary (torchinfo) |

---

## 📁 Output Files

| File | Description |
|---|---|
| `best_model.pth` | Best model checkpoint (saved during training) |
| `model_weights.pth` | Final model weights (used for size measurement) |
| `training_curves.png` | Loss and accuracy curves over epochs |
| `class_accuracy.png` | Per-class accuracy bar chart and histogram |

---

*Built with PyTorch · CIFAR-100 · Custom Architecture from Scratch*
