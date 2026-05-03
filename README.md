# CIFAR-100-Custom-CNN-From-Scratch
Yes, the file uploaded perfectly! I have the full notebook. Here's your complete structured analysis:

---

# 📘 CIFAR-100 Custom CNN — Complete Notebook Analysis

---

## 1. 🎯 Overall Objective

- **Problem:** Multi-class image classification on **100 categories** using a custom-built CNN — **no pretrained weights used anywhere**.
- **Goal:** Maximize test accuracy while keeping the model **lightweight** (~2M parameters, low FLOPs, small file size).
- **Model Name:** `EfficientLiteCNN` — inspired by EfficientNet's design principles but built from scratch.
- **Evaluation is multi-metric:** accuracy, generalization gap, class fairness, efficiency (FLOPs, latency, model size).

> ⭐ **Tell the examiner:** *"The goal was not just accuracy but an accuracy-efficiency tradeoff — we wanted a model that is both accurate and deployable on constrained hardware."*

---

## 2. 📦 Dataset Details

### Dataset Used
- **CIFAR-100** — 100 classes, 32×32 colour images
- **Train:** 50,000 samples → 391 batches
- **Test:** 10,000 samples → 79 batches
- **Batch Size:** 128

### Preprocessing — Training Set
| Augmentation | Purpose |
|---|---|
| `RandomCrop(32, padding=4)` | Spatial shift invariance |
| `RandomHorizontalFlip()` | Mirror symmetry robustness |
| `AutoAugment(CIFAR10 policy)` | Learned augmentation policy (rotate, shear, colour jitter, etc.) |
| `Normalize(MEAN, STD)` | Zero-mean / unit-variance using CIFAR-100 statistics: mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) |
| `Cutout(n_holes=1, length=16)` | Randomly masks a 16×16 patch to force distributed feature learning |

### Preprocessing — Test Set
- Only `ToTensor()` + `Normalize()` — **no augmentation** (to measure true accuracy)

> ⭐ **Tell the examiner:** *"Cutout forces the model to not rely on a single spatial region. AutoAugment is a learned policy — it was searched on CIFAR and applies a sequence of transforms automatically."*

---

## 3. 🏗️ Model Architecture — EfficientLiteCNN

### Building Blocks

**A. SEBlock (Squeeze-and-Excitation)**
- Global Average Pool → Flatten → Linear (squeeze) → GELU → Linear (excite) → Sigmoid
- Multiplies channel features by learned weights
- Reduction ratio = 4 (e.g., 128 channels → 32 → 128)
- **Purpose:** Channel attention — tells the model *which channels matter more*

**B. DepthwiseSeparableConv**
- Depthwise Conv (each channel separately) → BN → Pointwise Conv (1×1 mixing) → BN → GELU
- ~8–9× fewer FLOPs than standard convolution

**C. MBConv (Mobile Inverted Bottleneck)**
- **Expand** (1×1 conv, ratio=4) → **Depthwise Conv** (3×3) → **SEBlock** → **Project** (1×1 conv)
- Residual skip connection added when input and output dimensions match (stride=1, same channels)

### Full Architecture Flow

```
Input (3 × 32 × 32)
  ↓ Stem: Conv2d(3→32, 3×3) → BN → GELU
  ↓ Stage 1: MBConv(32→64, s=1) × 2                   [32×32]
  ↓ Stage 2: MBConv(64→128, s=2) + MBConv×2            [16×16]
  ↓ Stage 3: MBConv(128→192, s=2) + MBConv×2           [8×8]
  ↓ Stage 4: MBConv(192→256, s=2) + MBConv×1           [4×4]
  ↓ Head: Conv1×1(256→512) → BN → GELU → GAP → Flatten
  ↓ Dropout(0.3) → Linear(512→100)
```

### Key Design Choices
| Component | Value | Reason |
|---|---|---|
| Activation | **GELU** | Smoother than ReLU, better for image classification |
| Normalization | **BatchNorm** after every conv | Stabilizes training, allows higher LR |
| Pooling | **Global Average Pooling** | No large FC overhead, better spatial invariance |
| Dropout | **0.3** before classifier | Prevents co-adaptation of neurons |
| Weight Init | Kaiming Normal (conv), Truncated Normal (linear) | Proper initialization for deep networks |

> ⭐ **Tell the examiner:** *"Downsampling is done inside MBConv using stride=2 on the depthwise conv — this is more efficient than using MaxPool."*

---

## 4. 🏋️ Training Process

### Hyperparameters
| Parameter | Value |
|---|---|
| Epochs | **100** |
| Batch Size | **128** |
| Optimizer | **AdamW** (lr=1e-3, weight_decay=1e-4) |
| Loss Function | **CrossEntropyLoss with Label Smoothing = 0.1** |
| LR Scheduler | **CosineAnnealingWarmRestarts** (T_0=10, T_mult=2, eta_min=1e-5) |
| Mixup Alpha | **0.2** |

### Training Techniques Used

**1. Label Smoothing (0.1)**
- Instead of hard labels (0 or 1), targets are softened (e.g., 0.9 for correct, 0.001 for others)
- Prevents overconfidence, improves calibration

**2. Mixup Augmentation (alpha=0.2)**
- Blends two random images and their labels: `mixed_x = λ·x_a + (1-λ)·x_b`
- λ sampled from Beta(0.2, 0.2) distribution
- Loss = `λ·loss(pred, y_a) + (1-λ)·loss(pred, y_b)`
- Smooths decision boundaries, reduces memorization

**3. CosineAnnealingWarmRestarts**
- LR follows a cosine curve from 1e-3 → 1e-5, then **restarts** periodically
- T_0=10 means first restart at epoch 10, then T_mult=2 doubles interval each time
- Helps escape local minima

**4. Gradient Clipping (max_norm=1.0)**
- Clips gradients to prevent exploding gradient problem

**5. Best Model Checkpoint**
- Saves model weights (`best_model.pth`) whenever test accuracy improves

**6. Seed Fixed to 42** for full reproducibility

> ⭐ **Tell the examiner:** *"Mixup + Label Smoothing together form a powerful combination against overfitting — Mixup works on inputs, Label Smoothing works on targets."*

---

## 5. 📊 Evaluation

### Metrics Tracked
| Metric | Description |
|---|---|
| **Train/Test Accuracy (Top-1)** | % of correctly classified samples |
| **Top-5 Accuracy** | Whether correct class is in top 5 predictions |
| **Train/Test Loss** | CrossEntropy loss value |
| **Generalization Gap (Acc)** | Train Acc − Test Acc (lower = less overfit) |
| **Generalization Gap (Loss)** | Test Loss − Train Loss |
| **Per-class Accuracy** | Accuracy for each of the 100 classes individually |
| **Lowest / Highest Class Accuracy** | Fairness metrics |

### Efficiency Metrics (Section 9)
| Metric | How Measured |
|---|---|
| **Trainable Parameters** | `torchinfo` summary |
| **FLOPs** | `fvcore` FlopCountAnalysis |
| **Model Size (MB)** | `os.path.getsize` on saved `.pth` |
| **Latency (batch=1)** | Averaged over 200 runs using `time.perf_counter` |
| **Latency (batch=32)** | Averaged over 100 runs |
| **Avg Epoch Time / Total Time** | Tracked during training loop |

### Visualizations Produced
- **`training_curves.png`** — Loss vs Epoch + Accuracy vs Epoch (side-by-side line plots)
- **`class_accuracy.png`** — Two plots:
  - Bar chart of per-class accuracy sorted from lowest to highest (red = below 30%)
  - Histogram of class accuracy distribution with lowest/highest markers

> ⭐ **Tell the examiner:** *"The notebook loads the best checkpoint (not the last epoch) for final evaluation — this ensures we measure peak performance."*

---

## 6. 🔧 Techniques Applied

| Technique | Present? | Details |
|---|---|---|
| **Pruning** | ❌ No | Not implemented |
| **Quantization** | ❌ No | Not implemented |
| **Weight Sharing / Clustering** | ❌ No | Not implemented |
| **Knowledge Distillation** | ❌ No | Not implemented |
| **Mixup** | ✅ Yes | Alpha=0.2, applied in training loop |
| **CutOut** | ✅ Yes | 1 hole, 16×16 patch |
| **AutoAugment** | ✅ Yes | CIFAR10 policy |
| **Label Smoothing** | ✅ Yes | 0.1 smoothing factor |
| **SE Attention** | ✅ Yes | Channel recalibration in every MBConv |
| **Depthwise Separable Conv** | ✅ Yes | Core of MBConv blocks |
| **Gradient Clipping** | ✅ Yes | max_norm=1.0 |
| **Cosine LR + Warm Restarts** | ✅ Yes | T_0=10, T_mult=2 |

> ⭐ **Tell the examiner:** *"Pruning and quantization are not applied here — the focus was on designing an inherently efficient architecture rather than post-training compression."*

---

## 7. 💡 Key Observations

### Performance Insights
- **Epoch 1 output visible:** Train Loss=4.2554, Train Acc=6.99%, Test Acc=15.34% — expected for early training on 100 classes
- Test accuracy briefly exceeds train accuracy in epoch 1, which is normal because: test set has no Mixup/CutOut (cleaner data), and Mixup makes training harder
- Model ran on **CPU** (no GPU detected in the run), causing very slow epoch times (~3378s/epoch)

### Strengths
- Lightweight design (~2M params) with MBConv blocks — efficient for CIFAR-100
- Heavy regularization stack (Dropout + Label Smoothing + Mixup + CutOut + Weight Decay) reduces overfitting
- SE attention adds minimal parameters but improves accuracy significantly
- Multi-metric evaluation (fairness, efficiency, top-5) makes it a comprehensive study
- Best model is checkpointed — evaluation uses peak weights, not last epoch

### Limitations
- Ran on **CPU** — 100 epochs would take extremely long (estimated 90+ hours)
- Only 1 epoch output is captured in the notebook — full training results not visible
- No pruning/quantization — model not optimized for deployment beyond architecture design
- CIFAR-100 is inherently hard (100 fine-grained classes) — top-1 accuracy for a ~2M param model typically caps around 60–70%
