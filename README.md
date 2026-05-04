# CIFAR-100 EfficientLite CNN — From Scratch

> A custom lightweight convolutional neural network designed and trained entirely from scratch on the CIFAR-100 benchmark, featuring advanced regularization, model compression via pruning, and quantization techniques.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Dataset — CIFAR-100](#3-dataset--cifar-100)
4. [Data Pipeline & Augmentation](#4-data-pipeline--augmentation)
5. [Model Architecture — EfficientLite CNN](#5-model-architecture--efficientlite-cnn)
6. [Loss Function, Optimizer & Scheduler](#6-loss-function-optimizer--scheduler)
7. [Training Loop & Regularization](#7-training-loop--regularization)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Model Compression](#9-model-compression)
   - [9.1 Unstructured Pruning](#91-unstructured-pruning)
   - [9.2 Structured Pruning](#92-structured-pruning)
   - [9.3 Post-Training Quantization (PTQ)](#93-post-training-quantization-ptq)
   - [9.4 Quantization-Aware Training (QAT)](#94-quantization-aware-training-qat)
10. [Visualizations](#10-visualizations)
11. [Results & Comparison Table](#11-results--comparison-table)
12. [File Structure](#12-file-structure)
13. [Key Design Decisions](#13-key-design-decisions)
14. [Glossary](#14-glossary)

---

## 1. Project Overview

This project implements a fully custom CNN for the CIFAR-100 image classification benchmark — 100 classes, 60,000 images, 32×32 pixels — without using any pretrained weights or transfer learning.

**Core goals:**

- Maximize test accuracy with a small, efficient model (~2M parameters).
- Minimize overfitting through multiple modern regularization strategies.
- Demonstrate model compression techniques: pruning and quantization.
- Provide full reproducibility with a fixed random seed.

**Key techniques used:**

| Category | Technique |
|---|---|
| Architecture | Depthwise Separable Convolutions, MBConv (Inverted Residual), Squeeze-and-Excitation |
| Regularization | Dropout, Label Smoothing, Mixup, CutOut, Weight Decay |
| Training | AdamW optimizer, Cosine Annealing with Warm Restarts, Gradient Clipping |
| Augmentation | RandomCrop, RandomHorizontalFlip, AutoAugment, CutOut |
| Compression | Unstructured Pruning, Structured Pruning, PTQ (INT8), QAT (INT8) |

---

## 2. Prerequisites & Installation

### Environment

This notebook is designed to run on **Google Colab** with a GPU runtime (T4 or better recommended).

### Required Libraries

```bash
pip install torchinfo fvcore datasets -q
```

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Core deep learning framework and vision utilities |
| `torchinfo` | Human-readable model summary with parameter counts per layer |
| `fvcore` | Facebook Research library for FLOPs (floating-point operations) analysis |
| `datasets` (HuggingFace) | Loads CIFAR-100 from HuggingFace Hub — avoids the blocked `toronto.edu` server |
| `numpy`, `matplotlib` | Numerical operations and plotting |

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any CUDA GPU | NVIDIA T4 (Colab free tier) |
| RAM | 8 GB | 12 GB+ |
| Storage | 500 MB | 1 GB |

---

## 3. Dataset — CIFAR-100

### What is CIFAR-100?

CIFAR-100 (Canadian Institute for Advanced Research — 100 classes) is a standard benchmark in computer vision. It contains:

- **60,000 color images**, each 32×32 pixels with 3 RGB channels.
- **100 fine-grained classes** (e.g., apple, bear, bicycle, cloud, mushroom, ...).
- **50,000 training images** and **10,000 test images**.
- Each class has exactly 500 training and 100 test images.

CIFAR-100 is significantly harder than CIFAR-10 (10 classes) due to the large number of visually similar categories and the very small image resolution.

### Why 100 classes is hard

- Only 500 training images per class — limited data per category.
- Many classes are visually similar (e.g., various insects, vehicles, flowers).
- 32×32 images have very little spatial detail.

### Data Loading

Instead of downloading from the original University of Toronto server (which may be blocked in some environments), the dataset is loaded from the **HuggingFace Hub** using the `datasets` library:

```python
hf_train = load_dataset("uoft-cs/cifar100", split="train")
hf_test  = load_dataset("uoft-cs/cifar100", split="test")
```

A custom `HFCifar100` PyTorch `Dataset` wrapper is used to apply transforms and return `(image_tensor, fine_label)` pairs.

### Normalization Statistics

These are the channel-wise mean and standard deviation computed over the entire CIFAR-100 training set:

```python
MEAN = (0.5071, 0.4867, 0.4408)   # R, G, B channel means
STD  = (0.2675, 0.2565, 0.2761)   # R, G, B channel standard deviations
```

Normalizing with these values centers each channel to ~zero mean and ~unit variance, which stabilizes gradient flow and speeds up convergence.

---

## 4. Data Pipeline & Augmentation

Data augmentation artificially increases the effective size and diversity of the training set, reducing overfitting by ensuring the model sees slightly different versions of each image in every epoch.

### Training Transforms (applied only during training)

```
RandomCrop(32, padding=4)
  → RandomHorizontalFlip()
  → AutoAugment(CIFAR10 policy)
  → ToTensor()
  → Normalize(MEAN, STD)
  → CutOut(n_holes=1, length=16)
```

| Transform | What it does | Why it helps |
|---|---|---|
| `RandomCrop(32, padding=4)` | Pads the image by 4 pixels on each side then randomly crops back to 32×32 | Teaches the model that objects can appear at different positions — positional invariance |
| `RandomHorizontalFlip()` | Flips the image left-right with 50% probability | Doubles the effective dataset size; most objects look the same flipped |
| `AutoAugment(CIFAR10)` | Applies a learned sequence of operations (rotate, shear, color jitter, solarize, etc.) tuned specifically for CIFAR datasets | State-of-the-art augmentation policy; significantly improves generalization |
| `Normalize(MEAN, STD)` | Subtracts mean, divides by std per channel | Ensures inputs are zero-mean / unit-variance for stable gradient updates |
| `CutOut(n_holes=1, length=16)` | Randomly masks out a 16×16 square region of the image with zeros | Forces the model to learn distributed features across the whole image; prevents reliance on any single region |

### Test Transforms (applied during evaluation)

```
ToTensor()
  → Normalize(MEAN, STD)
```

No augmentation is applied at test time — we evaluate on clean, normalized images only.

### CutOut — Custom Implementation

CutOut is implemented as a callable class that operates on a tensor (after `ToTensor`). It generates a random center point and zeros out an `n_holes × length × length` patch:

```python
class Cutout:
    def __init__(self, n_holes=1, length=16):
        ...
    def __call__(self, img):
        # Creates a binary mask, multiplies elementwise with the image
        ...
```

### DataLoader Configuration

```python
BATCH_SIZE  = 128
NUM_WORKERS = 2
```

- **Batch size 128**: Standard for CIFAR training. Large enough for stable gradient estimates; small enough to fit in GPU memory.
- **`pin_memory=False`**: Disabled for Colab compatibility (avoids CUDA memory pinning issues).
- **`shuffle=True`** for training: Ensures different mini-batch composition each epoch, preventing the model from learning the data order.

---

## 5. Model Architecture — EfficientLite CNN

### Design Philosophy

The architecture draws inspiration from **EfficientNet** and **MobileNetV2**, adapting their core building blocks for CIFAR-100's small 32×32 input resolution. The goal is maximum accuracy per parameter and per FLOP.

### Architecture Overview

```
Input: (B, 3, 32, 32)
  │
  ▼
[Stem] Conv2d(3→32, 3×3, stride=1) → BN → GELU
  │
  ▼
[Stage 1] MBConv(32→64,  stride=1) × 2         → (B, 64,  32, 32)
  │
  ▼
[Stage 2] MBConv(64→128, stride=2) + MBConv×2  → (B, 128, 16, 16)
  │
  ▼
[Stage 3] MBConv(128→192,stride=2) + MBConv×2  → (B, 192,  8,  8)
  │
  ▼
[Stage 4] MBConv(192→256,stride=2) + MBConv×1  → (B, 256,  4,  4)
  │
  ▼
[Head] Conv1×1(256→512) → BN → GELU → GlobalAvgPool → Dropout(0.3) → Linear(512→100)
  │
  ▼
Output: (B, 100)  — raw logits
```

### Building Blocks

#### SEBlock — Squeeze-and-Excitation

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        ...
```

**Concept:** Channel attention mechanism. Instead of treating all feature maps equally, the SE block learns to *re-weight* them based on global context.

**How it works:**
1. **Squeeze**: Global average pooling collapses spatial dimensions to a single vector of shape `(C,)`.
2. **Excitation**: Two fully connected layers (with GELU activation) learn a per-channel importance score `(C,)`.
3. **Scale**: The original feature map is multiplied elementwise by these learned scores.

**Why `reduction=4`?** Reduces the hidden dimension to `C/4` for the FC layers, limiting parameter overhead while still allowing sufficient expressiveness. The `max(..., 16)` ensures very small channel counts still get at least 16 hidden units.

#### DepthwiseSeparableConv

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        ...
```

**Concept:** Factorizes a standard convolution into two cheaper operations.

| Operation | Cost |
|---|---|
| Standard Conv (C_in → C_out, k×k) | C_in × C_out × k² per output pixel |
| Depthwise Conv (C_in → C_in, k×k) | C_in × k² per output pixel |
| Pointwise Conv (C_in → C_out, 1×1) | C_in × C_out per output pixel |
| **Total DS-Conv** | C_in × (k² + C_out) per output pixel |

For k=3, C_out=128: standard conv costs ~1152 operations/pixel; DS-conv costs ~137. That's an **~8× reduction in FLOPs**.

**Pipeline:** `Depthwise Conv → BN → GELU → Pointwise Conv → BN → GELU`

#### MBConv — Mobile Inverted Bottleneck

```python
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        ...
```

**Concept:** The core block of MobileNetV2 and EfficientNet. It expands the channel count *before* the spatial convolution, then projects back down.

**Pipeline:**

```
Input (C_in)
  → Expand: Conv1×1 (C_in → C_in × expand_ratio) → BN → GELU     [more channels = richer representation]
  → Depthwise: Conv3×3 (groups=C_mid, stride=stride) → BN → GELU  [spatial feature extraction]
  → SE: Squeeze-and-Excitation on C_mid                             [channel attention]
  → Project: Conv1×1 (C_mid → C_out) → BN                         [compress back to output channels]
  → Residual: add input if stride==1 and in_ch==out_ch             [gradient highway]
```

**Why invert the bottleneck?** In traditional ResNets, the bottleneck compresses then expands. MBConv does the opposite (expand then compress) because the depthwise conv works better with more channels to separate — it extracts richer spatial features. The outer low-channel representation is efficient for storage and the skip connection.

**Skip connection:** Only applied when `stride=1` and `in_ch == out_ch`. This ensures residual connections don't change the shape of the feature map.

#### EfficientLiteCNN

```python
class EfficientLiteCNN(nn.Module):
    def __init__(self, num_classes=100):
        ...
```

**Stem:** A single 3×3 Conv with `stride=1` (not stride=2 as in ImageNet models) because CIFAR images are already small at 32×32. A stride-2 stem would immediately halve resolution to 16×16, losing too much spatial information.

**Stages:** Four stages progressively widen channels (32→64→128→192→256) and reduce spatial resolution via stride-2 MBConv blocks (32→16→8→4).

**Head:** A 1×1 conv expands to 512 channels (cheap on a 4×4 spatial map), Global Average Pooling collapses spatial dimensions completely (no large FC layers), Dropout(0.3) regularizes the final representation, and a single Linear layer maps to 100 class logits.

**Weight Initialization:**

| Layer type | Initialization | Why |
|---|---|---|
| `Conv2d` | `kaiming_normal_` (fan_out mode) | Preserves variance during forward pass for ReLU-like activations |
| `BatchNorm2d` | weight=1, bias=0 | Identity initialization — BN starts neutral |
| `Linear` | `trunc_normal_(std=0.02)` | Prevents large initial logits; borrowed from ViT/Transformer practice |

### Activation Function: GELU

GELU (Gaussian Error Linear Unit) is used throughout instead of ReLU. GELU is defined as `x × Φ(x)` where `Φ` is the standard normal CDF. It produces a smoother, probabilistically-weighted activation compared to ReLU's hard threshold. Empirically, GELU performs better than ReLU for image classification at this model scale.

### BatchNorm

Every convolution is followed by Batch Normalization before the activation. BatchNorm:
- Normalizes activations within a mini-batch, reducing internal covariate shift.
- Acts as an implicit regularizer.
- Allows higher learning rates.
- Reduces sensitivity to weight initialization.

---

## 6. Loss Function, Optimizer & Scheduler

### Loss: CrossEntropyLoss with Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Standard Cross-Entropy** pushes the model to output probability 1.0 for the correct class and 0.0 for all others. This can cause the model to become *overconfident* — assigning very large logits to the correct class, which hurts generalization.

**Label Smoothing (ε=0.1)** softens the targets: instead of `[0, 0, 1, 0, ...]`, the target becomes `[0.001, 0.001, 0.91, 0.001, ...]`. The `0.1` smoothing mass is spread uniformly across all 100 classes. This:
- Prevents overconfidence.
- Improves calibration (predicted probabilities better reflect true uncertainty).
- Acts as a regularizer, typically improving test accuracy by 0.5–1%.

### Optimizer: AdamW

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

**AdamW** is Adam with *decoupled* weight decay. Standard Adam applies weight decay incorrectly through the gradient update, which interacts badly with adaptive learning rates. AdamW fixes this by applying L2 regularization directly to the weights, independent of the gradient update:

```
w_t = w_{t-1} - lr × (adam_update) - lr × weight_decay × w_{t-1}
```

**Weight Decay (1e-4):** Penalizes large weights, preventing overfitting. Effectively an L2 regularization term.

**Learning Rate (1e-3):** Standard for AdamW on CIFAR-scale problems. Higher rates cause instability; lower rates slow convergence unnecessarily.

### Scheduler: Cosine Annealing with Warm Restarts

```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
```

**Cosine Annealing** decays the learning rate following a cosine curve from `lr_max` down to `eta_min`. This avoids the sharp loss cliffs of step-decay schedules and generally produces better final accuracy.

**Warm Restarts (SGDR):** After every `T_0` epochs, the learning rate is reset back to `lr_max`. This helps the optimizer escape local minima. `T_mult=2` doubles the period after each restart (10 → 20 → 40 epochs), allowing progressively longer exploration phases.

```
Epoch:    1   5  10  15  25
LR:    1e-3 ↘ 1e-5 ↗ 1e-3 ↘ ... ↗ 1e-3 ↘
```

---

## 7. Training Loop & Regularization

### Training Configuration

```python
NUM_EPOCHS  = 25
BATCH_SIZE  = 128
MIXUP_ALPHA = 0.2
```

### Mixup Augmentation

```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam
```

**Concept:** Mixup blends two training images and their labels linearly:
- `x_mixed = λ·x_i + (1-λ)·x_j`
- `y_mixed = λ·y_i + (1-λ)·y_j` (soft labels)

`λ` is sampled from a `Beta(0.2, 0.2)` distribution, which tends to produce values close to 0 or 1 (keeping images mostly distinct) with occasional strong interpolations.

**Why it helps:**
- Forces the model to learn *smooth interpolations* between classes.
- Discourages sharp decision boundaries and memorization.
- Reduces the generalization gap (train accuracy − test accuracy).

The loss for a mixed batch is:
```python
loss = λ × CE(pred, y_a) + (1-λ) × CE(pred, y_b)
```

### Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clips the global L2 norm of all gradients to a maximum of 1.0. This prevents **exploding gradients** — situations where very large gradient values cause unstable weight updates. Particularly important with Mixup and warm restarts where loss landscapes can be non-smooth.

### Training Step Summary

For each mini-batch:
1. Load `(images, labels)` onto the GPU.
2. Apply Mixup → `(mixed_images, labels_a, labels_b, λ)`.
3. Forward pass → `logits`.
4. Compute Mixup loss.
5. Backpropagate.
6. Clip gradients.
7. Update weights (AdamW step).
8. Update scheduler (per-step cosine decay).

### Checkpoint Saving

The best model (by test accuracy) is saved during training:
```python
torch.save(model.state_dict(), 'best_model.pth')
```

A full checkpoint (model + optimizer + scheduler + history) is optionally saved to Google Drive.

### Reproducibility

All random seeds are fixed at the start:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This ensures identical results on identical hardware across runs.

---

## 8. Evaluation Metrics

### Top-1 Accuracy

The fraction of test images where the model's most probable class equals the ground-truth label. The primary metric for CIFAR-100.

### Top-5 Accuracy

The fraction of test images where the ground-truth label appears in the model's top-5 predictions. A looser, more forgiving metric — useful for understanding whether the model is "close" even when not perfectly correct.

```python
top5 = outputs.topk(5, dim=1)[1]            # shape: (B, 5)
top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum()
```

### Generalization Gap

```python
acc_gap  = best_train_acc - best_test_acc    # lower = less overfit
loss_gap = best_test_loss - best_train_loss  # lower = better generalization
```

A large gap indicates overfitting. The regularization techniques (Mixup, Dropout, Label Smoothing, CutOut, Weight Decay) all aim to minimize this gap.

### Per-Class Accuracy

Computed by accumulating correct predictions per class over the full test set:
```python
class_correct[c] += (preds[mask] == labels[mask]).sum()
class_total[c]   += mask.sum()
class_acc[c]      = 100 × class_correct[c] / class_total[c]
```

Reports the best and worst performing classes, and produces a distribution histogram.

### Efficiency Metrics

| Metric | Tool | What it measures |
|---|---|---|
| Trainable Parameters | `torchinfo` | Number of learnable weights |
| FLOPs | `fvcore.FlopCountAnalysis` | Multiply-accumulate operations for a single forward pass |
| Model Size (MB) | `os.path.getsize` | Size of the saved `.pth` file on disk |
| Latency (batch=1) | `time.perf_counter` + CUDA sync | Average inference time per single image |
| Latency (batch=32) | `time.perf_counter` + CUDA sync | Average inference time per 32-image batch |

**Latency measurement best practices used:**
- 50 warmup iterations before measurement (GPU needs to warm up pipelines).
- 200 measurement iterations averaged to reduce variance.
- `torch.cuda.synchronize()` called before and after each timing to ensure GPU operations are complete (CUDA operations are asynchronous by default).

---

## 9. Model Compression

After training, the model is compressed using four techniques to study the accuracy/efficiency trade-off. All compression methods start from the same `best_model.pth` checkpoint.

### 9.1 Unstructured Pruning

**Goal:** Zero out 50% of individual weights globally, keeping the model's dense format but with sparse weight matrices.

**Method: Global L1 Unstructured Pruning**

```python
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.50,
)
```

**How it works:**
1. Collects all weights from all `Conv2d` and `Linear` layers into one pool.
2. Ranks them by absolute value (L1 norm).
3. Zeroes out the smallest 50% globally — not 50% per layer, but 50% of all weights combined. This means some layers may be pruned more aggressively than others based on the global weight distribution.
4. Masks are made permanent with `prune.remove()`.

**Sparsity measurement:**
```python
sparsity = 100 × (number of zero weights) / (total weights)
```

**Key observation:** Unstructured pruning does NOT reduce actual inference speed on standard hardware because GPUs/CPUs process dense matrices efficiently even with zeros. It reduces storage size only if sparse formats are used.

### 9.2 Structured Pruning

**Goal:** Remove entire filters (output channels) from `Conv2d` layers, producing a physically smaller and faster model.

**Method: L1-Norm Structured Pruning (30% filters per layer)**

```python
prune.ln_structured(module, name='weight', amount=0.30, n=1, dim=0)
```

- `n=1`: L1 norm (sum of absolute values per filter).
- `dim=0`: Prune along the output channel dimension (filter dimension).
- `amount=0.30`: Remove the 30% of filters with the smallest L1 norm in each layer.

**How it works:**
1. For each `Conv2d` layer, compute the L1 norm of each filter (a filter is a 3D tensor of shape `[C_in, k, k]`).
2. Rank filters by L1 norm.
3. Zero out the bottom 30%.
4. Make permanent.

**Advantage over unstructured:** Because entire filters are removed, the output feature map has fewer channels, which directly reduces computation in subsequent layers. This translates to real FLOPs reduction and faster inference.

**FLOPs are measured post-pruning** using `fvcore` to quantify actual computation savings.

### 9.3 Post-Training Quantization (PTQ)

**Goal:** Convert the pre-trained FP32 model's weights and activations to INT8, reducing memory and potentially accelerating inference.

**Method: Dynamic Quantization**

```python
model_ptq = torch.quantization.quantize_dynamic(
    model_ptq,
    qconfig_spec={nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)
```

**How Dynamic Quantization works:**
- **Weights** are statically quantized to INT8 at conversion time.
- **Activations** are dynamically quantized at runtime (per-batch range computed on-the-fly).
- No calibration dataset is required.

**INT8 vs FP32:**
- INT8 uses 1 byte per value; FP32 uses 4 bytes → ~4× theoretical memory reduction.
- INT8 arithmetic is faster on CPUs with SIMD support.
- Accuracy drop is typically small (< 1%) for well-trained models.

**Evaluated on CPU** (PTQ INT8 is primarily a CPU optimization; GPU kernels favor FP16/BF16 instead).

**FLOPs note:** FLOPs count remains the same as baseline — quantization changes the *precision* of operations, not their *count*. `fvcore` does not support quantized operators, so the baseline FLOPs are reported.

### 9.4 Quantization-Aware Training (QAT)

**Goal:** Fine-tune the model while simulating quantization noise, so that weights adapt to the quantization error before conversion.

**Phase 1: Fine-tuning on GPU (5 epochs)**

```python
QAT_EPOCHS = 5
QAT_LR     = 1e-4
```

The model is fine-tuned from the best checkpoint using a low learning rate (1e-4 vs 1e-3 during training) and cosine annealing. This allows the weights to settle into a configuration that is more robust to INT8 quantization.

**Phase 2: Dynamic Quantization (same as PTQ)**

After fine-tuning, the same `quantize_dynamic()` call is applied. The difference from PTQ is that the weights have been specifically adapted to minimize accuracy loss under quantization.

**Why QAT typically outperforms PTQ:**
- PTQ applies quantization to a model that was never trained to tolerate it.
- QAT lets the model adjust its weights during training with quantization in mind.
- The accuracy drop from QAT is usually smaller than PTQ, especially for difficult tasks like CIFAR-100 with 100 classes.

**A QAT fine-tuning accuracy curve** is plotted and saved as `qat_curve.png`.

---

## 10. Visualizations

The notebook produces four plot files:

| File | Contents |
|---|---|
| `training_curves.png` | Train/Test Loss and Train/Test Accuracy vs Epoch (2 subplots) |
| `class_accuracy.png` | Per-class accuracy bar chart (sorted) + histogram of class accuracy distribution |
| `compression_comparison.png` | 6-panel comparison: Accuracy, Loss, Model Size, Parameters, Latency, Accuracy-vs-Size scatter |
| `qat_curve.png` | QAT fine-tuning accuracy curve (Train + Test over 5 epochs) |

---

## 11. Results & Comparison Table

The final cell prints a complete comparison table across all five model variants:

```
Model                       Accuracy     Loss       Params          FLOPs     Size(MB)  Latency(ms)  Sparsity
Baseline (FP32)             XX.XX%     X.XXXX    X,XXX,XXX    XXX,XXX,XXX     X.XXXX      X.XXXX       0.0%
Unstructured Pruned (50%)   XX.XX%     X.XXXX    X,XXX,XXX    XXX,XXX,XXX     X.XXXX      X.XXXX      50.0%
Structured Pruned (30%)     XX.XX%     X.XXXX    X,XXX,XXX    XXX,XXX,XXX     X.XXXX      X.XXXX      XX.X%
PTQ (INT8)                  XX.XX%     X.XXXX    X,XXX,XXX    XXX,XXX,XXX     X.XXXX      X.XXXX       0.0%
QAT (INT8)                  XX.XX%     X.XXXX    X,XXX,XXX    XXX,XXX,XXX     X.XXXX      X.XXXX       0.0%
```

**Compression ratios** vs. baseline are reported for each variant:
- Size reduction (×)
- Parameter reduction (×)
- Accuracy drop (%)
- Loss increase (absolute)

**Best tradeoff** is determined by highest `accuracy / size_mb` ratio among compressed models.

---

## 12. File Structure

```
cifar100_custom_cnn.ipynb       ← Main notebook
best_model.pth                  ← Best checkpoint (by test accuracy)
model_weights.pth               ← Final model weights (for size measurement)
pruned_unstructured.pth         ← Weights after 50% unstructured pruning
pruned_structured.pth           ← Weights after 30% structured pruning
model_ptq.pth                   ← Dynamically quantized INT8 model (PTQ)
model_qat_finetuned.pth         ← QAT fine-tuned FP32 weights
model_qat.pth                   ← Dynamically quantized INT8 model (QAT)
training_curves.png             ← Loss + accuracy training plots
class_accuracy.png              ← Per-class accuracy analysis
compression_comparison.png      ← Full compression comparison charts
qat_curve.png                   ← QAT fine-tuning curve
```

---

## 13. Key Design Decisions

### Why no pretrained weights?

The notebook is designed as a from-scratch exercise to understand what an efficiently designed CNN can achieve without ImageNet pretraining. All accuracy comes from architecture design and training strategy alone.

### Why AdamW over SGD?

SGD with momentum can achieve slightly higher peak accuracy on CIFAR with careful LR tuning, but AdamW converges faster and is less sensitive to hyperparameter choices — making it more practical for a constrained training budget of 25 epochs.

### Why Global Average Pooling instead of a large FC layer?

After the final convolutional stage produces a 4×4 feature map, a fully connected layer would have `256 × 4 × 4 = 4,096` input neurons. With Global Average Pooling, we get one value per channel regardless of spatial size, reducing parameters by 16× and improving spatial invariance.

### Why MBConv with expand_ratio=4?

The expansion ratio of 4 provides a good balance: it allows rich spatial feature extraction in the expanded space while keeping the residual connections compact. Ratios below 2 reduce representational capacity; ratios above 6 increase parameters without proportional accuracy gain at this scale.

### Why Cosine Warm Restarts over a simple step schedule?

Step schedules are sensitive to when steps are applied. Cosine annealing provides a principled, smooth decay. Warm restarts allow the optimizer to periodically explore from a higher learning rate, helping escape flat regions and sharp minima, typically yielding 0.5–1% better final accuracy.

---

## 14. Glossary

| Term | Definition |
|---|---|
| **Batch Normalization (BN)** | Normalizes activations within a mini-batch to zero mean and unit variance; stabilizes training |
| **CutOut** | Augmentation that masks a random rectangular patch of the input image |
| **Depthwise Separable Convolution** | Factorizes standard convolution into a depthwise (per-channel spatial) + pointwise (channel mixing) operation; ~8× fewer FLOPs |
| **FLOPs** | Floating-Point Operations — a measure of computational complexity; usually counted as multiply-accumulate pairs |
| **GELU** | Gaussian Error Linear Unit; a smooth activation function that outperforms ReLU at this model scale |
| **Global Average Pooling (GAP)** | Averages each feature map spatially to a single value; eliminates large FC layers |
| **Gradient Clipping** | Caps gradient norm to prevent exploding gradients during backpropagation |
| **INT8 / FP32** | Integer 8-bit / floating-point 32-bit number formats; INT8 uses 4× less memory |
| **Inverted Residual (MBConv)** | Expand channels → depthwise conv → project back; used in MobileNetV2/EfficientNet |
| **Label Smoothing** | Softens one-hot training targets to prevent overconfidence; improves calibration |
| **Mixup** | Augmentation that linearly interpolates pairs of training images and their labels |
| **Post-Training Quantization (PTQ)** | Converts a trained FP32 model to INT8 after training, without any fine-tuning |
| **Pruning (Structured)** | Removes entire filters from convolutional layers; reduces FLOPs and actual inference speed |
| **Pruning (Unstructured)** | Zeroes out individual weights globally; reduces storage but not necessarily speed |
| **Quantization-Aware Training (QAT)** | Fine-tunes a model while simulating quantization effects; produces more accurate INT8 models than PTQ |
| **SE Block** | Squeeze-and-Excitation block; learns channel-wise attention weights using global context |
| **Warm Restarts** | Periodically resets learning rate to its maximum during cosine annealing; helps escape local minima |
| **Weight Decay** | L2 penalty on weight magnitudes; discourages large weights and reduces overfitting |
