# CIFAR-100 EfficientLite CNN — Complete Project Guide
### Training · Pruning · Quantization · Viva Preparation

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset — CIFAR-100](#4-dataset--cifar-100)
5. [Model Architecture — EfficientLite CNN](#5-model-architecture--efficientlite-cnn)
6. [Training Pipeline](#6-training-pipeline)
7. [Pruning — Deep Explanation](#7-pruning--deep-explanation)
8. [Quantization — Deep Explanation](#8-quantization--deep-explanation)
9. [Running the Notebook](#9-running-the-notebook)
10. [Metrics Explained](#10-metrics-explained)
11. [Expected Results](#11-expected-results)
12. [Common Issues & Fixes](#12-common-issues--fixes)
13. [Viva Questions & Answers](#13-viva-questions--answers)

---

## 1. Project Overview

This project trains a custom lightweight CNN called **EfficientLite** on the CIFAR-100 dataset from scratch (no pretrained weights), then applies four model compression techniques to reduce its size and improve inference speed while maintaining accuracy.

### Goals
- Build a parameter-efficient CNN (~2M params) that achieves strong accuracy on 100-class image classification
- Apply **Unstructured Pruning**, **Structured Pruning**, **Post-Training Quantization (PTQ)**, and **Quantization-Aware Training (QAT)**
- Compare all 5 variants (baseline + 4 compressed) across Accuracy, Loss, FLOPs, Model Size, and Latency

### Why this matters
Deep learning models are often too large and slow for real-world deployment on edge devices (phones, microcontrollers, embedded systems). Model compression techniques like pruning and quantization make it possible to deploy powerful models under strict memory and latency constraints.

---

## 2. Project Structure

```
cifar100_custom_cnn/
│
├── cifar100_custom_cnn.ipynb       # Main notebook — all sections
│
├── Checkpoints
│   ├── best_model.pth              # Best weights saved during training
│   ├── pruned_unstructured.pth     # Weights after unstructured pruning
│   ├── pruned_structured.pth       # Weights after structured pruning
│   ├── model_ptq.pth               # INT8 model after PTQ
│   └── model_qat.pth               # INT8 model after QAT fine-tuning
│
├── Plots
│   ├── training_curves.png         # Train/test loss and accuracy vs epoch
│   ├── class_accuracy.png          # Per-class accuracy distribution
│   ├── qat_curve.png               # QAT fine-tuning accuracy curve
│   └── compression_comparison.png  # Final comparison across all 5 variants
│
└── README.md                       # This file
```

---

## 3. Environment Setup

### Platform
- **Google Colab** with GPU runtime
- Set via: `Runtime → Change runtime type → T4 GPU`

### Libraries

| Library | Purpose |
|---------|---------|
| torch | Core deep learning framework |
| torchvision | Transforms, augmentations |
| torchinfo | Model summary, parameter counts |
| fvcore | FLOPs / multiply-add analysis |
| datasets | HuggingFace CIFAR-100 loader |
| matplotlib | Plotting training curves |
| numpy | Numerical operations |

### Installation
```bash
pip install torchinfo fvcore datasets -q
```

### Reproducibility
The following seeds are fixed at **42** to ensure identical results across runs:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 4. Dataset — CIFAR-100

### Overview
| Property | Value |
|----------|-------|
| Classes | 100 fine-grained classes |
| Train samples | 50,000 (500 per class) |
| Test samples | 10,000 (100 per class) |
| Image size | 32 × 32 pixels, RGB |
| Source | HuggingFace (`uoft-cs/cifar100`) |

### Why HuggingFace instead of torchvision?
The Toronto university server (`toronto.edu`) that hosts the original CIFAR-100 is often blocked in Colab. HuggingFace provides the same dataset reliably.

### Class Hierarchy
CIFAR-100 has **20 superclasses** (e.g., vehicles, animals, household items), each containing **5 fine classes** (e.g., "vehicles" → car, truck, bus, motorcycle, bicycle). The model predicts the **fine label** (0–99).

### Normalization Statistics
```
Mean = (0.5071, 0.4867, 0.4408)   # per channel (R, G, B)
Std  = (0.2675, 0.2565, 0.2761)
```
These are computed from the CIFAR-100 training set. Normalization centers each channel to zero mean and unit variance, which stabilizes training and speeds up convergence.

### Data Augmentations (Training only)

| Augmentation | What it does | Why |
|-------------|-------------|-----|
| `RandomCrop(32, padding=4)` | Pads image by 4px then crops randomly | Forces spatial invariance |
| `RandomHorizontalFlip` | 50% chance of horizontal flip | Doubles effective dataset size |
| `AutoAugment (CIFAR10)` | Applies learned augmentation policy (15+ ops) | Rotate, shear, color jitter etc. |
| `Normalize` | Subtract mean, divide by std | Zero-mean unit-variance inputs |
| `CutOut(16×16)` | Masks a random 16×16 patch to zero | Forces model to use whole image |

Test set uses **only Normalize** — no augmentation, since we want clean evaluation.

---

## 5. Model Architecture — EfficientLite CNN

### Design Philosophy
The goal is maximum accuracy per parameter — not just high accuracy. This is measured by the accuracy/efficiency tradeoff. No pretrained weights are used.

### Full Architecture Flow
```
Input: (B, 3, 32, 32)
│
├── Stem: Conv2d(3→32, k=3, pad=1) → BN → GELU
│   Output: (B, 32, 32, 32)
│
├── Stage 1: MBConv(32→64, stride=1) × 2
│   Output: (B, 64, 32, 32)
│
├── Stage 2: MBConv(64→128, stride=2) + MBConv(128→128) × 2
│   Output: (B, 128, 16, 16)    ← spatial: 32→16
│
├── Stage 3: MBConv(128→192, stride=2) + MBConv(192→192) × 2
│   Output: (B, 192, 8, 8)     ← spatial: 16→8
│
├── Stage 4: MBConv(192→256, stride=2) + MBConv(256→256)
│   Output: (B, 256, 4, 4)     ← spatial: 8→4
│
├── Head: Conv1×1(256→512) → BN → GELU → GAP → Flatten
│   Output: (B, 512)
│
├── Dropout(0.3)
│
└── Linear(512→100)
    Output: (B, 100)  ← class logits
```

### Building Blocks

#### 1. Depthwise Separable Convolution
Standard convolution applies one filter across all input channels simultaneously.
Depthwise separable splits this into two steps:
- **Depthwise conv:** Each filter operates on only ONE channel (groups=in_channels)
- **Pointwise conv:** 1×1 conv mixes channels

```
Standard Conv FLOPs:      C_in × C_out × K × K
Depthwise Sep FLOPs:      C_in × K × K  +  C_in × C_out
Speedup factor:           ~8-9× for K=3
```

#### 2. Squeeze-and-Excitation (SE) Block
A channel attention mechanism that asks: "Which channels are most important for this input?"

```
Input feature map (B, C, H, W)
  → Global Average Pool → (B, C)        # Squeeze: summarize spatial info
  → Linear(C → C/4) → GELU             # Bottleneck (reduction ratio = 4)
  → Linear(C/4 → C) → Sigmoid          # Excite: gate per channel (0 to 1)
  → Reshape to (B, C, 1, 1)
  → Multiply with input                 # Recalibrate channels
```
The output is input-adaptive — different images activate different channels.

#### 3. MBConv (Mobile Inverted Bottleneck)
The core building block, adapted from MobileNetV2/EfficientNet:
```
Input (C_in channels)
  → Expand: Conv1×1(C_in → C_in × 4)   # expand_ratio = 4
  → Depthwise Conv3×3
  → SE Block
  → Project: Conv1×1(C_in×4 → C_out)
  → (+ residual skip if stride=1 and C_in == C_out)
```
"Inverted" means it expands then compresses — opposite of traditional bottlenecks.

#### 4. BatchNorm
Applied after every convolution:
```
y = (x - mean) / sqrt(var + eps) × gamma + beta
```
Benefits: faster convergence, allows higher learning rates, acts as implicit regularizer.

#### 5. GELU Activation
`GELU(x) = x × Φ(x)` where Φ is the Gaussian CDF.
Smoother than ReLU near zero — doesn't hard-zero negative values, just attenuates them.

#### 6. Global Average Pooling (GAP)
Averages each feature map channel spatially:
```
(B, 512, 4, 4) → GAP → (B, 512)
```
Eliminates large FC layers. Provides spatial invariance — feature location doesn't matter.

#### 7. Weight Initialization
- Conv2d: Kaiming Normal — prevents vanishing/exploding gradients
- BatchNorm: weight=1, bias=0
- Linear: Truncated Normal (std=0.02)

---

## 6. Training Pipeline

### Loss Function — CrossEntropyLoss with Label Smoothing
Standard cross-entropy: `L = -log(p_correct_class)`

Label smoothing (ε=0.1) modifies the target distribution:
```
y_smooth = (1 - ε) × y_onehot + ε / num_classes
         = 0.9 for correct class, 0.001 for each wrong class
```
Prevents overconfidence, improves calibration, reduces overfitting on hard labels.

### Optimizer — AdamW
Adam with **decoupled weight decay**. Standard Adam conflates L2 regularization with the adaptive gradient update. AdamW decouples them, applying weight decay directly to parameters:
```
θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε) - α × λ × θ_{t-1}
```
Cleaner regularization. Less hyperparameter-sensitive than SGD.

### Scheduler — CosineAnnealingWarmRestarts
```
η_t = η_min + 0.5(η_max - η_min)(1 + cos(π × T_cur / T_0))
```
- `T_0=10`: first restart after 10 epochs
- `T_mult=2`: each cycle doubles in length (10 → 20 → 40 epochs)
- Restarts help escape local minima by temporarily increasing LR

### Mixup Augmentation
Creates virtual training samples by blending two images and their labels:
```
x_mixed = λ × x_i + (1-λ) × x_j       where λ ~ Beta(0.2, 0.2)
L = λ × CE(pred, y_i) + (1-λ) × CE(pred, y_j)
```
Forces smoother decision boundaries, reduces memorization.

### Gradient Clipping
`nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
Scales gradient vector if its L2 norm exceeds 1.0. Prevents exploding gradients.

### Hyperparameter Summary

| Hyperparameter | Value | Reason |
|----------------|-------|--------|
| Epochs | 50 | Sufficient convergence for ~2M param model |
| Batch Size | 128 | Standard for CIFAR on GPU |
| Optimizer | AdamW | Clean weight decay, less LR sensitivity |
| Learning Rate | 1e-3 | Standard for AdamW on CIFAR |
| Weight Decay | 1e-4 | L2 regularization |
| Label Smoothing | 0.1 | Prevents overconfidence |
| Mixup Alpha | 0.2 | Smoother decision boundaries |
| Dropout | 0.3 | Reduces co-adaptation |
| Grad Clip | 1.0 | Prevents exploding gradients |

---

## 7. Pruning — Deep Explanation

### What is Pruning?
Neural network pruning removes parameters (weights, neurons, filters) from a trained model to reduce size and computation with minimal accuracy loss.

### Core Motivation
Trained neural networks are typically **over-parameterized** — they have far more weights than strictly necessary to represent the learned function. The **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019) suggests that within any large network there exists a small "winning ticket" subnetwork that could achieve the same accuracy if trained in isolation. Pruning finds and keeps this subnetwork.

---

### Part 1 — Unstructured Pruning (Weight-level)

**What gets removed:** Individual scalar weights anywhere in the network, regardless of position.

**Method — L1 Magnitude Pruning:**
Remove weights with the smallest absolute value.
Assumption: small-magnitude weights contribute little to the output, so zeroing them causes minimal accuracy loss.

**Global vs Local:**
- Local: prune X% from each layer independently
- Global (used here): rank ALL weights across ALL layers together, zero the bottom X%

Global is better because it naturally protects important layers (early conv layers) from being over-pruned.

**PyTorch mechanism:**
```python
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.50)
```
PyTorch uses a weight_mask tensor. `prune.remove()` bakes zeros into the actual weights permanently.

**Why 50%?**
- Below 50%: minimal compression, negligible accuracy loss
- 50%: good compression, ~1–3% accuracy loss
- Above 80%: significant accuracy degradation without fine-tuning

**Key limitation:**
Zeroed weights are still stored as float32 zeros. File size doesn't shrink without sparse tensor formats (CSR, COO). Hardware speedup only with sparse BLAS kernels.

---

### Part 2 — Structured Pruning (Filter-level)

**What gets removed:** Entire filters (output channels) from Conv2d layers.

**Method — L1-norm Filter Ranking:**
```
L1_norm(filter_i) = Σ |w|  for all weights in filter_i
```
Filters with the lowest L1-norm are considered least important and are removed.

**Why structured pruning gives real speedup:**
```
Unstructured:  [w, 0, w, 0, 0, w, w, 0]  → irregular, needs sparse ops
Structured:    entire rows zeroed          → smaller dense matrix, standard ops
```
Removing filters directly reduces the number of output feature maps computed.

**Why 30% instead of 50%?**
Removing an entire filter is far more destructive than removing individual weights. At 30% filter pruning, accuracy remains acceptable. At 50%, accuracy drops steeply.

**Real FLOPs reduction:**
When output filters are pruned, all downstream computation on those feature maps is eliminated. FLOPs reduce proportionally to the fraction of filters removed.

---

### Iterative vs One-Shot Pruning
- **One-shot (used here):** Prune once, evaluate. Simple, fast.
- **Iterative:** Prune a little → fine-tune → prune more → repeat. Recovers more accuracy at high sparsity. Used in production.

---

## 8. Quantization — Deep Explanation

### What is Quantization?
Quantization maps continuous float32 values to a discrete lower-precision format (INT8). This reduces memory, speeds up computation, and lowers power consumption.

### The Math
For a float32 tensor with range [x_min, x_max], INT8 maps values to integers in [-128, 127]:
```
scale      = (x_max - x_min) / (q_max - q_min)
zero_point = round(q_min - x_min / scale)

x_quantized   = clamp(round(x / scale) + zero_point, q_min, q_max)
x_dequantized = (x_quantized - zero_point) × scale
```
Quantization error = x − x_dequantized. This rounding error causes accuracy loss.

### Precision Comparison

| Precision | Bits | Memory | Relative Speed |
|-----------|------|--------|----------------|
| float32 | 32 | 1× | 1× |
| float16 | 16 | 0.5× | ~2× |
| INT8 | 8 | 0.25× | ~4× |
| INT4 | 4 | 0.125× | ~8× |

---

### Part 3 — Post-Training Quantization (PTQ)

**What it is:** Convert a trained float32 model to INT8 without any retraining. Requires only a small calibration dataset.

**Steps in detail:**

1. **Move to CPU** — PyTorch static quantization (fbgemm backend) is CPU-only
2. **Fuse layers** — Fold Conv + BN into single operation, reducing quantization boundaries
3. **Insert observers** — Hooks that record activation min/max during calibration
4. **Calibrate** — Run 10 batches through model; observers collect statistics
5. **Convert** — Replace float32 ops with INT8 ops permanently

**Why calibration is needed:**
The quantization scale and zero_point depend on the actual range of activations. Without seeing real data, you cannot set these parameters accurately. 10 batches (~1280 images) is sufficient for CIFAR-100.

**Why PTQ loses accuracy:**
If calibration data is not representative, activation ranges are estimated incorrectly, causing clipping. Some layers (first conv, last FC, residual additions) are inherently more sensitive to quantization.

---

### Part 4 — Quantization-Aware Training (QAT)

**What it is:** Fine-tune the model with simulated quantization nodes inserted, so the model learns to compensate for INT8 rounding error before final conversion.

**Fake Quantization:**
```
x_fake = dequantize(quantize(x)) = round(x / scale) × scale
```
Same rounding error as real INT8, but implemented differentiably so gradients can flow.

**Steps in detail:**

1. **prepare_qat** — Insert fake quantization nodes after Conv and activations
2. **Fine-tune on GPU (5 epochs)** — Model sees quantization error and adapts weights (LR=1e-4)
3. **Convert on CPU** — Replace fake quantization with real INT8 ops

**Why QAT is more accurate than PTQ:**
The model experiences quantization error in every forward pass during fine-tuning and adjusts its weights to minimize it. PTQ applies quantization to a model trained with no awareness of rounding constraints.

**Why only 5 epochs?**
The model is already well-trained. QAT only makes small adjustments to improve INT8 robustness. More epochs risk overfitting.

---

### PTQ vs QAT Comparison

| Aspect | PTQ | QAT |
|--------|-----|-----|
| Retraining needed | No | Yes (5 epochs) |
| Calibration needed | Yes (10 batches) | No |
| Accuracy | Good | Better |
| Speed to deploy | Fast | Moderate |
| Best for | Quick deployment | Maximum accuracy at INT8 |

---

### Layer Fusion
Before quantization, consecutive layers are fused:
```
Conv2d + BatchNorm  →  single FusedConv (BN folded into weights)
```
Fusion: (1) eliminates BN at inference, (2) reduces quantization boundaries, (3) improves quantized accuracy.

### fbgemm Backend
FBGEMM (Facebook General Matrix Multiplication) is Meta's high-performance INT8 library for x86 CPUs. Uses AVX2/AVX512 SIMD instructions for vectorized INT8 arithmetic — gives real speedup over float32.

---

## 9. Running the Notebook

### Complete Run Order

```
Section 1  → Install libraries
Section 2  → Imports, seeds, device
Section 3  → Data loading + augmentations
Section 4  → Model definition (EfficientLite CNN)
Section 5  → Loss, optimizer, scheduler
Section 6  → Training (50 epochs) ← saves best_model.pth
Section 7  → Save checkpoint to Google Drive
Section 8  → Training plots
Section 9  → Per-class accuracy + Top-5
Section 10 → Efficiency metrics (FLOPs, size, latency)
Section 11 → Complete metrics summary table
Section 12 → Architecture summary (torchinfo)
─────────────────────────────────────────────────
Part 1     → Unstructured Pruning (50%)
Part 2     → Structured Pruning (30%)
Part 3     → Post-Training Quantization (PTQ)
Part 4     → Quantization-Aware Training (QAT)
Part 5     → Final comparison table + 4 plots
```

### Time Estimates (Colab T4 GPU)

| Step | Estimated Time |
|------|---------------|
| Training (50 epochs) | 40–60 min |
| Part 1 — Unstructured Pruning | 5–8 min |
| Part 2 — Structured Pruning | 5–8 min |
| Part 3 — PTQ (eval on CPU) | 3–5 min |
| Part 4 — QAT (fine-tune + eval) | 10–15 min |
| Part 5 — Comparison plots | < 1 min |

### Critical Rules
- Run sections strictly in order — later parts use variables from earlier parts
- Save to Google Drive (Section 7) — Colab storage resets on disconnect
- Keep tab active during training — Colab disconnects after 90 min idle
- Verify each part printed `✅ Done` before moving on

---

## 10. Metrics Explained

### Accuracy (Top-1)
```
Accuracy = (correct predictions / total predictions) × 100
```
Model's argmax prediction matches true label.

### Top-5 Accuracy
True label appears in model's 5 most confident predictions. Always higher than Top-1. Useful for CIFAR-100 where visually similar classes exist within superclasses.

### Loss (CrossEntropy)
```
CE Loss = -Σ y_true × log(y_pred)
```
Lower is better. Measures confidence and correctness of predicted probability distribution.

### FLOPs
Number of multiply-add operations per forward pass (measured by fvcore). Reflects computational cost — fewer FLOPs = faster on FLOPs-bound hardware. Quantization does not reduce FLOPs; only pruning/architecture changes do.

### Model Size (MB)
Size of saved `.pth` file (weights only). INT8 models are ~4× smaller than float32. Unstructured pruning does not shrink size on disk without sparse formats.

### Latency (ms)
Wall-clock inference time for batch=1, averaged over 100–200 runs after warmup. Measured on CPU for quantized models (where INT8 gives real speedup).

### Generalization Gap
```
Gap = Train Accuracy - Test Accuracy
```
Lower = less overfitting. Regularization techniques (dropout, weight decay, label smoothing, mixup) reduce this.

### Sparsity
```
Sparsity = (zero weights / total weights) × 100
```
Relevant only for pruned models.

---

## 11. Expected Results (Approximate)

| Variant | Accuracy | Loss | FLOPs | Size (MB) | Latency (ms) |
|---------|----------|------|-------|-----------|-------------|
| Baseline (FP32) | 55–63% | 1.8–2.2 | ~200M | 8–10 | baseline |
| Unstructured Pruned | 53–61% | 1.9–2.4 | ~200M | ~same | ~same |
| Structured Pruned | 50–58% | 2.0–2.5 | reduced | slightly less | ~same |
| PTQ (INT8) | 53–62% | 1.9–2.3 | ~200M | 2–3 | faster |
| QAT (INT8) | 54–63% | 1.8–2.2 | ~200M | 2–3 | faster |

> Actual results depend on GPU assigned by Colab and training randomness. Seed 42 is fixed.

---

## 12. Common Issues & Fixes

**`best_model.pth` not found**
Cause: Session reset or training didn't complete.
Fix: Rerun from Section 6. Always save to Drive in Section 7.

**Quantization error on GPU**
Cause: PyTorch static quantization requires CPU.
Fix: The code moves model to CPU automatically before prepare/convert.

**PTQ/QAT evaluation is very slow**
Cause: Quantized model evaluation runs on CPU.
Fix: Expected — wait 2–3 minutes. Do not interrupt.

**Variable not defined in Part 5**
Cause: A previous part errored or was skipped.
Fix: Run all parts in order. Each must print `✅ Done` before proceeding.

**Out of Memory (OOM) on GPU**
Cause: Batch size too large for assigned GPU.
Fix: Reduce `BATCH_SIZE` from 128 to 64 in Section 3.

**Colab disconnects during training**
Cause: Session timeout after 90 min idle.
Fix: Keep tab active. Checkpoint saves to Drive every time test accuracy improves.

---

## 13. Viva Questions & Answers

---

### 🔷 SECTION A — Dataset & Preprocessing

**Q1. Why use CIFAR-100 instead of CIFAR-10?**
CIFAR-100 has 100 classes with only 500 training images per class, versus CIFAR-10's 6000. This makes it significantly harder — the model must generalize from fewer examples and distinguish between visually similar fine-grained classes. It's a better benchmark for evaluating regularization, architecture quality, and compression robustness.

**Q2. Why normalize using dataset-specific mean and std instead of ImageNet stats?**
ImageNet stats are computed from large 224×224 natural images with different color distributions. CIFAR-100 images are 32×32 with different statistics. Using CIFAR-100 specific stats ensures inputs are truly zero-mean and unit-variance for this dataset, leading to more stable optimization and faster convergence.

**Q3. What is CutOut and why does it help?**
CutOut randomly masks a square patch of the input image to zero during training. It forces the model to make predictions from incomplete information, learning to use the whole image rather than relying on a single discriminative region. This reduces overfitting and improves generalization, especially on classes where one region dominates.

**Q4. What is Mixup and what problem does it solve?**
Mixup creates virtual training samples by linearly interpolating between two images and their labels. It prevents the model from being overconfident — the model learns to output smooth probability distributions rather than hard assignments. This acts as a strong data-space regularizer and reduces the generalization gap by smoothing decision boundaries between classes.

**Q5. Why use AutoAugment with the CIFAR-10 policy on CIFAR-100?**
The CIFAR-10 policy was found via Neural Architecture Search on CIFAR-10 but works well on CIFAR-100 because both datasets share similar low-level image statistics — small 32×32 color images. The policy includes 15+ operations like rotate, shear, and color jitter. Using the CIFAR-10 policy on CIFAR-100 is standard practice in the literature.

---

### 🔷 SECTION B — Model Architecture

**Q6. Why use depthwise separable convolutions instead of standard convolutions?**
Standard conv applies C_out filters of size (C_in × K × K) to every spatial location. Depthwise separable splits this: depthwise conv applies one filter per input channel (captures spatial patterns), then pointwise 1×1 conv mixes channels. For K=3, this gives approximately 8–9× fewer FLOPs with only a small accuracy penalty — a fundamental tradeoff in efficient architectures.

**Q7. What is the Squeeze-and-Excitation block doing and why is it useful?**
SE blocks implement channel attention. The squeeze step (global average pooling) collapses spatial dimensions to one value per channel, summarizing what each channel detected globally. The excite step (two FC layers + sigmoid) learns which channels are most informative for the current input. The output is a per-channel gate between 0–1 that recalibrates the feature map. This is input-adaptive — different images suppress or amplify different channels.

**Q8. Why is the bottleneck in MBConv called "inverted"?**
Traditional bottleneck blocks compress channels first (wide → narrow → wide), applying expensive 3×3 conv in the narrow space. MBConv does the opposite: it expands first (narrow → wide), applies depthwise conv in the wide space, then projects back (wide → narrow). The expansion provides a richer feature space for spatial filtering, while keeping residual connections at the narrow dimension to save memory.

**Q9. Why use Global Average Pooling instead of flattening before the classifier?**
Flattening a (B, 256, 4, 4) feature map gives a 4096-dimensional vector, requiring a large FC layer with hundreds of thousands of parameters. GAP averages each channel spatially to get (B, 256), requiring a much smaller FC. GAP also provides spatial invariance — the classifier sees average channel activation regardless of where a feature appears in the map.

**Q10. What is Kaiming initialization and why is it needed?**
Kaiming (He) initialization sets Conv weights from a normal distribution with std = sqrt(2 / fan_out). Without careful initialization, activations either vanish (all zeros) or explode (huge values) as they propagate through deep networks. Kaiming initialization maintains appropriate variance across layers when using ReLU-family activations, allowing stable gradient flow from the start of training.

**Q11. What does BatchNorm do and why is it placed after every convolution?**
BatchNorm normalizes activations across the batch: subtracts batch mean, divides by batch std, then applies learnable scale (gamma) and shift (beta). It prevents internal covariate shift — the changing distribution of activations during training. Benefits: allows higher learning rates, reduces sensitivity to initialization, acts as implicit regularization via batch noise, and speeds up convergence.

---

### 🔷 SECTION C — Training

**Q12. What is label smoothing and why does it help generalization?**
Label smoothing replaces the hard one-hot target (1.0 for correct, 0.0 for others) with a soft distribution: 0.9 for the correct class, 0.001 for each wrong class (with ε=0.1). This prevents the model from driving the correct class logit to infinity. Benefits: prevents overconfidence, improves calibration (predicted probabilities match true frequencies), and reduces overfitting on noisy or ambiguous labels.

**Q13. Why use AdamW instead of SGD or standard Adam?**
SGD with momentum requires careful LR scheduling and long warmup. Standard Adam applies weight decay through the gradient update, which conflates regularization with adaptive learning rates — they interact incorrectly. AdamW fixes this by applying weight decay directly to parameters, not through gradients. This gives cleaner regularization, less LR sensitivity, and consistently matches or outperforms SGD on CIFAR benchmarks.

**Q14. Explain the CosineAnnealingWarmRestarts scheduler.**
The learning rate follows a cosine curve from LR_max to eta_min over T_0 epochs, then restarts at LR_max. Each cycle doubles in length (T_mult=2): 10 → 20 → 40 epochs. The cosine shape gives smooth decay — no sharp loss cliffs like step decay. The warm restarts help the optimizer escape local minima by temporarily increasing LR, potentially finding better solutions in subsequent cycles.

**Q15. What is gradient clipping and when is it needed?**
Gradient clipping scales the gradient vector so its L2 norm doesn't exceed max_norm=1.0. If the gradient norm is larger, all gradients are scaled down proportionally. This prevents exploding gradients — situations where an unlucky batch or learning rate restart causes enormous gradient updates that destabilize training. It's especially important with AdamW during cosine restarts.

**Q16. What is the difference between train accuracy and test accuracy, and why is the gap important?**
Train accuracy measures how well the model fits the training data. Test accuracy measures generalization to unseen data. The gap (train − test) quantifies overfitting. A large gap means the model memorized training patterns that don't transfer to new data. All regularization techniques in this project (dropout, weight decay, label smoothing, mixup, augmentation) work together to minimize this gap.

---

### 🔷 SECTION D — Pruning

**Q17. What is the Lottery Ticket Hypothesis?**
Proposed by Frankle & Carlin (2019): within any randomly initialized neural network, there exists a small subnetwork (the "winning ticket") that, if trained in isolation with the same initialization, can match the full network's accuracy. Pruning attempts to discover this subnetwork in the already-trained model. This hypothesis motivates why high sparsity is achievable without destroying accuracy.

**Q18. What is the difference between unstructured and structured pruning?**
Unstructured pruning removes individual weights at arbitrary positions — the sparsity pattern is irregular. It achieves high compression ratios with low accuracy drop but needs sparse BLAS kernels for hardware speedup. Structured pruning removes entire filters, channels, or layers — the model is smaller and denser, giving real FLOPs reduction and speedup on all standard hardware without special sparse ops.

**Q19. Why does unstructured pruning at 50% not reduce model size on disk?**
The zeroed weights are still stored as float32 zeros in the `.pth` tensor. PyTorch's standard save format stores dense tensors. Real size reduction requires sparse storage formats (CSR, COO) that only store non-zero values and their indices. In this project we use standard `.pth` format, so size doesn't shrink for unstructured pruning.

**Q20. Why does structured pruning give real FLOPs reduction but unstructured doesn't?**
Structured pruning removes entire output filters from Conv2d layers. This means fewer feature maps are computed downstream, directly reducing multiply-add operations. Unstructured pruning zeroes individual weights in an irregular pattern — the same number of operations still runs (just multiplied by zero), which standard dense matrix multiply hardware cannot skip.

**Q21. What is global vs local pruning and which is better?**
Local pruning applies the same sparsity ratio to each layer independently. Global pruning ranks all weights across all layers together and removes the bottom X% globally. Global is generally better because different layers have different importance — early convolutional layers typically contain more critical weights and should be pruned less aggressively. Global pruning allocates sparsity naturally based on weight magnitude across the entire network.

**Q22. Why might accuracy improve slightly after very mild pruning?**
Mild pruning (10–20%) can act as a regularizer by removing noisy or redundant weights that contributed to overfitting. This sometimes reduces the generalization gap and slightly improves test accuracy. This phenomenon supports the idea that many weights in a trained network are redundant.

**Q23. What is iterative pruning and why is it better than one-shot pruning?**
Iterative pruning alternates between pruning and fine-tuning: prune a small amount → fine-tune → prune more → repeat. After each pruning step, fine-tuning allows remaining weights to adapt to the sparser structure. This recovers accuracy at high sparsity levels (70–90%) where one-shot pruning would cause significant degradation. This project uses one-shot pruning for simplicity.

---

### 🔷 SECTION E — Quantization

**Q24. What is quantization in neural networks?**
Quantization reduces the numerical precision of weights and activations from float32 (32-bit) to lower precision (typically INT8, 8-bit). Each float32 value takes 4 bytes; each INT8 takes 1 byte — 4× memory reduction. INT8 arithmetic is also faster on CPUs with SIMD instructions (AVX2/AVX512) and on specialized hardware (TPUs, mobile NPUs). The tradeoff is a small accuracy loss from rounding.

**Q25. Explain the quantization formula — what are scale and zero_point?**
For a float32 tensor with range [x_min, x_max]:
```
scale      = (x_max - x_min) / (q_max - q_min)
zero_point = round(q_min - x_min / scale)

x_int8 = clamp(round(x / scale) + zero_point, q_min, q_max)
x_float_approx = (x_int8 - zero_point) × scale
```
Scale maps the float range to the integer range. Zero_point maps float zero to an integer (needed for asymmetric quantization). Quantization error = x − x_float_approx, which comes from the rounding step.

**Q26. What is the difference between symmetric and asymmetric quantization?**
Symmetric quantization uses zero_point=0, mapping [-max_abs, +max_abs] to [-127, 127]. Simple computation but wastes range for non-symmetric distributions. Asymmetric quantization (used in fbgemm) uses zero_point ≠ 0, mapping [x_min, x_max] to [0, 255] (uint8). This is more accurate for activations after ReLU/GELU, which are non-symmetric.

**Q27. What is the role of calibration in PTQ?**
Calibration runs the model on a small representative dataset with observer hooks active. Observers record the minimum and maximum values of activations at each quantization point. These statistics determine scale and zero_point for each layer. Without calibration, you cannot set appropriate quantization parameters — wrong scale causes either clipping (values outside range mapped to max INT8) or poor resolution (range too large).

**Q28. Why does PTQ lose accuracy?**
Multiple sources of error: (1) activation ranges estimated from calibration may not cover rare extreme values, causing clipping; (2) some layers have wide or bimodal activation distributions that are hard to quantize accurately with uniform INT8; (3) residual addition layers are particularly sensitive since they combine quantized tensors with potentially mismatched scales; (4) first and last layers are conventionally kept at float32 but their boundary interactions still introduce error.

**Q29. What is fake quantization in QAT?**
Fake quantization simulates INT8 rounding during the float32 forward pass:
```
x_fake = dequantize(quantize(x)) = (round(x / scale) × scale)
```
This introduces the same rounding error as real INT8 but remains differentiable (using straight-through estimator for gradients). The model sees quantization error during training and its weights adjust to minimize it. After fine-tuning, converting to real INT8 produces better accuracy than PTQ.

**Q30. Why must quantization happen on CPU in PyTorch?**
PyTorch's static quantization uses the fbgemm backend, which is implemented for x86 CPUs using AVX2/AVX512 instructions. NVIDIA GPU quantization uses a different pathway (TensorRT or torch-tensorrt). For mobile deployment, the qnnpack backend is used. This project uses CPU quantization (fbgemm) as the standard research path. Fine-tuning in QAT still runs on GPU — only conversion and evaluation happen on CPU.

**Q31. What is layer fusion and why must it happen before quantization?**
Layer fusion merges consecutive operations (Conv + BN, or Conv + BN + ReLU) into a single fused operation. For quantization: (1) BN parameters are folded into Conv weights mathematically, eliminating BN at inference entirely; (2) fewer separate quantization boundaries means less accumulated rounding error; (3) fused ops are more amenable to INT8 acceleration. Without fusion, quantizing Conv and BN separately introduces unnecessary quantization error at their boundary.

**Q32. How much speedup does INT8 quantization give in practice?**
On modern x86 CPUs (AVX2): typically 1.5–3× speedup for inference, especially for large matrix multiplications. On mobile ARM CPUs with NEON: 2–4×. On dedicated hardware (Google TPU, Apple Neural Engine): 4–8×. The speedup is more pronounced for compute-bound operations (large batch sizes, large matrix multiplications) and less for memory-bound cases (small batch=1 inference where memory bandwidth dominates).

---

### 🔷 SECTION F — Evaluation & Metrics

**Q33. Why measure both Top-1 and Top-5 accuracy on CIFAR-100?**
CIFAR-100 has 20 superclasses each with 5 fine classes. A model might correctly identify the superclass but confuse the exact fine class — e.g., distinguishing "Persian cat" from "Siamese cat." Top-5 accuracy measures whether the correct label appears in the 5 most confident predictions, giving a more nuanced view of whether the model has learned semantically meaningful representations.

**Q34. Why doesn't unstructured pruning reduce FLOPs?**
FLOPs count the number of multiply-add operations. Unstructured pruning zeroes individual weights, but the matrix multiplication hardware still processes all elements — it just multiplies some by zero. Standard dense GPU/CPU math kernels cannot skip zero multiplications. FLOPs only reduce when you remove entire structural units (filters, layers) via structured pruning.

**Q35. Why measure latency after warmup runs?**
First runs are slower due to: CUDA kernel compilation and selection, memory allocation and caching, GPU clock frequency ramp-up (power management). Warmup runs (50 for batch=1) allow the system to reach steady state. Timing over 100–200 subsequent runs and averaging reduces variance from OS scheduling, memory access patterns, and GPU thermal throttling.

**Q36. What is the generalization gap and what are acceptable values?**
Generalization gap = Train Accuracy − Test Accuracy. On CIFAR-100 with ~2M parameter models: a gap below 10% indicates good regularization. A gap of 20–30% indicates significant overfitting. This project uses multiple complementary regularization techniques (dropout, weight decay, label smoothing, mixup, augmentation) specifically to minimize this gap.

**Q37. Why do quantized models show faster latency only on CPU, not GPU?**
GPU INT8 acceleration requires TensorRT or CUDA-specific quantization paths. PyTorch's fbgemm quantization is designed for CPU inference and uses x86 SIMD instructions (AVX2/AVX512) for INT8 arithmetic. On GPU, the overhead of moving the quantized model to CPU plus quantization ops dominates any arithmetic savings. For GPU INT8 speedup, TensorRT would be needed.

---

### 🔷 SECTION G — Design Decisions

**Q38. Why train from scratch instead of using pretrained weights?**
This project demonstrates that a well-designed architecture with proper training techniques can achieve strong accuracy without transfer learning. Training from scratch is a stricter test of architecture and regularization quality. It's also more appropriate here since CIFAR-100 images (32×32) differ significantly from ImageNet (224×224) — the learned low-level features may not transfer well, and the domain-specific augmentation policies are better suited to learning from scratch.

**Q39. Why target ~2M parameters for CIFAR-100?**
CIFAR-100 has only 50K training images (500 per class). A 100M parameter model would severely overfit. ~2M parameters sits at a sweet spot: sufficient capacity to distinguish 100 fine-grained classes, small enough to avoid severe overfitting with standard regularization, and efficient enough to make compression results meaningful. Starting from an already-efficient model makes pruning and quantization results more informative.

**Q40. What would you do differently to push accuracy above 70% on CIFAR-100?**
Several approaches would help:
(1) Use a pretrained backbone (EfficientNet-B4, ViT-S) with fine-tuning — transfer learning from ImageNet
(2) Increase model capacity with more MBConv blocks and wider channels
(3) Train for more epochs (100–200) with careful LR scheduling
(4) Use stronger augmentations: RandAugment, TrivialAugmentWide, RandomErasing
(5) Add Test-Time Augmentation (TTA) — average predictions over multiple augmented views at inference
(6) Use knowledge distillation from a larger teacher model

**Q41. What is knowledge distillation and how does it relate to this work?**
Knowledge distillation trains a small student model to mimic the soft probability outputs of a larger teacher model, rather than learning only from hard one-hot labels. The teacher's soft outputs contain richer information — class similarities and relationships. It's complementary to pruning and quantization: you can distill first to get a better small model, then prune and quantize that model for further compression. The combination often outperforms any single technique alone.

---

### 🔷 SECTION H — Broader Concepts

**Q42. What is the accuracy-efficiency tradeoff and how is it visualized?**
Every compression technique trades some accuracy for better efficiency (smaller size, fewer FLOPs, lower latency). The Pareto frontier is the set of models where you cannot improve one metric without hurting another. The scatter plot of Accuracy vs Model Size (Part 5) visualizes this — models in the top-left corner (high accuracy, small size) represent better tradeoffs. The goal of compression is to stay as close to the top-left as possible.

**Q43. What is the difference between model compression and model efficiency?**
Model compression reduces an existing trained model's size or FLOPs through post-training techniques (pruning, quantization, knowledge distillation). Model efficiency designs architectures to be computationally lean from the start (MobileNet, EfficientNet, SqueezeNet). This project does both: EfficientLite is an efficient architecture by design (depthwise sep conv, MBConv, GAP), then further compressed via pruning and quantization.

**Q44. Where would you deploy a quantized INT8 model?**
INT8 quantized models are ideal for:
- Mobile phones (ARM CPUs with NEON SIMD, Apple Neural Engine, Qualcomm Hexagon)
- Edge devices (Raspberry Pi, NVIDIA Jetson Nano)
- IoT sensors and microcontrollers (with extreme quantization INT4/binary)
- Cloud inference servers at scale (AVX512 INT8 on Intel Xeon)
- Browsers via WebAssembly (WASM SIMD)
- Autonomous systems requiring real-time inference under power constraints

**Q45. What is the difference between inference optimization and training optimization?**
Training optimization focuses on convergence speed, stability, and generalization — choice of optimizer, LR schedule, augmentation, regularization. Inference optimization focuses on deployment efficiency — model size, FLOPs, latency, power consumption. They have fundamentally different goals: training wants the best possible model regardless of cost; inference wants the best model within strict resource constraints. This project addresses both separately.

**Q46. What is sparse training and how does it differ from post-training pruning?**
Post-training pruning (used here) first trains a dense model, then removes weights. Sparse training (also called dynamic sparse training) starts with a sparse model and maintains sparsity throughout training — some weights are masked throughout, while the topology can evolve. Methods like RigL (Rigging the Lottery) achieve better accuracy at equivalent sparsity by dynamically updating which weights are active during training. Post-training pruning is simpler but sparse training is more powerful at very high sparsity.

**Q47. How do you choose pruning ratio and quantization precision for production?**
It depends on the deployment target:
- For mobile apps: INT8 + 50% structured pruning is a common starting point
- For microcontrollers with <1MB RAM: INT4 or binary quantization + 80%+ pruning
- For cloud servers: INT8 PTQ alone is often sufficient (minimal accuracy loss)
- Always validate on a held-out test set; measure accuracy, latency, and memory on the target hardware
- Use iterative pruning with fine-tuning for high sparsity targets

**Q48. What is the relationship between model capacity, dataset size, and regularization?**
Larger models have more capacity (can fit more complex functions) but need more data to generalize. With small datasets like CIFAR-100 (500 images/class), a large model without regularization will overfit. The right amount of regularization depends on the capacity-data ratio: more capacity or less data → need stronger regularization. This is why ~2M parameters combined with multiple regularization techniques (dropout, weight decay, label smoothing, mixup, augmentation) is a deliberate design choice.

---

*End of README — Good luck with your evaluation and viva! 🚀*
