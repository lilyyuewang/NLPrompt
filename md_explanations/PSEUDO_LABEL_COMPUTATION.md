# How Pseudo-Labels are Computed

## Overview

Pseudo-labels are computed using Optimal Transport (OT) to match image features with class prototypes (text features from prompts). The process involves several steps:

## Step-by-Step Process

### Step 1: Get Model Predictions

```python
# In OT_PL function
feat = model.image_encoder(inputs.cuda())  # Image features: [batch_size, feat_dim]
logits = model(inputs.cuda())              # Logits: [batch_size, num_classes]
out = logits.softmax(dim=-1)               # Probabilities: [batch_size, num_classes]
```

**What happens:**
- Image encoder extracts features from images
- Full model (image + text encoder) computes logits (similarity scores)
- Softmax converts logits to probabilities `P[i, j]` = probability sample `i` belongs to class `j`

### Step 2: Prepare Cost Matrix

```python
# Based on Pmode setting
if Pmode == 'out':
    P = out                    # Raw softmax probabilities
elif Pmode == 'logP':
    P = F.log_softmax(out, dim=1)  # Log probabilities (default)
elif Pmode == 'softmax':
    P = F.softmax(out, dim=1)      # Explicit softmax
```

**Cost Matrix:**
```python
M = -P  # Negative of prediction probabilities
```

**Why negative?**
- Higher probability → Lower cost → Easier to transport mass
- OT minimizes cost, so it prefers high-probability assignments

### Step 3: Set Up OT Problem

```python
# Source distribution (samples)
a = torch.ones((N_samples,)) / N_samples  # Uniform: each sample has weight 1/N

# Target distribution (classes)
b = torch.ones((num_classes,)) / num_classes * top_percent  # Scaled by budget
```

**Marginal Constraints:**
- Source (`a`): Uniform distribution over samples
- Target (`b`): Distribution over classes, scaled by `budget` (e.g., 0.3 = 30% of total mass)

### Step 4: Solve OT Problem with Sinkhorn

```python
coupling = ot.sinkhorn(a, b, M=-P, reg=reg_e, numItermax=1000, stopThr=1e-6)
```

**What Sinkhorn does:**
- Solves: `min_Q <M, Q>` subject to marginal constraints
- Returns coupling matrix `Q` (shape: `[N_samples, num_classes]`)
- Each entry `Q[i, j]` = amount of "mass" transported from sample `i` to class `j`

**Regularization (`reg_e`):**
- Entropy regularization (default: 0.01)
- Makes the solution smoother and more stable
- Smaller values → sharper coupling, larger values → smoother coupling

### Step 5: Compute Pseudo-Labels from Coupling Matrix

```python
# Normalize coupling matrix to get pseudo-label distribution
row_sum = torch.sum(couplings, 1).reshape((-1, 1))  # Sum over classes for each sample
pseudo_labels = torch.div(couplings, row_sum)        # Normalize: [N_samples, num_classes]
```

**Result:**
- `pseudo_labels[i, j]` = probability that sample `i` belongs to class `j` (normalized)
- Each row sums to 1.0 (probability distribution)

### Step 6: Get Hard Pseudo-Labels

```python
# Get the class with maximum coupling for each sample
max_value, argmax_plabels = torch.max(couplings, axis=1)
```

**Result:**
- `argmax_plabels[i]` = class index with highest coupling for sample `i`
- This is the **hard pseudo-label** (single class assignment)

## Complete Flow Diagram

```
Images
  ↓
Image Encoder (CLIP)
  ↓
Image Features: I ∈ [N, d]
  ↓
Text Encoder (CLIP with prompts)
  ↓
Text Features: T ∈ [C, d]
  ↓
Similarity: T · I^T ∈ [C, N]
  ↓
Model Predictions: P ∈ [N, C] (probabilities)
  ↓
Cost Matrix: M = -P ∈ [N, C]
  ↓
Sinkhorn Algorithm
  ↓
Coupling Matrix: Q ∈ [N, C]
  ↓
Normalize: pseudo_labels = Q / row_sum(Q)
  ↓
Hard Labels: argmax(Q, axis=1)
```

## Key Formulas

### From Paper (Equation 9)

The OT problem solved:
```
min < -log(T · I^T), Q >
Q ∈ R^{C×N}_+
s.t. Q·1_N = (1/C)·1_C,  Q^T·1_C = (1/N)·1_N
```

### In Code

```python
# Cost matrix (simplified - uses model predictions instead of direct similarity)
M = -P  # where P = model predictions

# OT problem
coupling = ot.sinkhorn(a, b, M=-P, reg=reg_e)

# Pseudo-labels
pseudo_labels = coupling / row_sum(coupling)  # Normalized
argmax_plabels = argmax(coupling, axis=1)     # Hard labels
```

## Differences from Paper

**Paper:**
- Uses direct similarity: `T · I^T` (text features × image features)
- Cost: `-log(T · I^T)`

**Code:**
- Uses model predictions: `P = softmax(logits)` (already includes similarity computation)
- Cost: `-P` (negative probabilities)

**Why different?**
- The model already computes `T · I^T` internally (text-image similarity)
- Using `-P` is equivalent but simpler (probabilities already normalized)

## Example

**Input:**
- 3 samples, 2 classes
- Model predictions `P`:
  ```
  P = [[0.7, 0.3],   # Sample 0: 70% class 0, 30% class 1
       [0.4, 0.6],   # Sample 1: 40% class 0, 60% class 1
       [0.2, 0.8]]   # Sample 2: 20% class 0, 80% class 1
  ```

**Cost Matrix:**
```
M = -P = [[-0.7, -0.3],
          [-0.4, -0.6],
          [-0.2, -0.8]]
```

**OT Coupling (simplified example):**
```
coupling = [[0.5, 0.1],   # Sample 0 → mostly class 0
            [0.2, 0.4],   # Sample 1 → mostly class 1
            [0.1, 0.7]]   # Sample 2 → mostly class 1
```

**Pseudo-Labels (normalized):**
```
pseudo_labels = [[0.83, 0.17],  # Sample 0: 83% class 0, 17% class 1
                 [0.33, 0.67],  # Sample 1: 33% class 0, 67% class 1
                 [0.13, 0.87]]  # Sample 2: 13% class 0, 87% class 1
```

**Hard Pseudo-Labels:**
```
argmax_plabels = [0, 1, 1]  # Sample 0→class 0, Sample 1→class 1, Sample 2→class 1
```

## Summary

1. **Model predictions** (`P`) → probabilities from CLIP model
2. **Cost matrix** (`M = -P`) → negative probabilities
3. **Sinkhorn algorithm** → solves OT problem, returns coupling matrix
4. **Normalize coupling** → `pseudo_labels = coupling / row_sum`
5. **Hard labels** → `argmax_plabels = argmax(coupling, axis=1)`

**Key insight:** Pseudo-labels come from the **optimal transport plan** that matches samples to classes while respecting marginal constraints. The coupling matrix represents how much "mass" flows from each sample to each class, and normalizing it gives the pseudo-label probability distribution.

