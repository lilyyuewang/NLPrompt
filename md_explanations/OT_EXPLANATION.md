# Optimal Transport (OT) Function Explanation

## Overview

The OT function in this codebase uses **Optimal Transport theory** (specifically the **Sinkhorn algorithm**) to identify clean vs noisy samples in a noisy label learning setting.

## Main Functions

### 1. `curriculum_structure_aware_PL(features, P, top_percent, ...)`

**What it does:**
- Solves an Optimal Transport problem to find the best matching between **samples** (data points) and **classes** (label categories)
- Uses the Sinkhorn algorithm, which is an efficient approximation of the optimal transport problem

**Inputs:**
- `features`: Normalized image features from the model encoder (shape: `[N_samples, feat_dim]`)
- `P`: Prediction probabilities from the model (shape: `[N_samples, N_classes]`)
  - Can be raw softmax output (`Pmode='out'`), log-softmax (`Pmode='logP'`), or softmax (`Pmode='softmax'`)
- `top_percent`: Budget - fraction of samples to select as "clean" (e.g., 0.3 means select top 30%)
- `reg_e`: Regularization parameter for Sinkhorn (entropy regularization, typically 0.01)

**What it returns:**
1. **`coupling`** (shape: `[N_samples, N_classes]`): 
   - The **optimal transport plan** or **coupling matrix**
   - Each entry `coupling[i, j]` represents how much "mass" is transported from sample `i` to class `j`
   - Higher values indicate stronger association between sample and class
   
2. **`selected_mask`** (shape: `[N_samples]`, boolean):
   - Binary mask indicating which samples are selected as "clean/high-confidence"
   - `True` = selected (predicted clean), `False` = rejected (predicted noisy)
   - Selected by taking top-k samples based on `max(coupling, dim=1)` values

**Physical Meaning:**

The Optimal Transport problem can be thought of as:
- **Source distribution (a)**: Uniform distribution over all samples (each sample has equal weight = 1/N)
- **Target distribution (b)**: Distribution over classes, scaled by `top_percent` (only `top_percent` fraction of total mass)
- **Cost matrix (M)**: `M = -P` (negative of prediction probabilities)
  - Higher prediction probability → Lower cost → More likely to transport mass
  - This means samples with high confidence predictions have lower transport cost

The Sinkhorn algorithm finds the optimal way to "transport" probability mass from samples to classes, minimizing the total cost while respecting the distribution constraints.

**Key insight:** Samples that have high coupling values (especially high max coupling) are those that:
1. Have high prediction confidence (low cost)
2. Are well-aligned with their predicted class
3. Are likely to be **clean samples** (correctly labeled)

### 2. `OT_PL(model, eval_loader, ...)`

**What it does:**
- Wrapper function that runs the model on all data, extracts features and predictions
- Calls `curriculum_structure_aware_PL` to get OT results
- Processes the coupling matrix to derive pseudo-labels and confidence scores

**What it returns:**
1. **`all_pseudo_labels`** (shape: `[N_samples, N_classes]`):
   - Normalized coupling matrix: `pseudo_labels[i, j] = coupling[i, j] / sum(coupling[i, :])`
   - Represents soft pseudo-label distribution for each sample
   
2. **`all_noisy_labels`** (shape: `[N_samples]`):
   - The noisy/corrupted labels from the dataset
   
3. **`all_gt_labels`** (shape: `[N_samples]`):
   - Ground truth clean labels (for evaluation)
   
4. **`all_selected_mask`** (shape: `[N_samples]`, boolean):
   - Which samples OT selected as clean (same as from `curriculum_structure_aware_PL`)
   
5. **`all_conf`** (shape: `[N_samples]`):
   - Confidence scores: `conf[i] = max(coupling[i, :]) / (1/N_classes)`
   - Higher values indicate higher confidence in the pseudo-label
   
6. **`all_argmax_plabels`** (shape: `[N_samples]`):
   - Hard pseudo-labels: `argmax(coupling[i, :])` - the class with highest coupling for each sample

## Physical Interpretation

### The Optimal Transport Problem

Think of it as a **logistics problem**:

1. **You have N samples** (source locations) with equal supply (1/N each)
2. **You have C classes** (destination locations) with limited capacity (`top_percent * N / C` each)
3. **Transport cost** from sample `i` to class `j` is `-P[i, j]` (negative prediction probability)
   - Lower cost = higher prediction probability = easier to transport
4. **Goal**: Find the optimal transport plan that minimizes total cost

### Why This Works for Noisy Label Detection

1. **Clean samples** typically have:
   - High prediction confidence (high `P[i, true_class]`)
   - Low transport cost to their true class
   - High coupling values → Selected as clean

2. **Noisy samples** typically have:
   - Low prediction confidence (low `P[i, noisy_label]`)
   - High transport cost (model doesn't agree with noisy label)
   - Low coupling values → Rejected as noisy

3. **The budget constraint** (`top_percent`):
   - Forces OT to select only the most confident samples
   - Acts as a filter: only samples with strong evidence are selected
   - Gradually increases during training (curriculum learning)

## Example Flow

```
Input: 1000 samples, 10 classes, budget=0.3 (select top 30%)

1. Model predicts: P[i, j] = probability sample i belongs to class j
2. Cost matrix: M[i, j] = -P[i, j]
3. Sinkhorn solves: Find coupling matrix that minimizes total cost
4. For each sample i: max_coupling[i] = max(coupling[i, :])
5. Select top 300 samples (30%) with highest max_coupling
6. selected_mask[i] = True if sample i is in top 300, else False
```

## Key Parameters

- **`reg_e`** (entropy regularization): Controls smoothness of the transport plan
  - Smaller values → sharper, more deterministic coupling
  - Larger values → smoother, more uniform coupling
  - Typical value: 0.01

- **`top_percent`** (budget): Fraction of samples to select
  - 0.3 = select top 30% most confident samples
  - Can be curriculum-scheduled (start low, increase over epochs)
  - **Important**: Budget is NOT automatically set to `(1 - noise_rate)`
  - Budget is an independent hyperparameter (default: 0.3 = 30%)
  - If you want to select `(1 - noise_rate)` samples, you need to set it manually

- **`Pmode`**: How to process model predictions
  - `'out'`: Use raw softmax output
  - `'logP'`: Use log-softmax (more numerically stable, emphasizes high-confidence predictions)
  - `'softmax'`: Explicit softmax

## Summary

**OT Function Purpose:**
- Identify which samples are likely clean vs noisy based on prediction confidence and feature similarity

**OT Function Returns:**
- `coupling`: Optimal transport plan (how samples map to classes)
- `selected_mask`: Binary selection of clean samples
- `pseudo_labels`: Soft label distribution derived from coupling
- `confidence`: Confidence scores for each sample

**Physical Meaning:**
- Finds optimal matching between samples and classes under budget constraints
- Samples with high coupling = high confidence = likely clean
- Samples with low coupling = low confidence = likely noisy

