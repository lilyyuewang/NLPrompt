# Comprehensive Guide to Optimal Transport in NLPrompt

This document provides a complete explanation of how Optimal Transport (OT) is used in NLPrompt for noisy label learning, including budget calculation, curriculum learning, confidence interpretation, coupling matrices, and pseudo-label computation.

---

## Table of Contents

1. [Optimal Transport Overview](#optimal-transport-overview)
2. [Budget Calculation](#budget-calculation)
3. [Curriculum Learning](#curriculum-learning)
4. [Confidence Explanation](#confidence-explanation)
5. [Coupling Matrix and Confidence](#coupling-matrix-and-confidence)
6. [Pseudo-Label Computation](#pseudo-label-computation)
7. [Complete Workflow](#complete-workflow)

---

## Optimal Transport Overview

### What is Optimal Transport?

Optimal Transport (OT) is a mathematical framework used to find the optimal way to "transport" probability mass from one distribution to another while minimizing a cost function. In NLPrompt, OT is used to identify clean vs noisy samples in a noisy label learning setting.

### Main Functions

#### 1. `curriculum_structure_aware_PL(features, P, top_percent, ...)`

**Purpose:**
- Solves an Optimal Transport problem to find the best matching between **samples** (data points) and **classes** (label categories)
- Uses the Sinkhorn algorithm, which is an efficient approximation of the optimal transport problem

**Inputs:**
- `features`: Normalized image features from the model encoder (shape: `[N_samples, feat_dim]`)
- `P`: Prediction probabilities from the model (shape: `[N_samples, N_classes]`)
  - Can be raw softmax output (`Pmode='out'`), log-softmax (`Pmode='logP'`), or softmax (`Pmode='softmax'`)
- `top_percent`: Budget - fraction of samples to select as "clean" (e.g., 0.3 means select top 30%)
- `reg_e`: Regularization parameter for Sinkhorn (entropy regularization, typically 0.01)

**Returns:**
1. **`coupling`** (shape: `[N_samples, N_classes]`): 
   - The **optimal transport plan** or **coupling matrix**
   - Each entry `coupling[i, j]` represents how much "mass" is transported from sample `i` to class `j`
   - Higher values indicate stronger association between sample and class
   
2. **`selected_mask`** (shape: `[N_samples]`, boolean):
   - Binary mask indicating which samples are selected as "clean/high-confidence"
   - `True` = selected (predicted clean), `False` = rejected (predicted noisy)
   - Selected by taking top-k samples based on `max(coupling, dim=1)` values

#### 2. `OT_PL(model, eval_loader, ...)`

**Purpose:**
- Wrapper function that runs the model on all data, extracts features and predictions
- Calls `curriculum_structure_aware_PL` to get OT results
- Processes the coupling matrix to derive pseudo-labels and confidence scores

**Returns:**
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

### Physical Interpretation

Think of OT as a **logistics problem**:

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

### Key Parameters

- **`reg_e`** (entropy regularization): Controls smoothness of the transport plan
  - Smaller values → sharper, more deterministic coupling
  - Larger values → smoother, more uniform coupling
  - Typical value: 0.01

- **`top_percent`** (budget): Fraction of samples to select
  - 0.3 = select top 30% most confident samples
  - Can be curriculum-scheduled (start low, increase over epochs)
  - **Important**: Budget is NOT automatically set to `(1 - noise_rate)`
  - Budget is an independent hyperparameter (default: 0.3 = 30%)

- **`Pmode`**: How to process model predictions
  - `'out'`: Use raw softmax output
  - `'logP'`: Use log-softmax (more numerically stable, emphasizes high-confidence predictions)
  - `'softmax'`: Explicit softmax

---

## Budget Calculation

### Function Overview

The `curriculum_scheduler` function gradually increases the budget from a starting value (`begin`) to an ending value (`end`) over `T` steps using different scheduling modes.

### Formula

```python
budget = begin + ratio * (end - begin)
```

Where:
- `begin`: Starting budget (default: `BEGIN_RATE = 0.3` = 30%)
- `end`: Final budget (default: `1.0` = 100%)
- `ratio`: Progress ratio from 0 to 1, calculated based on `mode`

### Parameters

- **`t`**: Current epoch/step (e.g., epoch 5)
- **`T`**: Total curriculum epochs (`CURRICLUM_EPOCH`, default: 0)
- **`begin`**: Starting budget (`BEGIN_RATE`, default: 0.3)
- **`end`**: Final budget (default: 1.0)
- **`mode`**: Scheduling mode (`CURRICLUM_MODE`, default: 'linear')

### Progress Ratio (`pho`)

```python
pho = t / T  # Progress from 0.0 to 1.0
```

### Scheduling Modes

#### 1. Linear Mode (`mode='linear'`)

```python
ratio = pho = t / T
budget = begin + (t/T) * (end - begin)
```

**Example** (begin=0.3, end=1.0, T=100):
- Epoch 0: `pho=0.0`, `ratio=0.0`, `budget = 0.3 + 0.0 * 0.7 = 0.3` (30%)
- Epoch 25: `pho=0.25`, `ratio=0.25`, `budget = 0.3 + 0.25 * 0.7 = 0.475` (47.5%)
- Epoch 50: `pho=0.5`, `ratio=0.5`, `budget = 0.3 + 0.5 * 0.7 = 0.65` (65%)
- Epoch 75: `pho=0.75`, `ratio=0.75`, `budget = 0.3 + 0.75 * 0.7 = 0.825` (82.5%)
- Epoch 100: `pho=1.0`, `ratio=1.0`, `budget = 0.3 + 1.0 * 0.7 = 1.0` (100%)

#### 2. Exponential Mode (`mode='exp'`)

```python
ratio = 1 - math.exp(-4 * pho)
budget = begin + (1 - exp(-4*t/T)) * (end - begin)
```

**Example** (begin=0.3, end=1.0, T=100):
- Epoch 0: `pho=0.0`, `ratio ≈ 0.0`, `budget ≈ 0.3` (30%)
- Epoch 25: `pho=0.25`, `ratio ≈ 0.632`, `budget ≈ 0.742` (74.2%)
- Epoch 50: `pho=0.5`, `ratio ≈ 0.865`, `budget ≈ 0.906` (90.6%)
- Epoch 75: `pho=0.75`, `ratio ≈ 0.950`, `budget ≈ 0.965` (96.5%)
- Epoch 100: `pho=1.0`, `ratio ≈ 0.982`, `budget ≈ 0.987` (98.7%)

*Exponential mode increases faster initially, then slows down*

#### 3. Custom Mode (`mode='customize'`)

```python
ratio = func(t, T)  # User-defined function
budget = begin + func(t, T) * (end - begin)
```

### Important Note: Default Configuration

**If `CURRICLUM_EPOCH = 0`:**

```python
if self.epoch < curriclum_epoch:  # 0 < 0 is False
    budget, pho = curriculum_scheduler(...)
else:
    budget, pho = 1., 1.  # Always executes!
```

When `CURRICLUM_EPOCH = 0`, the budget is **always set to 1.0** (100%), meaning OT selects **all samples** regardless of confidence!

**Current Configuration** (from `train.py`):
- `BEGIN_RATE = 0.3` (starts at 30%)
- `CURRICLUM_EPOCH = 0` (no curriculum, budget stays at 0.3... but actually becomes 1.0 due to the condition above)
- `CURRICLUM_MODE = 'linear'`

### Relationship to Noise Rate

**The budget is NOT automatically related to noise rate!**

- Budget is controlled by curriculum scheduler (independent hyperparameter)
- If you want `budget = 1 - noise_rate`, you need to:
  1. Set `BEGIN_RATE = 1 - noise_rate`, OR
  2. Modify the code to cap budget at `1 - noise_rate`

---

## Curriculum Learning

### Overview

When curriculum learning is enabled (`CURRICLUM_EPOCH > 0`), the budget **gradually increases** from `BEGIN_RATE` to `1.0` over the course of training, meaning OT selects more and more samples as "clean" over time.

### How It Works

The budget is recalculated **every epoch** in `before_epoch()`:

```python
if self.epoch < curriclum_epoch:
    budget, pho = curriculum_scheduler(self.epoch, curriclum_epoch, 
                                      begin=begin_rate, end=1, mode=curriclum_mode)
else:
    budget, pho = 1., 1.  # After curriculum, use 100%
```

### Example: Linear Curriculum Learning

**Configuration:**
- `BEGIN_RATE = 0.3` (start at 30%)
- `CURRICLUM_EPOCH = 100` (increase over 100 epochs)
- `CURRICLUM_MODE = 'linear'`

**Budget Over Time:**

| Epoch | Progress (pho) | Budget | Samples Selected |
|-------|----------------|--------|------------------|
| 0     | 0.0            | 0.30   | Top 30%          |
| 10    | 0.1            | 0.37   | Top 37%          |
| 25    | 0.25           | 0.475  | Top 47.5%        |
| 50    | 0.5            | 0.65   | Top 65%          |
| 75    | 0.75           | 0.825  | Top 82.5%        |
| 100   | 1.0            | 1.0    | Top 100% (all)   |
| 101+  | -              | 1.0    | Top 100% (all)   |

### Why This Makes Sense

#### Early Training (Low Budget)
- Model predictions are **uncertain** (not well-trained yet)
- Many samples have low confidence
- **Conservative selection**: Only select top 30% most confident samples
- Avoids including noisy samples that might confuse the model

#### Mid Training (Medium Budget)
- Model is learning and becoming more confident
- More samples have reliable predictions
- **Gradual expansion**: Increase to 50-70% selection
- Model can handle more samples as it improves

#### Late Training (High Budget)
- Model is well-trained and confident
- Most clean samples have high confidence
- **Full selection**: Select 80-100% of samples
- Model can distinguish clean from noisy effectively

### Physical Interpretation

Think of it as a **confidence threshold** that gradually relaxes:

- **Early epochs**: High threshold → Only very confident samples selected
- **Mid epochs**: Medium threshold → Moderately confident samples also selected  
- **Late epochs**: Low threshold → Most samples selected (model is confident)

### Exponential Mode Comparison

With `mode='exp'`, the budget increases **faster initially**, then slows down:

| Epoch | Linear Budget | Exponential Budget |
|-------|---------------|-------------------|
| 0     | 0.30          | 0.30              |
| 25    | 0.475         | 0.742             |
| 50    | 0.65          | 0.906             |
| 75    | 0.825         | 0.965             |
| 100   | 1.0           | 0.987             |

**Exponential mode**: Faster initial expansion, more conservative later.

---

## Confidence Explanation

### What Does "Confidence" Mean?

**Confidence is about classification** - how confident the model is about which class a sample belongs to. However, it's used as a **proxy** to infer whether a sample is clean or noisy.

### Classification Confidence (What OT Actually Computes)

The confidence comes from the model's **classification predictions**:

```python
# Model makes predictions
logits = model(inputs)
out = logits.softmax(dim=-1)  # P[i, j] = probability sample i belongs to class j

# OT computes coupling from predictions
coupling = ot.sinkhorn(..., M=-P, ...)  # Cost matrix = -P

# Confidence = max coupling value
max_value, argmax_plabels = torch.max(couplings, axis=1)
conf = max_value / (1/couplings.size(0))  # Normalized confidence
```

**What this measures:**
- How confident the model is about **which class** the sample belongs to
- High `conf` = Model is very sure about the class prediction
- Low `conf` = Model is uncertain about the class prediction

### Clean/Noisy Inference (How It's Used)

OT uses classification confidence as a **proxy** to infer clean/noisy status:

**Assumption:**
- **Clean samples** → Model sees correct label → Learns correct pattern → High classification confidence
- **Noisy samples** → Model sees wrong label → Confused/uncertain → Low classification confidence

**Selection Logic:**
```python
# Select top-k samples with highest classification confidence
max_values, _ = torch.max(coupling, 1)
topk_indices = torch.topk(max_values, topk_num)
selected_mask[topk_indices] = True  # Mark as "clean"
```

### The Three Types of Samples

Looking at `get_masks()` function:

```python
equal_label_mask = torch.eq(noisy_labels, argmax_plabels)

conf_l_mask = selected_mask AND (noisy_label == predicted_label)
conf_u_mask = selected_mask AND (noisy_label != predicted_label)
lowconf_u_mask = NOT selected_mask
```

**Interpretation:**

1. **`conf_l_mask`** (Confident Labeled):
   - Selected by OT (high classification confidence)
   - AND noisy label matches predicted label
   - **Interpretation**: Confident about classification AND label matches → Likely clean

2. **`conf_u_mask`** (Confident Unlabeled):
   - Selected by OT (high classification confidence)
   - BUT noisy label does NOT match predicted label
   - **Interpretation**: Confident about classification BUT label doesn't match → Likely noisy (model disagrees with label)

3. **`lowconf_u_mask`** (Low Confidence Unlabeled):
   - NOT selected by OT (low classification confidence)
   - **Interpretation**: Uncertain about classification → Likely noisy or ambiguous

### Key Insight

**Confidence is about classification, not directly about being clean.**

However, there's a **correlation**:
- High classification confidence → More likely to be clean
- Low classification confidence → More likely to be noisy

But this correlation is **not perfect**:
- A clean sample might have low confidence if it's ambiguous/hard
- A noisy sample might have high confidence if the wrong label happens to match the model's prediction

### Example Scenarios

#### Scenario 1: Clean Sample with High Confidence
- **Ground truth**: Class 5 (car)
- **Noisy label**: Class 5 (car) ✓
- **Model prediction**: Class 5 with 0.9 probability
- **OT coupling**: High coupling to class 5
- **Result**: Selected as clean (`conf_l_mask`)

#### Scenario 2: Noisy Sample with Low Confidence
- **Ground truth**: Class 5 (car)
- **Noisy label**: Class 3 (dog) ✗
- **Model prediction**: Class 5 with 0.4 probability (uncertain)
- **OT coupling**: Low coupling (uncertain)
- **Result**: Rejected (`lowconf_u_mask`)

#### Scenario 3: Noisy Sample with High Confidence (False Positive)
- **Ground truth**: Class 5 (car)
- **Noisy label**: Class 3 (dog) ✗
- **Model prediction**: Class 3 with 0.85 probability (confident but wrong!)
- **OT coupling**: High coupling to class 3
- **Result**: Selected as clean (`conf_u_mask`) - **Mistake!**

#### Scenario 4: Clean Sample with Low Confidence (False Negative)
- **Ground truth**: Class 5 (car)
- **Noisy label**: Class 5 (car) ✓
- **Model prediction**: Class 5 with 0.3 probability (uncertain, maybe ambiguous image)
- **OT coupling**: Low coupling
- **Result**: Rejected (`lowconf_u_mask`) - **Mistake!**

### Why This Works (Despite Imperfections)

1. **Most clean samples** have high classification confidence → Selected
2. **Most noisy samples** have low classification confidence → Rejected
3. **False positives/negatives** are minority → Overall effect is positive

The model learns to be confident about patterns it sees consistently (clean samples) and uncertain about inconsistent patterns (noisy samples).

---

## Coupling Matrix and Confidence

### Understanding `max_value` from Coupling Matrix

**Question:** Does `max_value, argmax_plabels = torch.max(couplings, axis=1)` return the highest probability belonging to a class as confidence?

**Answer:** **Not exactly.** `max_value` is the **maximum coupling value** (raw, unnormalized), not a probability. However, it does represent **confidence** - the strength of association between a sample and its most likely class.

### What is `couplings`?

`couplings` is the **optimal transport coupling matrix** from Sinkhorn algorithm:
- Shape: `[N_samples, N_classes]`
- Each entry `couplings[i, j]` = amount of "mass" transported from sample `i` to class `j`
- **Not normalized** - row sums can vary

### What Does `torch.max(couplings, axis=1)` Return?

```python
max_value, argmax_plabels = torch.max(couplings, axis=1)
```

**Returns:**
- `max_value`: Maximum coupling value for each sample (shape: `[N_samples]`)
- `argmax_plabels`: Index of the class with maximum coupling (shape: `[N_samples]`)

**Example:**
```python
# Sample 0: couplings = [0.05, 0.15, 0.02, ...]
#           max_value[0] = 0.15, argmax_plabels[0] = 1 (class 1)

# Sample 1: couplings = [0.01, 0.03, 0.12, ...]
#           max_value[1] = 0.12, argmax_plabels[1] = 2 (class 2)
```

### Is `max_value` a Probability?

**No, `max_value` is NOT a probability** because:
1. Couplings are **not normalized** - row sums can vary
2. `max_value` can be any positive value, not necessarily in [0, 1]

**However**, there IS a normalized probability version:

```python
row_sum = torch.sum(couplings, 1).reshape((-1, 1))
pseudo_labels = torch.div(couplings, row_sum)  # Normalized probabilities
max_probability = torch.max(pseudo_labels, axis=1)[0]  # This IS a probability
```

### What Does `max_value` Represent?

`max_value` represents:
- **Strength of coupling** to the most likely class
- **Confidence indicator** - higher value = stronger association = more confident
- **Raw transport mass** - not normalized, but proportional to confidence

### How is Confidence Calculated?

Looking at the code:

```python
max_value, argmax_plabels = torch.max(couplings, axis=1)
conf = max_value / (1 / couplings.size(0))  # Normalize by expected value
conf = torch.clip(conf, min=0, max=1.0)
```

**Confidence formula:**
- `conf = max_value / (1 / N_classes)`
- `conf = max_value * N_classes`
- Normalized to [0, 1] range

**Interpretation:**
- If `max_value = 1/N_classes` → `conf = 1.0` (baseline confidence)
- If `max_value > 1/N_classes` → `conf > 1.0` (clipped to 1.0) → High confidence
- If `max_value < 1/N_classes` → `conf < 1.0` → Low confidence

### Comparison: `max_value` vs `max_probability`

#### `max_value` (Raw Coupling)
```python
max_value, argmax = torch.max(couplings, axis=1)
# Example: max_value = 0.15 (raw coupling value)
# Not normalized, can be any positive value
```

#### `max_probability` (Normalized)
```python
pseudo_labels = couplings / row_sum
max_prob, argmax = torch.max(pseudo_labels, axis=1)
# Example: max_prob = 0.75 (75% probability)
# Normalized, always in [0, 1]
```

### Which One is Used for Selection?

**For selection** (in `curriculum_structure_aware_PL`):
```python
max_values, _ = torch.max(coupling, 1)  # Raw coupling values
topk_indices = torch.topk(max_values, topk_num)  # Select top-k
```

**Uses raw `max_value`**, not normalized probability, because:
- Relative ranking is what matters (top-k selection)
- Normalization doesn't change the ranking
- Raw values are sufficient for comparison

### Summary

| Aspect | `max_value` | `max_probability` |
|--------|-------------|------------------|
| **Type** | Raw coupling value | Normalized probability |
| **Range** | Any positive value | [0, 1] |
| **Meaning** | Strength of association | Probability of class |
| **Used for** | Selection (ranking) | Pseudo-label distribution |
| **Is it confidence?** | Yes, as confidence indicator | Yes, as probability |

**`max_value` is NOT the highest probability**, but it **does represent confidence**:
- It's the raw coupling strength to the most likely class
- Higher `max_value` = stronger association = higher confidence
- Used directly for ranking/selection (top-k)
- The normalized probability version (`pseudo_labels`) is computed separately

---

## Pseudo-Label Computation

### Overview

Pseudo-labels are computed using Optimal Transport (OT) to match image features with class prototypes (text features from prompts). The process involves several steps:

### Step-by-Step Process

#### Step 1: Get Model Predictions

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

#### Step 2: Prepare Cost Matrix

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

#### Step 3: Set Up OT Problem

```python
# Source distribution (samples)
a = torch.ones((N_samples,)) / N_samples  # Uniform: each sample has weight 1/N

# Target distribution (classes)
b = torch.ones((num_classes,)) / num_classes * top_percent  # Scaled by budget
```

**Marginal Constraints:**
- Source (`a`): Uniform distribution over samples
- Target (`b`): Distribution over classes, scaled by `budget` (e.g., 0.3 = 30% of total mass)

#### Step 4: Solve OT Problem with Sinkhorn

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

#### Step 5: Compute Pseudo-Labels from Coupling Matrix

```python
# Normalize coupling matrix to get pseudo-label distribution
row_sum = torch.sum(couplings, 1).reshape((-1, 1))  # Sum over classes for each sample
pseudo_labels = torch.div(couplings, row_sum)        # Normalize: [N_samples, num_classes]
```

**Result:**
- `pseudo_labels[i, j]` = probability that sample `i` belongs to class `j` (normalized)
- Each row sums to 1.0 (probability distribution)

#### Step 6: Get Hard Pseudo-Labels

```python
# Get the class with maximum coupling for each sample
max_value, argmax_plabels = torch.max(couplings, axis=1)
```

**Result:**
- `argmax_plabels[i]` = class index with highest coupling for sample `i`
- This is the **hard pseudo-label** (single class assignment)

### Complete Flow Diagram

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

### Key Formulas

#### From Paper (Equation 9)

The OT problem solved:
```
min < -log(T · I^T), Q >
Q ∈ R^{C×N}_+
s.t. Q·1_N = (1/C)·1_C,  Q^T·1_C = (1/N)·1_N
```

#### In Code

```python
# Cost matrix (simplified - uses model predictions instead of direct similarity)
M = -P  # where P = model predictions

# OT problem
coupling = ot.sinkhorn(a, b, M=-P, reg=reg_e)

# Pseudo-labels
pseudo_labels = coupling / row_sum(coupling)  # Normalized
argmax_plabels = argmax(coupling, axis=1)     # Hard labels
```

### Differences from Paper

**Paper:**
- Uses direct similarity: `T · I^T` (text features × image features)
- Cost: `-log(T · I^T)`

**Code:**
- Uses model predictions: `P = softmax(logits)` (already includes similarity computation)
- Cost: `-P` (negative probabilities)

**Why different?**
- The model already computes `T · I^T` internally (text-image similarity)
- Using `-P` is equivalent but simpler (probabilities already normalized)

### Example

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

---

## Complete Workflow

### End-to-End Process

1. **Model Training**:
   - Model learns from noisy labels
   - Generates predictions `P` for all samples

2. **OT Processing** (every epoch in `before_epoch()`):
   - Extract features and predictions
   - Calculate budget (curriculum learning)
   - Solve OT problem with Sinkhorn
   - Get coupling matrix

3. **Sample Selection**:
   - Calculate `max_value` for each sample
   - Select top `budget%` samples with highest `max_value`
   - Create `selected_mask`

4. **Pseudo-Label Generation**:
   - Normalize coupling matrix → `pseudo_labels` (soft)
   - `argmax(coupling)` → `argmax_plabels` (hard)

5. **Dataset Splitting**:
   - `conf_l_mask`: Selected AND noisy_label == predicted_label → Clean dataset
   - `conf_u_mask`: Selected AND noisy_label != predicted_label → Noisy dataset
   - `lowconf_u_mask`: Not selected → Noisy dataset

6. **Training**:
   - Clean dataset → CE loss
   - Noisy dataset → MAE loss (or ignored if MAE is disabled)

### Code Flow Summary

```
Model Predictions (P)
  ↓
Cost Matrix (M = -P)
  ↓
Budget Calculation (curriculum_scheduler)
  ↓
Sinkhorn Algorithm
  ↓
Coupling Matrix (Q)
  ↓
Max Coupling per Sample (max_value)
  ↓
Top-K Selection (selected_mask)
  ↓
Confidence Scores (conf)
  ↓
Pseudo-Labels (normalized Q)
  ↓
Hard Pseudo-Labels (argmax Q)
  ↓
Dataset Splitting (conf_l_mask, conf_u_mask, lowconf_u_mask)
```

### Key Takeaways

1. **OT identifies clean/noisy samples** based on prediction confidence
2. **Budget controls** how many samples are selected (curriculum learning)
3. **Confidence** is about classification, used as proxy for clean/noisy
4. **Coupling matrix** represents optimal transport plan
5. **Pseudo-labels** come from normalized coupling matrix
6. **Selection** uses raw coupling values (not normalized probabilities)

---

## Summary

This comprehensive guide covers:

- **Optimal Transport**: Mathematical framework for matching samples to classes
- **Budget Calculation**: How the fraction of selected samples is determined
- **Curriculum Learning**: Gradual increase in budget over training epochs
- **Confidence**: Classification confidence used as proxy for clean/noisy inference
- **Coupling Matrix**: Optimal transport plan representing sample-class associations
- **Pseudo-Labels**: Soft and hard labels derived from coupling matrix

Together, these components form a complete system for noisy label learning using Optimal Transport theory.

