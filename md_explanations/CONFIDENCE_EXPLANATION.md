# What Does "Confidence" Mean in OT?

## Short Answer

**Confidence is about classification** - how confident the model is about which class a sample belongs to. However, it's used as a **proxy** to infer whether a sample is clean or noisy.

## Detailed Explanation

### 1. Classification Confidence (What OT Actually Computes)

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

### 2. Clean/Noisy Inference (How It's Used)

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

### 3. The Three Types of Samples

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

## Key Insight

**Confidence is about classification, not directly about being clean.**

However, there's a **correlation**:
- High classification confidence → More likely to be clean
- Low classification confidence → More likely to be noisy

But this correlation is **not perfect**:
- A clean sample might have low confidence if it's ambiguous/hard
- A noisy sample might have high confidence if the wrong label happens to match the model's prediction

## Example Scenarios

### Scenario 1: Clean Sample with High Confidence
- **Ground truth**: Class 5 (car)
- **Noisy label**: Class 5 (car) ✓
- **Model prediction**: Class 5 with 0.9 probability
- **OT coupling**: High coupling to class 5
- **Result**: Selected as clean (`conf_l_mask`)

### Scenario 2: Noisy Sample with Low Confidence
- **Ground truth**: Class 5 (car)
- **Noisy label**: Class 3 (dog) ✗
- **Model prediction**: Class 5 with 0.4 probability (uncertain)
- **OT coupling**: Low coupling (uncertain)
- **Result**: Rejected (`lowconf_u_mask`)

### Scenario 3: Noisy Sample with High Confidence (False Positive)
- **Ground truth**: Class 5 (car)
- **Noisy label**: Class 3 (dog) ✗
- **Model prediction**: Class 3 with 0.85 probability (confident but wrong!)
- **OT coupling**: High coupling to class 3
- **Result**: Selected as clean (`conf_u_mask`) - **Mistake!**

### Scenario 4: Clean Sample with Low Confidence (False Negative)
- **Ground truth**: Class 5 (car)
- **Noisy label**: Class 5 (car) ✓
- **Model prediction**: Class 5 with 0.3 probability (uncertain, maybe ambiguous image)
- **OT coupling**: Low coupling
- **Result**: Rejected (`lowconf_u_mask`) - **Mistake!**

## Why This Works (Despite Imperfections)

1. **Most clean samples** have high classification confidence → Selected
2. **Most noisy samples** have low classification confidence → Rejected
3. **False positives/negatives** are minority → Overall effect is positive

The model learns to be confident about patterns it sees consistently (clean samples) and uncertain about inconsistent patterns (noisy samples).

## Summary

| Question | Answer |
|----------|--------|
| What does confidence measure? | **Classification confidence** - how sure the model is about which class |
| Does it directly measure "clean"? | **No**, it's a proxy based on correlation |
| How is it used? | High confidence → Selected as clean, Low confidence → Rejected as noisy |
| Is it perfect? | **No**, there can be false positives/negatives |
| Why does it work? | Clean samples tend to have high confidence, noisy samples tend to have low confidence |

## Code Flow

```
Model Prediction (P) 
  ↓
OT Coupling (from -P)
  ↓
Max Coupling per Sample
  ↓
Confidence Score
  ↓
Top-K Selection (high confidence)
  ↓
Inferred as "Clean"
```

**The confidence is about classification, but it's used to infer clean/noisy status based on the assumption that confident predictions correlate with clean labels.**

