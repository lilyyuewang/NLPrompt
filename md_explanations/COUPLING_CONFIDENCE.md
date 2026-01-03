# Understanding `max_value` from Coupling Matrix

## Question

Does `max_value, argmax_plabels = torch.max(couplings, axis=1)` return the highest probability belonging to a class as confidence?

## Answer

**Not exactly.** `max_value` is the **maximum coupling value** (raw, unnormalized), not a probability. However, it does represent **confidence** - the strength of association between a sample and its most likely class.

## What is `couplings`?

`couplings` is the **optimal transport coupling matrix** from Sinkhorn algorithm:
- Shape: `[N_samples, N_classes]`
- Each entry `couplings[i, j]` = amount of "mass" transported from sample `i` to class `j`
- **Not normalized** - row sums can vary

## What Does `torch.max(couplings, axis=1)` Return?

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

## Is `max_value` a Probability?

**No, `max_value` is NOT a probability** because:
1. Couplings are **not normalized** - row sums can vary
2. `max_value` can be any positive value, not necessarily in [0, 1]

**However**, there IS a normalized probability version:

```python
row_sum = torch.sum(couplings, 1).reshape((-1, 1))
pseudo_labels = torch.div(couplings, row_sum)  # Normalized probabilities
max_probability = torch.max(pseudo_labels, axis=1)[0]  # This IS a probability
```

## What Does `max_value` Represent?

`max_value` represents:
- **Strength of coupling** to the most likely class
- **Confidence indicator** - higher value = stronger association = more confident
- **Raw transport mass** - not normalized, but proportional to confidence

## How is Confidence Calculated?

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

## Comparison: `max_value` vs `max_probability`

### `max_value` (Raw Coupling)
```python
max_value, argmax = torch.max(couplings, axis=1)
# Example: max_value = 0.15 (raw coupling value)
# Not normalized, can be any positive value
```

### `max_probability` (Normalized)
```python
pseudo_labels = couplings / row_sum
max_prob, argmax = torch.max(pseudo_labels, axis=1)
# Example: max_prob = 0.75 (75% probability)
# Normalized, always in [0, 1]
```

## Which One is Used for Selection?

**For selection** (in `curriculum_structure_aware_PL`):
```python
max_values, _ = torch.max(coupling, 1)  # Raw coupling values
topk_indices = torch.topk(max_values, topk_num)  # Select top-k
```

**Uses raw `max_value`**, not normalized probability, because:
- Relative ranking is what matters (top-k selection)
- Normalization doesn't change the ranking
- Raw values are sufficient for comparison

## Summary

| Aspect | `max_value` | `max_probability` |
|--------|-------------|------------------|
| **Type** | Raw coupling value | Normalized probability |
| **Range** | Any positive value | [0, 1] |
| **Meaning** | Strength of association | Probability of class |
| **Used for** | Selection (ranking) | Pseudo-label distribution |
| **Is it confidence?** | Yes, as confidence indicator | Yes, as probability |

## Answer to Your Question

**`max_value` is NOT the highest probability**, but it **does represent confidence**:
- It's the raw coupling strength to the most likely class
- Higher `max_value` = stronger association = higher confidence
- Used directly for ranking/selection (top-k)
- The normalized probability version (`pseudo_labels`) is computed separately

**In essence:** `max_value` is a **confidence score** (raw coupling strength), not a probability, but it serves the same purpose - indicating how confident OT is about the sample's class assignment.

