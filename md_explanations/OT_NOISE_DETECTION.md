# Does OT Know the Noise Rate?

## Short Answer

**No, OT does NOT know or use the noise rate to determine how many samples are noisy.**

OT uses **confidence-based selection** - it selects the top `budget%` most confident samples, regardless of the actual noise rate.

## How Noise is Introduced

1. **Noise Rate is Set in Config** (`DATASET.NOISE_RATE`):
   - Used during data loading to corrupt labels
   - Example: `NOISE_RATE = 0.125` means 12.5% of labels are flipped
   - This happens in `dassl`'s `DatasetBase` class (not shown in this codebase)

2. **Ground Truth is Preserved**:
   - `gt_labels`: Original clean labels (stored as `gttarget`)
   - `noisy_labels`: Corrupted labels (stored as `label`)
   - Used for evaluation, but **not used by OT**

## How OT Works

OT **does NOT** use the noise rate. Instead, it:

1. **Gets Model Predictions** (`P`):
   - Model makes predictions on all samples
   - `P[i, j]` = probability sample `i` belongs to class `j`

2. **Computes Coupling Matrix**:
   - Uses Sinkhorn algorithm with cost matrix `M = -P`
   - Higher prediction confidence → Lower cost → Higher coupling

3. **Selects Top-K Based on Budget**:
   ```python
   max_values, _ = torch.max(coupling, 1)  # Max coupling per sample
   topk_num = int(total * top_percent)     # How many to select
   _, topk_indices = torch.topk(max_values, topk_num)  # Top-k indices
   selected_mask[topk_indices] = True      # Mark as selected
   ```

4. **Budget Controls Selection**:
   - `budget = 0.3` → Select top 30% most confident samples
   - `budget = 0.875` → Select top 87.5% most confident samples
   - **Budget is independent of noise rate!**

## Key Insight

**OT doesn't "know" how many samples are noisy.**

Instead, OT:
- Ranks samples by confidence (coupling values)
- Selects the top `budget%` as "clean"
- The rest are implicitly treated as "noisy" (rejected)

## Example Scenarios

### Scenario 1: 12.5% Noise, Budget = 0.3 (30%)

- **Actual**: 87.5% clean, 12.5% noisy
- **OT selects**: Top 30% most confident samples
- **Result**: OT selects fewer samples than there are clean samples
- **Implication**: Some clean samples are rejected (false negatives)

### Scenario 2: 12.5% Noise, Budget = 0.875 (87.5%)

- **Actual**: 87.5% clean, 12.5% noisy  
- **OT selects**: Top 87.5% most confident samples
- **Result**: OT selects approximately the same number as clean samples
- **Implication**: Better match, but may include some noisy samples (false positives)

### Scenario 3: 50% Noise, Budget = 0.3 (30%)

- **Actual**: 50% clean, 50% noisy
- **OT selects**: Top 30% most confident samples
- **Result**: OT selects fewer samples than clean samples
- **Implication**: Many clean samples are rejected

### Scenario 4: 50% Noise, Budget = 0.875 (87.5%)

- **Actual**: 50% clean, 50% noisy
- **OT selects**: Top 87.5% most confident samples
- **Result**: OT selects more samples than clean samples
- **Implication**: Many noisy samples are incorrectly selected as clean

## Why This Works

OT relies on the assumption that:
1. **Clean samples** have high prediction confidence → High coupling → Selected
2. **Noisy samples** have low prediction confidence → Low coupling → Rejected

The model learns to be confident about clean samples and uncertain about noisy samples, so OT can separate them based on confidence alone.

## Current Configuration Issue

In your current config:
- `CURRICLUM_EPOCH = 0` → Budget is always **1.0 (100%)**
- This means OT selects **ALL samples** regardless of confidence
- This defeats the purpose of OT for noise detection!

To fix: Set `CURRICLUM_EPOCH > 0` to enable curriculum learning.

## Summary

| Question | Answer |
|----------|--------|
| Does model know noise rate? | Yes, used to corrupt labels during data loading |
| Does OT use noise rate? | **No**, OT doesn't know or use the noise rate |
| How does OT determine noisy samples? | Based on **confidence scores** (coupling values) |
| How many samples are selected? | Controlled by **budget parameter**, not noise rate |
| What if budget ≠ (1 - noise_rate)? | OT still works, but may select wrong number of samples |

## Optimal Budget Setting

For best results, you might want to set:
```python
budget = min(budget_from_curriculum, 1.0 - noise_rate)
```

This ensures OT never selects more samples than there are clean samples, but this is **not currently implemented** in the code.

