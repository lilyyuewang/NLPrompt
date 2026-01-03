# Curriculum Learning in OT Selection

## Yes! OT Selects More Samples as Training Progresses

When curriculum learning is enabled (`CURRICLUM_EPOCH > 0`), the budget **gradually increases** from `BEGIN_RATE` to `1.0` over the course of training, meaning OT selects more and more samples as "clean" over time.

## How It Works

### Budget Schedule

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

**Visualization:**
```
Budget (%)
 100 |                    *
     |               *
     |          *
     |     *
  30 |*
     +----+----+----+----+----+----+----+----+----+----
     0   10   20   30   40   50   60   70   80   90  100
                              Epoch
```

## Why This Makes Sense

### Early Training (Low Budget)
- Model predictions are **uncertain** (not well-trained yet)
- Many samples have low confidence
- **Conservative selection**: Only select top 30% most confident samples
- Avoids including noisy samples that might confuse the model

### Mid Training (Medium Budget)
- Model is learning and becoming more confident
- More samples have reliable predictions
- **Gradual expansion**: Increase to 50-70% selection
- Model can handle more samples as it improves

### Late Training (High Budget)
- Model is well-trained and confident
- Most clean samples have high confidence
- **Full selection**: Select 80-100% of samples
- Model can distinguish clean from noisy effectively

## Physical Interpretation

Think of it as a **confidence threshold** that gradually relaxes:

- **Early epochs**: High threshold → Only very confident samples selected
- **Mid epochs**: Medium threshold → Moderately confident samples also selected  
- **Late epochs**: Low threshold → Most samples selected (model is confident)

## Current Configuration Issue

**Your current config:**
```python
CURRICLUM_EPOCH = 0  # No curriculum learning!
BEGIN_RATE = 0.3
```

**Result:**
- Budget is **always 1.0** (100%)
- OT selects **all samples** from the start
- No curriculum learning effect!

**To enable curriculum learning:**
```python
CURRICLUM_EPOCH = 50  # or 100, 200, etc.
BEGIN_RATE = 0.3
CURRICLUM_MODE = 'linear'  # or 'exp'
```

## Exponential Mode

With `mode='exp'`, the budget increases **faster initially**, then slows down:

| Epoch | Linear Budget | Exponential Budget |
|-------|---------------|-------------------|
| 0     | 0.30          | 0.30              |
| 25    | 0.475         | 0.742             |
| 50    | 0.65          | 0.906             |
| 75    | 0.825         | 0.965             |
| 100   | 1.0           | 0.987             |

**Exponential mode**: Faster initial expansion, more conservative later.

## Summary

✅ **Yes, OT selects more samples as training progresses** (if curriculum learning is enabled)

**Mechanism:**
1. Budget starts low (e.g., 30%)
2. Gradually increases over `CURRICLUM_EPOCH` epochs
3. Reaches 100% by the end of curriculum period
4. Stays at 100% for remaining epochs

**Benefits:**
- Early: Conservative selection (avoid noisy samples)
- Mid: Gradual expansion (model improving)
- Late: Full selection (model confident)

**Current Status:**
- Your config has `CURRICLUM_EPOCH = 0` → **No curriculum learning**
- Budget is always 100% → All samples selected from start
- To enable: Set `CURRICLUM_EPOCH > 0`

