# Budget Calculation in Curriculum Scheduler

## Function Overview

The `curriculum_scheduler` function gradually increases the budget from a starting value (`begin`) to an ending value (`end`) over `T` steps using different scheduling modes.

## Formula

```python
budget = begin + ratio * (end - begin)
```

Where:
- `begin`: Starting budget (default: `BEGIN_RATE = 0.3` = 30%)
- `end`: Final budget (default: `1.0` = 100%)
- `ratio`: Progress ratio from 0 to 1, calculated based on `mode`

## Parameters

- **`t`**: Current epoch/step (e.g., epoch 5)
- **`T`**: Total curriculum epochs (`CURRICLUM_EPOCH`, default: 0)
- **`begin`**: Starting budget (`BEGIN_RATE`, default: 0.3)
- **`end`**: Final budget (default: 1.0)
- **`mode`**: Scheduling mode (`CURRICLUM_MODE`, default: 'linear')

## Progress Ratio (`pho`)

```python
pho = t / T  # Progress from 0.0 to 1.0
```

## Scheduling Modes

### 1. Linear Mode (`mode='linear'`)
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

**Visualization:**
```
Budget
 1.0 |                    *
     |               *
     |          *
     |     *
 0.3 |*
     +----+----+----+----+----+----+----+----+----+----
     0   10   20   30   40   50   60   70   80   90  100
                              Epoch
```

### 2. Exponential Mode (`mode='exp'`)
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

**Visualization:**
```
Budget
 1.0 |                    *
     |               *
     |          *
     |     *
 0.3 |*
     +----+----+----+----+----+----+----+----+----+----
     0   10   20   30   40   50   60   70   80   90  100
                              Epoch
```
*Exponential mode increases faster initially, then slows down*

### 3. Custom Mode (`mode='customize'`)
```python
ratio = func(t, T)  # User-defined function
budget = begin + func(t, T) * (end - begin)
```

## Current Configuration

Based on `train.py`:
- `BEGIN_RATE = 0.3` (starts at 30%)
- `CURRICLUM_EPOCH = 0` (no curriculum, budget stays at 0.3)
- `CURRICLUM_MODE = 'linear'`

## Important Note

**If `CURRICLUM_EPOCH = 0`:**
```python
if self.epoch < curriclum_epoch:  # 0 < 0 is False
    budget, pho = curriculum_scheduler(...)
else:
    budget, pho = 1., 1.  # Always executes!
```

When `CURRICLUM_EPOCH = 0`, the budget is **always set to 1.0** (100%), meaning OT selects **all samples** regardless of confidence!

## Example Calculation Flow

**Scenario**: `BEGIN_RATE=0.3`, `CURRICLUM_EPOCH=50`, `CURRICLUM_MODE='linear'`, `epoch=25`

1. Calculate progress: `pho = 25 / 50 = 0.5`
2. Calculate ratio (linear): `ratio = 0.5`
3. Calculate budget: `budget = 0.3 + 0.5 * (1.0 - 0.3) = 0.3 + 0.5 * 0.7 = 0.65`
4. Result: **Budget = 65%** → Select top 65% most confident samples

## Relationship to Noise Rate

**The budget is NOT automatically related to noise rate!**

- Budget is controlled by curriculum scheduler (independent hyperparameter)
- If you want `budget = 1 - noise_rate`, you need to:
  1. Set `BEGIN_RATE = 1 - noise_rate`, OR
  2. Modify the code to cap budget at `1 - noise_rate`

## Summary

The budget calculation follows a **linear interpolation** (or exponential/custom) from `begin` to `end`:
- Starts at `BEGIN_RATE` (default: 30%)
- Gradually increases to 100% over `CURRICLUM_EPOCH` epochs
- If `CURRICLUM_EPOCH = 0`, budget is fixed at 100%
- The selected samples are the top `budget%` with highest coupling values

