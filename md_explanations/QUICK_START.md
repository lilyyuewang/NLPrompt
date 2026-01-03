# Quick Start: Parallel Training on 4 A40 GPUs

## 🚀 What I Created

Two scripts to run multiple training tasks in parallel on your 4 A40 GPUs:

1. **`parallel_train.py`** - Python script (recommended)
2. **`parallel_train.sh`** - Bash script (alternative)

## 📋 Current Configuration

**12 Tasks on Caltech101:**
- Noise rates: 0.125, 0.25, 0.375, 0.50, 0.625, 0.75
- Noise types: sym (symmetric), asym (asymmetric)
- Will run in **3 batches** of 4 tasks each (one task per GPU)

## ⚡ How to Run

### Option 1: Python Script (Recommended)

```bash
# Make it executable
chmod +x parallel_train.py

# Run with default settings
python parallel_train.py

# Or with custom data path (if needed)
python parallel_train.py --data-root ~/datasets/nlprompt/caltech-101 --gpus 0,1,2,3
```

### Option 2: Bash Script

```bash
# Make it executable
chmod +x parallel_train.sh

# Edit DATA_ROOT if needed (line 6)
# Then run:
./parallel_train.sh
```

## 📊 What Happens

```
Batch 1 (Tasks 1-4):
  GPU 0: caltech101, noise=sym_0.125
  GPU 1: caltech101, noise=sym_0.25
  GPU 2: caltech101, noise=sym_0.375
  GPU 3: caltech101, noise=sym_0.50
  
[Wait for all 4 to complete]

Batch 2 (Tasks 5-8):
  GPU 0: caltech101, noise=sym_0.625
  GPU 1: caltech101, noise=sym_0.75
  GPU 2: caltech101, noise=asym_0.125
  GPU 3: caltech101, noise=asym_0.25
  
[Wait for all 4 to complete]

Batch 3 (Tasks 9-12):
  GPU 0: caltech101, noise=asym_0.375
  GPU 1: caltech101, noise=asym_0.50
  GPU 2: caltech101, noise=asym_0.625
  GPU 3: caltech101, noise=asym_0.75
```

## 🎯 Key Benefits

- **4x faster** than running tasks sequentially
- **Automatic batching** - no manual intervention needed
- **GPU isolation** - each task uses its own GPU
- **Auto-skip** - won't rerun if results exist
- **Progress monitoring** - real-time status updates
- **Error handling** - one failure doesn't stop others

## 📝 Monitoring

### Check GPU Usage
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi
```

### View Logs
```bash
# Logs are saved to logs/ directory
tail -f logs/gpu0_caltech101_*.log
tail -f logs/gpu1_caltech101_*.log
```

### Check Progress
The script prints status updates every 10 seconds:
```
[00:15:23] Still running: 4 tasks on GPUs [0, 1, 2, 3]
✓ [GPU 2] Task completed: caltech101 (noise=sym_0.375)
  Time: 01:23:45
```

## 🔧 Customization

### Change Tasks

Edit `parallel_train.py` line 237, function `get_default_tasks()`:

```python
def get_default_tasks() -> List[TaskConfig]:
    tasks = []
    
    # Your custom tasks here
    for noise_rate in [0.1, 0.2, 0.3]:
        tasks.append(TaskConfig(
            dataset="caltech101",
            shots=16,
            noise_rate=noise_rate,
            noise_type="sym",
            num_classes=100,
            seed=1
        ))
    
    return tasks
```

### Use Different GPUs

```bash
# Use GPUs 4,5,6,7 instead
python parallel_train.py --gpus 4,5,6,7

# Use only 2 GPUs
python parallel_train.py --gpus 0,1
```

## 📁 Output Structure

Results saved to:
```
output/caltech101/NLPrompt/rn50_16shots/
├── noise_sym_0.125/seed1/
├── noise_sym_0.25/seed1/
├── noise_sym_0.375/seed1/
├── noise_sym_0.50/seed1/
├── noise_sym_0.625/seed1/
├── noise_sym_0.75/seed1/
├── noise_asym_0.125/seed1/
├── noise_asym_0.25/seed1/
├── noise_asym_0.375/seed1/
├── noise_asym_0.50/seed1/
├── noise_asym_0.625/seed1/
└── noise_asym_0.75/seed1/
```

## ⏱️ Expected Time

On 4x A40 GPUs:
- ~2-3 hours per batch
- ~6-9 hours total for all 12 tasks
- vs ~24-36 hours if running sequentially!

## 🐛 Troubleshooting

### Scripts won't run?
```bash
chmod +x parallel_train.py parallel_train.sh
python parallel_train.py  # Or run with python explicitly
```

### Want to rerun everything?
```bash
python parallel_train.py --no-skip-existing
```

### Need more details?
See `PARALLEL_TRAINING.md` for comprehensive documentation.

## 📚 Files Created

1. **`parallel_train.py`** - Main Python script (349 lines)
2. **`parallel_train.sh`** - Bash alternative (205 lines)
3. **`PARALLEL_TRAINING.md`** - Full documentation
4. **`QUICK_START.md`** - This file

## ✅ Ready to Go!

Just run:
```bash
python parallel_train.py
```

And watch your 4 A40s train 4 models simultaneously! 🚀

