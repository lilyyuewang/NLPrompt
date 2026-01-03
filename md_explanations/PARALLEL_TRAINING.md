# Parallel Training Guide for NLPrompt

## Hardware Setup
- **GPUs**: 4x NVIDIA A40 (48GB each)
- **Configuration**: One task per GPU for parallel training

## Overview

This guide explains how to run multiple training tasks in parallel across 4 GPUs. The current setup trains 12 different configurations on Caltech101:
- **6 noise rates**: 0.125, 0.25, 0.375, 0.50, 0.625, 0.75
- **2 noise types**: sym (symmetric), asym (asymmetric)
- **Total**: 12 tasks (will run in 3 batches of 4 tasks each)

## Available Scripts

### 1. Python Script: `parallel_train.py`
**Recommended for most users** - More flexible and provides better monitoring.

### 2. Bash Script: `parallel_train.sh`
Simpler alternative using pure bash.

---

## Quick Start

### Option 1: Using Python Script (Recommended)

```bash
# Basic usage with default settings
python parallel_train.py

# Specify custom data root and GPUs (default is ~/datasets/caltech-101)
python parallel_train.py --data-root ~/datasets/nlprompt/caltech-101 --gpus 0,1,2,3

# Run all tasks even if output exists
python parallel_train.py --no-skip-existing

# Use different GPUs (e.g., only use GPU 2 and 3)
python parallel_train.py --gpus 2,3
```

### Option 2: Using Bash Script

```bash
# Make executable (first time only)
chmod +x parallel_train.sh

# Run the script
./parallel_train.sh
```

---

## How It Works

### Task Distribution
With 4 GPUs and 12 tasks, the system automatically:
1. **Batch 1**: Runs tasks 1-4 on GPUs 0-3
2. **Batch 2**: Runs tasks 5-8 on GPUs 0-3 (after batch 1 completes)
3. **Batch 3**: Runs tasks 9-12 on GPUs 0-3 (after batch 2 completes)

### GPU Isolation
Each task is isolated to its own GPU using `CUDA_VISIBLE_DEVICES`:
```bash
CUDA_VISIBLE_DEVICES=0  # Task sees only GPU 0
CUDA_VISIBLE_DEVICES=1  # Task sees only GPU 1
# etc.
```

### Task Configuration

Current tasks test Caltech101 with varying noise:

| Task | Noise Type | Noise Rate | GPU (Batch 1) | GPU (Batch 2) | GPU (Batch 3) |
|------|------------|------------|---------------|---------------|---------------|
| 1    | sym        | 0.125      | 0             |               |               |
| 2    | sym        | 0.25       | 1             |               |               |
| 3    | sym        | 0.375      | 2             |               |               |
| 4    | sym        | 0.50       | 3             |               |               |
| 5    | sym        | 0.625      |               | 0             |               |
| 6    | sym        | 0.75       |               | 1             |               |
| 7    | asym       | 0.125      |               | 2             |               |
| 8    | asym       | 0.25       |               | 3             |               |
| 9    | asym       | 0.375      |               |               | 0             |
| 10   | asym       | 0.50       |               |               | 1             |
| 11   | asym       | 0.625      |               |               | 2             |
| 12   | asym       | 0.75       |               |               | 3             |

---

## Customization

### Modify Tasks (Python)

Edit `parallel_train.py`, function `get_default_tasks()` (around line 237):

```python
def get_default_tasks() -> List[TaskConfig]:
    tasks = []
    
    # Example: Different datasets
    datasets_configs = [
        ("caltech101", 100),
        ("oxford_pets", 37),
        ("oxford_flowers", 102),
    ]
    
    for dataset, num_classes in datasets_configs:
        tasks.append(TaskConfig(
            dataset=dataset,
            shots=16,
            noise_rate=0.5,
            noise_type="sym",
            num_classes=num_classes,
            seed=1
        ))
    
    return tasks
```

### Modify Tasks (Bash)

Edit `parallel_train.sh`, TASKS array (around line 11):

```bash
TASKS=(
    "caltech101 16 0.5 sym 100"
    "oxford_pets 16 0.5 sym 37"
    "oxford_flowers 16 0.5 sym 102"
    # Add more tasks...
)
```

### Change GPU Assignment

```bash
# Use different GPUs
python parallel_train.py --gpus 0,1,2,3  # Default
python parallel_train.py --gpus 4,5,6,7  # Use GPUs 4-7
python parallel_train.py --gpus 0,2      # Use only 2 GPUs

# In bash script, edit GPU_IDS array:
GPU_IDS=(0 1 2 3)  # Change to your preferred GPUs
```

---

## Command-Line Options (Python Script)

```bash
python parallel_train.py [OPTIONS]

Options:
  --data-root PATH          Dataset root directory (default: ~/datasets/nlprompt/caltech-101)
  --gpus GPU_LIST           Comma-separated GPU IDs (default: 0,1,2,3)
  --log-dir PATH            Log directory (default: logs)
  --skip-existing           Skip if output exists (default: True)
  --no-skip-existing        Force rerun all tasks
  --check-interval SECONDS  Status check interval (default: 10)
  --custom-tasks            Use custom task configuration
```

### Examples

```bash
# Monitor progress every 30 seconds
python parallel_train.py --check-interval 30

# Use different data location (if not using default ~/datasets/caltech-101)
python parallel_train.py --data-root /data/datasets/my-datasets

# Save logs to different directory
python parallel_train.py --log-dir my_experiment_logs

# Rerun all tasks (ignore existing results)
python parallel_train.py --no-skip-existing
```

---

## Monitoring

### Real-time Monitoring

The scripts provide real-time status updates:
```
================================================================================
[GPU 0] Launching task: TaskConfig(caltech101, shots=16, noise=sym_0.125)
[GPU 0] Output dir: output/caltech101/NLPrompt/rn50_16shots/noise_sym_0.125/seed1
[GPU 0] Log file: logs/gpu0_caltech101_20231228_143022.log
================================================================================

[00:05:30] Still running: 4 tasks on GPUs [0, 1, 2, 3]
✓ [GPU 2] Task completed: TaskConfig(caltech101, shots=16, noise=sym_0.375)
  Time: 01:23:45
```

### Check GPU Usage

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# More detailed monitoring
nvtop  # If installed

# Check specific GPUs
nvidia-smi -i 0,1,2,3
```

### View Logs

Logs are saved in the `logs/` directory:
```bash
# View live log
tail -f logs/gpu0_caltech101_*.log

# Check for errors
cat logs/gpu0_caltech101_*.err

# View all logs
ls -lh logs/
```

---

## Output Structure

Results are organized by task configuration:
```
output/
└── caltech101/
    └── NLPrompt/
        └── rn50_16shots/
            ├── noise_sym_0.125/
            │   └── seed1/
            │       ├── model-best.pth.tar
            │       ├── log.txt
            │       └── ...
            ├── noise_sym_0.25/
            │   └── seed1/
            ├── noise_asym_0.5/
            │   └── seed1/
            └── ...
```

---

## Troubleshooting

### Out of Memory Error

If you get OOM errors on A40 GPUs (48GB), try:
1. Reduce batch size in config file
2. Use fewer context tokens
3. Check if multiple processes are on same GPU

```bash
# Verify GPU isolation
nvidia-smi
# Should show 4 separate processes on 4 different GPUs
```

### Task Fails to Start

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"

# Verify GPU IDs
nvidia-smi -L
```

### Script Doesn't Execute

```bash
# Make scripts executable
chmod +x parallel_train.py parallel_train.sh

# Run with python explicitly
python parallel_train.py
```

### Tasks Run Sequentially Instead of Parallel

- Verify you're using the parallel scripts, not the original `train.py`
- Check that tasks are actually different (not skipped due to existing results)
- Use `--no-skip-existing` flag to force rerun

---

## Performance Tips

### A40 GPU Optimization

1. **Enable TensorFloat-32 (TF32)** for better performance:
   ```python
   # Already enabled in train.py line 147
   torch.backends.cudnn.benchmark = True
   ```

2. **Use AMP (Automatic Mixed Precision)**:
   ```yaml
   # In config file: configs/trainers/NLPrompt/rn50.yaml
   TRAINER:
     NLPROMPT:
       PREC: "amp"  # Use automatic mixed precision
   ```

3. **Maximize Batch Size**:
   With 48GB memory, you can use larger batches:
   ```yaml
   DATALOADER:
     TRAIN_X:
       BATCH_SIZE: 64  # Increase from default
   ```

### Expected Timing

Approximate times on A40 GPUs:
- **Caltech101 (100 epochs)**: ~2-3 hours per task
- **4 tasks in parallel**: ~2-3 hours per batch
- **All 12 tasks**: ~6-9 hours total

---

## Advanced Usage

### Running Multiple Seeds

```python
# In get_default_tasks()
tasks = []
seeds = [1, 2, 3]
for seed in seeds:
    for noise_rate in [0.125, 0.25, 0.375, 0.5]:
        tasks.append(TaskConfig(
            dataset="caltech101",
            noise_rate=noise_rate,
            noise_type="sym",
            num_classes=100,
            seed=seed  # Different seed
        ))
```

### Different Architectures

```python
TaskConfig(
    dataset="caltech101",
    cfg="vit_b16",  # Change architecture
    # ... other params
)
```

---

## Comparison: Original vs Parallel Training

### Original Method (`main.sh`)
```bash
# Sequential - one task at a time
./scripts/nlprompt/main.sh caltech101 16 0.5 sym 100
# Wait for completion...
./scripts/nlprompt/main.sh caltech101 16 0.25 sym 100
# Wait for completion...
# Total time: N × task_time
```

### Parallel Method (New Scripts)
```bash
# Parallel - 4 tasks simultaneously
python parallel_train.py
# All 4 run together
# Total time: (N/4) × task_time
```

**Speedup**: ~4x faster for multiple tasks!

---

## Safety Features

1. **Automatic Skip**: Won't rerun if results exist
2. **GPU Isolation**: Each task sees only its assigned GPU
3. **Error Logging**: Separate error logs per task
4. **Graceful Failure**: One task failing doesn't stop others
5. **Progress Tracking**: Real-time status updates

---

## Quick Reference

```bash
# Start parallel training (default config)
python parallel_train.py

# Use custom GPUs
python parallel_train.py --gpus 0,1,2,3

# Force rerun everything
python parallel_train.py --no-skip-existing

# Monitor GPUs
watch -n 1 nvidia-smi

# View logs
tail -f logs/gpu0_*.log

# Check output
ls -R output/
```

---

## Contact & Support

For issues or questions:
1. Check logs in `logs/` directory
2. Verify GPU availability with `nvidia-smi`
3. Review task configuration in the script
4. Check original `train.py` works: `python train.py --help`

