#!/usr/bin/env python3
"""
Parallel Training Script for NLPrompt
Trains 4 different tasks simultaneously on 4 different GPUs (one task per GPU)
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple


class TaskConfig:
    """Configuration for a single training task"""
    def __init__(self, dataset: str, shots: int, noise_rate: float, 
                 noise_type: str, num_classes: int, seed: int = 1,
                 cfg: str = "rn50", trainer: str = "NLPrompt"):
        self.dataset = dataset
        self.shots = shots
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.num_classes = num_classes
        self.seed = seed
        self.cfg = cfg
        self.trainer = trainer
    
    def get_output_dir(self, base_output: str = "output") -> str:
        """Generate output directory path for this task"""
        # Format with minimum 2 decimals, but keep more if needed
        rate_str = f"{self.noise_rate:.3f}".rstrip('0').rstrip('.')
        if '.' in rate_str and len(rate_str.split('.')[1]) < 2:
            rate_str = f"{self.noise_rate:.2f}"
        return f"{base_output}/{self.dataset}/{self.trainer}/{self.cfg}_{self.shots}shots/noise_{self.noise_type}_{rate_str}/seed{self.seed}"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'dataset': self.dataset,
            'shots': self.shots,
            'noise_rate': self.noise_rate,
            'noise_type': self.noise_type,
            'num_classes': self.num_classes,
            'seed': self.seed,
            'cfg': self.cfg,
            'trainer': self.trainer
        }
    
    def __repr__(self) -> str:
        # Format with minimum 2 decimals, but keep more if needed
        rate_str = f"{self.noise_rate:.3f}".rstrip('0').rstrip('.')
        if '.' in rate_str and len(rate_str.split('.')[1]) < 2:
            rate_str = f"{self.noise_rate:.2f}"
        return f"TaskConfig({self.dataset}, shots={self.shots}, noise={self.noise_type}_{rate_str})"


def build_training_command(task: TaskConfig, data_root: str, gpu_id: int, base_output: str = "output") -> Tuple[str, Dict]:
    """
    Build the training command and environment variables for a task
    
    Args:
        task: TaskConfig object with training parameters
        data_root: Root directory for datasets
        gpu_id: GPU ID to use for this task
        base_output: Base output directory name
    
    Returns:
        Tuple of (command_string, environment_dict)
    """
    output_dir = task.get_output_dir(base_output=base_output)
    
    # Build the command
    command = [
        "python", "train.py",
        "--root", data_root,
        "--seed", str(task.seed),
        "--trainer", task.trainer,
        "--dataset-config-file", f"configs/datasets/{task.dataset}.yaml",
        "--config-file", f"configs/trainers/{task.trainer}/{task.cfg}.yaml",
        "--output-dir", output_dir,
        "DATASET.NUM_SHOTS", str(task.shots),
        "DATASET.NOISE_RATE", str(task.noise_rate),
        "DATASET.NOISE_TYPE", task.noise_type,
        "DATASET.num_class", str(task.num_classes)
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Limit CPU threads to prevent oversubscription (64 CPUs / 4 GPUs = 16 threads per process)
    env['OMP_NUM_THREADS'] = '16'
    env['MKL_NUM_THREADS'] = '16'
    env['OPENBLAS_NUM_THREADS'] = '16'
    env['VECLIB_MAXIMUM_THREADS'] = '16'
    env['NUMEXPR_NUM_THREADS'] = '16'
    
    return command, env


def check_output_exists(task: TaskConfig, base_output: str = "output") -> bool:
    """Check if results already exist for this task and training is complete"""
    output_dir = task.get_output_dir(base_output=base_output)
    if not os.path.exists(output_dir):
        return False
    
    # Check if training completed by looking for "Finish training" in log
    log_file = os.path.join(output_dir, "log.txt")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                # Training is complete if we see "Finish training" message
                if "Finish training" in content:
                    return True
                # Also check if we reached the final epoch (200/200)
                if "epoch [200/200]" in content:
                    return True
        except Exception:
            pass
    
    # If directory exists but no completion indicator, consider it incomplete
    return False


def run_task(task: TaskConfig, data_root: str, gpu_id: int, 
             skip_existing: bool = True, log_dir: str = "logs", base_output: str = "output") -> subprocess.Popen:
    """
    Launch a training task on a specific GPU
    
    Args:
        task: TaskConfig object
        data_root: Root directory for datasets
        gpu_id: GPU ID to assign
        skip_existing: Skip if output directory already exists
        log_dir: Directory to save logs
        base_output: Base output directory name
    
    Returns:
        subprocess.Popen object for the running process
    """
    output_dir = task.get_output_dir(base_output=base_output)
    
    if skip_existing and check_output_exists(task, base_output=base_output):
        print(f"[GPU {gpu_id}] Results exist in {output_dir}. Skipping task: {task}")
        return None
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/gpu{gpu_id}_{task.dataset}_{timestamp}.log"
    err_file = f"{log_dir}/gpu{gpu_id}_{task.dataset}_{timestamp}.err"
    
    command, env = build_training_command(task, data_root, gpu_id, base_output=base_output)
    
    print(f"\n{'='*80}")
    print(f"[GPU {gpu_id}] Launching task: {task}")
    print(f"[GPU {gpu_id}] Output dir: {output_dir}")
    print(f"[GPU {gpu_id}] Log file: {log_file}")
    print(f"[GPU {gpu_id}] Command: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    # Open log files
    log_f = open(log_file, 'w')
    err_f = open(err_file, 'w')
    
    # Launch the process
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_f,
        stderr=err_f,
        cwd=os.getcwd()
    )
    
    # Store file handles with the process so we can close them later
    process.log_files = (log_f, err_f)
    process.task = task
    process.gpu_id = gpu_id
    
    return process


def get_default_tasks(trainer: str = "NLPrompt") -> List[TaskConfig]:
    """
    Define default tasks for parallel training
    
    Args:
        trainer: Name of the trainer to use (e.g., "NLPrompt" or "nlprompt_wo_mae")
    
    Modify this function to customize your training tasks
    """
    # Define datasets with their class counts
    # Note: Datasets are sorted by training time (fastest first) based on elapsed times from output/
    #       stanford_cars is placed last and will use GPUs 0,1,3 by default
    #       Run check_dataset_times.py to update this order based on actual training times
    datasets = [
        ("eurosat", 10),           # ~4m  9s (fastest)
        ("oxford_pets", 37),       # ~33m 31s
        ("dtd", 47),               # ~47m 14s
        ("ucf101", 101),           # ~59m  1s
        ("caltech101", 100),       # ~61m 35s
        ("oxford_flowers", 102),   # ~129m 16s
        ("stanford_cars", 196)     # ~304m 44s (slowest, uses GPUs 0,1,3)
    ]
    
    # Test different noise rates with sym and asym noise
    noise_rates = [0.125, 0.25, 0.375, 0.50, 0.625, 0.75]
    noise_types = ["sym", "asym"]
    
    tasks = []
    for dataset, num_classes in datasets:
        for noise_type in noise_types:
            for noise_rate in noise_rates:
                tasks.append(TaskConfig(
                        dataset=dataset,
                    shots=16,
                    noise_rate=noise_rate,
                    noise_type=noise_type,
                        num_classes=num_classes,
                    seed=1,
                    trainer=trainer
                ))
    
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Parallel training script for NLPrompt - trains multiple tasks on multiple GPUs"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="~/datasets/nlprompt",
        help="Root directory for datasets (dataset name is added automatically)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save training logs"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip tasks where output directory already exists"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Run all tasks even if output exists"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=10,
        help="Interval (seconds) to check process status"
    )
    parser.add_argument(
        "--custom-tasks",
        action="store_true",
        help="Use custom task configuration (modify get_default_tasks() function)"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="output_wo_mae",
        help="Base output directory name (default: output_wo_mae)"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="NLPrompt",
        help="Name of trainer to use (e.g., 'NLPrompt' or 'nlprompt_wo_mae')"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Get tasks with specified trainer
    tasks = get_default_tasks(trainer=args.trainer)
    
    print("\n" + "="*80)
    print("NLPrompt Parallel Training")
    print("="*80)
    print(f"Data root: {args.data_root}")
    print(f"Available GPUs: {gpu_ids}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"Trainer: {args.trainer}")
    print(f"Output base: {args.output_base}")
    print(f"Log directory: {args.log_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print("="*80 + "\n")
    
    # Check task and GPU counts
    num_tasks = len(tasks)
    num_gpus = len(gpu_ids)
    
    print(f"Total tasks: {num_tasks}")
    print(f"Available GPUs: {num_gpus}")
    
    if num_tasks > num_gpus:
        num_batches = (num_tasks + num_gpus - 1) // num_gpus
        print(f"Will run in {num_batches} batches")
    else:
        num_batches = 1
    print()
    
    # Display all tasks
    print("Tasks to run:")
    for i, task in enumerate(tasks):
        print(f"  Task {i+1}: {task}")
    print()
    
    # Run tasks in batches
    failed_count = 0
    task_idx = 0
    
    # Default GPU list for stanford_cars: [0, 1, 3]
    stanford_cars_gpus = [0, 1, 3]
    
    while task_idx < num_tasks:
        # Check if current batch contains stanford_cars tasks
        current_task = tasks[task_idx]
        is_stanford_cars = current_task.dataset == "stanford_cars"
        
        # Use different GPU list for stanford_cars
        if is_stanford_cars:
            active_gpu_ids = stanford_cars_gpus
            num_active_gpus = len(active_gpu_ids)
            print("=" * 80)
            print("Stanford Cars tasks detected - using GPUs:", active_gpu_ids)
            print("=" * 80)
        else:
            active_gpu_ids = gpu_ids
            num_active_gpus = num_gpus
        
        # Determine batch size
        remaining = num_tasks - task_idx
        batch_size = min(remaining, num_active_gpus)
        
        print("=" * 80)
        print(f"Launching batch: tasks {task_idx+1} to {task_idx+batch_size}")
        print("=" * 80)
        print()
        
        # Launch tasks in current batch
        processes = []
        for batch_idx in range(batch_size):
            i = task_idx + batch_idx
            gpu_id = active_gpu_ids[batch_idx]
            task = tasks[i]
            
            print(f"  GPU {gpu_id}: Task {i+1} - {task}")
            
            process = run_task(
                task=task,
                data_root=args.data_root,
                gpu_id=gpu_id,
                skip_existing=args.skip_existing,
                log_dir=args.log_dir,
                base_output=args.output_base
            )
            processes.append(process)
            time.sleep(2)  # Small delay between launches
        
        print()
        print("Batch launched. Waiting for completion...")
        print()
        
        # Monitor current batch
        active_processes = [p for p in processes if p is not None]
        
        if active_processes:
            batch_start = time.time()
            
            while active_processes:
                time.sleep(args.check_interval)
                
                for process in active_processes[:]:
                    poll_result = process.poll()
                    
                    if poll_result is not None:
                        elapsed = time.time() - batch_start
                        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                        
                        if poll_result == 0:
                            print(f"✓ [GPU {process.gpu_id}] Task completed: {process.task}")
                            print(f"  Time: {elapsed_str}")
                        else:
                            print(f"✗ [GPU {process.gpu_id}] Task failed (code {poll_result}): {process.task}")
                            print(f"  Check error log: {process.log_files[1].name}")
                            failed_count += 1
                        
                        process.log_files[0].close()
                        process.log_files[1].close()
                        active_processes.remove(process)
                
                if active_processes:
                    elapsed = time.time() - batch_start
                    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    print(f"[{elapsed_str}] Still running: {len(active_processes)} tasks")
        
        task_idx += batch_size
        
        if task_idx < num_tasks:
            print()
            print("Batch completed. Moving to next batch...")
            print()
    
    print()
    print("=" * 80)
    print("All tasks completed!")
    if failed_count == 0:
        print("Status: All tasks succeeded ✓")
    else:
        print(f"Status: {failed_count} task(s) failed ✗")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

