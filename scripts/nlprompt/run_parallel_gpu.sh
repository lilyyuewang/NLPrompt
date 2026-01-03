#!/bin/bash

# Parallel GPU Job Runner
# Runs training jobs across 4 GPUs, automatically starting next job when a GPU becomes free
# Usage: ./run_parallel_gpu.sh <dataset> <shots> <class> [noise_rates] [noise_types]
#
# Example: ./run_parallel_gpu.sh caltech101 16 100 "0.1 0.2 0.3 0.4 0.5" "sym asym"

set -e

# Configuration
DATA=~/datasets/nlprompt
TRAINER=NLPrompt
CFG=rn50
NUM_GPUS=4

# Parse arguments
DATASET=${1:-caltech101}
SHOTS=${2:-16}
CLASS=${3:-100}
NOISE_RATES=${4:-"0.1 0.2 0.3 0.4 0.5"}
NOISE_TYPES=${5:-"sym asym"}

echo "=========================================="
echo "Parallel GPU Job Runner"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Shots: $SHOTS"
echo "Class: $CLASS"
echo "Noise Rates: $NOISE_RATES"
echo "Noise Types: $NOISE_TYPES"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Create job queue: all combinations of noise_rate and noise_type
JOBS=()
for RATE in $NOISE_RATES; do
    for TYPE in $NOISE_TYPES; do
        JOBS+=("$RATE $TYPE")
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Total jobs: $TOTAL_JOBS"
echo ""

# Function to check if a GPU is free
is_gpu_free() {
    local gpu_id=$1
    # Check if any process is using this GPU
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk -v gpu="$gpu_id" -F', ' '$1==gpu && $2<100 {exit 0} {exit 1}'
}

# Function to wait for a free GPU
wait_for_free_gpu() {
    while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            if is_gpu_free $gpu; then
                echo $gpu
                return
            fi
        done
        sleep 5
    done
}

# Function to run a job on a specific GPU
run_job_on_gpu() {
    local gpu_id=$1
    local rate=$2
    local type=$3
    local job_num=$4
    
    local DIR="output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/noise_${type}_${rate}/seed1"
    
    if [ -d "$DIR" ]; then
        echo "[GPU $gpu_id] Job $job_num: Results exist in ${DIR}. Skipping."
        return 0
    fi
    
    echo "[GPU $gpu_id] Job $job_num: Starting (rate=$rate, type=$type)"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        --root ${DATA} \
        --seed 1 \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.NOISE_RATE ${rate} \
        DATASET.NOISE_TYPE ${type} \
        DATASET.num_class ${CLASS} \
        > "logs/gpu${gpu_id}_job${job_num}.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu_id] Job $job_num: Completed successfully (rate=$rate, type=$type)"
    else
        echo "[GPU $gpu_id] Job $job_num: Failed with exit code $exit_code (rate=$rate, type=$type)"
    fi
    return $exit_code
}

# Create logs directory
mkdir -p logs

# Track running jobs: array of PIDs indexed by GPU
declare -a RUNNING_PIDS
declare -a JOB_INDICES
for ((i=0; i<NUM_GPUS; i++)); do
    RUNNING_PIDS[$i]=0
    JOB_INDICES[$i]=-1
done

JOB_INDEX=0
COMPLETED=0

# Main loop: process all jobs
while [ $JOB_INDEX -lt $TOTAL_JOBS ] || [ $COMPLETED -lt $TOTAL_JOBS ]; do
    # Check for completed jobs and free GPUs
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        if [ ${RUNNING_PIDS[$gpu]} -ne 0 ]; then
            # Check if process is still running
            if ! kill -0 ${RUNNING_PIDS[$gpu]} 2>/dev/null; then
                # Process finished
                wait ${RUNNING_PIDS[$gpu]} 2>/dev/null
                exit_code=$?
                job_idx=${JOB_INDICES[$gpu]}
                IFS=' ' read -r rate type <<< "${JOBS[$job_idx]}"
                if [ $exit_code -eq 0 ]; then
                    echo "[GPU $gpu] Job $((job_idx+1))/$TOTAL_JOBS completed (rate=$rate, type=$type)"
                else
                    echo "[GPU $gpu] Job $((job_idx+1))/$TOTAL_JOBS failed (rate=$rate, type=$type, exit_code=$exit_code)"
                fi
                RUNNING_PIDS[$gpu]=0
                JOB_INDICES[$gpu]=-1
                ((COMPLETED++))
            fi
        fi
        
        # Start new job on free GPU
        if [ ${RUNNING_PIDS[$gpu]} -eq 0 ] && [ $JOB_INDEX -lt $TOTAL_JOBS ]; then
            IFS=' ' read -r rate type <<< "${JOBS[$JOB_INDEX]}"
            run_job_on_gpu $gpu $rate $type $((JOB_INDEX+1)) &
            RUNNING_PIDS[$gpu]=$!
            JOB_INDICES[$gpu]=$JOB_INDEX
            ((JOB_INDEX++))
        fi
    done
    
    # Wait a bit before checking again
    sleep 2
done

echo ""
echo "=========================================="
echo "All jobs completed!"
echo "=========================================="


