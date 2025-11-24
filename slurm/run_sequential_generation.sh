#!/usr/bin/env bash

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --account=envcomp
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/sequential_run_%j.log
#SBATCH --error=logs/sequential_run_%j.log
#SBATCH --job-name=sequential_run
#SBATCH --mail-user=kslurm@gmail.com
#SBATCH --mail-type=ALL

# Set environment variables
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=50
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1

echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "HF_HOME: $HF_HOME"

# Load modules
module purge
module load cuDNN/9.5.0.50-CUDA-12
module load GCC/13.3.0

echo "Loaded modules:"
module list

export CC=$(which gcc)
export CXX=$(which g++)

echo "CC: $CC"
echo "CXX: $CXX"

# Change to project directory
cd /p/project1/envcomp/yll/pThought

# Activate environment
source .venv/bin/activate

# WORKAROUND: Generate 8 samples by running 8 times with n=1
# This avoids the VLLM n>1 scheduler deadlock issue
echo "========================================"
echo "Starting Math Generation (Sequential)"
echo "========================================"

for i in {1..8}; do
    echo "--------------------------------"
    echo "Generating sample ${i}/8..."
    echo "--------------------------------"

    # For the first run, overwrite. For subsequent runs, resume from previous results
    if [ $i -eq 1 ]; then
        skythought generate \
            --task math_prm800k \
            --model Qwen/Qwen2-7B-Instruct \
            --backend vllm \
            --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.95 \
            --sampling-params max_tokens=4096,temperature=0.7 \
            --n 1 \
            --result-dir ./results/math_sequential \
            --batch-size 16 \
            --overwrite
    else
        skythought generate \
            --task math_prm800k \
            --model Qwen/Qwen2-7B-Instruct \
            --backend vllm \
            --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.95 \
            --sampling-params max_tokens=4096,temperature=0.7 \
            --n 1 \
            --result-dir ./results/math_sequential \
            --batch-size 16 \
            --resume-from ./results/math_sequential
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Sample ${i} failed!"
        exit 1
    fi

    echo "Sample ${i}/8 completed successfully"
done

echo "========================================"
echo "All 8 samples generated successfully!"
echo "Now scoring results..."
echo "========================================"

# Score the results
skythought score \
    --run-dir ./results/math_sequential \
    --task math_prm800k

echo "Run completed! Check ./results/math_sequential for outputs."
