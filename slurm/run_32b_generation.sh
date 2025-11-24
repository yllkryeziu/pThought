#!/usr/bin/env bash

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --account=envcomp
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/32b_run_%j.log
#SBATCH --error=logs/32b_run_%j.log
#SBATCH --job-name=32b_run
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

# 1. Run Math Generation with 32B model
echo "--------------------------------"
echo "--------------------------------"
echo "Starting Math Generation (32B)..."
echo "--------------------------------"
echo "--------------------------------"
skythought evaluate \
    --task math_prm800k \
    --model NovaSky-AI/Sky-T1-32B-Preview \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.90 \
    --sampling-params max_tokens=4096,temperature=1.0 \
    --n 8 \
    --result-dir ./results/math_32b \
    --batch-size 2 \
    --overwrite

# 2. Run Taco Generation with 32B model
echo "--------------------------------"
echo "--------------------------------"
echo "Starting TACO Generation (32B)..."
echo "--------------------------------"
echo "--------------------------------"
skythought evaluate \
    --task taco \
    --model NovaSky-AI/Sky-T1-32B-Preview \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.90 \
    --sampling-params max_tokens=4096,temperature=1.0 \
    --n 8 \
    --result-dir ./results/taco_32b \
    --batch-size 2 \
    --overwrite

# 3. Merge Results
echo "Merging results..."
python scripts/merge_results.py \
    --math-dir ./results/math_32b \
    --taco-dir ./results/taco_32b \
    --output ./results/merged_results_32b.json

echo "Run completed! Check output above for errors."
