#!/usr/bin/env bash

#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --account=envcomp
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/flash_test_%j.log
#SBATCH --error=logs/flash_test_%j.log
#SBATCH --job-name=flash_test

# Set cache directories
export HF_HOME=$PROJECT/yll/.cache/huggingface
export FLASHINFER_CACHE_DIR=$PROJECT/yll/.cache/flashinfer

echo "HF_HOME: $HF_HOME"
echo "FLASHINFER_CACHE_DIR: $FLASHINFER_CACHE_DIR"

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

# Test run with just 10 samples to verify everything works
skythought evaluate \
    --task math_prm800k \
    --model NovaSky-AI/Sky-T1-32B-Preview \
    --backend vllm \
    --backend-args tensor_parallel_size=4 \
    --sampling-params max_tokens=4096,temperature=1.0 \
    --n 8 \
    --result-dir ./results/math \
    --batch-size 32 \
    --overwrite \
    --as-test

echo "Test completed! Check output above for errors."