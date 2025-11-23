#!/usr/bin/env bash

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --account=envcomp
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/flash_run_%j.log
#SBATCH --error=logs/flash_run_%j.log
#SBATCH --job-name=flash_run
#SBATCH --mail-user=yll.kryeziu@tum.de
#SBATCH --mail-type=ALL

# Set cache directories (optional)
# export HF_HOME=$PROJECT/yll/.cache/huggingface
# export FLASHINFER_CACHE_DIR=$PROJECT/yll/.cache/flashinfer
export HF_HUB_OFFLINE=1

echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
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

# Force offline mode for datasets to avoid connection errors
export HF_DATASETS_OFFLINE=1

echo "CC: $CC"
echo "CXX: $CXX"

# Change to project directory
cd /p/project1/envcomp/yll/pThought

# Activate environment
source .venv/bin/activate

# 1. Run Math Generation
echo "Starting Math Generation..."
skythought evaluate \
    --task math_prm800k \
    --model NovaSky-AI/Sky-T1-32B-Preview \
    --backend vllm \
    --backend-args tensor_parallel_size=4 \
    --sampling-params max_tokens=4096,temperature=1.0 \
    --n 8 \
    --result-dir ./results/math \
    --batch-size 16 \
    --overwrite

# 2. Run Taco Generation
echo "Starting TACO Generation..."
skythought evaluate \
    --task taco \
    --model NovaSky-AI/Sky-T1-32B-Preview \
    --backend vllm \
    --backend-args tensor_parallel_size=4 \
    --sampling-params max_tokens=4096,temperature=1.0 \
    --n 8 \
    --result-dir ./results/taco \
    --batch-size 16 \
    --overwrite

# 3. Merge Results
echo "Merging results..."
python ../scripts/merge_results.py \
    --math-dir ./results/math \
    --taco-dir ./results/taco \
    --output ./results/merged_results.json

echo "Run completed! Check output above for errors."
