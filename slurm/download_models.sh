#!/usr/bin/env bash

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=envcomp
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/download_models_%j.log
#SBATCH --error=logs/download_models_%j.log
#SBATCH --job-name=download_models
#SBATCH --mail-user=kslurm@gmail.com
#SBATCH --mail-type=ALL

# IMPORTANT: This script needs internet access
# Remove offline mode settings to allow downloads
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

echo "Starting model downloads..."
echo "HF_HOME: $HF_HOME"

# Load modules
module purge
module load GCC/13.3.0

cd /p/project1/envcomp/yll/pThought
source .venv/bin/activate

# Download all models
echo ""
echo "========================================"
echo "Downloading Models"
echo "========================================"

# Download small model first for testing
echo "1. Downloading Qwen/Qwen3-0.6B (small model)..."
python scripts/download_model.py Qwen/Qwen3-0.6B

echo ""
echo "2. Downloading Qwen/Qwen2-7B-Instruct..."
python scripts/download_model.py Qwen/Qwen2-7B-Instruct

echo ""
echo "3. Downloading NovaSky-AI/Sky-T1-32B-Preview (large model)..."
python scripts/download_model.py NovaSky-AI/Sky-T1-32B-Preview

echo ""
echo "========================================"
echo "✓ All models downloaded successfully!"
echo "========================================"
echo ""
echo "You can now run inference jobs in offline mode."
