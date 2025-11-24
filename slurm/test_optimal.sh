#!/usr/bin/env bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --account=envcomp
#SBATCH --output=logs/test_optimal_%j.log
#SBATCH --job-name=test_optimal

export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

module purge
module load cuDNN/9.5.0.50-CUDA-12
module load GCC/13.3.0

cd /p/project1/envcomp/yll/pThought
source .venv/bin/activate

echo "Testing OPTIMAL config: batch_size=8, n=8, 4xA100..."
skythought evaluate \
    --task math500 \
    --model Qwen/Qwen3-0.6B \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.90 \
    --sampling-params max_tokens=4096,temperature=1.0 \
    --n 8 \
    --result-dir ./results/math_test_optimal \
    --batch-size 8 \
    --overwrite

echo "Test completed! Check logs above."
