#!/usr/bin/env bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --account=envcomp
#SBATCH --output=logs/test_n1_%j.log
#SBATCH --job-name=test_n1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

module purge
module load cuDNN/9.5.0.50-CUDA-12
module load GCC/13.3.0

cd /p/project1/envcomp/yll/pThought
source .venv/bin/activate

echo "Testing with n=1, batch_size=4..."
skythought evaluate \
    --task math500 \
    --model Qwen/Qwen2-7B-Instruct \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.90 \
    --sampling-params max_tokens=4096,temperature=0.7 \
    --n 1 \
    --result-dir ./results/math_test_n1 \
    --batch-size 4 \
    --overwrite

echo "Test completed! Check logs above."
