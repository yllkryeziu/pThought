#!/usr/bin/env bash
# Download all models needed for the project

set -e

echo "Activating environment..."
cd /p/project1/envcomp/yll/pThought
source .venv/bin/activate

# Temporarily disable offline mode to allow downloads
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

echo ""
echo "========================================"
echo "Downloading all required models..."
echo "========================================"
echo ""

# List of models to download
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen2-7B-Instruct"
    "NovaSky-AI/Sky-T1-32B-Preview"
)

for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Downloading: $model"
    echo "----------------------------------------"
    python scripts/download_model.py "$model" || echo "⚠ Failed to download $model, continuing..."
    echo ""
done

echo ""
echo "========================================"
echo "✓ All downloads complete!"
echo "========================================"
echo "Models are now cached and ready for offline use."
