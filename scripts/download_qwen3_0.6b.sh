#!/usr/bin/env bash
# Quick script to download Qwen/Qwen3-0.6B to cache

set -e

echo "Activating environment..."
cd /p/project1/envcomp/yll/pThought
source .venv/bin/activate

# Temporarily disable offline mode to allow downloads
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

echo "Downloading Qwen/Qwen3-0.6B..."
python scripts/download_model.py Qwen/Qwen3-0.6B

echo ""
echo "✓ Download complete!"
echo "Model is now cached and ready for offline use."
