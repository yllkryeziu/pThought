#!/usr/bin/env python3
"""
Quick script to download a model to HuggingFace cache
Usage: python scripts/download_model.py <model_name>
Example: python scripts/download_model.py Qwen/Qwen3-0.6B
"""

import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_name: str):
    print(f"Downloading {model_name}...")

    # Download full model snapshot (includes all files)
    print("Downloading model files...")
    snapshot_download(
        repo_id=model_name,
        local_files_only=False,
        resume_download=True,
    )

    # Also download tokenizer and config to ensure everything is cached
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"✓ Successfully downloaded {model_name}")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_model.py <model_name>")
        print("Example: python scripts/download_model.py Qwen/Qwen3-0.6B")
        sys.exit(1)

    model_name = sys.argv[1]
    download_model(model_name)
