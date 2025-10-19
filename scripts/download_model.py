#!/usr/bin/env python3
"""
Pre-download Whisper model for Docker image build.

This script downloads the faster-whisper model during Docker build time
to avoid runtime downloads and reduce cold start time on Cloud Run.
"""

import sys
from faster_whisper import WhisperModel

# Model configuration (matches app/config.py defaults)
MODEL_NAME = "small"
DEVICE = "cpu"  # Use CPU during build, GPU will be used at runtime
COMPUTE_TYPE = "int8"

def download_model():
    """Download and cache the Whisper model."""
    try:
        print(f"Downloading faster-whisper model: {MODEL_NAME}")
        print(f"Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")

        # Initialize model - this triggers download and caching
        model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )

        print(f"✓ Model '{MODEL_NAME}' downloaded and cached successfully")
        print(f"  Cache location: ~/.cache/huggingface/")

        return 0

    except Exception as e:
        print(f"✗ Failed to download model: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(download_model())
