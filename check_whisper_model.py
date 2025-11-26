#!/usr/bin/env python3
"""
Check if Whisper model is available locally.
"""

import os
import sys
from pathlib import Path

try:
    import whisper
except ImportError:
    print("[ERROR] openai-whisper is not installed")
    print("Install it with: pip install openai-whisper")
    sys.exit(1)

def check_model(model_size: str = "large-v3"):
    """Check if Whisper model is available."""
    print(f"Checking for Whisper model: {model_size}")
    
    # Get cache directory
    cache_dir = Path.home() / ".cache" / "whisper"
    model_file = cache_dir / f"{model_size}.pt"
    
    print(f"Cache directory: {cache_dir}")
    print(f"Model file: {model_file}")
    
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Model found! Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"[NOT FOUND] Model not found locally")
        print(f"  The model will be downloaded automatically on first use")
        print(f"  Make sure you have internet connection")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check Whisper model availability")
    parser.add_argument("--model", default="large-v3", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v3"],
                       help="Model size to check")
    args = parser.parse_args()
    
    check_model(args.model)

