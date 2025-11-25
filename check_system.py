#!/usr/bin/env python3
"""
System requirements checker for Audio Transcriber.
Checks Python, CUDA, FFmpeg, and available memory.
"""

import sys
import subprocess
import platform


def check_python_version():
    """Check Python version."""
    print("=" * 70)
    print("PYTHON VERSION")
    print("=" * 70)

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"Python version: {version_str}")

    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is sufficient (3.8+)")
        return True
    else:
        print("✗ Python version is too old (need 3.8+)")
        return False


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    print("\n" + "=" * 70)
    print("FFMPEG")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"FFmpeg: {version_line}")
        print("✓ FFmpeg is installed")
        return True

    except FileNotFoundError:
        print("✗ FFmpeg is NOT installed")
        print("\nInstallation instructions:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return False

    except Exception as e:
        print(f"✗ Error checking FFmpeg: {e}")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "=" * 70)
    print("CUDA / GPU")
    print("=" * 70)

    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)

            print(f"CUDA version: {cuda_version}")
            print(f"GPU devices: {device_count}")
            print(f"Primary GPU: {device_name}")

            # Check VRAM
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            print(f"GPU memory: {total_memory:.1f} GB")

            print("✓ CUDA is available")

            # Recommendations
            if total_memory < 4:
                print("\n⚠ Warning: Low GPU memory. Recommended models: tiny, base, small")
            elif total_memory < 6:
                print("\n→ Recommended models: small, medium")
            else:
                print("\n→ Recommended models: medium, large, large-v3")

            return True

        else:
            print("✗ CUDA is NOT available")
            print("→ Transcription will run on CPU (slower)")
            print("→ Recommended models: tiny, base, small")
            return False

    except ImportError:
        print("✗ PyTorch is not installed")
        print("Run: pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_memory():
    """Check available system memory."""
    print("\n" + "=" * 70)
    print("SYSTEM MEMORY")
    print("=" * 70)

    try:
        import psutil

        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        percent_used = memory.percent

        print(f"Total RAM: {total_gb:.1f} GB")
        print(f"Available RAM: {available_gb:.1f} GB")
        print(f"Used: {percent_used:.1f}%")

        if total_gb < 8:
            print("⚠ Warning: Low system memory (< 8 GB)")
            print("→ Recommended models: tiny, base")
            return False
        elif total_gb < 16:
            print("✓ Sufficient memory for most models")
            print("→ Recommended models: small, medium")
            return True
        else:
            print("✓ Excellent memory capacity")
            print("→ All models supported")
            return True

    except ImportError:
        print("⚠ psutil not installed (optional)")
        return None

    except Exception as e:
        print(f"✗ Error checking memory: {e}")
        return None


def check_disk_space():
    """Check available disk space."""
    print("\n" + "=" * 70)
    print("DISK SPACE")
    print("=" * 70)

    try:
        import psutil
        import os

        disk = psutil.disk_usage(os.getcwd())
        total_gb = disk.total / (1024**3)
        free_gb = disk.free / (1024**3)
        percent_used = disk.percent

        print(f"Total disk: {total_gb:.1f} GB")
        print(f"Free disk: {free_gb:.1f} GB")
        print(f"Used: {percent_used:.1f}%")

        if free_gb < 5:
            print("⚠ Warning: Low disk space (< 5 GB)")
            print("→ Models and cache require ~3-5 GB")
            return False
        else:
            print("✓ Sufficient disk space")
            return True

    except ImportError:
        print("⚠ psutil not installed (optional)")
        return None

    except Exception as e:
        print(f"✗ Error checking disk space: {e}")
        return None


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n" + "=" * 70)
    print("PYTHON DEPENDENCIES")
    print("=" * 70)

    required_packages = [
        "torch",
        "whisper",
        "nemo_toolkit",
        "librosa",
        "soundfile",
        "sklearn",
        "numpy"
    ]

    all_installed = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (not installed)")
            all_installed = False

    if not all_installed:
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")

    return all_installed


def print_system_info():
    """Print general system information."""
    print("\n" + "=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)

    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")


def main():
    """Run all checks."""
    print("\n" + "#" * 70)
    print("# AUDIO TRANSCRIBER - SYSTEM REQUIREMENTS CHECK")
    print("#" * 70 + "\n")

    results = {
        "python": check_python_version(),
        "ffmpeg": check_ffmpeg(),
        "cuda": check_cuda(),
        "memory": check_memory(),
        "disk": check_disk_space(),
        "dependencies": check_dependencies()
    }

    print_system_info()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")

    if results["python"] and results["ffmpeg"] and results["dependencies"]:
        print("\n✓ System is ready for audio transcription!")

        if not results["cuda"]:
            print("\nNote: Running on CPU (no CUDA). Processing will be slower.")
            print("Consider using smaller models: tiny, base, or small")

    else:
        print("\n✗ System is NOT ready. Please install missing requirements.")

    print("\n" + "#" * 70 + "\n")


if __name__ == "__main__":
    main()
