#!/bin/bash
# Setup script for Audio Transcriber with Speaker Diarization

set -e

echo "=================================================="
echo "Audio Transcriber Setup"
echo "=================================================="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo

# Check if FFmpeg is installed
echo "Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version 2>&1 | head -n1)
    echo "✓ FFmpeg is installed: $ffmpeg_version"
else
    echo "✗ FFmpeg is NOT installed"
    echo
    echo "Please install FFmpeg:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
    echo
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo

# Install requirements
echo "Installing Python dependencies..."
echo "This may take several minutes..."
echo
pip install -r requirements.txt

echo
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo
echo "To use the transcriber:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: python main_pipeline.py your_audio.wav --lang ru --speakers 4"
echo
echo "For more information, see README.md"
echo
