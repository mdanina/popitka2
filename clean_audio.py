#!/usr/bin/env python3
"""
Audio cleaning script using ffmpeg.
Removes silence, converts to mono, and normalizes sample rate.
"""

import sys
import os
import subprocess
from pathlib import Path


def clean_audio(input_path: str, output_path: str = None) -> str:
    """
    Clean audio file: convert to mono, 16kHz sample rate, remove silence.

    Args:
        input_path: Path to input audio file
        output_path: Path to output file (optional, auto-generated if None)

    Returns:
        Path to cleaned audio file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.wav"
    else:
        output_path = Path(output_path)

    print(f"Cleaning audio: {input_path}")
    print(f"Output will be saved to: {output_path}")

    # FFmpeg command for audio cleaning
    cmd = [
        "ffmpeg",
        "-i", str(input_path),           # Input file
        "-ac", "1",                       # Convert to mono (1 channel)
        "-ar", "16000",                   # Sample rate 16 kHz
        "-af",                            # Audio filters
        "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-35dB:detection=peak",
        "-y",                             # Overwrite output file
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Audio cleaned successfully: {output_path}")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        print(f"Error cleaning audio: {e}")
        print(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  MacOS: brew install ffmpeg")
        raise


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python clean_audio.py <input_audio_file> [output_file]")
        print("Example: python clean_audio.py meeting.mp3")
        print("         python clean_audio.py meeting.mp3 meeting_cleaned.wav")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        cleaned_file = clean_audio(input_file, output_file)
        print(f"\n✓ Done! Cleaned audio saved to: {cleaned_file}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
