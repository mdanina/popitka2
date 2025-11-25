#!/usr/bin/env python3
"""
Main pipeline for audio transcription with speaker diarization.
Combines audio cleaning, Whisper transcription, NeMo diarization, and output formatting.
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
# Get the Python executable that's running this script
PYTHON_EXE = sys.executable


def run_command(cmd: list, description: str):
    """
    Run a command and handle errors.

    Args:
        cmd: Command as list of strings
        description: Description of what the command does
    """
    print(f"\n{'=' * 70}")
    print(f"STEP: {description}")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}")
        print(f"Command: {' '.join(cmd)}")
        return False


def transcribe_with_diarization(
    audio_path: str,
    language: str = "ru",
    max_speakers: int = 6,
    clean_audio: bool = True,
    whisper_model: str = "large-v3"
):
    """
    Run complete transcription pipeline with speaker diarization.

    Args:
        audio_path: Path to input audio file
        language: Language code (ru, en)
        max_speakers: Maximum number of speakers
        clean_audio: Whether to clean audio first
        whisper_model: Whisper model size to use

    Returns:
        True if successful, False otherwise
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        print(f"✗ Error: Audio file not found: {audio_path}")
        return False

    print(f"\n{'#' * 70}")
    print(f"# AUDIO TRANSCRIPTION WITH SPEAKER DIARIZATION")
    print(f"{'#' * 70}")
    print(f"\nInput file: {audio_path}")
    print(f"Language: {language}")
    print(f"Max speakers: {max_speakers}")
    print(f"Clean audio: {clean_audio}")
    print(f"Whisper model: {whisper_model}")

    # Step 1: Clean audio (optional)
    if clean_audio:
        cleaned_path = audio_path.parent / f"{audio_path.stem}_cleaned.wav"

        if not run_command(
            ["python", "clean_audio.py", str(audio_path)],
            "Audio cleaning"
        ):
            print("Warning: Audio cleaning failed, continuing with original file")
            working_audio = audio_path
        else:
            working_audio = cleaned_path
    else:
        working_audio = audio_path

    # Step 2: Whisper transcription
    json_file = working_audio.parent / f"{working_audio.stem}.json"

    if not run_command(
        [
            "python", "transcribe_whisper.py",
            str(working_audio),
            "--lang", language,
            "--model", whisper_model
        ],
        "Whisper transcription"
    ):
        return False

    if not json_file.exists():
        print(f"✗ Error: Transcription file not created: {json_file}")
        return False

    # Step 3: Speaker diarization
    tagged_json = working_audio.parent / f"{working_audio.stem}_tagged.json"

    if not run_command(
        [
            "python", "diarize_nemo.py",
            str(working_audio),
            str(json_file),
            "--max-speakers", str(max_speakers)
        ],
        "Speaker diarization"
    ):
        return False

    if not tagged_json.exists():
        print(f"✗ Error: Tagged transcript not created: {tagged_json}")
        return False

    # Step 4: Convert to readable formats
    if not run_command(
        ["python", "convert_to_readable.py", str(tagged_json)],
        "Format conversion"
    ):
        return False

    # Summary
    print(f"\n{'#' * 70}")
    print(f"# PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'#' * 70}")

    output_base = working_audio.stem
    print(f"\nGenerated files:")
    print(f"  - {output_base}.json - Whisper segments (raw)")
    print(f"  - {output_base}_tagged.json - Tagged with speakers")
    print(f"  - {output_base}_transcript.txt - Plain text transcript")
    print(f"  - {output_base}_transcript.md - Markdown transcript")
    print(f"  - {output_base}_detailed.md - Detailed Markdown with stats")

    return True


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Complete audio transcription pipeline with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Russian audio with up to 3 speakers
  python main_pipeline.py meeting.wav --lang ru --speakers 3

  # English audio without cleaning
  python main_pipeline.py interview.mp3 --lang en --no-clean

  # Fast transcription with medium model
  python main_pipeline.py lecture.wav --model medium

Full pipeline:
  1. Clean audio (remove silence, convert to mono 16kHz)
  2. Transcribe with Whisper
  3. Perform speaker diarization with NeMo
  4. Generate readable outputs (TXT, Markdown)
        """
    )

    parser.add_argument(
        "audio_path",
        help="Path to audio file (WAV, MP3, FLAC, M4A, etc.)"
    )

    parser.add_argument(
        "--lang",
        choices=["ru", "en"],
        default="ru",
        help="Audio language (default: ru)"
    )

    parser.add_argument(
        "--speakers",
        type=int,
        default=6,
        choices=range(1, 7),
        metavar="1-6",
        help="Maximum number of speakers (default: 6)"
    )

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip audio cleaning step"
    )

    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
        default="large-v3",
        help="Whisper model size (default: large-v3)"
    )

    args = parser.parse_args()

    # Run pipeline
    success = transcribe_with_diarization(
        audio_path=args.audio_path,
        language=args.lang,
        max_speakers=args.speakers,
        clean_audio=not args.no_clean,
        whisper_model=args.model
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
