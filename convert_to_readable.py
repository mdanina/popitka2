#!/usr/bin/env python3
"""
Convert tagged JSON transcription to readable formats (TXT and Markdown).
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import timedelta


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS timestamp.
    Correctly handles audio longer than 24 hours.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (HH:MM:SS)
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def convert_to_txt(segments: list, output_path: str):
    """
    Convert tagged segments to plain text format.

    Args:
        segments: List of transcription segments with speakers
        output_path: Path to save TXT file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            timestamp = format_timestamp(seg["start"])
            speaker = seg.get("speaker", "Unknown")
            text = seg["text"].strip()

            f.write(f"[{timestamp}] {speaker}: {text}\n")

    print(f"✓ TXT saved to: {output_path}")


def convert_to_markdown(segments: list, output_path: str, audio_name: str = None):
    """
    Convert tagged segments to Markdown format with better structure.

    Args:
        segments: List of transcription segments with speakers
        output_path: Path to save Markdown file
        audio_name: Name of the audio file (for title)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        if audio_name:
            f.write(f"# Transcription: {audio_name}\n\n")
        else:
            f.write("# Transcription\n\n")

        # Calculate duration
        if segments:
            duration = segments[-1]["end"]
            f.write(f"**Duration:** {format_timestamp(duration)}\n\n")

            # Count speakers
            speakers = set(seg.get("speaker", "Unknown") for seg in segments)
            f.write(f"**Speakers:** {len(speakers)}\n\n")

        f.write("---\n\n")

        # Transcription content
        f.write("## Transcription\n\n")

        current_speaker = None
        for seg in segments:
            timestamp = format_timestamp(seg["start"])
            speaker = seg.get("speaker", "Unknown")
            text = seg["text"].strip()

            # Add speaker header if changed
            if speaker != current_speaker:
                f.write(f"\n### {speaker}\n\n")
                current_speaker = speaker

            f.write(f"**[{timestamp}]** {text}\n\n")

    print(f"✓ Markdown saved to: {output_path}")


def convert_to_detailed_markdown(segments: list, output_path: str, audio_name: str = None):
    """
    Convert tagged segments to detailed Markdown with statistics.

    Args:
        segments: List of transcription segments with speakers
        output_path: Path to save Markdown file
        audio_name: Name of the audio file (for title)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        title = audio_name if audio_name else "Audio Transcription"
        f.write(f"# {title}\n\n")

        # Metadata section
        f.write("## Metadata\n\n")

        if segments:
            duration = segments[-1]["end"]
            f.write(f"- **Duration:** {format_timestamp(duration)}\n")
            f.write(f"- **Total segments:** {len(segments)}\n")

            # Speaker statistics
            speakers = {}
            for seg in segments:
                speaker = seg.get("speaker", "Unknown")
                duration_seg = seg["end"] - seg["start"]

                if speaker not in speakers:
                    speakers[speaker] = {"count": 0, "duration": 0}

                speakers[speaker]["count"] += 1
                speakers[speaker]["duration"] += duration_seg

            f.write(f"- **Speakers:** {len(speakers)}\n\n")

            # Speaker details
            f.write("### Speaker Statistics\n\n")
            for speaker, stats in sorted(speakers.items()):
                f.write(f"- **{speaker}:**\n")
                f.write(f"  - Segments: {stats['count']}\n")
                f.write(f"  - Total speaking time: {format_timestamp(stats['duration'])}\n")

        f.write("\n---\n\n")

        # Transcription
        f.write("## Full Transcription\n\n")

        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            speaker = seg.get("speaker", "Unknown")
            text = seg["text"].strip()

            f.write(f"**[{start} - {end}] {speaker}:**\n\n")
            f.write(f"{text}\n\n")

    print(f"✓ Detailed Markdown saved to: {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert tagged JSON transcription to readable formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_readable.py meeting_tagged.json
  python convert_to_readable.py meeting_tagged.json --format txt
  python convert_to_readable.py meeting_tagged.json --format detailed
        """
    )

    parser.add_argument(
        "json_path",
        help="Path to tagged JSON file"
    )

    parser.add_argument(
        "--format",
        choices=["txt", "markdown", "detailed", "all"],
        default="all",
        help="Output format (default: all)"
    )

    args = parser.parse_args()

    # Load JSON
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"✗ Error: File not found: {json_path}")
        sys.exit(1)

    print(f"Loading: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        print("✗ Error: No segments found in JSON")
        sys.exit(1)

    print(f"✓ Loaded {len(segments)} segments")

    # Get audio name from filename
    audio_name = json_path.stem.replace("_tagged", "")

    # Convert to requested formats
    base_path = json_path.parent / audio_name

    if args.format in ["txt", "all"]:
        convert_to_txt(segments, str(base_path) + "_transcript.txt")

    if args.format in ["markdown", "all"]:
        convert_to_markdown(segments, str(base_path) + "_transcript.md", audio_name)

    if args.format in ["detailed", "all"]:
        convert_to_detailed_markdown(segments, str(base_path) + "_detailed.md", audio_name)

    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()
