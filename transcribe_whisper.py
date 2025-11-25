#!/usr/bin/env python3
"""
Audio transcription using OpenAI Whisper.
Supports multiple languages and model sizes.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import whisper


def transcribe_audio(
    audio_path: str,
    language: str = "ru",
    model_size: str = "large-v3",
    temperature: float = 0.0,
    beam_size: int = None,
    condition_on_previous: bool = False,
    initial_prompt: str = ""
) -> dict:
    """
    Transcribe audio file using Whisper.

    Args:
        audio_path: Path to audio file
        language: Language code (ru, en, etc.)
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
        temperature: Sampling temperature (0 = deterministic)
        beam_size: Beam search size (None = greedy)
        condition_on_previous: Use previous text as context
        initial_prompt: Initial prompt to guide the model

    Returns:
        Dictionary with transcription results
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Using device: {device.upper()} - {device_name}")

    # Load model
    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size, device=device)

    # Prepare transcription options
    options = {
        "language": language,
        "temperature": temperature,
        "condition_on_previous_text": condition_on_previous,
    }

    if beam_size is not None:
        options["beam_size"] = beam_size

    if initial_prompt:
        options["initial_prompt"] = initial_prompt

    # Transcribe
    print(f"Transcribing audio: {audio_path}")
    print(f"Language: {language}, Model: {model_size}")
    start_time = time.time()

    result = model.transcribe(audio_path, **options)

    elapsed = round(time.time() - start_time, 2)
    print(f"✓ Transcription completed in {elapsed}s")

    return result


def save_results(result: dict, audio_path: str):
    """
    Save transcription results to JSON and TXT files.

    Args:
        result: Transcription result from Whisper
        audio_path: Original audio file path (used for output naming)
    """
    base = Path(audio_path).stem

    # Save segments as JSON
    json_path = Path(audio_path).parent / f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    print(f"✓ Segments saved to: {json_path}")

    # Save full text as TXT
    txt_path = Path(audio_path).parent / f"{base}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"✓ Full text saved to: {txt_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_whisper.py meeting.wav
  python transcribe_whisper.py meeting.wav --lang en --model medium
  python transcribe_whisper.py meeting.wav --prompt "Topics: AI, ML, NLP"
        """
    )

    parser.add_argument(
        "audio_path",
        help="Path to audio file"
    )

    parser.add_argument(
        "--lang",
        default="ru",
        choices=["ru", "en", "de", "fr", "es", "it", "pt", "nl", "pl", "uk"],
        help="Audio language (default: ru)"
    )

    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
        help="Whisper model size (default: large-v3)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, 0-1 (default: 0 for deterministic)"
    )

    parser.add_argument(
        "--beam_size",
        type=int,
        help="Beam search size (default: None, uses greedy decoding)"
    )

    parser.add_argument(
        "--condition",
        action="store_true",
        help="Condition on previous text for context"
    )

    parser.add_argument(
        "--prompt",
        default="",
        help="Initial prompt to guide transcription (topics, names, terms)"
    )

    args = parser.parse_args()

    try:
        # Transcribe
        result = transcribe_audio(
            audio_path=args.audio_path,
            language=args.lang,
            model_size=args.model,
            temperature=args.temperature,
            beam_size=args.beam_size,
            condition_on_previous=args.condition,
            initial_prompt=args.prompt
        )

        # Save results
        save_results(result, args.audio_path)

        print("\n✓ Transcription complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
