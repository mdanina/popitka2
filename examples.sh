#!/bin/bash
# Example commands for using the audio transcriber

# ==============================================================================
# BASIC USAGE EXAMPLES
# ==============================================================================

# Example 1: Russian audio with automatic speaker detection (up to 6 speakers)
# python main_pipeline.py meeting.wav --lang ru --speakers 6

# Example 2: English audio with 3 speakers
# python main_pipeline.py interview.mp3 --lang en --speakers 3

# Example 3: Fast transcription with medium model
# python main_pipeline.py lecture.wav --model medium --lang ru

# Example 4: Skip audio cleaning step
# python main_pipeline.py podcast.wav --lang en --no-clean

# ==============================================================================
# STEP-BY-STEP USAGE EXAMPLES
# ==============================================================================

# Step 1: Clean audio (optional)
# python clean_audio.py my_audio.mp3
# Output: my_audio_cleaned.wav

# Step 2: Transcribe with Whisper
# python transcribe_whisper.py my_audio_cleaned.wav --lang ru --model large-v3
# Output: my_audio_cleaned.json, my_audio_cleaned.txt

# Step 3: Diarize speakers
# python diarize_nemo.py my_audio_cleaned.wav my_audio_cleaned.json --max-speakers 4
# Output: my_audio_cleaned_tagged.json

# Step 4: Convert to readable formats
# python convert_to_readable.py my_audio_cleaned_tagged.json
# Output: my_audio_cleaned_transcript.txt, my_audio_cleaned_transcript.md, my_audio_cleaned_detailed.md

# ==============================================================================
# ADVANCED EXAMPLES
# ==============================================================================

# With initial prompt for better accuracy
# python transcribe_whisper.py meeting.wav --lang ru --prompt "Topics: AI, machine learning"

# With beam search
# python transcribe_whisper.py meeting.wav --beam_size 5

# Generate only specific format
# python convert_to_readable.py meeting_tagged.json --format txt

# ==============================================================================
# MODEL SIZE COMPARISON
# ==============================================================================

# Tiny model (fastest, lowest quality)
# python main_pipeline.py test.wav --model tiny --lang ru

# Medium model (balanced)
# python main_pipeline.py test.wav --model medium --lang ru

# Large-v3 model (slowest, best quality)
# python main_pipeline.py test.wav --model large-v3 --lang ru
