#!/usr/bin/env python3
"""
Speaker diarization using NVIDIA NeMo.
Identifies and labels different speakers in audio.
"""

import signal
if not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils import validate_audio_file


def extract_embeddings(
    wav: np.ndarray,
    sr: int,
    model: EncDecSpeakerLabelModel,
    window_length: float = 3.0,
    step_length: float = 1.5
):
    """
    Extract speaker embeddings from audio using sliding window.

    Args:
        wav: Audio waveform
        sr: Sample rate
        model: NeMo speaker embedding model
        window_length: Window size in seconds
        step_length: Step size in seconds

    Returns:
        Tuple of (embeddings array, timestamps list)
    """
    embeddings = []
    timestamps = []
    current_time = 0.0
    total_duration = len(wav) / sr

    # Calculate total number of windows for progress bar
    num_windows = int((total_duration - window_length) / step_length) + 1

    print(f"Extracting embeddings (window: {window_length}s, step: {step_length}s)...")

    with tqdm(total=num_windows, desc="Extracting embeddings", unit="window") as pbar:
        while current_time + window_length <= total_duration:
            # Extract segment
            start_sample = int(current_time * sr)
            end_sample = int((current_time + window_length) * sr)
            segment = wav[start_sample:end_sample]

            # Save to temporary file for NeMo
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, segment, sr)
                tmp_path = tmp.name

            try:
                # Get embedding
                with torch.no_grad():
                    embedding = model.get_embedding(tmp_path).cpu().numpy().squeeze()

                # Normalize (with protection against division by zero)
                norm = np.linalg.norm(embedding)
                if norm > 1e-10:  # Avoid division by zero
                    embedding = embedding / norm
                    embeddings.append(embedding)
                    timestamps.append((current_time, current_time + window_length))
                else:
                    pbar.write(f"Warning: Zero embedding at {current_time:.1f}s, skipping")

            finally:
                os.remove(tmp_path)

            current_time += step_length
            pbar.update(1)

    if len(embeddings) == 0:
        raise RuntimeError(
            "No valid embeddings extracted. Audio may be too short, "
            "silent, or corrupted."
        )

    print(f"[OK] Extracted {len(embeddings)} embeddings")
    return np.stack(embeddings), timestamps


def auto_cluster_speakers(embeddings: np.ndarray, max_speakers: int = 10):
    """
    Automatically determine optimal number of speakers and cluster them.

    Args:
        embeddings: Speaker embeddings array
        max_speakers: Maximum number of speakers to consider

    Returns:
        Array of speaker labels

    Raises:
        RuntimeError: If clustering fails for all speaker counts
    """
    print(f"Clustering speakers (max: {max_speakers})...")

    if len(embeddings) < 2:
        raise ValueError("Need at least 2 embeddings for clustering")

    best_labels = None
    best_score = -1
    best_k = 2

    for k in range(2, min(max_speakers + 1, len(embeddings) + 1)):
        try:
            labels = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                random_state=42
            ).fit_predict(embeddings)

            score = silhouette_score(embeddings, labels)

            if score > best_score:
                best_labels = labels
                best_score = score
                best_k = k

        except Exception as e:
            print(f"Warning: Clustering with {k} speakers failed: {e}")
            continue

    # If all clustering attempts failed, raise error
    if best_labels is None:
        raise RuntimeError(
            f"Failed to cluster speakers. Tried 2-{max_speakers} speakers. "
            "Try different audio or adjust max_speakers parameter."
        )

    print(f"[OK] Detected {best_k} speakers (silhouette score: {best_score:.3f})")
    return best_labels


def merge_segments(timestamps, labels, gap_threshold: float = 0.5):
    """
    Merge consecutive segments from the same speaker.

    Args:
        timestamps: List of (start, end) tuples
        labels: Speaker labels for each segment
        gap_threshold: Maximum gap to merge (seconds)

    Returns:
        List of merged segments with speaker labels
    """
    if len(timestamps) == 0:
        return []

    merged = []
    current_segment = {
        "speaker": int(labels[0]),
        "start": timestamps[0][0],
        "end": timestamps[0][1]
    }

    for (start, end), label in zip(timestamps[1:], labels[1:]):
        label = int(label)

        # Same speaker and close enough - extend segment
        if label == current_segment["speaker"] and start <= current_segment["end"] + gap_threshold:
            current_segment["end"] = end
        else:
            # Different speaker or gap too large - save and start new segment
            merged.append(current_segment)
            current_segment = {
                "speaker": label,
                "start": start,
                "end": end
            }

    # Add last segment
    merged.append(current_segment)

    print(f"[OK] Merged into {len(merged)} continuous segments")
    return merged


def assign_speakers_to_transcript(whisper_segments, diarization_segments):
    """
    Assign speaker labels to Whisper transcription segments.

    Args:
        whisper_segments: List of transcription segments from Whisper
        diarization_segments: List of diarization segments with speaker labels

    Returns:
        List of transcription segments with speaker labels added
    """
    print("Assigning speakers to transcript segments...")

    tagged_segments = []

    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Find overlapping diarization segment
        speaker = "Unknown"
        max_overlap = 0

        for diar_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(seg_start, diar_seg["start"])
            overlap_end = min(seg_end, diar_seg["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                speaker = f"Speaker_{diar_seg['speaker'] + 1}"

        # Add speaker to segment
        tagged_segment = {**seg, "speaker": speaker}
        tagged_segments.append(tagged_segment)

    print(f"[OK] Tagged {len(tagged_segments)} segments with speakers")
    return tagged_segments


def diarize_and_tag(
    audio_path: str,
    whisper_json_path: str,
    max_speakers: int = 6
):
    """
    Perform speaker diarization and tag transcription.

    Args:
        audio_path: Path to audio file
        whisper_json_path: Path to Whisper transcription JSON
        max_speakers: Maximum number of speakers

    Returns:
        List of tagged segments

    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If files are invalid
        RuntimeError: If diarization fails
    """
    # Validate files
    audio_path = validate_audio_file(audio_path)

    whisper_json_path = Path(whisper_json_path)
    if not whisper_json_path.is_file():
        raise FileNotFoundError(f"Whisper JSON not found: {whisper_json_path}")

    # Load NeMo model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    print("Loading NeMo speaker embedding model...")

    try:
        model = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        ).to(device).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: You may need to set HF_TOKEN environment variable")
        raise

    # Load audio
    print(f"Loading audio: {audio_path}")
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(wav) / sr
    print(f"[OK] Audio loaded: {duration:.1f}s")

    # Extract embeddings
    embeddings, timestamps = extract_embeddings(wav, sr, model)

    # Cluster speakers
    labels = auto_cluster_speakers(embeddings, max_speakers)

    # Merge segments
    diarization_segments = merge_segments(timestamps, labels)

    # Load Whisper transcription
    print(f"Loading transcription: {whisper_json_path}")
    with open(whisper_json_path, "r", encoding="utf-8") as f:
        whisper_segments = json.load(f)

    # Assign speakers
    tagged_segments = assign_speakers_to_transcript(whisper_segments, diarization_segments)

    return tagged_segments


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Speaker diarization using NVIDIA NeMo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diarize_nemo.py meeting.wav meeting.json
  python diarize_nemo.py meeting.wav meeting.json --max-speakers 4
        """
    )

    parser.add_argument(
        "audio_path",
        help="Path to audio file"
    )

    parser.add_argument(
        "whisper_json",
        help="Path to Whisper transcription JSON"
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=6,
        help="Maximum number of speakers (default: 6)"
    )

    args = parser.parse_args()

    try:
        # Perform diarization
        tagged_segments = diarize_and_tag(
            audio_path=args.audio_path,
            whisper_json_path=args.whisper_json,
            max_speakers=args.max_speakers
        )

        # Save results
        output_path = Path(args.whisper_json).with_stem(
            Path(args.whisper_json).stem + "_tagged"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tagged_segments, f, ensure_ascii=False, indent=2)

        print(f"\n[OK] Tagged transcript saved to: {output_path}")

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
