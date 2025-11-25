#!/usr/bin/env python3
"""
Utility functions for audio transcription pipeline.
Includes validation, logging setup, and helper functions.
"""

import logging
from pathlib import Path
from typing import Optional


# Valid audio file extensions
VALID_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Default maximum file size (in MB)
DEFAULT_MAX_FILE_SIZE_MB = 500


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Configure logging for the application.

    Args:
        level: Logging level (logging.INFO, logging.DEBUG, etc.)
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_audio_file(
    file_path: str,
    max_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    must_exist: bool = True
) -> Path:
    """
    Validate audio file path and properties.

    Args:
        file_path: Path to audio file
        max_size_mb: Maximum allowed file size in MB
        must_exist: If True, file must exist

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If file doesn't exist and must_exist=True
        ValueError: If file is invalid (wrong format, too large, etc.)
        PermissionError: If file is not readable
    """
    # Resolve path (protects against path traversal)
    try:
        path = Path(file_path).resolve(strict=must_exist)
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    except RuntimeError as e:
        raise ValueError(f"Invalid file path: {file_path} - {e}")

    # If file must exist, perform additional checks
    if must_exist:
        # Check if it's a file (not directory)
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Check file extension
        if path.suffix.lower() not in VALID_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Invalid audio format: {path.suffix}\n"
                f"Supported formats: {', '.join(sorted(VALID_AUDIO_EXTENSIONS))}"
            )

        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)\n"
                f"Consider splitting the audio into smaller segments."
            )

        # Check if file is empty
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty: {path}")

        # Check if file is readable
        if not path.is_file() or not path.exists():
            raise FileNotFoundError(f"Cannot access file: {path}")

        try:
            with open(path, 'rb') as f:
                f.read(1)  # Try to read first byte
        except PermissionError:
            raise PermissionError(f"No read permission for file: {path}")
        except Exception as e:
            raise ValueError(f"Cannot read file: {path} - {e}")

    return path


def validate_output_path(file_path: str, overwrite: bool = True) -> Path:
    """
    Validate output file path.

    Args:
        file_path: Path where output will be saved
        overwrite: If False, raise error if file already exists

    Returns:
        Resolved Path object

    Raises:
        FileExistsError: If file exists and overwrite=False
        PermissionError: If directory is not writable
        ValueError: If path is invalid
    """
    try:
        path = Path(file_path).resolve()
    except RuntimeError as e:
        raise ValueError(f"Invalid output path: {file_path} - {e}")

    # Check if parent directory exists and is writable
    parent = path.parent
    if not parent.exists():
        raise ValueError(f"Output directory doesn't exist: {parent}")

    if not parent.is_dir():
        raise ValueError(f"Parent path is not a directory: {parent}")

    # Check write permission
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"No write permission for directory: {parent}")

    # Check if file already exists
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}\n"
            f"Use overwrite=True to replace it."
        )

    return path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB", "500 KB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


# Add missing import
import os
