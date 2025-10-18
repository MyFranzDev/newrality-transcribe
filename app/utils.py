"""Utility functions for file validation and processing."""

import os
import uuid
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException, status
from app.config import settings


def generate_request_id() -> str:
    """Generate a unique request ID for logging and tracing."""
    return str(uuid.uuid4())


def validate_audio_file(file: UploadFile) -> None:
    """
    Validate uploaded audio file.

    Args:
        file: Uploaded file from request

    Raises:
        HTTPException: If file is invalid (wrong format, too large, etc.)
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded"
        )

    # Check if filename exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename"
        )

    # Extract file extension
    file_ext = Path(file.filename).suffix.lstrip('.').lower()

    # Check if format is allowed
    allowed_formats = settings.allowed_formats_list
    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {file_ext}. Allowed formats: {', '.join(allowed_formats)}"
        )


async def save_upload_to_temp(file: UploadFile) -> Tuple[str, int]:
    """
    Save uploaded file to /tmp directory.

    Args:
        file: Uploaded file from request

    Returns:
        Tuple of (temp_file_path, file_size_bytes)

    Raises:
        HTTPException: If file is too large or cannot be saved
    """
    # Generate unique filename
    file_ext = Path(file.filename).suffix if file.filename else ""
    temp_filename = f"audio_{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join("/tmp", temp_filename)

    # Read and save file with size check
    file_size = 0
    max_size = settings.max_file_size_bytes

    try:
        with open(temp_path, "wb") as temp_file:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                file_size += len(chunk)

                # Check file size limit
                if file_size > max_size:
                    # Clean up partial file
                    temp_file.close()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    max_mb = settings.max_file_size_mb
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Maximum size: {max_mb}MB"
                    )

                temp_file.write(chunk)

        return temp_path, file_size

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )


def cleanup_temp_file(file_path: str) -> None:
    """
    Remove temporary file from /tmp.

    Args:
        file_path: Path to temporary file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        # Log error but don't raise - cleanup is best-effort
        pass
