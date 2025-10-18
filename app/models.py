"""Pydantic models for request and response validation."""

from typing import Optional, List
from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Request parameters for transcription (from query params)."""

    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'it', 'en'). Auto-detected if not provided."
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0.0 = deterministic, 1.0 = creative)"
    )
    beam_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Beam size for decoding (higher = more accurate but slower)"
    )


class TranscriptionSegment(BaseModel):
    """A single transcription segment with timing."""

    id: int = Field(description="Segment ID")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text for this segment")


class TranscriptionResponse(BaseModel):
    """Response from transcription endpoint."""

    text: str = Field(description="Full transcribed text")
    language: str = Field(description="Detected or specified language")
    duration: float = Field(description="Processing time in seconds")
    segments: Optional[List[TranscriptionSegment]] = Field(
        default=None,
        description="Detailed segments with timestamps (optional)"
    )


class HealthCheckResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(description="Service status (healthy/degraded/unhealthy)")
    model: str = Field(description="Loaded Whisper model")
    device: str = Field(description="Compute device (cuda/cpu)")
    version: str = Field(description="Service version")


class ModelsResponse(BaseModel):
    """Response from models list endpoint."""

    models: List[str] = Field(description="Available Whisper models")
    active: str = Field(description="Currently active model")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(description="Error type/category")
    detail: str = Field(description="Human-readable error message")
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for debugging"
    )
