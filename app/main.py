"""FastAPI application entry point for Newrality Transcribe."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.models import (
    TranscriptionResponse,
    HealthCheckResponse,
    ModelsResponse,
    ErrorResponse,
)
from app.auth import verify_api_key
from app.utils import (
    generate_request_id,
    validate_audio_file,
    save_upload_to_temp,
    cleanup_temp_file,
)
from app.transcription import transcription_service

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info(
        "Starting Newrality Transcribe service",
        version=__version__,
        model=settings.whisper_model
    )
    yield
    # Shutdown
    logger.info("Shutting down Newrality Transcribe service")


# Create FastAPI application
app = FastAPI(
    title="Newrality Transcribe",
    description="Self-hosted audio transcription microservice using faster-whisper",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint (no authentication required).

    Returns service status and model information.
    """
    model_info = transcription_service.get_model_info()

    return HealthCheckResponse(
        status="healthy" if model_info["loaded"] else "degraded",
        model=model_info["model"],
        device=model_info["device"],
        version=__version__
    )


@app.get("/api/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    List available Whisper models (no authentication required).

    Returns the active model and list of available models.
    """
    available_models = ["tiny", "base", "small", "medium", "large"]

    return ModelsResponse(
        models=available_models,
        active=settings.whisper_model
    )


@app.post(
    "/api/v1/transcribe",
    response_model=TranscriptionResponse,
    dependencies=[Depends(verify_api_key)]
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query(
        None,
        description="Language code (e.g., 'it', 'en'). Auto-detected if not provided."
    ),
    temperature: float = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0.0 = deterministic)"
    ),
    beam_size: int = Query(
        None,
        ge=1,
        le=10,
        description="Beam size for decoding"
    ),
    include_segments: bool = Query(
        False,
        description="Include detailed segments with timestamps"
    )
):
    """
    Transcribe an audio file to text.

    Requires X-API-Key header for authentication.

    Supports multiple audio formats: mp3, wav, m4a, ogg, flac, webm.
    Maximum file size: 25MB (configurable).
    """
    request_id = generate_request_id()
    temp_file_path = None

    try:
        logger.info(
            "Received transcription request",
            request_id=request_id,
            filename=file.filename,
            content_type=file.content_type,
            language=language
        )

        # Validate audio file
        validate_audio_file(file)

        # Save uploaded file to temp directory
        temp_file_path, file_size = await save_upload_to_temp(file)

        logger.info(
            "File saved to temp",
            request_id=request_id,
            temp_path=temp_file_path,
            file_size_bytes=file_size
        )

        # Perform transcription
        text, detected_language, duration, segments = transcription_service.transcribe(
            audio_path=temp_file_path,
            language=language,
            temperature=temperature,
            beam_size=beam_size,
            include_segments=include_segments
        )

        logger.info(
            "Transcription successful",
            request_id=request_id,
            duration_seconds=duration,
            text_length=len(text)
        )

        return TranscriptionResponse(
            text=text,
            language=detected_language,
            duration=duration,
            segments=segments
        )

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, etc.)
        raise

    except Exception as e:
        logger.error(
            "Transcription request failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )

    finally:
        # Always cleanup temp file
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
            logger.debug(
                "Temp file cleaned up",
                request_id=request_id,
                temp_path=temp_file_path
            )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler for consistent error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            detail=exc.detail,
            request_id=generate_request_id()
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False
    )
