"""Whisper transcription service using faster-whisper."""

import time
import threading
from typing import Optional, List, Tuple
from faster_whisper import WhisperModel
import structlog
from app.config import settings
from app.models import TranscriptionSegment

logger = structlog.get_logger()


class TranscriptionService:
    """Manages Whisper model and transcription operations."""

    def __init__(self):
        """Initialize the transcription service."""
        self.model: Optional[WhisperModel] = None
        self.model_name = settings.whisper_model
        self.device = settings.whisper_device
        self.compute_type = settings.whisper_compute_type
        self.loading = False
        self.load_error: Optional[str] = None
        self._load_lock = threading.Lock()

    def load_model_async(self) -> None:
        """Start loading the Whisper model in a background thread."""
        if self.model is not None or self.loading:
            return  # Already loaded or loading

        def _load():
            with self._load_lock:
                if self.model is not None:
                    return  # Another thread already loaded it

                self.loading = True
                try:
                    logger.info(
                        "Loading Whisper model in background",
                        model=self.model_name,
                        device=self.device,
                        compute_type=self.compute_type
                    )

                    self.model = WhisperModel(
                        self.model_name,
                        device=self.device,
                        compute_type=self.compute_type
                    )

                    logger.info("Whisper model loaded successfully")

                except Exception as e:
                    error_msg = f"Failed to load Whisper model: {str(e)}"
                    logger.error("Failed to load Whisper model", error=str(e))
                    self.load_error = error_msg
                finally:
                    self.loading = False

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def wait_for_model(self, timeout: int = 120) -> None:
        """Wait for the model to finish loading."""
        start_time = time.time()
        while self.model is None and self.loading and time.time() - start_time < timeout:
            time.sleep(0.5)

        if self.load_error:
            raise RuntimeError(self.load_error)

        if self.model is None:
            raise RuntimeError("Model loading timeout or model not started")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        beam_size: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        include_segments: bool = False
    ) -> Tuple[str, str, float, Optional[List[TranscriptionSegment]]]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'it', 'en'). Auto-detected if None.
            temperature: Sampling temperature (0.0-1.0)
            beam_size: Beam size for decoding
            initial_prompt: Optional text to guide transcription
            include_segments: Whether to include detailed segments in response

        Returns:
            Tuple of (transcribed_text, detected_language, duration_seconds, segments)

        Raises:
            Exception: If transcription fails
        """
        # Wait for model to finish loading if it's still loading
        if not self.model:
            logger.info("Model not ready, waiting for background loading to complete...")
            self.wait_for_model()

        # Use defaults from settings if not provided
        language = language or settings.default_language
        temperature = temperature if temperature is not None else settings.default_temperature
        beam_size = beam_size or settings.default_beam_size

        start_time = time.time()

        try:
            logger.info(
                "Starting transcription",
                audio_path=audio_path,
                language=language,
                temperature=temperature,
                beam_size=beam_size,
                initial_prompt=initial_prompt
            )

            # Perform transcription
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=language,
                temperature=temperature,
                beam_size=beam_size,
                initial_prompt=initial_prompt,
                vad_filter=settings.enable_vad_filter,
                word_timestamps=False  # Disable for faster processing
            )

            # Collect segments and build full text
            segments_list = []
            full_text_parts = []

            for segment in segments_iter:
                full_text_parts.append(segment.text.strip())

                if include_segments:
                    segments_list.append(
                        TranscriptionSegment(
                            id=segment.id,
                            start=segment.start,
                            end=segment.end,
                            text=segment.text.strip()
                        )
                    )

            # Join all segments into full text
            full_text = " ".join(full_text_parts)

            # Get detected language
            detected_language = info.language if info.language else language

            duration = time.time() - start_time

            logger.info(
                "Transcription completed",
                duration_seconds=duration,
                detected_language=detected_language,
                text_length=len(full_text),
                segments_count=len(segments_list) if include_segments else None
            )

            return (
                full_text,
                detected_language,
                duration,
                segments_list if include_segments else None
            )

        except Exception as e:
            logger.error("Transcription failed", error=str(e), audio_path=audio_path)
            raise Exception(f"Transcription failed: {str(e)}")

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded": self.model is not None
        }


# Global transcription service instance
transcription_service = TranscriptionService()
