"""Configuration management using Pydantic Settings."""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Configuration
    api_key: str = Field(
        default="vff_dev_reportbabel_demo",
        description="Primary API key for authentication"
    )
    allowed_api_keys: str = Field(
        default="vff_dev_reportbabel_demo",
        description="Comma-separated list of allowed API keys"
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # Whisper Model Configuration
    whisper_model: str = Field(
        default="small",
        description="Whisper model size (tiny, base, small, medium, large)"
    )
    whisper_device: str = Field(
        default="auto",
        description="Device for inference (auto, cuda, cpu)"
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="Compute type for inference (int8, float16, float32)"
    )

    # Transcription Settings
    default_language: str = Field(
        default="it",
        description="Default language for transcription"
    )
    default_temperature: float = Field(
        default=0.0,
        description="Default temperature for sampling"
    )
    default_beam_size: int = Field(
        default=5,
        description="Default beam size for decoding"
    )

    # File Upload Limits
    max_file_size_mb: int = Field(
        default=25,
        description="Maximum file size in megabytes"
    )
    allowed_audio_formats: str = Field(
        default="mp3,wav,m4a,ogg,flac,webm",
        description="Comma-separated list of allowed audio formats"
    )

    # Performance
    enable_vad_filter: bool = Field(
        default=True,
        description="Enable Voice Activity Detection filter"
    )

    @property
    def allowed_api_keys_list(self) -> List[str]:
        """Parse allowed API keys from comma-separated string."""
        return [key.strip() for key in self.allowed_api_keys.split(",") if key.strip()]

    @property
    def allowed_formats_list(self) -> List[str]:
        """Parse allowed audio formats from comma-separated string."""
        return [fmt.strip().lower() for fmt in self.allowed_audio_formats.split(",") if fmt.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


# Global settings instance
settings = Settings()
