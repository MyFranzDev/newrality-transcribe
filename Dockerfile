# Single-stage Dockerfile for faster-whisper transcription service
# Optimized for Google Cloud Run with GPU support
# Using single-stage to avoid issues with copying HuggingFace cache

FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper model (critical for cold start optimization)
# This downloads ~926MB model into /root/.cache/huggingface
COPY scripts/download_model.py /tmp/download_model.py
RUN python /tmp/download_model.py && \
    echo "Model pre-downloaded successfully" && \
    du -sh /root/.cache/huggingface

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Create app user and set permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /tmp /root/.cache && \
    chown -R appuser:appuser /tmp /app /root/.cache

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application
CMD ["python", "-m", "app.main"]
