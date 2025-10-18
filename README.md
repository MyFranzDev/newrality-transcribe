# Newrality Transcribe

Self-hosted audio transcription microservice using faster-whisper (Whisper Small model). Provides drop-in replacement for OpenAI Whisper API to reduce costs and maintain data privacy for VoiceFormFiller SaaS platform.

## Features

- **Fast transcription** with faster-whisper (4x faster than OpenAI Whisper)
- **GPU acceleration** support (NVIDIA T4 on Cloud Run)
- **Multiple audio formats** (mp3, wav, m4a, ogg, flac, webm)
- **API key authentication** for security
- **Structured JSON logging** for production monitoring
- **Stateless design** with in-memory file processing
- **Voice Activity Detection** (VAD) for improved accuracy

## Quick Start

### 1. Prerequisites

- Python 3.11+
- (Optional) NVIDIA GPU with CUDA support for faster processing

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/MyFranzDev/newrality-transcribe.git
cd newrality-transcribe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` to customize settings:

```bash
# API Configuration
API_KEY=your_secure_api_key_here
ALLOWED_API_KEYS=key1,key2,key3

# Whisper Model
WHISPER_MODEL=small
WHISPER_DEVICE=auto  # auto, cuda, cpu
WHISPER_COMPUTE_TYPE=int8

# Transcription
DEFAULT_LANGUAGE=it
```

### 4. Run the Service

```bash
# Start the server
python -m app.main

# Or use uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The service will be available at `http://localhost:8080`.

## API Usage

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "model": "small",
  "device": "cuda",
  "version": "0.1.0"
}
```

### List Available Models

```bash
curl http://localhost:8080/api/v1/models
```

### Transcribe Audio

```bash
curl -X POST http://localhost:8080/api/v1/transcribe \
  -H "X-API-Key: vff_dev_reportbabel_demo" \
  -F "file=@path/to/audio.mp3" \
  -F "language=it"
```

Response:
```json
{
  "text": "Questo è un esempio di trascrizione audio.",
  "language": "it",
  "duration": 1.23
}
```

#### Optional Parameters

- `language` (string): Language code (e.g., 'it', 'en'). Auto-detected if not provided.
- `temperature` (float, 0.0-1.0): Sampling temperature (default: 0.0 for deterministic)
- `beam_size` (int, 1-10): Beam size for decoding (default: 5)
- `include_segments` (bool): Include detailed segments with timestamps (default: false)

### Example with Segments

```bash
curl -X POST http://localhost:8080/api/v1/transcribe \
  -H "X-API-Key: vff_dev_reportbabel_demo" \
  -F "file=@audio.mp3" \
  -F "include_segments=true"
```

Response:
```json
{
  "text": "Questo è un esempio di trascrizione.",
  "language": "it",
  "duration": 1.23,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Questo è un esempio di trascrizione."
    }
  ]
}
```

## Integration with VoiceFormFiller

Update your VoiceFormFiller backend `.env`:

```bash
# Replace OpenAI Whisper with self-hosted
TRANSCRIBE_SERVICE_URL=http://localhost:8080/api/v1
TRANSCRIBE_API_KEY=vff_dev_reportbabel_demo
USE_SELF_HOSTED_WHISPER=true
```

## Development

### Project Structure

```
newrality-transcribe/
├── app/
│   ├── __init__.py          # Package version
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Environment configuration
│   ├── models.py            # Pydantic request/response models
│   ├── transcription.py     # Whisper service logic
│   ├── auth.py              # API key validation
│   └── utils.py             # File validation helpers
├── tests/                   # Unit and integration tests
├── scripts/                 # Deployment and utility scripts
├── .env.example             # Example environment variables
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## Deployment

### Local Docker (TODO)

```bash
# Build Docker image
docker build -t newrality-transcribe .

# Run container
docker run -p 8080:8080 \
  -e API_KEY=your_key \
  newrality-transcribe
```

### Google Cloud Run (TODO)

```bash
# Deploy to Cloud Run with GPU
gcloud run deploy newrality-transcribe \
  --region us-central1 \
  --gpu 1 \
  --gpu-type nvidia-tesla-t4 \
  --memory 2Gi \
  --timeout 300s
```

## Performance

### Model Comparison

| Model  | Size  | Speed      | Accuracy | VRAM   |
|--------|-------|------------|----------|--------|
| tiny   | 39M   | 32x faster | Low      | ~1GB   |
| base   | 74M   | 16x faster | Medium   | ~1GB   |
| small  | 244M  | 4x faster  | Good     | ~2GB   |
| medium | 769M  | 2x faster  | Better   | ~5GB   |
| large  | 1550M | 1x faster  | Best     | ~10GB  |

**Current**: `small` model (balanced speed/accuracy for Italian)

### Optimization Tips

- Use `WHISPER_COMPUTE_TYPE=int8` for faster inference with minimal accuracy loss
- Enable `ENABLE_VAD_FILTER=true` to remove silence and improve accuracy
- Use GPU (`WHISPER_DEVICE=cuda`) for 4x speed improvement
- Set `temperature=0.0` for deterministic, reproducible results

## Troubleshooting

### Model Download Issues

If the model fails to download on first run, check your internet connection and firewall settings. The model is downloaded from Hugging Face.

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

If False, install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

If you encounter OOM errors:
- Use a smaller model (`WHISPER_MODEL=base`)
- Use int8 quantization (`WHISPER_COMPUTE_TYPE=int8`)
- Reduce `DEFAULT_BEAM_SIZE` (e.g., 3 instead of 5)

## Contributing

See `.context` for development guidelines and roadmap.

## License

Proprietary - Newrality 2025

## Support

For issues, contact: francescozazza@gmail.com
