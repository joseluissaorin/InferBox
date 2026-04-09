# InferBox - GPU inference server
# Build:  docker build -t inferbox:latest .
# Run:    docker run --rm --gpus all -p 8811:8811 \
#           -v ~/.cache/huggingface:/root/.cache/huggingface \
#           -e INFERBOX_API_KEY=your-key \
#           inferbox:latest

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential cmake git curl ca-certificates \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch \
    && pip install -r requirements.txt

# Optional extras (image gen, peft, bitsandbytes, otel)
RUN pip install \
    diffusers \
    peft \
    bitsandbytes \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-http \
    || true

# Copy app
COPY inferbox/ ./inferbox/
COPY models.yaml ./models.yaml

EXPOSE 8811

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8811/v1/health || exit 1

CMD ["uvicorn", "inferbox.server:app", "--host", "0.0.0.0", "--port", "8811"]
