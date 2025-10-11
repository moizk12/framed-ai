# Simple and reliable Dockerfile for HF Spaces
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Create app directory and set as WORKDIR first
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories (use /data which is writable in HF Spaces)
RUN mkdir -p /data/uploads /data/results /data/tmp /data/models /data/hf/hub /data/hf/transformers /data/torch /data/cache /data/Ultralytics

# Set environment variables
ENV DATA_ROOT=/data \
    HF_HOME=/data/hf \
    HUGGINGFACE_HUB_CACHE=/data/hf/hub \
    TRANSFORMERS_CACHE=/data/hf/transformers \
    TORCH_HOME=/data/torch \
    XDG_CACHE_HOME=/data/cache \
    YOLO_CONFIG_DIR=/data/Ultralytics \
    ULTRALYTICS_CFG=/data/Ultralytics/settings.json \
    UPLOAD_DIR=/data/uploads

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run application (no user switching to avoid permission issues)
CMD ["gunicorn", "-k", "gthread", "--threads", "4", "-w", "1", \
     "--timeout", "120", "--keep-alive", "5", \
     "-b", "0.0.0.0:7860", "run:app"]