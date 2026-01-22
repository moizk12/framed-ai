# Production Dockerfile for FRAMED
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

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories under /data/framed
RUN mkdir -p \
    /data/framed/models \
    /data/framed/uploads \
    /data/framed/cache \
    /data/framed/results \
    /data/framed/tmp \
    /data/framed/Ultralytics && \
    chmod -R 777 /data/framed

# Set environment variables for centralized runtime paths
ENV FRAMED_DATA_DIR=/data/framed \
    HF_HOME=/data/framed/cache \
    TRANSFORMERS_CACHE=/data/framed/cache \
    HUGGINGFACE_HUB_CACHE=/data/framed/cache \
    TORCH_HOME=/data/framed/cache \
    XDG_CACHE_HOME=/data/framed/cache \
    YOLO_CONFIG_DIR=/data/framed/Ultralytics \
    ULTRALYTICS_CFG=/data/framed/Ultralytics/settings.json

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run application with gunicorn (single worker for ML safety)
CMD ["gunicorn", "-k", "gthread", "--threads", "4", "-w", "1", \
     "--timeout", "120", "--keep-alive", "5", \
     "-b", "0.0.0.0:7860", "run:app"]