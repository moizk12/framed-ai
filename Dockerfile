# Production Track A runtime.
FROM python:3.11.13-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install the hash-locked public runtime dependency graph.
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Copy application code
COPY . .

# Create runtime directories under /data/framed
RUN groupadd --system framed && useradd --system --gid framed --home /app framed && mkdir -p \
    /data/framed/models \
    /data/framed/uploads \
    /data/framed/cache \
    /data/framed/results \
    /data/framed/tmp \
    /data/framed/Ultralytics && \
    chown -R framed:framed /data/framed /app

# Set environment variables for centralized runtime paths
ENV FRAMED_DATA_DIR=/data/framed \
    FRAMED_ENV=production \
    FRAMED_PUBLIC_BETA_ONLY=true \
    PUBLIC_AUTO_MIGRATE=false \
    FRAMED_COGNITION_V1=false \
    HF_HOME=/data/framed/cache \
    TRANSFORMERS_CACHE=/data/framed/cache \
    HUGGINGFACE_HUB_CACHE=/data/framed/cache \
    TORCH_HOME=/data/framed/cache \
    XDG_CACHE_HOME=/data/framed/cache \
    YOLO_CONFIG_DIR=/data/framed/Ultralytics \
    ULTRALYTICS_CFG=/data/framed/Ultralytics/settings.json

# Expose port
EXPOSE 7860

ARG FRAMED_VERSION=dev
ARG FRAMED_BUILD_SHA=unknown
ENV FRAMED_VERSION=${FRAMED_VERSION} FRAMED_BUILD_SHA=${FRAMED_BUILD_SHA}

USER framed

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=5)"

# Migrate first, then replace the bootstrap process with gunicorn.
CMD ["sh", "-c", "python -m framed.public_migrations && exec gunicorn -k gthread --threads 4 -w 1 --timeout 120 --keep-alive 5 -b 0.0.0.0:${PORT:-7860} run:app"]
