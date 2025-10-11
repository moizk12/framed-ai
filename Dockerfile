# Simplified single-stage Dockerfile for HF Spaces
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

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

# Create non-root user and ensure app directory exists
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -m -d /home/appuser appuser && \
    mkdir -p /home/appuser/app && \
    chown -R appuser:appgroup /home/appuser

# Create all data directories with proper permissions
RUN mkdir -p \
    /data/uploads \
    /data/results \
    /data/tmp \
    /data/models \
    /data/hf/hub \
    /data/hf/transformers \
    /data/torch \
    /data/cache \
    /data/Ultralytics && \
    chown -R appuser:appgroup /data && \
    chmod -R 755 /data

# Set working directory (this directory was created above)
WORKDIR /home/appuser/app

# Copy requirements first (for better caching)
COPY --chown=appuser:appgroup requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appgroup . .

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

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run application
CMD ["gunicorn", "-k", "gthread", "--threads", "4", "-w", "1", \
     "--timeout", "120", "--keep-alive", "5", \
     "-b", "0.0.0.0:7860", "run:app"]