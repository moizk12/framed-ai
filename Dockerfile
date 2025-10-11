# ===================================
# FRAMED - Fixed Dockerfile for HF Spaces
# ===================================

FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# ✅ CREATE DATA DIRECTORIES AS ROOT (fixes permission issues)
RUN mkdir -p \
    /data/uploads \
    /data/results \
    /data/tmp \
    /data/models \
    /data/hf/hub \
    /data/hf/transformers \
    /data/torch \
    /data/cache \
    /data/Ultralytics \
    && chown -R appuser:appgroup /data \
    && chmod -R 755 /data

USER appuser
WORKDIR /home/appuser/app

# ===================================
# Builder stage
# ===================================
FROM base as builder

COPY --chown=appuser:appgroup requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ===================================
# Final stage
# ===================================
FROM base as final

COPY --from=builder /home/appuser/.local /home/appuser/.local

ENV PATH=/home/appuser/.local/bin:$PATH \
    DATA_ROOT=/data \
    HF_HOME=/data/hf \
    HUGGINGFACE_HUB_CACHE=/data/hf/hub \
    TRANSFORMERS_CACHE=/data/hf/transformers \
    TORCH_HOME=/data/torch \
    XDG_CACHE_HOME=/data/cache \
    YOLO_CONFIG_DIR=/data/Ultralytics \
    ULTRALYTICS_CFG=/data/Ultralytics/settings.json \
    UPLOAD_DIR=/data/uploads

COPY --chown=appuser:appgroup . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["gunicorn", "-k", "gthread", "--threads", "4", "-w", "1", \
     "--timeout", "120", "--keep-alive", "5", \
     "-b", "0.0.0.0:7860", "run:app"]