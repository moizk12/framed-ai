# ---- Multi-stage build for optimized image size ----
FROM python:3.11-slim as base

# Set environment variables for best practices
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies including Git LFS
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Create a non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Create and set permissions for data directory
RUN mkdir -p /data && chown -R appuser:appgroup /data

# Switch to the non-root user
USER appuser

# Set the working directory
WORKDIR /home/appuser/app

# ---- Builder Stage ----
FROM base as builder

# Copy only the requirements file to leverage Docker layer caching
COPY --chown=appuser:appgroup requirements.txt .

# Install Python dependencies to user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Final Stage ----
FROM base as final

# Copy installed packages from the builder stage
COPY --from=builder /home/appuser/.local /home/appuser/.local

# Add the local user's bin to the PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appgroup . .

# Set cache environment variables for writable paths
ENV DATA_ROOT=/data \
    HF_HOME=/data/hf \
    HUGGINGFACE_HUB_CACHE=/data/hf/hub \
    TRANSFORMERS_CACHE=/data/hf/transformers \
    TORCH_HOME=/data/torch \
    XDG_CACHE_HOME=/data/cache \
    YOLO_CONFIG_DIR=/data/Ultralytics \
    ULTRALYTICS_CFG=/data/Ultralytics/settings.json

# Expose the port the app will run on
EXPOSE 7860

# Command to run the application using Gunicorn with gthread worker
CMD ["gunicorn", "-k", "gthread", "-w", "2", "-b", "0.0.0.0:7860", "run:app"]