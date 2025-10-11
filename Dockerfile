# Dockerfile

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV DATA_ROOT=/app/app_data 

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create app data directory and set permissions
RUN mkdir -p /app/app_data && chmod 755 /app/app_data

# Create non-root user with proper permissions
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN chown -R user:user /app
USER user

EXPOSE 7860

# Use Python directly
CMD ["python", "run.py"]