# --- FRAMED: Dockerfile (CPU only, fast, stable) ---
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV/DeepFace/YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Python deps (install CPU torch explicitly first to avoid pulling CUDA wheels)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# app requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

# runtime
ENV PYTHONUNBUFFERED=1
# HF Spaces expects the app on 7860 by default
EXPOSE 7860
CMD ["gunicorn","-w","2","-k","gthread","-t","120","-b","0.0.0.0:7860","app:app"]
