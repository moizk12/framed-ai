FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# python deps (install CPU torch wheels first for reliability on Spaces)
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

# gunicorn serves Flask app object named "app" in app.py
CMD ["gunicorn","-w","2","-k","gthread","-t","120","--log-level","debug","-b","0.0.0.0:7860","app:app"]



