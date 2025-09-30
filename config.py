import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEEPFACE_ENABLE = os.getenv("DEEPFACE_ENABLE", "true").lower() == "true"
GPU = os.getenv("GPU", "false").lower() == "true"
