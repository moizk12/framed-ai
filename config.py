import os

# Load .env if python-dotenv available (so OPENAI_API_KEY works without manual export)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEEPFACE_ENABLE = os.getenv("DEEPFACE_ENABLE", "true").lower() == "true"
GPU = os.getenv("GPU", "false").lower() == "true"
