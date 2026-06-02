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

FRAMED_LOCAL_BASE_URL = os.getenv("FRAMED_LOCAL_BASE_URL", "http://localhost:1234/v1").strip()
FRAMED_LOCAL_API_KEY = os.getenv("FRAMED_LOCAL_API_KEY", "lm-studio")
FRAMED_LOCAL_MODEL_A = os.getenv("FRAMED_LOCAL_MODEL_A", "").strip()
FRAMED_LOCAL_MODEL_B = os.getenv("FRAMED_LOCAL_MODEL_B", "").strip()
FRAMED_STRICT_LOCAL = os.getenv("FRAMED_STRICT_LOCAL", "false").lower() == "true"
FRAMED_AB_MODEL = os.getenv("FRAMED_AB_MODEL", "").strip()
_gemma_budget = os.getenv("FRAMED_GEMMA_IMAGE_TOKEN_BUDGET", "").strip()
FRAMED_GEMMA_IMAGE_TOKEN_BUDGET = int(_gemma_budget) if _gemma_budget.isdigit() else None
