# REF:D1 runtime data dirs, HF/transformers cache env, ensure_directories
import os
import tempfile

DEFAULT_BASE_DATA_DIR = os.path.join(tempfile.gettempdir(), "framed")
BASE_DATA_DIR = os.getenv("FRAMED_DATA_DIR", DEFAULT_BASE_DATA_DIR)

MODEL_DIR = os.path.join(BASE_DATA_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
CACHE_DIR = os.path.join(BASE_DATA_DIR, "cache")
ANALYSIS_CACHE_DIR = os.path.join(BASE_DATA_DIR, "analysis_cache")
EXPRESSION_CACHE_DIR = os.path.join(BASE_DATA_DIR, "expression_cache")

CACHE_VERSION = 2

PERCEPTION_MAX_WORKERS = min(4, int(os.environ.get("FRAMED_PERCEPTION_WORKERS", "4")))

DATA_ROOT = BASE_DATA_DIR
UPLOAD_FOLDER = UPLOAD_DIR
RESULTS_FOLDER = os.path.join(BASE_DATA_DIR, "results")
TMP_FOLDER = os.path.join(BASE_DATA_DIR, "tmp")
MODELS_DIR = MODEL_DIR

HF_HOME = CACHE_DIR
TRANSFORMERS_CACHE = CACHE_DIR
HUGGINGFACE_HUB_CACHE = CACHE_DIR
TORCH_HOME = CACHE_DIR
XDG_CACHE_HOME = CACHE_DIR

YOLO_CONFIG_DIR = os.path.join(BASE_DATA_DIR, "Ultralytics")
ULTRALYTICS_CFG = os.path.join(YOLO_CONFIG_DIR, "settings.json")


def ensure_directories():
    """Create runtime directories; on BASE_DATA_DIR failure, fall back to temp."""
    directories = [
        BASE_DATA_DIR,
        MODEL_DIR,
        UPLOAD_DIR,
        CACHE_DIR,
        ANALYSIS_CACHE_DIR,
        EXPRESSION_CACHE_DIR,
        RESULTS_FOLDER,
        TMP_FOLDER,
        YOLO_CONFIG_DIR,
    ]
    for p in directories:
        try:
            os.makedirs(p, exist_ok=True)
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot create directory {p}: {e}")
            if p == BASE_DATA_DIR:
                os.makedirs(DEFAULT_BASE_DATA_DIR, exist_ok=True)
                print(f"Using fallback base directory: {DEFAULT_BASE_DATA_DIR}")


os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR
os.environ["ULTRALYTICS_CFG"] = ULTRALYTICS_CFG

ensure_directories()

ECHO_MEMORY_PATH = os.path.join(BASE_DATA_DIR, "echo_memory.json")
