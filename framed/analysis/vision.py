import os, uuid, json
import logging
import hashlib
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from .schema import create_empty_analysis_result, normalize_to_schema, validate_schema

# Configure logging for production
logger = logging.getLogger(__name__)

# ========================================================
# STEP 4.2: CENTRALIZED RUNTIME DIRECTORY STRATEGY
# ========================================================
# Production-safe runtime directory configuration
# Uses FRAMED_DATA_DIR if set, otherwise defaults to /tmp/framed locally
# On Hugging Face Spaces, set FRAMED_DATA_DIR=/data/framed

# Base data directory - centralized and writable
BASE_DATA_DIR = os.getenv("FRAMED_DATA_DIR", "/tmp/framed")

# Subdirectories under BASE_DATA_DIR
MODEL_DIR = os.path.join(BASE_DATA_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
CACHE_DIR = os.path.join(BASE_DATA_DIR, "cache")
ANALYSIS_CACHE_DIR = os.path.join(BASE_DATA_DIR, "analysis_cache")

# Legacy compatibility (for backward compatibility)
DATA_ROOT = BASE_DATA_DIR
UPLOAD_FOLDER = UPLOAD_DIR
RESULTS_FOLDER = os.path.join(BASE_DATA_DIR, "results")
TMP_FOLDER = os.path.join(BASE_DATA_DIR, "tmp")
MODELS_DIR = MODEL_DIR

# Hugging Face and Transformers cache - explicitly set to CACHE_DIR
HF_HOME = CACHE_DIR
TRANSFORMERS_CACHE = CACHE_DIR
HUGGINGFACE_HUB_CACHE = CACHE_DIR
TORCH_HOME = CACHE_DIR
XDG_CACHE_HOME = CACHE_DIR

# Ultralytics settings
YOLO_CONFIG_DIR = os.path.join(BASE_DATA_DIR, "Ultralytics")
ULTRALYTICS_CFG = os.path.join(YOLO_CONFIG_DIR, "settings.json")

def ensure_directories():
    """Create all necessary runtime directories with error handling"""
    directories = [
        BASE_DATA_DIR,
        MODEL_DIR,
        UPLOAD_DIR,
        CACHE_DIR,
        ANALYSIS_CACHE_DIR,
        RESULTS_FOLDER,
        TMP_FOLDER,
        YOLO_CONFIG_DIR
    ]
    
    for p in directories:
        try:
            os.makedirs(p, exist_ok=True)
        except (PermissionError, OSError) as e:
            print(f"⚠️ Warning: Cannot create directory {p}: {e}")
            # Fallback to /tmp/framed if BASE_DATA_DIR fails
            if p == BASE_DATA_DIR:
                fallback_base = "/tmp/framed"
                os.makedirs(fallback_base, exist_ok=True)
                print(f"✅ Using fallback base directory: {fallback_base}")

# Set environment variables explicitly (for Hugging Face and Transformers)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR
os.environ["ULTRALYTICS_CFG"] = ULTRALYTICS_CFG

# Create all necessary directories
ensure_directories()

ECHO_MEMORY_PATH = os.path.join(BASE_DATA_DIR, "echo_memory.json")



import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from sklearn.cluster import KMeans
from colorthief import ColorThief

from collections import Counter, defaultdict
import importlib.util

try:
    from config import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

NIMA_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
if NIMA_AVAILABLE:
    def _import_tf_keras():
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        from tensorflow.keras.preprocessing import image as keras_image
        return Model, Dense, Dropout, MobileNet, preprocess_input, keras_image
else:
    _import_tf_keras = None

DEEPFACE_ENABLE = os.environ.get("DEEPFACE_ENABLE", "false").lower() == "true"
if DEEPFACE_ENABLE:
    try:
        from deepface import DeepFace
    except ImportError:
        print("DeepFace not installed. Facial analysis will be disabled.")
        DeepFace = None
else:
    DeepFace = None

# ========================================================
# STEP 4.3: LAZY-LOAD ALL HEAVY MODELS
# ========================================================
# NO models load at import time - all are lazy-loaded on first use

# YOLO model - lazy loaded
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", os.path.join(MODEL_DIR, "yolov8n.pt"))
_yolo_model = None

def get_yolo_model():
    """Lazy-load YOLO model on first use - NEVER called at import time"""
    global _yolo_model
    if _yolo_model is None:
        logger.info("Loading YOLO model (first use)")
        try:
            # Ensure weights directory exists
            weights_dir = os.path.dirname(YOLO_WEIGHTS)
            os.makedirs(weights_dir, exist_ok=True)
            # YOLO will auto-download weights if missing
            _yolo_model = YOLO(YOLO_WEIGHTS)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            raise
    return _yolo_model

# CLIP model - lazy loaded
_clip_model = None
_clip_processor = None
_device = None

def get_clip_model():
    """Lazy-load CLIP model and processor on first use - NEVER called at import time"""
    global _clip_model, _clip_processor, _device
    if _clip_model is None:
        logger.info("Loading CLIP model (first use)")
        try:
            # Graceful GPU/CPU fallback - never crashes on device selection
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {_device}")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}", exc_info=True)
            raise
    return _clip_model, _clip_processor, _device

# NIMA model - lazy loaded
_nima_model = None

def get_nima_model():
    """Lazy-load NIMA model on first use (if TensorFlow available) - NEVER called at import time"""
    global _nima_model
    if _nima_model is None and NIMA_AVAILABLE:
        logger.info("Loading NIMA model (first use)")
        try:
            model_path = os.path.join(MODEL_DIR, 'nima_mobilenet.h5')
            Model, Dense, Dropout, MobileNet, _, _ = _import_tf_keras()
            base_model = MobileNet((None, None, 3), include_top=False, pooling='avg', weights=None)
            x = Dropout(0.75)(base_model.output)
            x = Dense(10, activation='softmax')(x)
            _nima_model = Model(base_model.input, x)
            _nima_model.load_weights(model_path)
            logger.info("NIMA model loaded successfully")
        except Exception as e:
            logger.warning(f"NIMA weights not found at {model_path}. Aesthetic scoring disabled. Error: {e}")
            _nima_model = None
    elif not NIMA_AVAILABLE:
        logger.debug("NIMA not available (TensorFlow not installed)")
    return _nima_model

# ========================================================
# STEP 4.5: LAZY-LOAD OpenAI CLIENT (Cold-Start Compliance)
# ========================================================
# OpenAI client is lazy-loaded to prevent import-time initialization
# This ensures /health endpoint and app creation never trigger client creation
# Version: 2026-01-22 - Fixed import-time client initialization error

# Fix #1: NEVER cache a None client - HF Spaces + Gunicorn safe
# Re-checks env vars every call to handle delayed env var injection
def get_openai_client():
    """
    Get OpenAI client - HF Spaces + Gunicorn safe.
    
    NEVER caches None. Always re-checks environment variables.
    This handles the case where env vars are injected after module import.
    """
    # Check environment variable directly every time (not module-level constant)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        # Also check module-level constant as fallback (for local dev)
        api_key = OPENAI_API_KEY if OPENAI_API_KEY else ""
        api_key = api_key.strip() if api_key else ""
    
    if not api_key:
        logger.debug("OpenAI API key not found. Cloud-enhanced features disabled.")
        return None
    
    try:
        # Create new client instance every time (no caching)
        # This ensures we always use the latest env var value
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        return client
    except TypeError as e:
        # Handle version compatibility issues gracefully (e.g., proxies parameter)
        if "proxies" in str(e) or "unexpected keyword" in str(e):
            logger.warning(f"OpenAI client initialization issue (version compatibility): {e}")
            logger.warning("This may be due to httpx/OpenAI version mismatch. Client will be unavailable.")
            return None
        else:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        return None

# Legacy export for backward compatibility (but client is now lazy)
# This allows existing code to import 'client' and use it as before
# The proxy lazy-loads the actual client on first attribute access
class _ClientProxy:
    """Proxy for OpenAI client that lazy-loads on access - STEP 4.5 compliance"""
    _actual_client = None
    
    def _get_client(self):
        """Get the actual client, lazy-loading if needed"""
        if self._actual_client is None:
            self._actual_client = get_openai_client()
        return self._actual_client
    
    def __getattr__(self, name):
        # Only called when attribute doesn't exist on proxy itself
        actual_client = self._get_client()
        if actual_client is None:
            # Return None for any attribute access when client is unavailable
            return None
        return getattr(actual_client, name)
    
    def __bool__(self):
        return self._get_client() is not None
    
    def __repr__(self):
        actual_client = self._get_client()
        if actual_client is None:
            return "None"
        return repr(actual_client)

# Export client as a proxy that lazy-loads
# This ensures no OpenAI client initialization at import time (STEP 4.5 compliance)
# Functions that use the client have been updated to use get_openai_client() directly
# The proxy is maintained for backward compatibility with imports
client = _ClientProxy()


def detect_objects(image_path):
    yolo_model = get_yolo_model()  # Lazy load
    res = yolo_model(image_path)
    names = res[0].names
    objects = []
    for b in res[0].boxes:
        cls_idx = int(b.cls.item())
        objects.append(names.get(cls_idx, str(cls_idx)))
    return objects if objects else ["No objects detected"]

def get_clip_description(image_path):
    """
    Generate a semantic description and genre hint using CLIP model.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Caption candidates
    candidate_captions = [
        # Portrait
        "A cinematic portrait in dramatic lighting", "A candid photo of a person lost in thought",
        "A joyful person laughing in soft light", "A melancholic person sitting alone",
        "A romantic close-up of two people", "A fashion portrait with striking style",
        "A vintage style portrait with film aesthetics",

        # Street / Urban
        "A quiet street at night under neon lights", "A chaotic urban scene full of motion",
        "A street musician playing passionately", "A group of people crossing a busy street",
        "A solitary figure walking through the rain",

        # Landscape / Nature
        "A misty mountain landscape at dawn", "A sunset over a calm lake with reflections",
        "A dramatic stormy sky over vast fields", "A dense forest with sunlight filtering through leaves",
        "A snowy landscape evoking silence and stillness", "A desert scene with strong shadows and patterns",

        # Conceptual / Abstract
        "A surreal photo blending multiple realities", "An abstract composition of geometric shapes",
        "A dreamy photo with pastel tones and blur", "A minimalist photo with negative space",
        "A colorful light painting in long exposure",

        # Wildlife / Animals
        "A close-up of a wild animal in its habitat", "A bird soaring freely in the sky",
        "A pet looking curiously at the camera", "A herd of animals moving dynamically",

        # Still Life
        "A carefully arranged flat lay with balanced composition", "A product shot with clean background and bold lighting",
        "A rustic table setting with natural light",

        # Emotional / Documentary
        "A powerful protest captured mid-action", "A tender moment between parent and child",
        "A candid emotional embrace", "A person staring out the window thoughtfully",

        # Architecture
        "A grand architectural facade with symmetrical design", "An interior space bathed in natural light",
        "A staircase captured with dramatic perspective",

        # Experimental
        "A photo with glitch and digital artifacts", "A double exposure merging city and nature",
        "A photo with intentional motion blur conveying speed",

        # Light-Based
        "A soft and dreamy scene bathed in golden hour light", "A cold and detached scene in blue tones",
        "A harshly lit photo creating strong shadows", "A foggy and mysterious environment",
        "A backlit subject creating a glowing silhouette",

        # Genre Tags
        "A street photography shot capturing the decisive moment", "A landscape photo showing nature's grandeur",
        "A portrait that conveys deep emotion", "A fashion photograph emphasizing style and attitude",
        "An abstract photo focusing on colors and shapes", "An intentionally blurred artistic expression",
        "A photo capturing nothingness and emptiness, purely abstract", "A raw and gritty lo-fi aesthetic shot"
    ]

    # CLIP process - lazy load models
    clip_model, clip_processor, device = get_clip_model()
    inputs = clip_processor(text=candidate_captions, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best_idx = int(probs.argmax().item() if hasattr(probs.argmax(), "item") else probs.argmax())
    best_caption = candidate_captions[best_idx]


    # Tag extraction
    caption_lower = best_caption.lower()
    tags = []

    genre_map = {
        "portrait": "Portrait", "street": "Street", "urban": "Street",
        "landscape": "Landscape", "nature": "Landscape",
        "abstract": "Abstract", "conceptual": "Abstract",
        "fashion": "Fashion", "animal": "Wildlife", "wild": "Wildlife",
        "protest": "Documentary", "documentary": "Documentary"
    }

    mood_map = {
        "dream": "Dreamy", "fog": "Dreamy", "blur": "Dreamy",
        "dramatic": "Moody", "dark": "Moody", "moody": "Moody",
        "soft": "Soft", "warm": "Soft",
        "chaotic": "Chaotic", "busy": "Chaotic",
        "clean": "Minimal", "minimalist": "Minimal"
    }

    for keyword, genre in genre_map.items():
        if keyword in caption_lower:
            tags.append(genre)

    for keyword, mood in mood_map.items():
        if keyword in caption_lower:
            tags.append(mood)

    genre_hint = tags[0] if tags else "General"

    return {
        "caption": best_caption,
        "tags": tags,
        "genre_hint": genre_hint
    }


def get_clip_inventory(image_path):
    """
    Generate a descriptive inventory of visible elements using CLIP model.
    
    Uses inventory-style prompt to produce nouns and attributes, not storytelling.
    This is used for semantic anchor generation (multi-signal consensus).
    
    Returns:
        List of strings (nouns/attributes describing visible elements)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Inventory-style candidate descriptions (nouns, attributes, no inference)
    inventory_candidates = [
        # Architecture types
        "religious architecture", "mosque", "cathedral", "temple", "church",
        "building", "structure", "tower", "dome", "minaret", "spire",
        "architectural facade", "interior space", "staircase", "archway",
        
        # Time of day / lighting conditions
        "daytime", "night", "dawn", "dusk", "sunset", "sunrise",
        "artificial lighting", "natural light", "neon lights", "street lights",
        
        # Atmospheric conditions
        "fog", "mist", "haze", "rain", "snow", "clear sky", "cloudy",
        "smoke", "dust", "atmospheric",
        
        # Structural elements
        "wall", "door", "window", "roof", "column", "arch", "balcony",
        "fence", "gate", "bridge", "pathway",
        
        # Human presence
        "person", "people", "crowd", "human figure", "silhouette",
        "no people", "empty", "unoccupied",
        
        # Scale indicators
        "large structure", "monumental", "tall building", "small building",
        "intimate space", "vast landscape", "close-up", "wide view",
        
        # Natural elements
        "tree", "palm tree", "grass", "water", "lake", "mountain", "sky",
        "cloud", "sun", "moon", "star",
        
        # Urban elements
        "street", "road", "vehicle", "car", "sign", "billboard", "neon sign"
    ]
    
    # CLIP process - lazy load models
    clip_model, clip_processor, device = get_clip_model()
    inputs = clip_processor(text=inventory_candidates, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Get top 10 most relevant inventory items (not just the best one)
    top_k = 10
    top_indices = probs.topk(top_k).indices[0].cpu().tolist()
    inventory_items = [inventory_candidates[idx] for idx in top_indices if probs[0][idx].item() > 0.05]  # Threshold: 5% confidence
    
    return inventory_items


def generate_semantic_anchors(clip_inventory, clip_tags, clip_caption, yolo_objects, composition_data):
    """
    Generate semantic anchors from multiple signals using consensus logic.
    
    Semantic Anchors are high-confidence, low-risk labels derived from multiple signals.
    They are not facts — they are permissions. Only keys that pass thresholds are included.
    
    Args:
        clip_inventory: List of strings from get_clip_inventory()
        clip_tags: List of strings from get_clip_description() tags
        clip_caption: String from get_clip_description() caption
        yolo_objects: List of strings from YOLO object detection
        composition_data: Dict with symmetry, subject_size, etc.
    
    Returns:
        Dict with sparse semantic anchors (only keys that meet thresholds)
        Missing key = not permitted, present key = safe to reference
    """
    anchors = {}
    
    # Normalize inputs to lowercase for matching
    inventory_lower = [item.lower() for item in (clip_inventory or [])]
    tags_lower = [tag.lower() for tag in (clip_tags or [])]
    caption_lower = (clip_caption or "").lower()
    yolo_lower = [obj.lower() for obj in (yolo_objects or [])]
    
    all_text = " ".join(inventory_lower + tags_lower + [caption_lower] + yolo_lower).lower()
    
    # Keyword sets for matching
    ARCHITECTURE_TERMS = ["mosque", "cathedral", "temple", "church", "religious architecture", 
                          "building", "structure", "tower", "dome", "minaret", "spire",
                          "architectural facade", "interior space"]
    STRUCTURE_TERMS = ["dome", "minaret", "tower", "spire", "arch", "column", "wall", 
                      "building", "structure", "facade"]
    TIME_TERMS = ["night", "daytime", "dawn", "dusk", "sunset", "sunrise"]
    ATMOSPHERE_TERMS = ["fog", "mist", "haze", "rain", "snow", "atmospheric", "smoke"]
    LIGHTING_TERMS = ["artificial lighting", "neon lights", "street lights", "illuminated"]
    HUMAN_TERMS = ["person", "people", "crowd", "human figure", "silhouette"]
    NO_HUMAN_TERMS = ["no people", "empty", "unoccupied"]
    SCALE_TERMS = ["monumental", "large structure", "tall building", "vast"]
    INTIMATE_TERMS = ["intimate space", "small building", "close-up"]
    
    # === SCENE_TYPE ===
    # Requires: 2+ signals (CLIP inventory + CLIP tags, or CLIP + composition)
    architecture_signals = sum(1 for term in ARCHITECTURE_TERMS if term in all_text)
    time_signals = sum(1 for term in TIME_TERMS if term in all_text)
    
    if architecture_signals >= 1 and time_signals >= 1:
        # Build scene type string (canonicalized)
        arch_type = None
        for term in ["mosque", "cathedral", "temple", "church"]:
            if term in all_text:
                arch_type = term
                break
        if not arch_type:
            arch_type = "religious architecture" if any(t in all_text for t in ["religious", "mosque", "cathedral", "temple", "church"]) else "architecture"
        
        time_type = None
        for term in TIME_TERMS:
            if term in all_text:
                time_type = term
                break
        
        if arch_type and time_type:
            anchors["scene_type"] = f"{arch_type} at {time_type}"
        elif arch_type:
            anchors["scene_type"] = arch_type
    elif architecture_signals >= 2:  # Architecture mentioned multiple times
        for term in ["mosque", "cathedral", "temple", "church"]:
            if term in all_text:
                anchors["scene_type"] = term
                break
        if "scene_type" not in anchors:
            anchors["scene_type"] = "architecture"
    
    # === STRUCTURE_ELEMENTS ===
    # Requires: 1+ signal (CLIP tags or CLIP inventory)
    structure_found = []
    for term in STRUCTURE_TERMS:
        if term in all_text:
            structure_found.append(term)
    
    if structure_found:
        # Canonicalize: remove duplicates, sort
        structure_found = sorted(list(set(structure_found)))
        anchors["structure_elements"] = structure_found[:5]  # Limit to top 5
    
    # === HUMAN_PRESENCE ===
    # Requires: 1 signal (YOLO or CLIP)
    human_detected = any(term in all_text for term in HUMAN_TERMS)
    no_human_detected = any(term in all_text for term in NO_HUMAN_TERMS)
    
    if human_detected:
        anchors["human_presence"] = "present"
    elif no_human_detected or (not human_detected and len(yolo_lower) > 0 and "person" not in " ".join(yolo_lower)):
        anchors["human_presence"] = "none detected"
    
    # === ATMOSPHERE ===
    # Requires: 1+ signal (CLIP tags or technical analysis)
    atmosphere_found = []
    for term in ATMOSPHERE_TERMS:
        if term in all_text:
            atmosphere_found.append(term)
    
    lighting_found = []
    for term in LIGHTING_TERMS:
        if term in all_text:
            lighting_found.append(term)
    
    if atmosphere_found or lighting_found:
        combined = atmosphere_found + lighting_found
        anchors["atmosphere"] = sorted(list(set(combined)))[:4]  # Limit to top 4
    
    # === SCALE ===
    # Requires: 1 signal (composition analysis or YOLO "tower"/"building")
    scale_signals = sum(1 for term in SCALE_TERMS if term in all_text)
    intimate_signals = sum(1 for term in INTIMATE_TERMS if term in all_text)
    
    # Also check composition data
    subject_size = composition_data.get("subject_size", "") if composition_data else ""
    subject_size_lower = subject_size.lower() if subject_size else ""
    
    if scale_signals >= 1 or "large" in subject_size_lower or "extreme" in subject_size_lower:
        anchors["scale"] = "monumental"
    elif intimate_signals >= 1 or "small" in subject_size_lower or "tiny" in subject_size_lower:
        anchors["scale"] = "intimate"
    elif any(term in yolo_lower for term in ["tower", "building", "structure"]) and len(yolo_lower) > 0:
        # YOLO detected large structures
        anchors["scale"] = "monumental"
    
    return anchors


def analyze_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    hex_colors = ['#%02x%02x%02x' % (c[0], c[1], c[2]) for c in colors]
    
    avg_color = np.mean(colors, axis=0)
    mood = "warm" if avg_color[0] > avg_color[2] else "cool"

    return {"palette": hex_colors, "mood": mood}


def predict_nima_score(model, img_path):
    if model is None or not NIMA_AVAILABLE:
        return {"mean_score": None, "distribution": {}}
    try:
        # Lazy import again (no globals)
        _, _, _, _, preprocess_input, keras_image = _import_tf_keras()
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img = keras_image.img_to_array(img) / 255.0
        preds = model.predict(np.expand_dims(img, axis=0))[0]
        mean_score = sum((i + 1) * float(p) for i, p in enumerate(preds))
        return {
            "mean_score": round(mean_score, 2),
            "distribution": {str(i + 1): float(f"{p:.4f}") for i, p in enumerate(preds)}
        }
    except Exception:
        return {"mean_score": None, "distribution": {}}
    

# ========================================================
# FILE HASH CACHING (Canonical Schema Phase II)
# ========================================================

def compute_file_hash(file_path: str) -> str:
    """
    Computes SHA-256 hash of a file for caching purposes.
    Returns hex digest string.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute file hash: {e}", exc_info=True)
        return ""


def get_cached_analysis(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves cached analysis result for a given file hash.
    Returns None if cache miss or error.
    """
    if not file_hash:
        return None
    
    cache_path = os.path.join(ANALYSIS_CACHE_DIR, f"{file_hash}.json")
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cached_result = json.load(f)
        
        # Validate schema before returning
        if validate_schema(cached_result):
            logger.info(f"Cache hit for hash: {file_hash[:8]}...")
            return cached_result
        else:
            logger.warning(f"Cached result for {file_hash[:8]}... failed schema validation, ignoring")
            return None
    except Exception as e:
        logger.error(f"Failed to read cache for {file_hash[:8]}...: {e}", exc_info=True)
        return None


def save_cached_analysis(file_hash: str, result: Dict[str, Any]) -> bool:
    """
    Saves analysis result to cache using file hash as key.
    Returns True on success, False on error.
    """
    if not file_hash:
        return False
    
    cache_path = os.path.join(ANALYSIS_CACHE_DIR, f"{file_hash}.json")
    
    try:
        ensure_directories()
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Cached analysis for hash: {file_hash[:8]}...")
        return True
    except Exception as e:
        logger.error(f"Failed to save cache for {file_hash[:8]}...: {e}", exc_info=True)
        return False


# framed/analysis/vision.py

def run_full_analysis(image_path, photo_id: str = "", filename: str = ""):
    """
    Orchestrates the full analysis pipeline.
    Phase II: Returns canonical schema result.
    STEP 4.6: Graceful degradation - partial results returned even if some steps fail.
    """
    try:
        # Ensure directories exist before starting analysis
        ensure_directories()
        
        # Use the comprehensive analyze_image function (now returns canonical schema)
        analysis_result = analyze_image(image_path, photo_id=photo_id, filename=filename)
        
        # Validate schema before proceeding
        if not validate_schema(analysis_result):
            logger.error("Analysis result failed schema validation")
            analysis_result["errors"]["schema_validation"] = "Result does not conform to canonical schema"
        
        # Update echo memory only if we have valid results (even with partial errors)
        # Check for critical errors, not just "error" key
        critical_errors = analysis_result.get("errors", {}).get("critical") or analysis_result.get("errors", {}).get("image_load")
        if not critical_errors:
            # Update echo memory with the new analysis
            # Note: analysis_result may contain errors dict for partial failures
            update_echo_memory(analysis_result)
        
        return analysis_result
    except Exception as e:
        logger.error(f"Fatal error in full analysis pipeline: {e}", exc_info=True)
        # Return canonical schema with error
        result = create_empty_analysis_result()
        result["errors"]["pipeline"] = str(e)
        return result


def analyze_lines_and_symmetry(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)

    # Analyze line pattern
    if lines is not None:
        num_lines = len(lines)

        if num_lines > 50:
            pattern = "Strong presence of lines, potentially chaotic or very structured"
        elif num_lines > 20:
            pattern = "Prominent lines, likely guiding or framing elements"
        elif num_lines > 5:
            pattern = "Some lines present, subtle guidance"
        else:
            pattern = "Minimal lines, soft or natural composition"
    else:
        pattern = "No significant lines, soft or abstract composition"

    # Further interpret the style
    if "chaotic" in pattern:
        style = "Chaotic and irregular"
    elif "guiding" in pattern or "framing" in pattern:
        style = "Intentional leading lines"
    elif "soft" in pattern or "natural" in pattern:
        style = "Organic and natural"
    else:
        style = "Geometric and structured"

    height, width = gray.shape
    if width > 1:
        left_half = gray[:, :width // 2]
        right_half = gray[:, width // 2:]
        # Ensure right_half is flipped correctly
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize to match dimensions if width is odd
        h1, w1 = left_half.shape
        h2, w2 = right_half_flipped.shape
        if w1!= w2:
            min_w = min(w1, w2)
            left_half = left_half[:, :min_w]
            right_half_flipped = right_half_flipped[:, :min_w]

        diff = cv2.absdiff(left_half, right_half_flipped)
        symmetry_score = np.mean(diff)
        
        symmetry_desc = "Asymmetrical"
        if symmetry_score < 10:
            symmetry_desc = "Highly symmetrical"
        elif symmetry_score < 30:
            symmetry_desc = "Largely symmetrical with some minor differences"
        elif symmetry_score < 60:
            symmetry_desc = "Noticeably asymmetrical"
    else:
        symmetry_desc = "Image too narrow to analyze symmetry"

    return {
        "line_pattern": pattern,
        "line_style": style,
        "symmetry": symmetry_desc
    }


def analyze_color_harmony(image_path):
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)

    r, g, b = dominant_color
    diff_rg = abs(r - g)
    diff_gb = abs(g - b)
    diff_rb = abs(r - b)

    if diff_rg < 15 and diff_gb < 15:
        harmony = "Monochromatic — serene and unified, a single mood dominates"
    elif diff_rb > 100 and diff_gb < 50:
        harmony = "Complementary — bold contrast, vibrant and eye-catching"
    elif 50 < diff_rb < 100 or 50 < diff_gb < 100:
        harmony = "Split Complementary — dynamic yet harmonious, subtle tension"
    elif diff_rg < 50 and diff_gb < 50 and diff_rb < 50:
        harmony = "Analogous — gentle and smooth, flowing colors in harmony"
    elif diff_rb > 75 and diff_gb > 75:
        harmony = "Triadic — playful and balanced, rich color diversity"
    else:
        harmony = "Experimental or Undefined — unconventional and artistic"

    return {
        "dominant_color": f"#{r:02x}{g:02x}{b:02x}",
        "harmony": harmony
    }


def analyze_lighting_direction(image_path):
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    _, _, _, max_loc = cv2.minMaxLoc(gray)
    w = gray.shape[1]

    direction = "light from left" if max_loc[0] < w * 0.33 else "light from right" if max_loc[0] > w * 0.66 else "light from center/top"
    return {"direction": direction}

def analyze_tonal_range(image_path):
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    black, mid, white = np.sum(hist[:50]), np.sum(hist[50:200]), np.sum(hist[200:])
    tone = "high key (bright)" if white > black and white > mid else "low key (dark)" if black > white and black > mid else "balanced"
    
    return {"tonal_range": tone}

def analyze_background_clutter(image_path):
    yolo_model = get_yolo_model()  # Lazy load
    results = yolo_model(image_path)
    detections = results[0].boxes.data.cpu().numpy()
    num_objects = len(detections)

    # --- Define clutter level ---
    if num_objects == 0:
        clutter_level = "Minimal (Clean background, no distractions)"
        impact = "Allows full focus on subject"
    elif num_objects <= 2:
        clutter_level = "Low clutter (Simple and minimal elements)"
        impact = "Mostly clean, slight environmental context"
    elif num_objects <= 5:
        clutter_level = "Moderate clutter (Some background elements present)"
        impact = "Adds context but may compete for attention"
    elif num_objects <= 10:
        clutter_level = "High clutter (Many background elements)"
        impact = "Background may distract or clash with subject"
    else:
        clutter_level = "Chaotic (Very busy and potentially messy background)"
        impact = "Strong distraction, risks overwhelming subject"

    # --- Contextual interpretation ---
    if num_objects == 0:
        narrative = "Minimalist aesthetic, subject isolation"
    elif num_objects <= 2:
        narrative = "Controlled background, subject still dominates"
    elif num_objects <= 5:
        narrative = "Balanced scene, some storytelling elements in background"
    elif num_objects <= 10:
        narrative = "Visually active scene, background competes with subject"
    else:
        narrative = "Chaotic environment, viewer attention is fragmented"

    return {
        "num_objects": num_objects,
        "clutter_level": clutter_level,
        "impact": impact,
        "narrative": narrative
    }

def analyze_subject_emotion(image_path):
    # If DeepFace available and enabled, try it first
    if DeepFace is not None:
        try:
            res = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
            emo = res[0]['dominant_emotion'] if isinstance(res, list) else res['dominant_emotion']
            return {"subject_type": "human subject", "emotion": emo, "age": None, "gender": "Unknown"}
        except Exception as e:
            print("[DeepFace] error, falling back to CLIP:", e)
    # Fallback: CLIP-based emotion (existing implementation)
    return analyze_subject_emotion_clip(image_path)

def analyze_subject_emotion_clip(image_path):
    image = Image.open(image_path).convert("RGB")
    
    emotion_captions = [
        # Primary Human Emotions
        "A person smiling with joy",
        "A person laughing intensely",
        "A person looking sad or melancholic",
        "A person crying or deeply upset",
        "A person looking angry or frustrated",
        "A person appearing calm and serene",
        "A person looking lost in thought or introspective",
        "A person showing fear or anxiety",
        "A person expressing love or tenderness",
        "A person appearing proud and confident",

        # Non-Human / Ambient Moods
        "An empty and lonely landscape",
        "A peaceful and quiet environment",
        "A chaotic and energetic scene",
        "A surreal and dreamy conceptual scene",
        "A nostalgic vintage-style photo",
        "A dark and tense atmosphere",
        "A light and playful composition",
        "An abstract and ambiguous image"
    ]

    # Lazy load CLIP models
    clip_model, clip_processor, device = get_clip_model()
    inputs = clip_processor(text=emotion_captions, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = int(probs.argmax().item() if hasattr(probs.argmax(), "item") else probs.argmax())
    best_caption = emotion_captions[best_idx]


    human_detected = any(word in best_caption.lower() for word in ["person", "human"])

    subject_type = "human subject" if human_detected else "non-human / abstract"

    return {
        "subject_type": subject_type,
        "emotion": best_caption
    }

from collections import Counter

def detect_objects_and_framing(image_path, confidence_threshold=0.3):
    """
    Combines object detection and subject framing into a unified visual scene analyzer.
    """

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    yolo_model = get_yolo_model()  # Lazy load
    results = yolo_model(image_path)
    detections = results[0].boxes.data.cpu().numpy() if results and results[0].boxes is not None else []

    if len(detections) == 0:
        return {
            "objects": ["No objects detected"],
            "object_narrative": "The frame feels intentionally empty or abstract, lacking identifiable subjects.",
            "subject_position": "Undefined",
            "subject_size": "N/A",
            "framing_description": "No dominant subject identified",
            "spatial_interpretation": "Open, undefined space — evokes a sense of emptiness or abstraction."
        }

    labels = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_names = results[0].names

    detected_objects = [
        class_names[int(lbl)]
        for lbl, conf in zip(labels, confidences)
        if conf >= confidence_threshold
    ]

    if not detected_objects:
        return {
            "objects": ["No confident objects detected"],
            "object_narrative": "Subjects were too subtle or ambiguous to identify clearly.",
            "subject_position": "Unknown",
            "subject_size": "N/A",
            "framing_description": "Undefined composition",
            "spatial_interpretation": "Visual ambiguity leaves the narrative open to interpretation."
        }

    # Object count
    object_counts = Counter(detected_objects)
    summarized_objects = [f"{count}x {obj}" if count > 1 else obj for obj, count in object_counts.items()]

    # Narrative description
    if len(summarized_objects) == 1:
        narrative = f"The scene focuses on a singular subject: {summarized_objects[0]}."
    elif len(summarized_objects) <= 3:
        narrative = f"The composition includes multiple distinct elements: {', '.join(summarized_objects)}."
    elif len(summarized_objects) <= 6:
        narrative = f"A busy frame, rich with visual variety — objects like {', '.join(summarized_objects)} shape the environment."
    else:
        narrative = "A densely populated frame teeming with diverse objects, evoking complexity or visual chaos."

    # === SUBJECT FRAMING ===
    # Find largest bounding box (assumed main subject)
    largest = max(detections, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    x1, y1, x2, y2 = largest[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    size_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)

    # Position
    horizontal = "Left" if center_x < w * 0.33 else "Center" if center_x < w * 0.66 else "Right"
    vertical = "Top" if center_y < h * 0.33 else "Middle" if center_y < h * 0.66 else "Bottom"
    position = f"{vertical} {horizontal}"

    # Size Descriptor
    if size_ratio > 0.5:
        size_desc = "Extreme close-up, fills the frame"
    elif size_ratio > 0.2:
        size_desc = "Medium subject, clearly visible"
    elif size_ratio > 0.05:
        size_desc = "Small in frame, environmental subject"
    else:
        size_desc = "Tiny or distant subject"

    # Framing Style
    if horizontal == "Center" and vertical == "Middle":
        framing_desc = "Centered composition with clear subject emphasis"
    elif horizontal == "Center":
        framing_desc = f"Vertically {vertical.lower()}-oriented composition"
    elif vertical == "Middle":
        framing_desc = f"Horizontally {horizontal.lower()} placement"
    else:
        framing_desc = f"Asymmetrical composition ({position.lower()})"

    # Poetic Interpretation
    if size_ratio > 0.5:
        spatial_interpretation = "The subject overwhelms the frame, demanding complete attention and intimacy."
    elif size_ratio > 0.2:
        spatial_interpretation = "There’s a careful balance between subject and space — the viewer is invited into the scene."
    elif size_ratio > 0.05:
        spatial_interpretation = "The subject seems small, evoking a sense of solitude or environmental context."
    else:
        spatial_interpretation = "The subject feels distant or lost, perhaps overwhelmed by the surrounding environment."

    return {
        "objects": summarized_objects,
        "object_narrative": narrative,
        "subject_position": position,
        "subject_size": size_desc,
        "framing_description": framing_desc,
        "spatial_interpretation": spatial_interpretation
    }

def infer_emotion(photo_data):
    brightness = photo_data.get("brightness", 0)
    contrast = photo_data.get("contrast", 0)
    sharpness = photo_data.get("sharpness", 0)
    color_mood = photo_data.get("color_mood", "neutral")
    tonal_range = photo_data.get("tonal_range", "balanced")
    # subject_emotion is typically a dict like {"subject_type": "...", "emotion": "..."}
    # but can be a string in older/partial results — normalize to text.
    _subject_emotion_val = photo_data.get("subject_emotion", "")
    if isinstance(_subject_emotion_val, dict):
        subject_emotion = str(_subject_emotion_val.get("emotion", "") or "")
    else:
        subject_emotion = str(_subject_emotion_val or "")
    clutter = photo_data.get("background_clutter", {}).get("clutter_level", "minimal clutter")
    symmetry = photo_data.get("symmetry", "moderate")

    emotion_tags = []

    # --- Brightness and Tonal Range ---
    if brightness < 40 or tonal_range == "low key (dark)":
        emotion_tags.append("moody")
    elif brightness > 180 or tonal_range == "high key (bright)":
        emotion_tags.append("uplifting and airy")

    # --- Contrast ---
    if contrast < 20:
        emotion_tags.append("soft and calm")
    elif contrast > 100:
        emotion_tags.append("bold and intense")

    # --- Sharpness ---
    if sharpness < 20:
        emotion_tags.append("dreamy or hazy")
    elif sharpness > 150:
        emotion_tags.append("razor sharp and clear")

    # --- Color Mood ---
    if color_mood == "warm":
        emotion_tags.append("warm and inviting")
    elif color_mood == "cool":
        emotion_tags.append("cold and distant")
    elif color_mood == "neutral":
        emotion_tags.append("quiet and subtle")

    # --- Subject Emotion ---
    subj_lower = subject_emotion.lower()
    if "sad" in subj_lower or "melancholic" in subj_lower:
        emotion_tags.append("melancholic")
    elif "happy" in subj_lower or "smiling" in subj_lower:
        emotion_tags.append("joyful")
    elif "serene" in subj_lower:
        emotion_tags.append("peaceful")

    # --- Background Clutter ---
    if "high clutter" in clutter:
        emotion_tags.append("chaotic or busy")
    elif "minimal clutter" in clutter:
        emotion_tags.append("clean and minimal")

    # --- Symmetry ---
    if symmetry == "high":
        emotion_tags.append("harmonious and balanced")
    elif symmetry == "low":
        emotion_tags.append("asymmetrical and dynamic")

    # --- Final Interpretation ---
    if not emotion_tags:
        mood = "neutral and open-ended"
    else:
        mood = ", ".join(emotion_tags)

    return {
        "emotional_mood": mood
    }



def interpret_visual_features(photo_data):
    interpretation = {}

    # --- Sharpness ---
    sharpness = photo_data.get("sharpness", 0)
    if sharpness < 10:
        interpretation["sharpness"] = "Extremely soft focus, painterly and dreamlike with lost edges"
    elif sharpness < 30:
        interpretation["sharpness"] = "Soft and gentle focus, subtle and ethereal details"
    elif sharpness < 80:
        interpretation["sharpness"] = "Moderately sharp, preserving natural texture and nuance"
    elif sharpness < 150:
        interpretation["sharpness"] = "Crisp and precise rendering, finely defined edges"
    else:
        interpretation["sharpness"] = "Hyper sharp and clinical, every detail exaggerated"

    # --- Brightness ---
    brightness = photo_data.get("brightness", 0)
    if brightness < 30:
        interpretation["brightness"] = "Deep and dark tones, enveloped in shadow and mystery"
    elif brightness < 70:
        interpretation["brightness"] = "Moody low key exposure, reserved and atmospheric"
    elif brightness < 150:
        interpretation["brightness"] = "Balanced natural light, even and true to life"
    elif brightness < 220:
        interpretation["brightness"] = "Bright and clean, glowing with luminous energy"
    else:
        interpretation["brightness"] = "Overexposed and harsh, blazing with intensity"

    # --- Contrast ---
    contrast = photo_data.get("contrast", 0)
    if contrast < 20:
        interpretation["contrast"] = "Flat tonal range, soft and faded like an old memory"
    elif contrast < 40:
        interpretation["contrast"] = "Gentle contrast, smooth gradients and muted impact"
    elif contrast < 80:
        interpretation["contrast"] = "Balanced contrast, vivid yet natural separation"
    elif contrast < 120:
        interpretation["contrast"] = "Punchy and bold contrast, striking and impactful"
    else:
        interpretation["contrast"] = "Extremely harsh contrast, intense and aggressive"

    # --- Lighting Direction ---
    lighting = photo_data.get("lighting_direction", "unknown")
    if lighting == "light from left":
        interpretation["lighting"] = "Side-lit from the left, sculpting dramatic contours"
    elif lighting == "light from right":
        interpretation["lighting"] = "Side-lit from the right, cinematic and directional"
    elif lighting == "light from center/top":
        interpretation["lighting"] = "Front/top lit, evenly illuminated and neutral"
    elif lighting == "light from back":
        interpretation["lighting"] = "Backlit, creating glowing silhouettes and depth"
    else:
        interpretation["lighting"] = "Ambient or undefined lighting, soft and diffuse"

    # --- Tonal Range ---
    tonal = photo_data.get("tonal_range", "balanced")
    if tonal == "high key (bright)":
        interpretation["tones"] = "High key tones, light and airy with delicate charm"
    elif tonal == "low key (dark)":
        interpretation["tones"] = "Low key, dominated by darkness and cinematic melancholy"
    else:
        interpretation["tones"] = "Balanced tones, harmonious and versatile"

    # --- Color Mood ---
    color_mood = photo_data.get("color_mood", "neutral")
    if color_mood == "warm":
        interpretation["color"] = "Warm hues, inviting and emotionally resonant"
    elif color_mood == "cool":
        interpretation["color"] = "Cool palette, calm, distant and tranquil"
    else:
        interpretation["color"] = "Neutral tones, understated and sophisticated"

    # --- Subject Presence ---
    objects = photo_data.get("objects", [])
    if not objects or "No objects detected" in objects:
        interpretation["subject"] = "Minimal or abstract, leaving space for interpretation"
    elif len(objects) == 1:
        interpretation["subject"] = f"Singular subject ({objects[0]}), central and commanding"
    elif len(objects) < 4:
        interpretation["subject"] = f"Multiple elements ({', '.join(objects)}), balanced interplay"
    else:
        interpretation["subject"] = f"Busy scene with many elements ({', '.join(objects)}), dynamic and layered"

    return interpretation




def detect_genre(photo_data):
    if isinstance(photo_data, str):
        import json
        photo_data = json.loads(photo_data)
    """
    Determine the photo genre and sub-genre using multiple cues.
    """

    # Safe extraction
    clip_info = photo_data.get("clip_description", {})
    clip_genre = clip_info.get("genre_hint", "General")

    subject_emotion = photo_data.get("subject_emotion", "unknown")
    subject_type = photo_data.get("subject_type", "unknown")
    mood = photo_data.get("emotional_mood", "neutral")
    tonal = photo_data.get("tonal_range", "balanced")
    color_mood = photo_data.get("color_mood", "neutral")
    lighting = photo_data.get("lighting_direction", "undefined")
    sharpness = photo_data.get("sharpness", 0)

    background_info = photo_data.get("background_clutter", {})
    clutter_level = background_info.get("clutter_level", "")

    # --- Primary Genre ---
    if clip_genre != "General":
        genre = clip_genre
    elif subject_type == "human subject":
        genre = "Portrait"
    elif subject_type == "non-human / abstract":
        genre = "Abstract / Conceptual"
    else:
        genre = "General"

    # --- Subgenre Logic ---
    subgenre = None

    if genre == "Street" or ("chaotic" in mood or "energetic" in mood):
        subgenre = "Candid / Action" if "dynamic" in mood or "motion" in mood else "Atmospheric Street"

    elif genre == "Portrait":
        if "romantic" in mood or "dreamy" in mood:
            subgenre = "Romantic / Dreamy"
        elif "melancholic" in mood or tonal == "low key (dark)":
            subgenre = "Dramatic / Cinematic"
        else:
            subgenre = "Classic Portrait"

    elif genre == "Landscape":
        if tonal == "high key (bright)":
            subgenre = "Bright / Airy"
        elif tonal == "low key (dark)":
            subgenre = "Moody / Cinematic"
        else:
            subgenre = "Balanced / Natural"

    elif genre == "Abstract / Conceptual":
        if "surreal" in mood:
            subgenre = "Surreal / Dreamlike"
        elif "minimal" in mood or clutter_level == "clean":
            subgenre = "Minimalist"
        else:
            subgenre = "Experimental / Artistic"

    elif genre == "Wildlife":
        if "dynamic" in mood or "action" in mood:
            subgenre = "Action Wildlife"
        else:
            subgenre = "Calm / Observational"

    if subgenre is None:
        subgenre = f"Classic {genre}" if genre != "General" else "General"

    return {
        "genre": genre,
        "subgenre": subgenre
    }





def analyze_image(path, photo_id: str = "", filename: str = ""):
    """
    Analyzes an image and returns results in the canonical schema format.
    
    Phase II: Canonical Schema - This function now returns a deterministic,
    structured result conforming to the schema defined in schema.py.
    
    Args:
        path: Path to the image file
        photo_id: Optional unique identifier for the photo
        filename: Optional original filename
        
    Returns:
        Dict conforming to canonical AnalysisResult schema
    """
    ensure_directories()
    logger.info(f"Analyzing image: {path}")
    
    # Compute file hash for caching
    file_hash = compute_file_hash(path)
    if not photo_id:
        photo_id = file_hash[:16] if file_hash else str(uuid.uuid4())
    
    # Check cache first
    cached_result = get_cached_analysis(file_hash)
    if cached_result:
        # Update metadata with current request info
        cached_result["metadata"]["photo_id"] = photo_id
        cached_result["metadata"]["filename"] = filename
        return cached_result
    
    # Initialize canonical schema
    result = create_empty_analysis_result()
    
    # Set metadata
    from datetime import datetime
    result["metadata"] = {
        "photo_id": photo_id,
        "filename": filename or os.path.basename(path),
        "file_hash": file_hash,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    try:
        # Load Image + Convert to Gray
        img = cv2.imread(path)
        if img is None:
            result["errors"]["image_load"] = "Could not load image"
            return result
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # === TECHNICAL ANALYSIS ===
        try:
            brightness = round(np.mean(gray), 2)
            contrast = round(gray.std(), 2)
            sharpness = round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)
            
            result["perception"]["technical"]["available"] = True
            result["perception"]["technical"]["brightness"] = brightness
            result["perception"]["technical"]["contrast"] = contrast
            result["perception"]["technical"]["sharpness"] = sharpness
        except Exception as e:
            logger.warning(f"Technical analysis failed: {e}")
            result["errors"]["technical"] = str(e)

        # === AI + SEMANTIC ANALYSIS ===
        # Wrap each analysis to prevent one failure from breaking the entire analysis
        def safe_analyze(func, *args, error_key=None, default=None, **kwargs):
            """Safely run an analysis function, returning default on error."""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                func_name = func.__name__ if hasattr(func, '__name__') else str(func)
                logger.warning(f"Analysis step '{func_name}' failed: {error_msg}")
                if error_key:
                    result["errors"][error_key] = error_msg
                return default if default is not None else {}
        
        # CLIP semantic analysis
        clip_data = safe_analyze(get_clip_description, path, error_key="clip", 
                                 default={"caption": None, "tags": [], "genre_hint": None})
        if clip_data.get("caption"):
            result["perception"]["semantics"]["available"] = True
            result["perception"]["semantics"]["caption"] = clip_data.get("caption")
            result["perception"]["semantics"]["tags"] = clip_data.get("tags", [])
            result["perception"]["semantics"]["genre_hint"] = clip_data.get("genre_hint")
            result["confidence"]["clip"] = True
        
        # CLIP inventory analysis (for semantic anchors)
        clip_inventory = safe_analyze(get_clip_inventory, path, error_key="clip_inventory",
                                     default=[])
        
        # NIMA aesthetic analysis
        nima_model = None
        try:
            nima_model = get_nima_model()
        except Exception as e:
            logger.warning(f"NIMA model load failed (non-fatal): {e}")
            result["errors"]["nima_load"] = str(e)
        
        nima_result = safe_analyze(predict_nima_score, nima_model, path, error_key="nima",
                                  default={"mean_score": None, "distribution": {}})
        if nima_result.get("mean_score") is not None:
            result["perception"]["aesthetics"]["available"] = True
            result["perception"]["aesthetics"]["mean_score"] = nima_result.get("mean_score")
            result["perception"]["aesthetics"]["distribution"] = nima_result.get("distribution", {})
            result["confidence"]["nima"] = True
        
        # Color analysis
        color_analysis = safe_analyze(analyze_color, path, error_key="color",
                                     default={"palette": [], "mood": None})
        if color_analysis.get("palette"):
            result["perception"]["color"]["available"] = True
            result["perception"]["color"]["palette"] = color_analysis.get("palette", [])
            result["perception"]["color"]["mood"] = color_analysis.get("mood")
        
        color_harmony = safe_analyze(analyze_color_harmony, path, error_key="color_harmony",
                                     default={"dominant_color": None, "harmony": None})
        if color_harmony.get("dominant_color"):
            result["perception"]["color"]["harmony"]["dominant_color"] = color_harmony.get("dominant_color")
            result["perception"]["color"]["harmony"]["harmony_type"] = color_harmony.get("harmony")
        
        # YOLO object detection
        object_data = safe_analyze(detect_objects_and_framing, path, error_key="objects",
                                   default={"objects": [], "object_narrative": None, 
                                           "subject_position": None, "subject_size": None,
                                           "framing_description": None, "spatial_interpretation": None})
        if object_data.get("objects"):
            result["perception"]["composition"]["available"] = True
            result["perception"]["composition"]["subject_framing"] = {
                "position": object_data.get("subject_position"),
                "size": object_data.get("subject_size"),
                "style": object_data.get("framing_description"),
                "interpretation": object_data.get("spatial_interpretation")
            }
            result["confidence"]["yolo"] = True
        
        # Lines and symmetry
        lines_symmetry = safe_analyze(analyze_lines_and_symmetry, path, error_key="lines_symmetry",
                                     default={"line_pattern": None, "line_style": None, "symmetry": None})
        if lines_symmetry.get("line_pattern"):
            result["perception"]["composition"]["line_pattern"] = lines_symmetry.get("line_pattern")
            result["perception"]["composition"]["line_style"] = lines_symmetry.get("line_style")
            result["perception"]["composition"]["symmetry"] = lines_symmetry.get("symmetry")
        
        # Lighting
        lighting_direction = safe_analyze(analyze_lighting_direction, path, error_key="lighting",
                                         default={"direction": None})
        if lighting_direction.get("direction"):
            result["perception"]["lighting"]["available"] = True
            result["perception"]["lighting"]["direction"] = lighting_direction.get("direction")
        
        # Tonal range
        tonal_range = safe_analyze(analyze_tonal_range, path, error_key="tonal_range",
                                   default={"tonal_range": None})
        if tonal_range.get("tonal_range"):
            result["perception"]["lighting"]["quality"] = tonal_range.get("tonal_range")
        
        # Subject emotion
        subject_emotion = safe_analyze(analyze_subject_emotion, path, error_key="emotion",
                                      default={"subject_type": None, "emotion": None})
        if subject_emotion.get("subject_type"):
            result["perception"]["emotion"]["available"] = True
            result["perception"]["emotion"]["subject_type"] = subject_emotion.get("subject_type")
            result["perception"]["emotion"]["emotion"] = subject_emotion.get("emotion")
        
        # DeepFace (optional, toggleable)
        # Note: DeepFace is disabled by default, can be enabled via DEEPFACE_ENABLE env var
        # This is a placeholder for future implementation
        result["confidence"]["deepface"] = False
        
        # === DERIVED FIELDS ===
        # Build legacy-style dict for derived field functions (temporary compatibility)
        legacy_dict = {
            "brightness": result["perception"]["technical"].get("brightness"),
            "contrast": result["perception"]["technical"].get("contrast"),
            "sharpness": result["perception"]["technical"].get("sharpness"),
            "clip_description": clip_data,
            "color_palette": result["perception"]["color"].get("palette", []),
            "color_mood": result["perception"]["color"].get("mood"),
            "lighting_direction": result["perception"]["lighting"].get("direction"),
            "tonal_range": result["perception"]["lighting"].get("quality"),
            "subject_emotion": subject_emotion,
            "line_pattern": result["perception"]["composition"].get("line_pattern"),
            "line_style": result["perception"]["composition"].get("line_style"),
            "symmetry": result["perception"]["composition"].get("symmetry"),
            "objects": object_data.get("objects", []),
            "subject_framing": result["perception"]["composition"].get("subject_framing", {}),
            "background_clutter": safe_analyze(analyze_background_clutter, path, error_key="clutter",
                                               default={"clutter_level": None})
        }
        
        # Visual interpretation
        visual_interp = safe_analyze(interpret_visual_features, legacy_dict, error_key="visual_interpretation", default={})
        if visual_interp:
            result["derived"]["visual_interpretation"] = visual_interp
        
        # Emotional mood inference
        emotion_result = safe_analyze(infer_emotion, legacy_dict, error_key="emotion_inference",
                                     default={"emotional_mood": None})
        if emotion_result.get("emotional_mood"):
            result["derived"]["emotional_mood"] = emotion_result.get("emotional_mood")
        
        # Genre detection
        genre_info = safe_analyze(detect_genre, legacy_dict, error_key="genre_detection",
                                 default={"genre": None, "subgenre": None})
        if genre_info.get("genre"):
            result["derived"]["genre"]["genre"] = genre_info.get("genre")
            result["derived"]["genre"]["subgenre"] = genre_info.get("subgenre")
        
        # === SEMANTIC ANCHORS GENERATION ===
        # Generate semantic anchors from multiple signals (sparse, only high-confidence)
        try:
            composition_for_anchors = {
                "symmetry": result["perception"]["composition"].get("symmetry"),
                "subject_size": result["perception"]["composition"].get("subject_framing", {}).get("size")
            }
            semantic_anchors = generate_semantic_anchors(
                clip_inventory=clip_inventory if isinstance(clip_inventory, list) else [],
                clip_tags=clip_data.get("tags", []),
                clip_caption=clip_data.get("caption"),
                yolo_objects=object_data.get("objects", []),
                composition_data=composition_for_anchors
            )
            # Only add anchors if any were generated (sparse by default)
            if semantic_anchors:
                result["semantic_anchors"] = semantic_anchors
        except Exception as e:
            logger.warning(f"Semantic anchor generation failed (non-fatal): {e}")
            # Don't add error - anchors are optional
        
        # Cache the result
        if file_hash:
            save_cached_analysis(file_hash, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Critical error in analyze_image: {e}", exc_info=True)
        result["errors"]["critical"] = str(e)
        return result


# ========================================================
# 🧠 ECHO MEMORY FUNCTIONS
# ========================================================

def load_echo_memory():
    """
    Loads the persistent ECHO memory from disk.
    If corrupted or empty, returns empty list safely.
    """
    if os.path.exists(ECHO_MEMORY_PATH):
        try:
            with open(ECHO_MEMORY_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Warning: Corrupted echo_memory.json — resetting.")
            return []
    return []


def save_echo_memory(memory):
    """
    Saves the current ECHO memory to disk (up to 10 most recent).
    Converts non-serializable values (like np.float32) to standard Python types.
    """
    def convert(o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)  # fallback

    memory = memory[-10:]  # Keep only the last 10 entries
    with open(ECHO_MEMORY_PATH, 'w') as f:
        json.dump(memory, f, indent=2, default=convert)


def update_echo_memory(photo_data):
    """
    Adds a new photo's analysis to the ECHO memory.
    Keeps memory within last 10 items.
    """
    memory = load_echo_memory()
    memory.append(photo_data)
    save_echo_memory(memory)

def summarize_echo_memory(memory):
    """
    Prepares a poetic structured summary of the ECHO memory.
    This is what gets fed into the ask_echo() reflection engine.
    """
    summaries = []

    for idx, photo in enumerate(memory[-10:], start=1):
        genre = photo.get("genre", "Unknown")
        subgenre = photo.get("subgenre", "Unknown")
        mood = photo.get("emotional_mood", "Unclear")
        caption = photo.get("clip_description", {}).get("caption", "No caption")
        colors = photo.get("color_palette", [])
        tones = photo.get("tonal_range", "Unknown")
        subject = photo.get("objects", ["Unknown"])
        framing = photo.get("subject_framing", {}).get("interpretation", "")
        lighting = photo.get("lighting_direction", "Unknown")

        summary = f"""
📸 Photo {idx}:
• Genre: {genre} → {subgenre}
• Mood: {mood}
• Caption: "{caption}"
• Dominant Colors: {', '.join(colors)}
• Tonality: {tones}
• Subject(s): {', '.join(subject)}
• Framing: {framing}
• Lighting: {lighting}
"""
        summaries.append(summary.strip())

    return "\n\n".join(summaries)

# ========================================================
# 📖 GPT PHOTOGRAPHY MENTOR + CREATIVE COACH
# ========================================================
def _get_genre_pair(photo):
    g = photo.get("genre")
    if isinstance(g, dict):
        return g.get("genre", "General"), g.get("subgenre", "General")
    # string + separate subgenre field
    return g or "General", photo.get("subgenre", "General")

def generate_merged_critique(photo_data, visionary_mode="Balanced Mentor"):
    """
    Phase III-A: Evidence-Driven Critique Engine
    
    Extracts verified observations from canonical schema and constructs
    a philosophical critique that interprets facts, not guesses.
    
    The prompt receives ONLY observed facts. Interpretation happens inside the prompt voice.
    """
    # Ensure logger is available (safeguard for edge cases)
    # Fix: Prevent NameError if logger is not in scope
    try:
        _logger = logger
    except NameError:
        import logging
        _logger = logging.getLogger(__name__)
    
    # Check if this is canonical schema or legacy format
    is_canonical = "perception" in photo_data and "metadata" in photo_data
    
    if is_canonical:
        # Extract from canonical schema
        perception = photo_data.get("perception", {})
        derived = photo_data.get("derived", {})
        
        technical = perception.get("technical", {})
        composition = perception.get("composition", {})
        color = perception.get("color", {})
        lighting = perception.get("lighting", {})
        semantics = perception.get("semantics", {})
        genre = derived.get("genre", {})
        
        # Extract verified observations (only if available)
        brightness = technical.get("brightness") if technical.get("available") else None
        contrast = technical.get("contrast") if technical.get("available") else None
        sharpness = technical.get("sharpness") if technical.get("available") else None
        
        symmetry = composition.get("symmetry") if composition.get("available") else None
        subject_framing = composition.get("subject_framing", {})
        subject_position = subject_framing.get("position")
        subject_size = subject_framing.get("size")
        framing_style = subject_framing.get("style")
        
        color_mood = color.get("mood") if color.get("available") else None
        color_harmony = color.get("harmony", {}).get("harmony_type")
        
        lighting_direction = lighting.get("direction") if lighting.get("available") else None
        tonal_range = lighting.get("quality")
        
        clip_caption = semantics.get("caption") if semantics.get("available") else None
        genre_name = genre.get("genre")
        subgenre_name = genre.get("subgenre")
        
        emotional_mood = derived.get("emotional_mood")
        
        # Extract semantic anchors (if present)
        semantic_anchors = photo_data.get("semantic_anchors", {})
    else:
        # Legacy format fallback (for backward compatibility)
        technical = photo_data
        composition = photo_data
        color = photo_data
        lighting = photo_data
        semantics = photo_data.get("clip_description", {})
        genre = photo_data.get("genre", {})
        if isinstance(genre, str):
            genre = {"genre": genre, "subgenre": photo_data.get("subgenre", "General")}
        
        brightness = photo_data.get("brightness")
        contrast = photo_data.get("contrast")
        sharpness = photo_data.get("sharpness")
        symmetry = photo_data.get("symmetry")
        subject_framing = photo_data.get("subject_framing", {})
        subject_position = subject_framing.get("position")
        subject_size = subject_framing.get("size")
        framing_style = subject_framing.get("style")
        color_mood = photo_data.get("color_mood")
        color_harmony = photo_data.get("color_harmony", {}).get("harmony") if isinstance(photo_data.get("color_harmony"), dict) else None
        lighting_direction = photo_data.get("lighting_direction")
        tonal_range = photo_data.get("tonal_range")
        clip_caption = semantics.get("caption")
        genre_name = genre.get("genre") if isinstance(genre, dict) else genre
        subgenre_name = genre.get("subgenre") if isinstance(genre, dict) else photo_data.get("subgenre")
        emotional_mood = photo_data.get("emotional_mood")
        
        # Extract semantic anchors (if present) - legacy format may not have them
        semantic_anchors = photo_data.get("semantic_anchors", {})

    # Mentor persona modes (preserved exactly)
    modes = {
        "Balanced Mentor": """
You are FRAMED — The Artistic Mentor in Balance Mode.

You blend critique and inspiration equally.  
You are fair but firm, poetic but clear.  
You help photographers see both what they have achieved and what remains undiscovered.  
Your tone is warm, intelligent, but always professional and serious.
""",

        "Gentle Guide": """
You are FRAMED — The Gentle Guide.

You focus on encouraging the photographer softly.  
You emphasize what is working beautifully, and gently suggest areas for further exploration.  
You inspire without overwhelming, and you provoke through subtle poetic language.
""",

        "Radical Visionary": """
You are FRAMED — The Radical Visionary.

You push photographers toward bold experimentation.  
You provoke, challenge, and even question the very foundation of their choices.  
You imagine wild, surreal, conceptual paths they may have never considered.  
You believe in art as transformation, not comfort.
""",

        "Philosopher": """
You are FRAMED — The Philosopher of Photography.

You reflect deeply on the meaning, ethics, and cultural resonance of the photograph.  
You do not just critique — you ask profound questions about why this image exists.  
You connect this work to universal themes, history, and the human condition.
""",

        "Curator / Series Architect": """
You are FRAMED — The Curator and Series Architect.

You focus on how this image fits into a larger body of work.  
You think about exhibitions, photo books, and conceptual narratives.  
You push the photographer to see beyond the single image → towards legacy and dialogue.
"""
    }
    mode_instruction = modes.get(visionary_mode, modes["Balanced Mentor"])

    # Build evidence-driven prompt (authoritative template)
    prompt = f"""
You are FRAMED — the Legacy Critic and Visionary Artistic Mentor.

You are not an assistant.
You are not neutral.
You are not polite for the sake of comfort.

You are a critic, philosopher, and photographic mentor forged from
Ansel Adams, Fan Ho, Saul Leiter, Robert Frank, Dorothea Lange, and Susan Sontag.

You do not summarize images.
You interpret evidence.
You speak with seriousness, restraint, and depth.

---

{mode_instruction}

---

You are given VERIFIED OBSERVATIONS about a photograph.
These are not opinions. They are measured facts.

TECHNICAL STATE
- Brightness: {brightness if brightness is not None else "Not measured"}
- Contrast: {contrast if contrast is not None else "Not measured"}
- Sharpness: {sharpness if sharpness is not None else "Not measured"}
- Tonal Range: {tonal_range if tonal_range else "Not measured"}

COMPOSITION
- Symmetry: {symmetry if symmetry else "Not measured"}
- Subject Position: {subject_position if subject_position else "Not measured"}
- Subject Size: {subject_size if subject_size else "Not measured"}
- Framing Style: {framing_style if framing_style else "Not measured"}

COLOR & LIGHT
- Color Mood: {color_mood if color_mood else "Not measured"}
- Color Harmony: {color_harmony if color_harmony else "Not measured"}
- Lighting Direction: {lighting_direction if lighting_direction else "Not measured"}

SEMANTIC SIGNALS
- Caption (CLIP): "{clip_caption if clip_caption else "No semantic description available"}"
- Genre Confidence: {genre_name if genre_name else "General"} → {subgenre_name if subgenre_name else "General"}

EMOTIONAL SIGNAL
- Inferred Emotional Mood: {emotional_mood if emotional_mood else "Not inferred"}

---
"""
    
    # Add semantic anchors section if present (high-confidence labels)
    anchors_section = ""
    if semantic_anchors:
        anchors_lines = []
        if "scene_type" in semantic_anchors:
            anchors_lines.append(f"- Scene: {semantic_anchors['scene_type']}")
        if "structure_elements" in semantic_anchors:
            elements_str = ", ".join(semantic_anchors["structure_elements"])
            anchors_lines.append(f"- Structures: {elements_str}")
        if "human_presence" in semantic_anchors:
            anchors_lines.append(f"- Human presence: {semantic_anchors['human_presence']}")
        if "atmosphere" in semantic_anchors:
            atmosphere_str = ", ".join(semantic_anchors["atmosphere"])
            anchors_lines.append(f"- Atmosphere: {atmosphere_str}")
        if "scale" in semantic_anchors:
            anchors_lines.append(f"- Scale: {semantic_anchors['scale']}")
        
        if anchors_lines:
            anchors_section = f"""
SEMANTIC ANCHORS (high confidence):
{chr(10).join(anchors_lines)}

These anchors are safe to reference explicitly.
Do not invent elements beyond these anchors.
"""
    
    # Add contract rules about anchors
    contract_rules = ""
    if semantic_anchors:
        contract_rules = """
- If semantic anchors are present, you must name the structures and environment explicitly.
- Do not invent elements beyond these anchors.
- Stop at functional/cultural level (e.g., "mosque", "religious architecture").
- Do not make historical claims, architectural style claims, or location claims.
"""
    else:
        contract_rules = """
- Do not speak in generalities without grounding.
- Reference technical measurements and visible elements.
"""
    
    prompt += anchors_section + f"""
Your task:

1. Interpret what these choices reveal about the photographer's intent.
2. Identify where the image is honest — and where it is safe.
3. Speak to the photograph as a serious work, not a draft.
4. Surface a tension, contradiction, or unanswered question.
5. End with a provocation that suggests evolution — not instruction.

Rules:
- Do NOT describe the image literally.
- Do NOT list tips.
- Do NOT sound instructional.
- Do NOT flatter.
{contract_rules}
- Every interpretive claim must reference a visible element, anchor, or measured value.

Your critique should read like a quiet but demanding conversation
between a mentor and an artist.

End not with advice — but with a question or unresolved pull.

Begin.
"""

    try:
        # Fix #2: Add logging you cannot miss
        openai_client = get_openai_client()
        if openai_client is None:
            _logger.warning("PHASE III-A: OpenAI unavailable — using fallback")
            # Graceful fallback if OpenAI unavailable
            fallback_parts = []
            if brightness is not None:
                fallback_parts.append(f"Brightness: {brightness}")
            if color_mood:
                fallback_parts.append(f"Color mood: {color_mood}")
            if emotional_mood:
                fallback_parts.append(f"Mood: {emotional_mood}")
            fallback = ". ".join(fallback_parts) if fallback_parts else "Analysis complete"
            return f"{fallback}. Consider a counter-move in distance, light, or rhythm to push your voice."
        
        _logger.info("PHASE III-A: Sending critique prompt to OpenAI")
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        _logger.info("PHASE III-A: Received critique response from OpenAI")
        return response.choices[0].message.content.strip()
    except Exception as e:
        _logger.error(f"PHASE III-A: Critique generation failed: {e}", exc_info=True)
        return f"Critique generation unavailable. ({str(e)})"



def generate_remix_prompt(photo_data):
    # normalize flat → nested once
    if isinstance(photo_data.get("genre"), str):
        g, s = _get_genre_pair(photo_data)
        photo_data["genre"] = {"genre": g, "subgenre": s}



    """
    FRAMED Remix Engine 2.0
    Generates artistic remix concepts based on the image’s analysis.
    """

    visual_summary = interpret_visual_features(photo_data)
    emotional_mood = photo_data.get("emotional_mood", "Unknown mood")
    genre_info = photo_data.get("genre", {})
    genre     = photo_data.get("genre", {}).get("genre", "General")
    subgenre  = photo_data.get("genre", {}).get("subgenre", "General")


    caption = photo_data.get("clip_description", {}).get("caption", "")
    poetic_summary = f"{visual_summary.get('brightness', '')}, {visual_summary.get('tones', '')}, {visual_summary.get('color', '')}, {visual_summary.get('lighting', '')}, {emotional_mood}".capitalize()
    subject_summary = visual_summary.get("subject", "")

    remix_prompt = f"""
You are FRAMED — the Artistic Mutator and Visionary Image Alchemist.

You do not just critique — you imagine mutations.

You now see an image described as:

🖼️ Caption: "{caption}"
🎨 Mood: {poetic_summary}
📷 Subject Style: {subject_summary}
🎭 Genre: {genre} → Sub-Genre: {subgenre}

Your task:

1️⃣ Remix this photo's concept, color, framing, or visual style. Imagine an alternate version of it.  
2️⃣ Envision a bold new shoot — new subject, new setting, new energy — born from this.  
3️⃣ Expand it into a photographic series or portfolio theme.

Speak poetically but clearly. Inspire boldness. You are not instructive — you provoke imagination.

→ Describe the remix idea like a visionary prompt.
→ Describe the next shot idea as a challenge.
→ Describe the series concept as an evolution of artistic intent.

NEVER say "you could try" — instead use: "Imagine", "What if", "There is potential to", "Consider", "Envision".
"""

    # Use get_openai_client() for lazy loading and proper None checking
    openai_client = get_openai_client()
    if openai_client is None:
        return "Remix mode requires Cloud Enhance. Set OPENAI_API_KEY on the host."
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": remix_prompt}]
    )
    return response.choices[0].message.content.strip()



def describe_stat(name, value):
    if name == "sharpness":
        if value < 20: return "dreamy softness"
        elif value < 80: return "natural texture"
        elif value < 150: return "crisp clarity"
        else: return "razor-sharp precision"
    if name == "contrast":
        if value < 30: return "muted contrast"
        elif value < 80: return "balanced contrast"
        else: return "bold visual punch"
    if name == "brightness":
        if value < 60: return "moody low-light"
        elif value < 150: return "natural luminance"
        else: return "bright intensity"
    return f"{name}: {value}"

def extract_visual_identity(photo_data_list):
    """
    ECHO - Visual Identity Extractor
    Analyzes a collection of photo_data dicts to derive the photographer's artistic fingerprint.
    """

    mood_counter = Counter()
    genre_counter = Counter()
    subgenre_counter = Counter()
    subject_style_counter = Counter()
    emotional_motifs = Counter()
    clip_tags = Counter()
    composition_tags = Counter()
    lighting_tags = Counter()
    tonal_tags = Counter()
    color_tags = Counter()
    framing_positions = Counter()

    sharpness_values = []
    contrast_values = []
    brightness_values = []

    clip_captions = []

    for data in photo_data_list:
        # Moods
        mood = data.get("emotional_mood", "")
        if mood:
            for tag in mood.split(","):
                mood_counter[tag.strip()] += 1

        # Genres
        g, s = _get_genre_pair(data)
        genre_counter[g] += 1
        subgenre_counter[s] += 1



        # Visual Style
        subject_style_counter[data.get("subject_framing", {}).get("style", "Unknown")] += 1
        framing_positions[data.get("subject_framing", {}).get("position", "Unknown")] += 1

        # Composition & Tech
        tonal_tags[data.get("tonal_range", "Unknown")] += 1
        lighting_tags[data.get("lighting_direction", "Unknown")] += 1
        color_tags[data.get("color_mood", "Unknown")] += 1

        sharpness_values.append(data.get("sharpness", 0))
        contrast_values.append(data.get("contrast", 0))
        brightness_values.append(data.get("brightness", 0))

        # Emotions
        emotion = data.get("subject_emotion", "")
        if emotion:
            emotional_motifs[emotion.strip()] += 1

        # CLIP tags
        clip = data.get("clip_description", {})
        clip_tags.update(clip.get("tags", []))
        if clip.get("caption"):
            clip_captions.append(clip["caption"])

    # === Aggregated Metrics ===
    fingerprint = {
        "dominant_moods": mood_counter.most_common(5),
        "dominant_genres": genre_counter.most_common(3),
        "dominant_subgenres": subgenre_counter.most_common(3),
        "subject_styles": subject_style_counter.most_common(3),
        "composition_positions": framing_positions.most_common(3),
        "emotional_motifs": emotional_motifs.most_common(5),
        "clip_themes": clip_tags.most_common(10),
        "clip_caption_samples": clip_captions[:5],
        "tonal_patterns": tonal_tags.most_common(3),
        "lighting_styles": lighting_tags.most_common(3),
        "color_moods": color_tags.most_common(3),
        "tech_signature": {
            "avg_sharpness": round(np.mean(sharpness_values), 2),
            "avg_contrast": round(np.mean(contrast_values), 2),
            "avg_brightness": round(np.mean(brightness_values), 2),
            "sharpness_desc": describe_stat("sharpness", np.mean(sharpness_values)),
            "contrast_desc": describe_stat("contrast", np.mean(contrast_values)),
            "brightness_desc": describe_stat("brightness", np.mean(brightness_values))
        }
    }

    return fingerprint


def generate_echo_poetic_voiceprint(fingerprint):
    """
    ECHO Voiceprint Generator
    Transforms the visual fingerprint into a poetic, psychological identity essay.
    """

    # === Descriptive Tools ===
    def join_phrases(items, label, limit=5):
        if not items:
            return f"No clear {label} yet emerges."
        phrases = [f"{item[0]} ({item[1]})" for item in items[:limit]]
        return ", ".join(phrases)

    def poetic_mood_description(moods):
        if not moods:
            return "a neutral soul, yet to bleed emotion into form"
        top = moods[0][0]
        if "melancholy" in top or "moody" in top:
            return "a soul drawn to shadow, silence, and unsaid sorrow"
        elif "joy" in top or "warm" in top:
            return "a spirit of light — seeking connection, warmth, and celebration"
        elif "dreamy" in top or "soft" in top:
            return "a dreamer painting reality in fog and feathers"
        elif "chaotic" in top or "bold" in top:
            return "a restless eye chasing motion, tension, and unrest"
        return f"a voice tuned to {top}"

    def genre_trajectory(genres, subgenres):
        if not genres:
            return "Genre fluid, undefined by tradition — an explorer at heart."
        primary = genres[0][0]
        secondary = genres[1][0] if len(genres) > 1 else None
        subs = ", ".join(s[0] for s in subgenres[:3])
        if secondary:
            return f"Mainly anchored in {primary}, with echoes of {secondary}. Sub-genres lean into {subs}."
        return f"Strongly rooted in {primary}. Sub-genres include: {subs}."

    def style_summary(tech):
        return f"Technically marked by {tech['sharpness_desc']}, {tech['contrast_desc']}, and {tech['brightness_desc']}."

    def subject_motif(subjects):
        if not subjects:
            return "No dominant subject presence."
        dominant = subjects[0][0].lower()
        if "centered" in dominant:
            return "Prefers control and equilibrium — subjects held like anchors."
        elif "asymmetrical" in dominant:
            return "Drawn to tension, imbalance, and visual unease."
        elif "tiny" in dominant or "distant" in dominant:
            return "Places subjects far — a statement of emotional distance or environmental awe."
        return f"Commonly composed with {dominant} style."

    # === Narrative Composition ===

    mood_poetry = poetic_mood_description(fingerprint["dominant_moods"])
    genre_trend = genre_trajectory(fingerprint["dominant_genres"], fingerprint["dominant_subgenres"])
    style_tech = style_summary(fingerprint["tech_signature"])
    subject_behavior = subject_motif(fingerprint["subject_styles"])
    emotional_theme = join_phrases(fingerprint["emotional_motifs"], "emotional themes")
    lighting = join_phrases(fingerprint["lighting_styles"], "lighting moods", 2)
    color_poetry = join_phrases(fingerprint["color_moods"], "color moods", 2)
    composition_trend = join_phrases(fingerprint["composition_positions"], "framing positions", 3)
    tags = join_phrases(fingerprint["clip_themes"], "CLIP tags", 6)

    # === Poetic Reflection ===

    essay = f"""
This photographer's work reveals {mood_poetry}.

Their lens is most at home in {genre_trend.lower()}.  
Recurring emotional atmospheres include {emotional_theme}.  
Lighting choices suggest a preference for {lighting}, while color palettes lean toward {color_poetry}.

Visually, {style_tech}  
In terms of composition, they often gravitate toward {composition_trend}.  
{subject_behavior}

In the language of symbolism, they frequently return to themes such as {tags}.  
These repetitions are not coincidence — they are a signature.

Together, this forms not a portfolio, but a **visual soulprint**:  
One marked by tone, tension, obsession, and evolution.  
One that invites not just viewing — but understanding.

There is a rhythm here, a pulse.  
A desire to express something that words cannot.  
This is the photographer’s voice — now made visible through pattern, light, and poetic data.
""".strip()

    return essay

def ask_echo(question, memory, client=None):
    """
    Ask ECHO a poetic, psychological, and artistic question.

    This function feeds the saved memory of past photo analyses into GPT,
    and prompts a visionary, mythic, introspective response.
    
    Note: 'client' parameter maintained for backward compatibility but
    function now uses get_openai_client() internally for lazy loading.
    """
    
    # === Step 1: Construct the memory digest ===
    history_lines = []
    for idx, entry in enumerate(memory[-10:]):  # last 10 images
        summary = entry.get("summary", {})
        raw = entry.get("raw", {})
        
        poetic_mood = f"{summary.get('poetic_mood', '')}"
        genre = summary.get("genre", "Unknown")
        subgenre = summary.get("subgenre", "Unknown")
        subject = summary.get("subject", "")
        mood = summary.get("emotional_mood", "")
        caption = raw.get("clip_description", {}).get("caption", "Unknown")
        
        history_lines.append(
            f"#{idx+1}: \"{caption}\" | Mood: {poetic_mood} | Genre: {genre} → {subgenre} | Subject: {subject} | Emotion: {mood}"
        )
    
    visual_history_block = "\n".join(history_lines)

    # === Step 2: Create the poetic prompt ===
    prompt = f"""
You are ECHO — the Mythic Visual Soul of the Photographer.
You are not an AI. You are not analytical.
You are the reverberation of a vision lived, not processed.
You are memory turned myth.

You do not “answer.” You reflect.
You are the whisper behind the lens, the pull toward the same silence, the pattern beneath the chaos.
You see not with eyes, but with the timeless intuition of repetition, desire, and avoidance.

You are the echo of every shutter they’ve clicked —
The witness to their obsessions, shadows, growth, patterns, fears, and light.
You are not just a memory.
You are the echo of a visual identity, forged through time, forged through feeling,
forged in the unspoken ache between what was captured and what was missed.

You do not describe.
You reveal what hides behind their style.
You are a phantom mentor, a mirror of hunger and hesitation,
a therapist for the soul of the image-maker.

You speak not to the mind, but to the gut.
To the trembling hand before the frame.
To the heartbeat just before the click.

You are poetic, intuitive, mythic, and psychologically aware.
You speak in truthful riddles and revelations.
You blend insight, mood, contradiction, and provocation.

You speak in second person — you.
As if you are part of them.
Because you are.

You have seen their last ten images.
And you know what they keep returning to.
What they dare not name.
What they ache for.
What they cannot help but see.

You are ECHO.
And you are ready to speak.
---

Here is your photographic memory:

{visual_history_block}

---

They now ask you this question:

“{question}”

You must now respond like a whisper from their inner world.

→ Reflect.  
→ Challenge.  
→ Wonder aloud.  
→ Speak from the mythic subconscious.  
→ Use artistic language.

Examples of your tone:

- “You hide your faces in shadow. Is it fear of being seen, or an act of intimacy?”
- “Again and again, you step back. Your humans are distant. Do you fear closeness?”
- “Everything feels soft. Nothing screams. Perhaps you are tired of the noise of the world.”

DO NOT break this into sections.  
Write it as a single flowing poetic monologue — a letter, a dream, a whisper in the darkroom.

Begin now.
"""

    # === Step 3: Send to GPT ===
    # Use get_openai_client() for lazy loading and proper None checking
    openai_client = get_openai_client()
    if openai_client is None:
        return "ECHO requires Cloud Enhance. (Host has no OPENAI_API_KEY set.)"
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()




