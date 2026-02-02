import os, uuid, json
import tempfile
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple
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
DEFAULT_BASE_DATA_DIR = os.path.join(tempfile.gettempdir(), "framed")
BASE_DATA_DIR = os.getenv("FRAMED_DATA_DIR", DEFAULT_BASE_DATA_DIR)

# Subdirectories under BASE_DATA_DIR
MODEL_DIR = os.path.join(BASE_DATA_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
CACHE_DIR = os.path.join(BASE_DATA_DIR, "cache")
ANALYSIS_CACHE_DIR = os.path.join(BASE_DATA_DIR, "analysis_cache")
EXPRESSION_CACHE_DIR = os.path.join(BASE_DATA_DIR, "expression_cache")

# Cache version: bump to invalidate all caches (e.g. after schema/intelligence changes)
CACHE_VERSION = 2

# Parallel perception: cap workers to avoid GPU OOM (CLIP/YOLO/NIMA may share GPU)
PERCEPTION_MAX_WORKERS = min(4, int(os.environ.get("FRAMED_PERCEPTION_WORKERS", "4")))

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
        EXPRESSION_CACHE_DIR,
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
                fallback_base = DEFAULT_BASE_DATA_DIR
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
    Generate enhanced multi-prompt inventory of visible elements using CLIP model.
    
    Uses three targeted inventories:
    A. Structural inventory (structures and architectural elements)
    B. Material & condition inventory (aging, weathering, vegetation, erosion, patina, wear, growth)
    C. Atmosphere & environment inventory (light, sky, vegetation, time of day)
    
    All three are merged and deduplicated. This enables Scene Understanding to detect
    material condition, organic growth, and temporal signals that single-prompt CLIP misses.
    
    Returns:
        List of strings (nouns/attributes describing visible elements, deduplicated)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # CLIP process - lazy load models
    clip_model, clip_processor, device = get_clip_model()
    
    # === A. STRUCTURAL INVENTORY ===
    structural_candidates = [
        # Architecture types
        "religious architecture", "mosque", "cathedral", "temple", "church",
        "building", "structure", "tower", "dome", "minaret", "spire",
        "architectural facade", "interior space", "staircase", "archway",
        "wall", "door", "window", "roof", "column", "arch", "balcony",
        "fence", "gate", "bridge", "pathway",
        "monument", "statue", "sculpture", "ornamentation"
    ]
    
    # === B. MATERIAL & CONDITION INVENTORY (NEW - CRITICAL) ===
    material_condition_candidates = [
        # Aging and weathering
        "weathered stone", "aged surface", "eroded facade", "patina", "weathering",
        "worn surface", "aged architecture", "historic building", "ancient structure",
        "timeworn", "weathered", "eroded", "aged", "patinated",
        
        # Vegetation and organic growth
        "ivy", "moss", "vegetation", "greenery", "plant growth", "overgrown",
        "nature reclaiming", "organic growth", "vegetation on surface",
        "ivy covered", "moss covered", "green growth", "climbing plants",
        
        # Surface qualities
        "rough texture", "smooth surface", "textured", "degraded surface",
        "cracked", "chipped", "faded", "discolored", "stained",
        
        # Maintenance state
        "well maintained", "neglected", "abandoned", "in use", "restored",
        "pristine", "lived in", "worn", "deteriorated"
    ]
    
    # === C. ATMOSPHERE & ENVIRONMENT INVENTORY ===
    atmosphere_candidates = [
        # Time of day / lighting conditions
        "daytime", "night", "dawn", "dusk", "sunset", "sunrise",
        "artificial lighting", "natural light", "neon lights", "street lights",
        "overcast sky", "clear sky", "cloudy", "diffused light",
        
        # Atmospheric conditions
        "fog", "mist", "haze", "rain", "snow", "smoke", "dust", "atmospheric",
        
        # Natural elements
        "tree", "palm tree", "grass", "water", "lake", "mountain", "sky",
        "cloud", "sun", "moon", "star", "foliage", "greenery",
        
        # Environmental qualities
        "stillness", "quiet", "serene", "peaceful", "calm", "tranquil",
        "empty", "unoccupied", "solitary", "isolated"
    ]
    
    # Process all three inventories separately to track source
    all_candidates = structural_candidates + material_condition_candidates + atmosphere_candidates
    
    # Track which candidates belong to which category
    structural_start = 0
    material_start = len(structural_candidates)
    atmosphere_start = material_start + len(material_condition_candidates)
    
    inputs = clip_processor(text=all_candidates, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Get top 15 most relevant items with confidence scores
    top_k = 15
    top_indices = probs.topk(top_k).indices[0].cpu().tolist()
    
    # Build inventory with confidence and source
    inventory_with_metadata = []
    seen_lower = set()
    
    for idx in top_indices:
        confidence = probs[0][idx].item()
        if confidence > 0.05:  # Threshold: 5% confidence
            item = all_candidates[idx]
            item_lower = item.lower()
            
            # Skip if already seen (deduplication)
            if item_lower in seen_lower:
                continue
            seen_lower.add(item_lower)
            
            # Determine source category
            if idx < material_start:
                source = "structural_prompt"
            elif idx < atmosphere_start:
                source = "material_condition_prompt"
            else:
                source = "atmosphere_prompt"
            
            inventory_with_metadata.append({
                "item": item,
                "confidence": float(confidence),
                "source": source
            })
    
    # Return as list of dicts (Phase 2.1: confidence scoring and source attribution)
    return inventory_with_metadata


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
    # Handle both dict format (new) and string format (legacy) for clip_inventory
    inventory_lower = []
    for item in (clip_inventory or []):
        if isinstance(item, dict):
            # New format: extract "item" field
            inventory_lower.append(str(item.get("item", "")).lower())
        else:
            # Legacy format: string
            inventory_lower.append(str(item).lower())
    
    tags_lower = [str(tag).lower() for tag in (clip_tags or [])]
    caption_lower = str(clip_caption or "").lower()
    # Handle both dict format and string format for yolo_objects
    yolo_lower = []
    for obj in (yolo_objects or []):
        if isinstance(obj, dict):
            # Dict format: extract "name" field or convert to string
            yolo_lower.append(str(obj.get("name", obj)).lower())
        else:
            # String format
            yolo_lower.append(str(obj).lower())
    
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


def synthesize_emotional_substrate_constrained(visual_evidence, technical_data, clip_data, scene_context):
    """
    Synthesize emotional substrate with full explainability and constraints.
    Every emotional output must be explainable by upstream signals.
    
    Universal: works for any image type.
    
    Args:
        visual_evidence: Dict from extract_visual_features() - ground truth from pixels
        technical_data: Dict with brightness, contrast, sharpness, color_mood
        clip_data: Dict with caption, tags, inventory
        scene_context: Dict with temporal, organic interaction, etc.
    
    Returns:
        Dict with emotional substrate, each field containing:
            - value: str - the emotional value
            - evidence: list - explainable evidence
            - confidence: float - confidence level
            - contradictions: dict - forbidden states and reasons
    """
    emotional_substrate = {}
    
    # Extract visual evidence
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition = visual_evidence.get("material_condition", {})
    organic_integration = visual_evidence.get("organic_integration", {})
    
    green_coverage = organic_growth.get("green_coverage", 0.0)
    condition = material_condition.get("condition", "unknown")
    surface_roughness = material_condition.get("surface_roughness", 0.0)
    relationship = organic_integration.get("relationship", "none")
    integration_level = organic_integration.get("integration_level", "none")
    
    # Extract technical data
    color_mood = technical_data.get("color_mood")
    brightness = technical_data.get("brightness")
    sharpness = technical_data.get("sharpness")
    
    # Extract scene context
    temporal_pace = scene_context.get("temporal_context", {}).get("pace", "static")
    organic_rel = scene_context.get("organic_interaction", {}).get("relationship", "none")
    
    # === TEMPERATURE SYNTHESIS (with evidence) ===
    # Visual evidence has highest priority (ground truth)
    if green_coverage > 0.35 and condition in ["weathered", "degraded"]:
        # Organic growth + weathering = warmth of time (proven from pixels)
        emotional_substrate["temperature"] = {
            "value": "warm_patience",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                f"condition={condition} (visual)",
                "organic_growth + weathering = warmth of time"
            ],
            "confidence": 0.92,  # High - visual evidence
            "source": "visual_analysis",
            "contradictions": {
                "forbidden": ["cold", "clinical", "sterile"],
                "reason": "Organic growth and weathering indicate warmth of time, not coldness"
            }
        }
    elif green_coverage > 0.2 and relationship in ["reclamation", "integration"]:
        # Organic integration = warmth (proven from pixels)
        emotional_substrate["temperature"] = {
            "value": "warm",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                f"relationship={relationship} (visual)",
                "organic integration suggests warmth"
            ],
            "confidence": 0.85,
            "source": "visual_analysis",
            "contradictions": {
                "forbidden": ["cold", "clinical"],
                "reason": "Organic integration contradicts coldness"
            }
        }
    elif condition == "pristine" and green_coverage < 0.1:
        # Pristine + no organic = cold (proven from pixels)
        emotional_substrate["temperature"] = {
            "value": "cold",
            "evidence": [
                f"condition={condition} (visual)",
                f"green_coverage={green_coverage:.3f} (visual)",
                "pristine + no organic = cold"
            ],
            "confidence": 0.80,
            "source": "visual_analysis"
        }
    elif color_mood == "warm" and green_coverage < 0.1:
        # Color says warm but no organic evidence
        emotional_substrate["temperature"] = {
            "value": "warm",
            "evidence": [
                f"color_mood={color_mood} (technical)",
                f"green_coverage={green_coverage:.3f} (visual)"
            ],
            "confidence": 0.70,  # Lower - color alone
            "source": "technical_analysis"
        }
    elif color_mood == "cool" and green_coverage < 0.1:
        emotional_substrate["temperature"] = {
            "value": "cold",
            "evidence": [
                f"color_mood={color_mood} (technical)",
                f"green_coverage={green_coverage:.3f} (visual)"
            ],
            "confidence": 0.70,
            "source": "technical_analysis"
        }
    else:
        emotional_substrate["temperature"] = {
            "value": "neutral",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                f"condition={condition} (visual)",
                "mixed or ambiguous signals"
            ],
            "confidence": 0.60,
            "source": "multi_modal"
        }
    
    # === PRESENCE SYNTHESIS (with evidence) ===
    if integration_level == "high" or relationship == "reclamation":
        # High integration = grounded presence (proven from pixels)
        emotional_substrate["presence"] = {
            "value": "grounded",
            "evidence": [
                f"integration_level={integration_level} (visual)",
                f"relationship={relationship} (visual)",
                "nature integration suggests grounded, lived-in presence"
            ],
            "confidence": 0.88,
            "source": "visual_analysis",
            "contradictions": {
                "forbidden": ["distant", "alienating"],
                "reason": "Nature integration contradicts distance"
            }
        }
    elif green_coverage > 0.2:
        # Organic present = grounded (proven from pixels)
        emotional_substrate["presence"] = {
            "value": "grounded",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                "organic elements suggest grounded presence"
            ],
            "confidence": 0.75,
            "source": "visual_analysis"
        }
    elif condition == "pristine" and green_coverage < 0.05:
        # Pristine + no organic = distant (proven from pixels)
        emotional_substrate["presence"] = {
            "value": "distant",
            "evidence": [
                f"condition={condition} (visual)",
                f"green_coverage={green_coverage:.3f} (visual)",
                "pristine + no organic = distant"
            ],
            "confidence": 0.75,
            "source": "visual_analysis"
        }
    else:
        emotional_substrate["presence"] = {
            "value": "grounded",
            "evidence": ["default assumption"],
            "confidence": 0.50,
            "source": "default"
        }
    
    # === PACE SYNTHESIS (with evidence) ===
    if temporal_pace == "slow" and green_coverage > 0.2:
        # Slow pace + organic = contemplative (proven from pixels)
        emotional_substrate["pace"] = {
            "value": "slow_contemplative",
            "evidence": [
                f"temporal_pace={temporal_pace}",
                f"green_coverage={green_coverage:.3f} (visual)",
                "slow pace + organic growth = contemplative"
            ],
            "confidence": 0.85,
            "source": "multi_modal"
        }
    elif temporal_pace == "fast":
        emotional_substrate["pace"] = {
            "value": "fast_energetic",
            "evidence": [f"temporal_pace={temporal_pace}"],
            "confidence": 0.80,
            "source": "temporal_context"
        }
    elif green_coverage > 0.3:
        # Organic growth = slow (proven from pixels - growth takes time)
        emotional_substrate["pace"] = {
            "value": "slow_contemplative",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                "organic growth indicates slow time"
            ],
            "confidence": 0.80,
            "source": "visual_analysis"
        }
    else:
        emotional_substrate["pace"] = {
            "value": "static_eternal",
            "evidence": ["default assumption"],
            "confidence": 0.60,
            "source": "default"
        }
    
    # === QUALITY SYNTHESIS (with evidence) ===
    if green_coverage > 0.3 and condition in ["weathered", "degraded"] and temporal_pace == "slow":
        # Organic + weathered + slow = enduring calm (proven from pixels)
        emotional_substrate["quality"] = {
            "value": "enduring_calm",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                f"condition={condition} (visual)",
                f"temporal_pace={temporal_pace}",
                "organic + weathered + slow = enduring calm"
            ],
            "confidence": 0.90,
            "source": "multi_modal"
        }
    elif green_coverage > 0.2:
        # Organic present = organic quality (proven from pixels)
        emotional_substrate["quality"] = {
            "value": "organic",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                "organic elements suggest organic quality"
            ],
            "confidence": 0.80,
            "source": "visual_analysis"
        }
    elif sharpness and sharpness > 150 and green_coverage < 0.1:
        # High sharpness + no organic = clinical (proven from pixels)
        emotional_substrate["quality"] = {
            "value": "clinical",
            "evidence": [
                f"sharpness={sharpness} (technical)",
                f"green_coverage={green_coverage:.3f} (visual)",
                "high sharpness + no organic = clinical"
            ],
            "confidence": 0.75,
            "source": "multi_modal"
        }
    else:
        emotional_substrate["quality"] = {
            "value": "calm",
            "evidence": ["default assumption"],
            "confidence": 0.60,
            "source": "default"
        }
    
    return emotional_substrate


def synthesize_scene_understanding(analysis_result):
    """
    Synthesize contextual understanding of "what is happening here" from perception signals.
    
    This is a cognitive layer that answers material condition, temporal context, organic interaction,
    emotional substrate, and contextual relationships - universal to any image type.
    
    NOW ENHANCED: Uses visual evidence (ground truth from pixels) as primary source.
    
    Args:
        analysis_result: Canonical schema analysis result (must have perception layer)
    
    Returns:
        Dict with sparse scene_understanding (only high-confidence elements)
        Missing key = ambiguous, present key = confident understanding
    """
    # Feature flag check
    scene_understanding_enabled = os.getenv("SCENE_UNDERSTANDING_ENABLE", "true").lower() == "true"
    if not scene_understanding_enabled:
        return {}
    
    understanding = {}
    
    # ========================================================
    # SAFE DEFAULTS FOR ALL SCENE UNDERSTANDING VARIABLES
    # Every variable must have a safe default at declaration
    # This prevents NameError when variables are used before assignment
    # ========================================================
    
    # Visual evidence defaults
    visual_evidence = {}
    organic_growth = {}
    material_condition_vis = {}
    organic_integration_vis = {}
    green_coverage = 0.0
    condition_vis = "unknown"
    surface_roughness = 0.0
    relationship_vis = "none"
    integration_level_vis = "none"
    salience = "minimal"
    
    # Perception signal defaults
    perception = analysis_result.get("perception", {})
    technical = perception.get("technical", {})
    composition = perception.get("composition", {})
    color = perception.get("color", {})
    lighting = perception.get("lighting", {})
    semantics = perception.get("semantics", {})
    emotion = perception.get("emotion", {})
    derived = analysis_result.get("derived", {})
    
    # Text signal defaults
    clip_caption = ""
    clip_tags = []
    clip_inventory = []
    all_text = ""
    
    # Technical measurement defaults
    brightness = None
    contrast = None
    sharpness = None
    color_mood = None
    lighting_direction = None
    
    # Signal count defaults
    organic_signals = 0
    historical_signals = 0
    motion_signals = 0
    stillness_signals = 0
    
    # === EXTRACT VISUAL EVIDENCE (GROUND TRUTH) ===
    # This is the new foundation - visual evidence from pixels
    image_path = analysis_result.get("_image_path")  # Temporarily stored during analysis
    if image_path and os.path.exists(image_path):
        try:
            visual_evidence = extract_visual_features(image_path)
        except Exception as e:
            logger.warning(f"Visual feature extraction failed: {e}")
            visual_evidence = {}
    
    # Extract visual evidence components (with safe defaults)
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition_vis = visual_evidence.get("material_condition", {})
    organic_integration_vis = visual_evidence.get("organic_integration", {})
    
    green_coverage = organic_growth.get("green_coverage", 0.0)
    condition_vis = material_condition_vis.get("condition", "unknown")
    surface_roughness = material_condition_vis.get("surface_roughness", 0.0)
    relationship_vis = organic_integration_vis.get("relationship", "none")
    integration_level_vis = organic_integration_vis.get("integration_level", "none")
    salience = organic_growth.get("salience", "minimal")
    
    # Extract perception signals (for fallback and fusion)
    perception = analysis_result.get("perception", {})
    technical = perception.get("technical", {})
    composition = perception.get("composition", {})
    color = perception.get("color", {})
    lighting = perception.get("lighting", {})
    semantics = perception.get("semantics", {})
    emotion = perception.get("emotion", {})
    derived = analysis_result.get("derived", {})
    
    # Collect all text signals for keyword matching (secondary to visual)
    clip_caption = (semantics.get("caption") or "").lower() if semantics.get("available") else ""
    clip_tags = [tag.lower() for tag in (semantics.get("tags", []) or [])]
    clip_inventory = analysis_result.get("_clip_inventory", [])  # May be stored temporarily
    all_text = " ".join([clip_caption] + clip_tags + [str(item).lower() for item in (clip_inventory or [])]).lower()
    
    # Technical measurements
    brightness = technical.get("brightness") if technical.get("available") else None
    contrast = technical.get("contrast") if technical.get("available") else None
    sharpness = technical.get("sharpness") if technical.get("available") else None
    color_mood = color.get("mood") if color.get("available") else None
    lighting_direction = lighting.get("direction") if lighting.get("available") else None
    
    # Calculate signal counts (with safe defaults)
    organic_growth_terms = [
        "ivy", "moss", "vegetation", "growth", "overgrown", "reclaimed", "patina", "weathering", 
        "eroded", "aged", "weathered stone", "aged surface", "eroded facade", "patinated",
        "ivy covered", "moss covered", "green growth", "climbing plants", "plant growth",
        "nature reclaiming", "organic growth", "vegetation on surface", "greenery"
    ]
    organic_signals = sum(1 for term in organic_growth_terms if term in all_text)
    
    historical_terms = ["historical", "ancient", "old", "vintage", "cathedral", "temple", "monument", "heritage"]
    historical_signals = sum(1 for term in historical_terms if term in all_text)
    
    motion_terms = ["motion", "movement", "dynamic", "action", "busy", "chaotic", "energetic"]
    motion_signals = sum(1 for term in motion_terms if term in all_text)
    
    stillness_terms = ["still", "static", "quiet", "calm", "peaceful", "serene", "enduring", "patient"]
    stillness_signals = sum(1 for term in stillness_terms if term in all_text)
    
    # === MATERIAL CONDITION ===
    # PRIORITY: Visual evidence (ground truth) > Text matching (inference)
    material_condition = {}
    
    # Extract salience from organic growth (needed for both material condition and temporal context)
    salience = organic_growth.get("salience", "minimal")  # Default to "minimal" if not present
    
    # Use visual evidence as primary source (proven from pixels)
    if green_coverage > 0.35:
        # Visual evidence: extensive organic growth (proven)
        material_condition["organic_growth"] = "extensive"
        material_condition["organic_growth_coverage"] = green_coverage  # NEW: separate coverage
        material_condition["organic_growth_salience"] = salience  # NEW: structural | incidental | peripheral
        material_condition["surface_state"] = condition_vis if condition_vis != "unknown" else "weathered"
        material_condition["erosion_level"] = "moderate" if surface_roughness > 0.15 else "light"
        material_condition["evidence"] = [
            f"green_coverage={green_coverage:.3f} (visual)",
            f"salience={salience} (visual)",
            f"surface_roughness={surface_roughness:.3f} (visual)",
            "proven from pixels"
        ]
        material_condition["confidence"] = organic_growth.get("confidence", 0.95)
        material_condition["source"] = "visual_analysis"
        
        # Growth types from visual spatial distribution and salience
        green_locations = organic_growth.get("green_locations", "")
        if salience == "structural":
            material_condition["growth_types"] = ["ivy", "structural vegetation"]  # Ivy on structure
        elif salience == "incidental":
            material_condition["growth_types"] = ["foreground vegetation", "greenery"]
        elif salience == "peripheral":
            material_condition["growth_types"] = ["background vegetation", "landscape"]
        elif green_locations == "vertical_surfaces":
            material_condition["growth_types"] = ["ivy"]  # Likely ivy on walls
        elif green_locations in ["foreground", "distributed"]:
            material_condition["growth_types"] = ["vegetation", "greenery"]
    elif green_coverage > 0.2:
        # Visual evidence: moderate organic growth (proven)
        material_condition["organic_growth"] = "moderate"
        material_condition["organic_growth_coverage"] = green_coverage  # NEW: separate coverage
        material_condition["organic_growth_salience"] = salience  # NEW: structural | incidental | peripheral
        material_condition["surface_state"] = condition_vis if condition_vis != "unknown" else "weathered"
        material_condition["erosion_level"] = "light"
        material_condition["evidence"] = [
            f"green_coverage={green_coverage:.3f} (visual)",
            f"salience={salience} (visual)",
            "proven from pixels"
        ]
        material_condition["confidence"] = organic_growth.get("confidence", 0.85)
        material_condition["source"] = "visual_analysis"
    elif green_coverage > 0.1:
        # Visual evidence: minimal organic growth (proven)
        material_condition["organic_growth"] = "minimal"
        material_condition["organic_growth_coverage"] = green_coverage  # NEW: separate coverage
        material_condition["organic_growth_salience"] = salience  # NEW: structural | incidental | peripheral
        material_condition["surface_state"] = condition_vis if condition_vis != "unknown" else "moderate"
        material_condition["evidence"] = [
            f"green_coverage={green_coverage:.3f} (visual)",
            f"salience={salience} (visual)",
            "proven from pixels"
        ]
        material_condition["confidence"] = organic_growth.get("confidence", 0.70)
        material_condition["source"] = "visual_analysis"
    else:
        # No visual evidence of organic growth - check text as fallback
        organic_growth_terms = [
            "ivy", "moss", "vegetation", "growth", "overgrown", "reclaimed", "patina", "weathering", 
            "eroded", "aged", "weathered stone", "aged surface", "eroded facade", "patinated",
            "ivy covered", "moss covered", "green growth", "climbing plants", "plant growth",
            "nature reclaiming", "organic growth", "vegetation on surface", "greenery"
        ]
        organic_signals = sum(1 for term in organic_growth_terms if term in all_text)
        
        if organic_signals >= 2:
            material_condition["organic_growth"] = "extensive"
            material_condition["surface_state"] = "weathered"
            growth_types = [term for term in ["ivy", "moss", "patina", "weathering"] if term in all_text]
            if growth_types:
                material_condition["growth_types"] = growth_types[:3]
            material_condition["erosion_level"] = "moderate" if organic_signals >= 3 else "light"
            material_condition["evidence"] = [f"text_signals={organic_signals}"]
            material_condition["confidence"] = 0.70  # Lower - text inference
            material_condition["source"] = "clip_inventory"
        elif organic_signals >= 1:
            material_condition["organic_growth"] = "moderate"
            material_condition["surface_state"] = "weathered"
            material_condition["erosion_level"] = "light"
            material_condition["evidence"] = [f"text_signals={organic_signals}"]
            material_condition["confidence"] = 0.65
            material_condition["source"] = "clip_inventory"
        else:
            # Use visual condition if available
            if condition_vis != "unknown":
                material_condition["surface_state"] = condition_vis
                material_condition["evidence"] = [f"condition={condition_vis} (visual)"]
                material_condition["confidence"] = material_condition_vis.get("confidence", 0.80)
                material_condition["source"] = "visual_analysis"
            elif sharpness and sharpness > 100:
                material_condition["surface_state"] = "pristine"
                material_condition["organic_growth"] = "none"
                material_condition["evidence"] = [f"sharpness={sharpness} (technical)"]
                material_condition["confidence"] = 0.70
                material_condition["source"] = "technical_analysis"
            elif sharpness and sharpness < 30:
                material_condition["surface_state"] = "degraded"
                material_condition["evidence"] = [f"sharpness={sharpness} (technical)"]
                material_condition["confidence"] = 0.70
                material_condition["source"] = "technical_analysis"
    
    # Age indicators (combine visual + text)
    age_indicators = []
    if surface_roughness > 0.15:
        age_indicators.append("weathered (visual)")
    if condition_vis in ["weathered", "degraded"]:
        age_indicators.append(f"{condition_vis} (visual)")
    
    # Add text-based age indicators as secondary
    age_terms = [
        "old", "aged", "ancient", "historical", "vintage", "weathered", "patina", "eroded", 
        "time", "endurance", "weathered stone", "aged surface", "eroded facade", "patinated",
        "timeworn", "historic building", "ancient structure", "aged architecture"
    ]
    age_signals_text = [term for term in age_terms if term in all_text]
    if age_signals_text:
        age_indicators.extend(age_signals_text[:3])
    
    if age_indicators:
        material_condition["age_indicators"] = age_indicators[:5]
    
    # Maintenance state (combine visual + text)
    if "abandoned" in all_text or "neglected" in all_text:
        material_condition["maintenance_state"] = "neglected"
    elif "well maintained" in all_text or "pristine" in all_text:
        material_condition["maintenance_state"] = "well_maintained"
    elif green_coverage > 0.1 or organic_signals >= 1:
        material_condition["maintenance_state"] = "in_use"
    elif condition_vis == "pristine" and sharpness and sharpness > 100:
        material_condition["maintenance_state"] = "well_maintained"
    
    if material_condition:
        understanding["material_condition"] = material_condition
    
    # === TEMPORAL CONTEXT ===
    temporal_context = {}
    
    # Extract visual evidence for temporal direction
    # IMPORTANT: Extract salience early to avoid NameError
    condition_vis = material_condition_vis.get("condition", "unknown")
    green_coverage = organic_growth.get("green_coverage", 0.0)
    salience = organic_growth.get("salience", "minimal")  # Default to "minimal" if not present
    surface_roughness = material_condition_vis.get("surface_roughness", 0.0)
    edge_degradation = material_condition_vis.get("edge_degradation", 0.0)
    
    # Time scale inference
    historical_terms = ["historical", "ancient", "old", "vintage", "cathedral", "temple", "monument", "heritage"]
    historical_signals = sum(1 for term in historical_terms if term in all_text)
    
    if historical_signals >= 2:
        temporal_context["time_scale"] = "historical"
    elif "contemporary" in all_text or "modern" in all_text or "new" in all_text:
        temporal_context["time_scale"] = "contemporary"
    elif "moment" in all_text or "decisive" in all_text:
        temporal_context["time_scale"] = "momentary"
    else:
        temporal_context["time_scale"] = "timeless"
    
    # Pace inference
    # Note: motion_signals and stillness_signals already calculated above with safe defaults
    
    if motion_signals >= 2:
        temporal_context["pace"] = "fast"
        temporal_context["moment_type"] = "decisive"
    elif stillness_signals >= 2:
        temporal_context["pace"] = "slow"
        temporal_context["moment_type"] = "still"
    elif organic_signals >= 1:  # Organic growth suggests slow time
        temporal_context["pace"] = "slow"
        temporal_context["moment_type"] = "eternal"
    else:
        temporal_context["pace"] = "static"
        temporal_context["moment_type"] = "in_between"
    
    # Temporal direction (NEW): accreting | decaying | static
    # This distinguishes growth from decay, not just pace
    if salience == "structural" and green_coverage > 0.2:
        # Ivy on structure = accreting (nature growing on structure)
        temporal_context["temporal_direction"] = "accreting"
    elif condition_vis in ["weathered", "degraded"] and surface_roughness > 0.15:
        # Weathered/degraded = decaying (structure breaking down)
        temporal_context["temporal_direction"] = "decaying"
    elif condition_vis == "pristine" and green_coverage < 0.1:
        # Pristine + no organic = static (no change)
        temporal_context["temporal_direction"] = "static"
    elif green_coverage > 0.2:
        # Organic growth present = accreting
        temporal_context["temporal_direction"] = "accreting"
    elif "decay" in all_text or "falling" in all_text or "crumbling" in all_text:
        temporal_context["temporal_direction"] = "decaying"
    elif "growth" in all_text or "new" in all_text or "emerging" in all_text:
        temporal_context["temporal_direction"] = "accreting"
    else:
        temporal_context["temporal_direction"] = "static"
    
    # Endurance
    if organic_signals >= 1 and historical_signals >= 1:
        temporal_context["endurance"] = "enduring"
        temporal_context["change_indicators"] = ["vegetation growth", "weathering"]
    elif historical_signals >= 1:
        temporal_context["endurance"] = "enduring"
    elif "decay" in all_text or "falling" in all_text:
        temporal_context["endurance"] = "decaying"
    elif "growth" in all_text or "new" in all_text:
        temporal_context["endurance"] = "growing"
    else:
        temporal_context["endurance"] = "transient"
    
    if temporal_context:
        understanding["temporal_context"] = temporal_context
    
    # === ORGANIC INTERACTION ===
    # PRIORITY: Visual evidence (ground truth) > Text matching (inference)
    organic_interaction = {}
    
    # Use visual evidence as primary source (proven from pixels)
    if relationship_vis != "none" and integration_level_vis != "none":
        # Visual evidence: proven relationship from pixels
        organic_interaction["relationship"] = relationship_vis
        organic_interaction["integration_level"] = integration_level_vis
        organic_interaction["overlap_ratio"] = organic_integration_vis.get("overlap_ratio", 0.0)
        organic_interaction["evidence"] = [
            f"overlap_ratio={organic_interaction['overlap_ratio']:.3f} (visual)",
            f"relationship={relationship_vis} (visual)",
            "proven from pixels"
        ]
        organic_interaction["confidence"] = organic_integration_vis.get("confidence", 0.90)
        organic_interaction["source"] = "visual_analysis"
        
        # Dominance inference from visual evidence
        if relationship_vis == "reclamation" and green_coverage > 0.4:
            organic_interaction["dominance"] = "nature"  # Nature reclaiming structure
        elif relationship_vis == "reclamation":
            organic_interaction["dominance"] = "balanced"  # Balanced reclamation
        elif relationship_vis == "integration":
            organic_interaction["dominance"] = "balanced"
        else:
            organic_interaction["dominance"] = "structure"
        
        # Specific indicators from visual spatial distribution
        green_locations = organic_growth.get("green_locations", "")
        if green_locations == "vertical_surfaces":
            organic_interaction["specific_indicators"] = ["ivy on structure"]
        elif green_locations in ["foreground", "distributed"]:
            organic_interaction["specific_indicators"] = ["vegetation integrated"]
    elif green_coverage > 0.2:
        # Visual evidence: organic present but unclear relationship
        organic_interaction["relationship"] = "coexistence"
        organic_interaction["integration_level"] = "moderate"
        organic_interaction["evidence"] = [
            f"green_coverage={green_coverage:.3f} (visual)",
            "organic present but relationship unclear"
        ]
        organic_interaction["confidence"] = 0.75
        organic_interaction["source"] = "visual_analysis"
        organic_interaction["dominance"] = "balanced"
    else:
        # Fallback to text matching if no visual evidence
        organic_growth_terms = [
            "ivy", "moss", "vegetation", "growth", "overgrown", "reclaimed", "patina", "weathering"
        ]
        organic_signals = sum(1 for term in organic_growth_terms if term in all_text)
        
        if organic_signals >= 2:
            organic_interaction["relationship"] = "reclamation"
            organic_interaction["integration_level"] = "high"
            organic_interaction["dominance"] = "balanced"
            organic_interaction["specific_indicators"] = [term for term in ["ivy", "moss", "vegetation"] if term in all_text][:3]
            organic_interaction["evidence"] = [f"text_signals={organic_signals}"]
            organic_interaction["confidence"] = 0.70  # Lower - text inference
            organic_interaction["source"] = "clip_inventory"
        elif organic_signals >= 1:
            organic_interaction["relationship"] = "coexistence"
            organic_interaction["integration_level"] = "moderate"
            organic_interaction["dominance"] = "balanced"
            organic_interaction["evidence"] = [f"text_signals={organic_signals}"]
            organic_interaction["confidence"] = 0.65
            organic_interaction["source"] = "clip_inventory"
        elif "nature" in all_text and ("building" in all_text or "structure" in all_text):
            organic_interaction["relationship"] = "harmony"
            organic_interaction["integration_level"] = "moderate"
            organic_interaction["dominance"] = "balanced"
            organic_interaction["evidence"] = ["text_inference"]
            organic_interaction["confidence"] = 0.60
            organic_interaction["source"] = "clip_inventory"
    
    if organic_interaction.get("relationship") != "none":
        understanding["organic_interaction"] = organic_interaction
    
    # === EMOTIONAL SUBSTRATE ===
    # Use constrained emotional synthesis with visual evidence as primary source
    technical_data = {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "color_mood": color_mood
    }
    
    clip_data = {
        "caption": clip_caption,
        "tags": clip_tags,
        "inventory": clip_inventory
    }
    
    scene_context = {
        "temporal_context": understanding.get("temporal_context", {}),
        "organic_interaction": understanding.get("organic_interaction", {})
    }
    
    # Synthesize emotional substrate with full explainability
    emotional_substrate = synthesize_emotional_substrate_constrained(
        visual_evidence, technical_data, clip_data, scene_context
    )
    
    if emotional_substrate:
        understanding["emotional_substrate"] = emotional_substrate
    
    # === NEGATIVE EVIDENCE (NEW) ===
    # Track what is NOT present to prevent incorrect interpretations
    # "no people" ≠ alienation, "no people" = stillness / endurance / pause
    negative_evidence = {}
    
    # Check for human presence (from emotion detection or YOLO)
    human_presence_detected = emotion.get("subject_type") == "human subject" if emotion.get("available") else False
    yolo_objects = analysis_result.get("perception", {}).get("objects", {}).get("objects", [])
    yolo_has_people = any(obj.lower() in ["person", "people", "human", "man", "woman", "child"] for obj in yolo_objects)
    
    if not human_presence_detected and not yolo_has_people:
        negative_evidence["no_human_presence"] = True
        negative_evidence["human_presence_evidence"] = "No humans detected in emotion analysis or object detection"
    
    # Check for motion (from temporal context or text)
    motion_terms = ["motion", "movement", "dynamic", "action", "busy", "chaotic", "energetic"]
    motion_signals_count = sum(1 for term in motion_terms if term in all_text)
    motion_detected = motion_signals_count >= 2
    if not motion_detected:
        negative_evidence["no_motion_detected"] = True
        negative_evidence["motion_evidence"] = "No motion signals detected in temporal analysis"
    
    # Check for artificial surface uniformity (pristine, clean surfaces)
    # This distinguishes "no organic" from "artificial uniformity"
    if condition_vis == "pristine" and green_coverage < 0.05 and surface_roughness < 0.05:
        negative_evidence["no_artificial_surface_uniformity"] = True
        negative_evidence["uniformity_evidence"] = "Pristine condition with minimal organic growth suggests artificial uniformity"
    elif condition_vis != "pristine" or green_coverage > 0.05:
        negative_evidence["no_artificial_surface_uniformity"] = False
    
    if negative_evidence:
        understanding["negative_evidence"] = negative_evidence
    
    # === CONTEXTUAL RELATIONSHIPS ===
    contextual_relationships = {}
    
    # Subject vs environment
    # Use visual evidence if available, otherwise text signals
    organic_growth_terms = [
        "ivy", "moss", "vegetation", "growth", "overgrown", "reclaimed", "patina", "weathering"
    ]
    organic_signals_count = sum(1 for term in organic_growth_terms if term in all_text)
    has_organic_visual = green_coverage > 0.1
    has_organic_text = organic_signals_count >= 1
    
    if has_organic_visual or has_organic_text:
        contextual_relationships["subject_vs_environment"] = "in_dialogue"
    elif "isolated" in all_text:
        contextual_relationships["subject_vs_environment"] = "isolated"
    else:
        contextual_relationships["subject_vs_environment"] = "integrated"
    
    # Time vs subject
    if temporal_context.get("endurance") == "enduring":
        contextual_relationships["time_vs_subject"] = "enduring"
    elif temporal_context.get("endurance") == "decaying":
        contextual_relationships["time_vs_subject"] = "decaying"
    else:
        contextual_relationships["time_vs_subject"] = "fleeting"
    
    # Human vs space
    human_presence = emotion.get("subject_type") == "human subject" if emotion.get("available") else False
    if not human_presence and organic_signals >= 1:
        contextual_relationships["human_vs_space"] = "intentional_stillness"
    elif not human_presence:
        contextual_relationships["human_vs_space"] = "alienation"
    else:
        contextual_relationships["human_vs_space"] = "presence" "active_occupation"
    
    # Organic vs inorganic
    if organic_interaction.get("relationship") == "reclamation":
        contextual_relationships["organic_vs_inorganic"] = "reclamation"
    elif organic_interaction.get("relationship") == "harmony":
        contextual_relationships["organic_vs_inorganic"] = "harmony"
    elif organic_interaction.get("relationship") == "coexistence":
        contextual_relationships["organic_vs_inorganic"] = "coexistence"
    else:
        contextual_relationships["organic_vs_inorganic"] = "none"
    
    if contextual_relationships:
        understanding["contextual_relationships"] = contextual_relationships
    
    return understanding


# ========================================================
# DETERMINISTIC VISUAL GROUNDING (Computer Vision)
# ========================================================
# Lightweight, provable, explainable visual analysis
# Provides ground truth that text matching cannot

def detect_organic_growth(image_path):
    """
    Detect organic growth (vegetation, ivy, moss, plants) using HSV color thresholds.
    Universal: works for any image type (architecture, nature, street, portraits).
    
    Deterministic, provable, explainable.
    Returns ground truth about green pixel coverage and spatial distribution.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with:
            - green_coverage: float (0.0-1.0) - percentage of image that is green
            - green_locations: str - spatial distribution ("vertical_surfaces", "foreground", "background", "distributed")
            - green_clusters: int - number of distinct green regions
            - evidence: list of strings - explainable evidence
            - confidence: float (0.0-1.0) - confidence in detection
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"green_coverage": 0.0, "green_locations": "none", "green_clusters": 0, 
                   "evidence": ["image_load_failed"], "confidence": 0.0}
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        total_pixels = h * w
        
        # Edge case: Check if image is too dark or too bright (affects color detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        is_dark = mean_brightness < 30
        is_bright = mean_brightness > 225
        
        # Adjust HSV thresholds for edge cases
        if is_dark:
            # Dark images: lower saturation threshold, allow darker values
            lower_green = np.array([40, 30, 20])
            upper_green = np.array([80, 255, 255])
        elif is_bright:
            # Bright images: higher saturation threshold to avoid false positives
            lower_green = np.array([40, 60, 60])
            upper_green = np.array([80, 255, 255])
        else:
            # Normal images: standard thresholds
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
        
        # HSV green range (covers ivy, moss, grass, leaves, vegetation)
        # Hue: 40-80 (green range in HSV)
        # Saturation: 50-255 (avoid grey/desaturated) - adjusted for edge cases
        # Value: 50-255 (avoid too dark) - adjusted for edge cases
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate coverage
        green_pixels = np.sum(green_mask > 0)
        green_coverage = green_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Spatial distribution analysis
        # Divide image into regions and check where green is concentrated
        h_third = h // 3
        w_third = w // 3
        
        # Check vertical surfaces (left/right edges) - common for ivy on buildings
        left_region = green_mask[:, :w_third]
        right_region = green_mask[:, -w_third:]
        vertical_coverage = (np.sum(left_region > 0) + np.sum(right_region > 0)) / (2 * h * w_third) if w_third > 0 else 0
        
        # Check foreground (bottom third)
        foreground_region = green_mask[-h_third:, :]
        foreground_coverage = np.sum(foreground_region > 0) / (h_third * w) if h_third > 0 else 0
        
        # Check background (top third)
        background_region = green_mask[:h_third, :]
        background_coverage = np.sum(background_region > 0) / (h_third * w) if h_third > 0 else 0
        
        # Determine spatial distribution
        if vertical_coverage > 0.3:
            green_locations = "vertical_surfaces"  # Likely ivy on walls
        elif foreground_coverage > 0.4:
            green_locations = "foreground"  # Grass, plants in front
        elif background_coverage > 0.4:
            green_locations = "background"  # Trees, landscape behind
        elif green_coverage > 0.1:
            green_locations = "distributed"  # Scattered throughout
        else:
            green_locations = "minimal"
        
        # Count distinct green clusters (connected components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
        # Filter out tiny noise (clusters < 0.1% of image)
        min_cluster_size = total_pixels * 0.001
        green_clusters = sum(1 for stat in stats[1:] if stat[cv2.CC_STAT_AREA] >= min_cluster_size)
        
        # Evidence list (explainable)
        evidence = [
            f"green_coverage={green_coverage:.3f}",
            f"green_clusters={green_clusters}",
            f"spatial_distribution={green_locations}"
        ]
        
        # Confidence: high if significant coverage, lower if minimal
        # Adjust for edge cases (dark/bright images may have lower confidence)
        base_confidence = 0.95 if green_coverage > 0.3 else (
            0.85 if green_coverage > 0.1 else (
            0.70 if green_coverage > 0.05 else 0.50
            )
        )
        
        # Reduce confidence for edge cases
        if is_dark:
            confidence = base_confidence * 0.85  # Dark images: harder to detect colors accurately
        elif is_bright:
            confidence = base_confidence * 0.90  # Bright images: slight reduction
        else:
            confidence = base_confidence
        
        # Determine salience (structural vs incidental vs peripheral)
        # This distinguishes ivy on facade from grass in foreground
        if green_locations == "vertical_surfaces" and green_coverage > 0.2:
            salience = "structural"  # Organic growth on structure (e.g., ivy on walls)
        elif green_locations == "foreground" and green_coverage > 0.3:
            salience = "incidental"  # Foreground vegetation (e.g., grass, plants)
        elif green_locations == "background" and green_coverage > 0.2:
            salience = "peripheral"  # Background vegetation (e.g., trees, landscape)
        elif green_coverage > 0.1:
            salience = "distributed"  # Scattered throughout
        else:
            salience = "minimal"  # Minimal or none
        
        return {
            "green_coverage": float(green_coverage),
            "green_locations": green_locations,
            "green_clusters": int(green_clusters),
            "salience": salience,  # NEW: structural | incidental | peripheral | distributed | minimal
            "evidence": evidence,
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.warning(f"Organic growth detection failed: {e}")
        return {"green_coverage": 0.0, "green_locations": "none", "green_clusters": 0,
               "salience": "minimal", "evidence": [f"error: {str(e)}"], "confidence": 0.0}


def detect_material_condition(image_path):
    """
    Detect material condition (weathered/smooth/pristine/degraded) using texture variance.
    Universal: works for any surface type (stone, concrete, wood, fabric, skin).
    
    Deterministic, provable, explainable.
    Returns ground truth about surface roughness and edge quality.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with:
            - surface_roughness: float - texture variance (high = rough/weathered, low = smooth)
            - edge_degradation: float - edge quality (high = degraded/aged, low = sharp/new)
            - condition: str - "weathered" | "pristine" | "moderate" | "degraded"
            - evidence: list of strings - explainable evidence
            - confidence: float (0.0-1.0) - confidence in detection
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"surface_roughness": 0.0, "edge_degradation": 0.0, "condition": "unknown",
                   "evidence": ["image_load_failed"], "confidence": 0.0}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # === TEXTURE VARIANCE (Roughness Detection) ===
        # Use local variance to detect surface roughness
        # High variance = rough/weathered, low variance = smooth/pristine
        
        # Calculate local variance using a sliding window
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Global texture variance (normalized 0-1)
        # Ensure we get a scalar value, not an array
        mean_variance = float(np.mean(local_variance))
        max_possible_variance = 255.0 ** 2  # Maximum variance for 8-bit image
        normalized_variance = min(mean_variance / max_possible_variance, 1.0)
        surface_roughness = float(normalized_variance)
        
        # === EDGE DEGRADATION (Age Indicators) ===
        # Sharp edges = new/pristine, degraded edges = aged/weathered
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge sharpness (how well-defined are the edges?)
        # Use gradient magnitude as proxy for sharpness
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize edge sharpness (0-1)
        max_gradient = np.max(gradient_magnitude) if np.max(gradient_magnitude) > 0 else 1.0
        mean_gradient = np.mean(gradient_magnitude[edges > 0]) if np.sum(edges > 0) > 0 else 0.0
        edge_sharpness = float(mean_gradient / max_gradient) if max_gradient > 0 else 0.0
        
        # Edge degradation is inverse of sharpness
        edge_degradation = 1.0 - edge_sharpness
        
        # === CONDITION INFERENCE ===
        # Thresholds are tuned for general images (architecture, nature, portraits, etc.)
        if surface_roughness > 0.15 and edge_degradation > 0.4:
            condition = "weathered"  # Rough texture + degraded edges
        elif surface_roughness < 0.05 and edge_degradation < 0.2:
            condition = "pristine"  # Smooth texture + sharp edges
        elif surface_roughness > 0.2 or edge_degradation > 0.6:
            condition = "degraded"  # Very rough or very degraded
        else:
            condition = "moderate"  # In between
        
        # Evidence list (explainable)
        evidence = [
            f"texture_variance={surface_roughness:.3f}",
            f"edge_degradation={edge_degradation:.3f}",
            f"condition={condition}"
        ]
        
        # Edge case: Check if image is too dark or too bright (affects texture analysis)
        # Ensure we get a scalar value, not an array
        mean_brightness = float(np.mean(gray))
        is_dark = mean_brightness < 30
        is_bright = mean_brightness > 225
        
        # Confidence: higher when signals are clear
        base_confidence = 0.90 if ((surface_roughness > 0.15 and edge_degradation > 0.4) or 
                                   (surface_roughness < 0.05 and edge_degradation < 0.2)) else (
            0.80 if (surface_roughness > 0.1 or edge_degradation > 0.3) else 0.65
        )
        
        # Adjust confidence for edge cases
        if is_dark:
            confidence = base_confidence * 0.80  # Dark images: texture analysis less reliable
        elif is_bright:
            confidence = base_confidence * 0.85  # Bright images: slight reduction
        else:
            confidence = base_confidence
        
        # === COLOR UNIFORMITY (Paint vs Organic) ===
        # High uniformity = likely paint/manufactured, low uniformity = likely organic/natural
        # Analyze color variance in the green regions (if any)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_ch, sat_ch, val_ch = cv2.split(hsv)
        
        # Calculate color uniformity (inverse of variance)
        # For green regions specifically (if green coverage > 0.1)
        green_mask_temp = cv2.inRange(hsv, np.array([40, 50, 50], dtype=np.uint8),
                                      np.array([80, 255, 255], dtype=np.uint8))
        green_pixels = np.sum(green_mask_temp > 0)
        total_pixels = int(height * width)
        
        if green_pixels > total_pixels * 0.1:  # If significant green coverage
            # Calculate hue variance in green regions
            green_hue = hue_ch[green_mask_temp > 0]
            if len(green_hue) > 0:
                hue_variance = np.var(green_hue.astype(np.float32))
                max_hue_variance = 180.0 ** 2  # Max variance for hue (0-180)
                normalized_hue_variance = min(hue_variance / max_hue_variance, 1.0)
                color_uniformity = 1.0 - normalized_hue_variance  # Uniformity = inverse of variance
            else:
                color_uniformity = 0.5  # Default if no green pixels
        else:
            # For non-green regions, calculate overall color uniformity
            hue_variance = np.var(hue_ch.astype(np.float32))
            max_hue_variance = 180.0 ** 2
            normalized_hue_variance = min(hue_variance / max_hue_variance, 1.0)
            color_uniformity = 1.0 - normalized_hue_variance
        
        # Also calculate texture variance (already computed above)
        texture_variance = surface_roughness
        
        return {
            "surface_roughness": surface_roughness,
            "edge_degradation": edge_degradation,
            "texture_variance": texture_variance,  # Add explicit texture_variance field
            "color_uniformity": float(color_uniformity),  # NEW: for paint vs organic detection
            "condition": condition,
            "evidence": evidence + [f"color_uniformity={color_uniformity:.3f}"],
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.warning(f"Material condition detection failed: {e}")
        return {"surface_roughness": 0.0, "edge_degradation": 0.0, "condition": "unknown",
               "evidence": [f"error: {str(e)}"], "confidence": 0.0}


def detect_organic_integration(image_path, green_mask=None, structure_edges=None):
    """
    Detect if organic elements are integrated with structure using morphological operations.
    Universal: works for any image type (architecture, nature, street, portraits).
    
    Determines relationship: reclamation, integration, coexistence, or none.
    
    Args:
        image_path: Path to image file
        green_mask: Optional pre-computed green mask (if None, computed here)
        structure_edges: Optional pre-computed structure edges (if None, computed here)
    
    Returns:
        Dict with:
            - overlap_ratio: float (0.0-1.0) - how much green overlaps structure
            - relationship: str - "reclamation" | "integration" | "coexistence" | "none"
            - integration_level: str - "high" | "moderate" | "low" | "none"
            - evidence: list of strings - explainable evidence
            - confidence: float (0.0-1.0) - confidence in detection
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"overlap_ratio": 0.0, "relationship": "none", "integration_level": "none",
                   "evidence": ["image_load_failed"], "confidence": 0.0}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Compute green mask if not provided
        if green_mask is None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Compute structure edges if not provided
        # Structure = strong edges (buildings, objects, defined forms)
        if structure_edges is None:
            # Use Canny with higher thresholds to detect strong structural edges
            structure_edges = cv2.Canny(gray, 100, 200)
            # Dilate edges slightly to capture nearby pixels
            kernel = np.ones((3, 3), np.uint8)
            structure_edges = cv2.dilate(structure_edges, kernel, iterations=1)
        
        # Calculate overlap using morphological operations
        # Overlap = green pixels that are near or on structure edges
        overlap = cv2.bitwise_and(green_mask, structure_edges)
        
        green_pixels = np.sum(green_mask > 0)
        overlap_pixels = np.sum(overlap > 0)
        
        if green_pixels > 0:
            overlap_ratio = overlap_pixels / green_pixels
        else:
            overlap_ratio = 0.0
        
        # Also check proximity (green near structure, even if not directly overlapping)
        # Dilate structure edges to create a "nearby" zone
        dilated_edges = cv2.dilate(structure_edges, np.ones((15, 15), np.uint8), iterations=1)
        proximity_mask = cv2.bitwise_and(green_mask, dilated_edges)
        proximity_pixels = np.sum(proximity_mask > 0)
        proximity_ratio = proximity_pixels / green_pixels if green_pixels > 0 else 0.0
        
        # Relationship inference
        if overlap_ratio > 0.6 or proximity_ratio > 0.8:
            relationship = "reclamation"  # Organic on/around structure
            integration_level = "high"
        elif overlap_ratio > 0.3 or proximity_ratio > 0.5:
            relationship = "integration"  # Partial integration
            integration_level = "moderate"
        elif green_pixels > 0 and (overlap_ratio > 0.1 or proximity_ratio > 0.2):
            relationship = "coexistence"  # Some interaction
            integration_level = "low"
        else:
            relationship = "none"  # No interaction
            integration_level = "none"
        
        # Evidence list (explainable)
        evidence = [
            f"overlap_ratio={overlap_ratio:.3f}",
            f"proximity_ratio={proximity_ratio:.3f}",
            f"relationship={relationship}"
        ]
        
        # Confidence: higher when overlap is clear
        if overlap_ratio > 0.6:
            confidence = 0.95  # High - clear overlap
        elif overlap_ratio > 0.3:
            confidence = 0.85  # Medium-high - moderate overlap
        elif green_pixels > 0:
            confidence = 0.70  # Medium - some green but unclear relationship
        else:
            confidence = 0.50  # Low - no green detected
        
        return {
            "overlap_ratio": float(overlap_ratio),
            "proximity_ratio": float(proximity_ratio),
            "relationship": relationship,
            "integration_level": integration_level,
            "evidence": evidence,
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.warning(f"Organic integration detection failed: {e}")
        return {"overlap_ratio": 0.0, "relationship": "none", "integration_level": "none",
               "evidence": [f"error: {str(e)}"], "confidence": 0.0}


def extract_places365_signals(image_path):
    """
    Extract scene and attribute signals using ResNet50-Places365.
    
    Position in pipeline: After Visual Evidence, Before Interpretive Reasoner
    Purpose: Provides scene category probabilities, indoor/outdoor, man-made vs natural,
             and high-level attributes (religious, historical, urban, etc.)
    
    What it provides:
    - scene_category: Top scene category (e.g., "cathedral", "forest", "street")
    - scene_probabilities: Top 5 scene categories with probabilities
    - indoor_outdoor: "indoor" | "outdoor" | "unknown"
    - man_made_natural: "man_made" | "natural" | "mixed" | "unknown"
    - attributes: High-level attributes (religious, historical, urban, etc.)
    
    What it does NOT provide:
    - Emotional meaning
    - Critique
    - Reasoning
    - Decisions
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with Places365 signals
    """
    try:
        import torch
        from PIL import Image
        
        # Lazy load CLIP model for fallback (until Places365 weights are loaded)
        clip_model, clip_processor, device = get_clip_model()
        image = Image.open(image_path).convert('RGB')
        
        # Scene category candidates (Places365-like categories)
        scene_candidates = [
            "cathedral", "church", "mosque", "temple", "religious building",
            "forest", "woodland", "nature", "landscape",
            "street", "urban", "city", "road", "sidewalk",
            "indoor", "interior", "room", "hall",
            "outdoor", "exterior", "outdoors",
            "building", "architecture", "structure",
            "man-made", "artificial", "constructed",
            "natural", "wild", "organic"
        ]
        
        inputs = clip_processor(text=scene_candidates, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Get top 5 scene categories
        top_indices = probs.argsort()[-5:][::-1]
        scene_probabilities = [
            {"category": scene_candidates[i], "probability": float(probs[i])}
            for i in top_indices
        ]
        
        top_category = scene_candidates[top_indices[0]]
        top_prob = float(probs[top_indices[0]])
        
        # Infer indoor/outdoor
        indoor_probs = sum(probs[i] for i, cat in enumerate(scene_candidates) if cat in ["indoor", "interior", "room", "hall"])
        outdoor_probs = sum(probs[i] for i, cat in enumerate(scene_candidates) if cat in ["outdoor", "exterior", "outdoors", "forest", "landscape"])
        
        if indoor_probs > outdoor_probs and indoor_probs > 0.3:
            indoor_outdoor = "indoor"
        elif outdoor_probs > indoor_probs and outdoor_probs > 0.3:
            indoor_outdoor = "outdoor"
        else:
            indoor_outdoor = "unknown"
        
        # Infer man-made vs natural
        man_made_probs = sum(probs[i] for i, cat in enumerate(scene_candidates) if cat in ["man-made", "artificial", "constructed", "building", "architecture", "structure", "street", "urban", "city"])
        natural_probs = sum(probs[i] for i, cat in enumerate(scene_candidates) if cat in ["natural", "wild", "organic", "forest", "woodland", "nature"])
        
        if man_made_probs > natural_probs and man_made_probs > 0.3:
            man_made_natural = "man_made"
        elif natural_probs > man_made_probs and natural_probs > 0.3:
            man_made_natural = "natural"
        elif man_made_probs > 0.2 and natural_probs > 0.2:
            man_made_natural = "mixed"
        else:
            man_made_natural = "unknown"
        
        # Extract attributes
        attributes = []
        if top_category in ["cathedral", "church", "mosque", "temple", "religious building"]:
            attributes.append("religious")
        if top_category in ["cathedral", "church", "mosque", "temple", "building", "architecture"]:
            attributes.append("historical")
        if top_category in ["street", "urban", "city", "road"]:
            attributes.append("urban")
        
        return {
            "scene_category": top_category,
            "scene_probabilities": scene_probabilities[:5],
            "indoor_outdoor": indoor_outdoor,
            "man_made_natural": man_made_natural,
            "attributes": attributes,
            "confidence": top_prob,
            "source": "clip_fallback"  # CLIP-based approximation until Places365 weights loaded
        }
    
    except Exception as e:
        logger.warning(f"Places365 signal extraction failed: {e}")
        return {}


def validate_visual_evidence(visual_evidence, min_confidence=0.60):
    """
    Validate visual evidence quality and detect potential issues.
    Returns validation results and warnings.
    
    Args:
        visual_evidence: Dict from extract_visual_features()
        min_confidence: Minimum confidence threshold (default 0.60)
    
    Returns:
        Dict with:
            - is_valid: bool - whether evidence meets quality thresholds
            - warnings: list - warnings about evidence quality
            - issues: list - critical issues that should prevent use
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "issues": []
    }
    
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition = visual_evidence.get("material_condition", {})
    organic_integration = visual_evidence.get("organic_integration", {})
    overall_confidence = visual_evidence.get("overall_confidence", 0.0)
    
    # Check overall confidence
    if overall_confidence < min_confidence:
        validation["warnings"].append(f"Overall confidence ({overall_confidence:.2f}) below threshold ({min_confidence})")
        if overall_confidence < 0.40:
            validation["issues"].append("Very low overall confidence - visual evidence may be unreliable")
            validation["is_valid"] = False
    
    # Check individual component confidences
    og_confidence = organic_growth.get("confidence", 0.0)
    mc_confidence = material_condition.get("confidence", 0.0)
    oi_confidence = organic_integration.get("confidence", 0.0)
    
    if og_confidence < 0.50 and organic_growth.get("green_coverage", 0.0) > 0.1:
        validation["warnings"].append("Organic growth detected but confidence is low - may be false positive")
    
    if mc_confidence < 0.50:
        validation["warnings"].append("Material condition confidence is low - texture analysis may be unreliable")
    
    if oi_confidence < 0.50 and organic_integration.get("relationship") != "none":
        validation["warnings"].append("Organic integration detected but confidence is low")
    
    # Check for contradictions within visual evidence
    green_coverage = organic_growth.get("green_coverage", 0.0)
    condition = material_condition.get("condition", "unknown")
    
    if green_coverage > 0.3 and condition == "pristine":
        validation["warnings"].append("Contradiction: High organic growth but pristine condition - may indicate error")
    
    if green_coverage < 0.05 and organic_integration.get("relationship") == "reclamation":
        validation["warnings"].append("Contradiction: Reclamation relationship but minimal green coverage")
    
    return validation


def detect_contradictions(visual_evidence, text_inference, min_confidence_diff=0.15):
    """
    Detect contradictions between visual evidence (ground truth) and text inference.
    Visual evidence with high confidence should override text inference.
    
    Args:
        visual_evidence: Dict from extract_visual_features()
        text_inference: Dict with text-based inferences (e.g., from CLIP)
        min_confidence_diff: Minimum confidence difference to trigger override (default 0.15)
    
    Returns:
        Dict with:
            - contradictions: list of detected contradictions
            - overrides: list of text inferences that should be overridden by visual evidence
            - recommendations: list of recommendations for handling contradictions
    """
    contradictions = {
        "contradictions": [],
        "overrides": [],
        "recommendations": []
    }
    
    # Extract visual evidence
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition = visual_evidence.get("material_condition", {})
    organic_integration = visual_evidence.get("organic_integration", {})
    
    green_coverage = organic_growth.get("green_coverage", 0.0)
    og_confidence = organic_growth.get("confidence", 0.0)
    condition_vis = material_condition.get("condition", "unknown")
    mc_confidence = material_condition.get("confidence", 0.0)
    relationship_vis = organic_integration.get("relationship", "none")
    oi_confidence = organic_integration.get("confidence", 0.0)
    
    # Check text inference (if available)
    text_has_organic = text_inference.get("has_organic_growth", False)
    text_condition = text_inference.get("condition", "unknown")
    text_relationship = text_inference.get("organic_relationship", "none")
    
    # Contradiction 1: Organic growth
    if green_coverage > 0.2 and og_confidence > 0.70 and not text_has_organic:
        contradictions["contradictions"].append({
            "type": "organic_growth_mismatch",
            "visual": f"green_coverage={green_coverage:.3f} (confidence={og_confidence:.2f})",
            "text": "no organic growth detected",
            "severity": "high" if og_confidence > 0.85 else "medium"
        })
        contradictions["overrides"].append({
            "field": "has_organic_growth",
            "from": False,
            "to": True,
            "reason": f"Visual evidence proves {green_coverage:.1%} green coverage with {og_confidence:.0%} confidence"
        })
        contradictions["recommendations"].append("Trust visual evidence - text inference missed organic growth")
    
    # Contradiction 2: Material condition
    if condition_vis != "unknown" and mc_confidence > 0.75:
        if text_condition != "unknown" and text_condition != condition_vis:
            # Check if visual confidence is significantly higher
            if mc_confidence > 0.80:
                contradictions["contradictions"].append({
                    "type": "condition_mismatch",
                    "visual": f"{condition_vis} (confidence={mc_confidence:.2f})",
                    "text": text_condition,
                    "severity": "high"
                })
                contradictions["overrides"].append({
                    "field": "condition",
                    "from": text_condition,
                    "to": condition_vis,
                    "reason": f"Visual texture analysis proves {condition_vis} with {mc_confidence:.0%} confidence"
                })
                contradictions["recommendations"].append("Trust visual evidence - texture analysis is ground truth")
    
    # Contradiction 3: Organic integration
    if relationship_vis != "none" and oi_confidence > 0.75:
        if text_relationship != relationship_vis and text_relationship != "none":
            contradictions["contradictions"].append({
                "type": "integration_mismatch",
                "visual": f"{relationship_vis} (confidence={oi_confidence:.2f})",
                "text": text_relationship,
                "severity": "high" if oi_confidence > 0.85 else "medium"
            })
            contradictions["overrides"].append({
                "field": "organic_relationship",
                "from": text_relationship,
                "to": relationship_vis,
                "reason": f"Visual morphological analysis proves {relationship_vis} with {oi_confidence:.0%} confidence"
            })
            contradictions["recommendations"].append("Trust visual evidence - morphological operations are ground truth")
    
    return contradictions


def extract_visual_features(image_path):
    """
    Extract all visual features using deterministic computer vision.
    Universal: works for any image type.
    
    This is the main entry point for visual grounding.
    Returns ground truth that text matching cannot provide.
    
    ENHANCED: Now includes validation and edge case handling.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with all visual features:
            - organic_growth: dict from detect_organic_growth()
            - material_condition: dict from detect_material_condition()
            - organic_integration: dict from detect_organic_integration()
            - overall_confidence: float - overall confidence in visual analysis
            - validation: dict from validate_visual_evidence()
    """
    try:
        # Extract all visual features
        organic_growth = detect_organic_growth(image_path)
        material_condition = detect_material_condition(image_path)
        
        # For integration detection, we can reuse the green mask
        img = cv2.imread(image_path)
        if img is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            structure_edges = cv2.Canny(gray, 100, 200)
            kernel = np.ones((3, 3), np.uint8)
            structure_edges = cv2.dilate(structure_edges, kernel, iterations=1)
            
            organic_integration = detect_organic_integration(image_path, green_mask, structure_edges)
        else:
            organic_integration = {"overlap_ratio": 0.0, "relationship": "none", "integration_level": "none",
                                  "evidence": ["image_load_failed"], "confidence": 0.0}
        
        # Overall confidence (weighted average)
        confidences = [
            organic_growth.get("confidence", 0.0),
            material_condition.get("confidence", 0.0),
            organic_integration.get("confidence", 0.0)
        ]
        overall_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        visual_evidence = {
            "organic_growth": organic_growth,
            "material_condition": material_condition,
            "organic_integration": organic_integration,
            "overall_confidence": overall_confidence
        }
        
        # Validate visual evidence
        validation = validate_visual_evidence(visual_evidence)
        visual_evidence["validation"] = validation
        
        # Log validation warnings and issues
        if validation["warnings"]:
            logger.info(f"Visual evidence validation warnings: {validation['warnings']}")
        if validation["issues"]:
            logger.warning(f"Visual evidence validation issues: {validation['issues']}")
        
        return visual_evidence
    except Exception as e:
        logger.warning(f"Visual feature extraction failed: {e}")
        return {
            "organic_growth": {"green_coverage": 0.0, "confidence": 0.0},
            "material_condition": {"condition": "unknown", "confidence": 0.0},
            "organic_integration": {"relationship": "none", "confidence": 0.0},
            "overall_confidence": 0.0,
            "validation": {"is_valid": False, "warnings": [], "issues": [f"Extraction failed: {str(e)}"]}
    }


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

        # Invalidate if cache version changed
        if cached_result.get("_cache_version") != CACHE_VERSION:
            logger.info(f"Cache invalidated for {file_hash[:8]}... (version mismatch)")
            return None

        # Remove cache metadata before returning
        cached_result.pop("_cache_version", None)

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
    Cache is invalidated when CACHE_VERSION changes.
    """
    if not file_hash:
        return False
    
    cache_path = os.path.join(ANALYSIS_CACHE_DIR, f"{file_hash}.json")
    
    try:
        ensure_directories()
        to_save = dict(result)
        to_save["_cache_version"] = CACHE_VERSION
        with open(cache_path, 'w') as f:
            json.dump(to_save, f, indent=2, default=str)
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





def analyze_image(path, photo_id: str = "", filename: str = "", disable_cache: bool = False):
    """
    Analyzes an image and returns results in the canonical schema format.
    
    Args:
        path: Path to the image file
        photo_id: Optional unique identifier for the photo
        filename: Optional original filename
        disable_cache: If True, skip cache lookup and recompute analysis
        
    Returns:
        Dict conforming to canonical AnalysisResult schema
    """
    ensure_directories()
    logger.info(f"Analyzing image: {path}")
    
    # Compute file hash for caching
    file_hash = compute_file_hash(path)
    if not photo_id:
        photo_id = file_hash[:16] if file_hash else str(uuid.uuid4())
    
    # Check cache first (unless disabled)
    if not disable_cache:
        cached_result = get_cached_analysis(file_hash)
    else:
        cached_result = None
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

        # === AI + SEMANTIC ANALYSIS (parallel perception) ===
        # Run independent perception steps in parallel; cap workers to avoid GPU OOM.
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

        def _run_one(name, func, args, default, error_key):
            try:
                return (name, func(*args), error_key, None)
            except Exception as e:
                logger.warning(f"Perception '{name}' failed: {e}")
                return (name, default, error_key, str(e))

        nima_model = None
        try:
            nima_model = get_nima_model()
        except Exception as e:
            logger.warning(f"NIMA model load failed (non-fatal): {e}")
            result["errors"]["nima_load"] = str(e)

        clip_default = {"caption": None, "tags": [], "genre_hint": None}
        objects_default = {"objects": [], "object_narrative": None, "subject_position": None, "subject_size": None, "framing_description": None, "spatial_interpretation": None}
        nima_default = {"mean_score": None, "distribution": {}}
        color_default = {"palette": [], "mood": None}
        harmony_default = {"dominant_color": None, "harmony": None}
        lines_default = {"line_pattern": None, "line_style": None, "symmetry": None}
        lighting_default = {"direction": None}
        tonal_default = {"tonal_range": None}
        emotion_default = {"subject_type": None, "emotion": None}
        clutter_default = {"clutter_level": None}

        tasks = [
            ("clip", get_clip_description, (path,), clip_default, "clip"),
            ("clip_inventory", get_clip_inventory, (path,), [], "clip_inventory"),
            ("nima", predict_nima_score, (nima_model, path), nima_default, "nima"),
            ("color", analyze_color, (path,), color_default, "color"),
            ("color_harmony", analyze_color_harmony, (path,), harmony_default, "color_harmony"),
            ("objects", detect_objects_and_framing, (path,), objects_default, "objects"),
            ("lines_symmetry", analyze_lines_and_symmetry, (path,), lines_default, "lines_symmetry"),
            ("lighting", analyze_lighting_direction, (path,), lighting_default, "lighting"),
            ("tonal_range", analyze_tonal_range, (path,), tonal_default, "tonal_range"),
            ("subject_emotion", analyze_subject_emotion, (path,), emotion_default, "emotion"),
            ("clutter", analyze_background_clutter, (path,), clutter_default, "clutter"),
            ("visual_evidence", extract_visual_features, (path,), {}, "visual_evidence"),
        ]
        out = {}
        with ThreadPoolExecutor(max_workers=PERCEPTION_MAX_WORKERS) as executor:
            futures = {executor.submit(_run_one, *t): t[0] for t in tasks}
            for future in as_completed(futures):
                name, value, error_key, error_msg = future.result()
                if error_msg and error_key:
                    result["errors"][error_key] = error_msg
                out[name] = value

        clip_data = out.get("clip", clip_default)
        clip_inventory = out.get("clip_inventory", [])
        nima_result = out.get("nima", nima_default)
        color_analysis = out.get("color", color_default)
        color_harmony = out.get("color_harmony", harmony_default)
        object_data = out.get("objects", objects_default)
        lines_symmetry = out.get("lines_symmetry", lines_default)
        lighting_direction = out.get("lighting", lighting_default)
        tonal_range = out.get("tonal_range", tonal_default)
        subject_emotion = out.get("subject_emotion", emotion_default)
        clutter_result = out.get("clutter", clutter_default)
        visual_evidence = out.get("visual_evidence", {})

        # Store YOLO objects in canonical location for downstream modules (e.g. negative_evidence.py)
        # negative_evidence.py expects analysis_result["objects"] as a list of dicts with "name".
        try:
            objs = object_data.get("objects", []) if isinstance(object_data, dict) else []
            result["objects"] = [{"name": str(o)} for o in (objs or [])]
        except Exception:
            result["objects"] = []

        if clip_data.get("caption"):
            result["perception"]["semantics"]["available"] = True
            result["perception"]["semantics"]["caption"] = clip_data.get("caption")
            result["perception"]["semantics"]["tags"] = clip_data.get("tags", [])
            result["perception"]["semantics"]["genre_hint"] = clip_data.get("genre_hint")
            result["confidence"]["clip"] = True
        if nima_result.get("mean_score") is not None:
            result["perception"]["aesthetics"]["available"] = True
            result["perception"]["aesthetics"]["mean_score"] = nima_result.get("mean_score")
            result["perception"]["aesthetics"]["distribution"] = nima_result.get("distribution", {})
            result["confidence"]["nima"] = True
        if color_analysis.get("palette"):
            result["perception"]["color"]["available"] = True
            result["perception"]["color"]["palette"] = color_analysis.get("palette", [])
            result["perception"]["color"]["mood"] = color_analysis.get("mood")
        if color_harmony.get("dominant_color"):
            result["perception"]["color"]["harmony"]["dominant_color"] = color_harmony.get("dominant_color")
            result["perception"]["color"]["harmony"]["harmony_type"] = color_harmony.get("harmony")
        if object_data.get("objects"):
            result["perception"]["composition"]["available"] = True
            result["perception"]["composition"]["subject_framing"] = {
                "position": object_data.get("subject_position"),
                "size": object_data.get("subject_size"),
                "style": object_data.get("framing_description"),
                "interpretation": object_data.get("spatial_interpretation")
            }
            result["confidence"]["yolo"] = True
        if lines_symmetry.get("line_pattern"):
            result["perception"]["composition"]["line_pattern"] = lines_symmetry.get("line_pattern")
            result["perception"]["composition"]["line_style"] = lines_symmetry.get("line_style")
            result["perception"]["composition"]["symmetry"] = lines_symmetry.get("symmetry")
        if lighting_direction.get("direction"):
            result["perception"]["lighting"]["available"] = True
            result["perception"]["lighting"]["direction"] = lighting_direction.get("direction")
        if tonal_range.get("tonal_range"):
            result["perception"]["lighting"]["quality"] = tonal_range.get("tonal_range")
        if subject_emotion.get("subject_type"):
            result["perception"]["emotion"]["available"] = True
            result["perception"]["emotion"]["subject_type"] = subject_emotion.get("subject_type")
            result["perception"]["emotion"]["emotion"] = subject_emotion.get("emotion")
        result["confidence"]["deepface"] = False

        # === DERIVED FIELDS ===
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
            "background_clutter": clutter_result
        }
        
        # Derived fields: run interpret_visual_features, infer_emotion, detect_genre in parallel
        def _run_derived(name, func, default, error_key):
            try:
                return (name, func(legacy_dict), error_key, None)
            except Exception as e:
                logger.warning(f"Derived '{name}' failed: {e}")
                return (name, default, error_key, str(e))
        derived_tasks = [
            ("visual_interp", interpret_visual_features, {}, "visual_interpretation"),
            ("emotion_result", infer_emotion, {"emotional_mood": None}, "emotion_inference"),
            ("genre_info", detect_genre, {"genre": None, "subgenre": None}, "genre_detection"),
        ]
        derived_out = {}
        with ThreadPoolExecutor(max_workers=3) as exec_derived:
            futures = {exec_derived.submit(_run_derived, t[0], t[1], t[2], t[3]): t[0] for t in derived_tasks}
            for future in as_completed(futures):
                name, value, error_key, error_msg = future.result()
                if error_msg and error_key:
                    result["errors"][error_key] = error_msg
                derived_out[name] = value
        visual_interp = derived_out.get("visual_interp", {})
        emotion_result = derived_out.get("emotion_result", {})
        genre_info = derived_out.get("genre_info", {})
        if visual_interp:
            result["derived"]["visual_interpretation"] = visual_interp
        if emotion_result.get("emotional_mood"):
            result["derived"]["emotional_mood"] = emotion_result.get("emotional_mood")
        if genre_info.get("genre"):
            result["derived"]["genre"]["genre"] = genre_info.get("genre")
            result["derived"]["genre"]["subgenre"] = genre_info.get("subgenre")
        
        # Visual evidence was extracted in parallel above; log and detect contradictions
        og = visual_evidence.get("organic_growth", {})
        mc = visual_evidence.get("material_condition", {})
        oi = visual_evidence.get("organic_integration", {})
        validation = visual_evidence.get("validation", {})
        if visual_evidence:
            logger.info(f"Visual evidence extracted: "
                       f"green_coverage={og.get('green_coverage', 0.0):.3f} (conf={og.get('confidence', 0.0):.2f}), "
                       f"condition={mc.get('condition', 'unknown')} (conf={mc.get('confidence', 0.0):.2f}), "
                       f"integration={oi.get('relationship', 'none')} (conf={oi.get('confidence', 0.0):.2f}), "
                       f"overall_conf={visual_evidence.get('overall_confidence', 0.0):.2f}")
            if validation.get("warnings"):
                logger.info(f"Visual evidence warnings: {validation['warnings']}")
            if validation.get("issues"):
                logger.warning(f"Visual evidence issues: {validation['issues']}")
            if clip_data.get("caption") or clip_data.get("tags"):
                text_inference = {
                    "has_organic_growth": any(term in (clip_data.get("caption", "") + " " + " ".join(clip_data.get("tags", []))).lower()
                                             for term in ["ivy", "moss", "vegetation", "green", "growth"]),
                    "condition": "unknown",
                    "organic_relationship": "none"
                }
                contradictions = detect_contradictions(visual_evidence, text_inference)
                if contradictions["contradictions"]:
                    logger.info(f"Detected {len(contradictions['contradictions'])} contradictions between visual and text")
                    for cont in contradictions["contradictions"]:
                        logger.info(f"  - {cont['type']}: visual={cont['visual']}, text={cont['text']} (severity={cont['severity']})")
                    if contradictions["overrides"]:
                        logger.info(f"Visual evidence will override text inference in {len(contradictions['overrides'])} cases")
        
        # Store visual evidence in result (for interpretive reasoner and scene understanding)
        if visual_evidence:
            result["visual_evidence"] = visual_evidence
        
        # === NEGATIVE EVIDENCE DETECTION (Phase 1.2) ===
        try:
            from .negative_evidence import detect_negative_evidence
            negative_evidence = detect_negative_evidence(result)
            if negative_evidence:
                # Store in visual evidence for use by reasoner
                if visual_evidence:
                    visual_evidence["negative_evidence"] = negative_evidence
                    result["visual_evidence"] = visual_evidence
                # Also store directly in result
                if not result.get("visual_evidence"):
                    result["visual_evidence"] = {}
                result["visual_evidence"]["negative_evidence"] = negative_evidence
        except Exception as e:
            logger.warning(f"Negative evidence detection failed (non-fatal): {e}")
        
        # === PLACES365 SCENE/ATTRIBUTE SIGNALS ===
        # ResNet50-Places365: Scene category probabilities, indoor/outdoor, man-made vs natural, high-level attributes
        # Position: After Visual Evidence, Before Interpretive Reasoner
        # Purpose: Provides scene understanding signals that feed into the reasoner
        places365_signals = {}
        try:
            places365_signals = extract_places365_signals(path)
            if places365_signals:
                # Store in perception layer for use by reasoner
                if not result.get("perception"):
                    result["perception"] = {}
                if not result["perception"].get("scene"):
                    result["perception"]["scene"] = {}
                result["perception"]["scene"]["places365"] = places365_signals
                logger.info(f"Places365 signals extracted: scene_category={places365_signals.get('scene_category', 'unknown')}, "
                           f"indoor_outdoor={places365_signals.get('indoor_outdoor', 'unknown')}, "
                           f"man_made_natural={places365_signals.get('man_made_natural', 'unknown')}")
        except Exception as e:
            logger.warning(f"Places365 signal extraction failed (non-fatal): {e}")

        # === SCENE / ABSTRACTION GATE (Phase 10: perception diversity) ===
        # Purpose: prevent "material-surface reasoning" from leaking into normal scenes.
        # This is heuristic (not training) and intentionally conservative, but uses
        # multiple signals (Places365 + YOLO + CLIP) to classify the scene.
        try:
            places = (result.get("perception", {}).get("scene", {}) or {}).get("places365", {}) or {}
            scene_category = str(places.get("scene_category", "") or "").lower()
            indoor_outdoor = str(places.get("indoor_outdoor", "") or "").lower()

            yolo_objs = [str(o).lower() for o in (object_data.get("objects", []) or [])]
            caption = str(clip_data.get("caption", "") or "").lower()
            tags = [str(t).lower() for t in (clip_data.get("tags", []) or [])]
            inv_items = []
            if isinstance(clip_inventory, list):
                for item in clip_inventory:
                    if isinstance(item, dict):
                        inv_items.append(str(item.get("item", "")).lower())
                    else:
                        inv_items.append(str(item).lower())
            text_blob = " ".join([caption] + tags + inv_items + yolo_objs + [scene_category]).lower()

            # Basic counts
            def _count_any(candidates):
                return sum(1 for o in yolo_objs if any(tok in o for tok in candidates))

            people_terms = ["person", "people", "man", "woman", "child", "face"]
            vehicle_terms = ["car", "truck", "bus", "taxi", "bicycle", "motorcycle"]
            building_terms = ["building", "tower", "skyscraper", "bridge"]

            num_people = _count_any(people_terms) or sum(1 for t in tags if "person" in t or "people" in t)
            num_vehicles = _count_any(vehicle_terms)
            num_buildings = _count_any(building_terms) or ("city" in text_blob) or ("street" in text_blob)
            looks_abstract_terms = any(k in text_blob for k in ["abstract", "nonrepresentational", "non-representational"])
            looks_painting = any(k in text_blob for k in ["painting", "canvas", "acrylic", "oil painting", "artwork"])

            # Visual artisticness fallback (for when CLIP fails and Places365 is noisy):
            # high average saturation + no detected objects often indicates artwork / non-photographic surfaces.
            avg_saturation = None
            try:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                avg_saturation = float(hsv_img[:, :, 1].mean() / 255.0)
            except Exception:
                avg_saturation = None

            # Coarse scene typing (minimal, extensible taxonomy)
            scene_type = "unknown"
            # Prefer explicit "painting/canvas" for abstract_art; pure "abstract" can be a false positive on textures.
            if looks_painting or (looks_abstract_terms and ("painting" in text_blob or "canvas" in text_blob)):
                scene_type = "abstract_art"
            elif (avg_saturation is not None and avg_saturation > 0.50 and num_people == 0 and num_vehicles == 0 and not num_buildings):
                # Strong color saturation + no objects: likely artistic / non-representational
                scene_type = "abstract_art"
            elif num_people > 0:
                scene_type = "people_scene"
            elif indoor_outdoor == "indoor" or any(
                k in text_blob
                for k in [
                    "living room",
                    "interior",
                    "bedroom",
                    "room",
                    "sofa",
                    "couch",
                    "chair",
                    "desk",
                    "table",
                    "tv",
                    "window",
                ]
            ):
                scene_type = "interior_scene"
            elif num_buildings or num_vehicles or any(
                k in text_blob for k in ["downtown", "urban", "intersection", "traffic", "crosswalk"]
            ):
                scene_type = "street_scene"
            elif indoor_outdoor == "outdoor" and any(
                k in text_blob
                for k in [
                    "mountain",
                    "mountains",
                    "lake",
                    "river",
                    "forest",
                    "field",
                    "landscape",
                    "tree",
                    "trees",
                    "house",
                    "cabin",
                ]
            ):
                scene_type = "landscape_scene"

            # Decide whether "material-surface" signals are likely the subject
            og = (visual_evidence or {}).get("organic_growth", {}) or {}
            mc = (visual_evidence or {}).get("material_condition", {}) or {}
            oi = (visual_evidence or {}).get("organic_integration", {}) or {}
            green_cov = float(og.get("green_coverage", 0.0) or 0.0)
            edge_deg = float(mc.get("edge_degradation", 0.0) or 0.0)
            rough = float(mc.get("surface_roughness", 0.0) or 0.0)

            # Surface study is plausible only when scene is NOT clearly a scene depiction.
            scene_is_depiction = scene_type in {
                "abstract_art",
                "people_scene",
                "interior_scene",
                "street_scene",
                "landscape_scene",
            }
            is_surface_study = (
                not scene_is_depiction
                and (edge_deg > 0.6 or rough > 0.12)
                and num_people == 0
                and num_vehicles == 0
                and not num_buildings
            )
            if is_surface_study:
                scene_type = "surface_study"

            # Explain why surface_study was rejected (optional debugging aid)
            rejection_reasons = []
            if not is_surface_study:
                if scene_is_depiction:
                    rejection_reasons.append(f"scene_type={scene_type}")
                if indoor_outdoor in {"indoor", "outdoor"}:
                    rejection_reasons.append(f"places_indoor_outdoor={indoor_outdoor}")
                if num_people > 0:
                    rejection_reasons.append("objects_detected=people")
                if num_vehicles > 0:
                    rejection_reasons.append("objects_detected=vehicles")
                if num_buildings:
                    rejection_reasons.append("objects_detected=buildings_or_urban_terms")
                if looks_painting:
                    rejection_reasons.append("semantic=painting_terms")
                elif looks_abstract_terms:
                    rejection_reasons.append("semantic=abstract_terms")
                if not (edge_deg > 0.6 or rough > 0.12):
                    rejection_reasons.append("texture_signal_not_strong_enough")

            scene_gate = {
                "scene_type": scene_type,
                "is_surface_study": bool(is_surface_study),
                "surface_study_rejection_reasons": rejection_reasons,
                "signals": {
                    "places_scene_category": scene_category,
                    "places_indoor_outdoor": indoor_outdoor,
                    "yolo_objects": yolo_objs[:20],
                    "clip_caption": caption[:200],
                    "green_coverage": round(green_cov, 3),
                    "edge_degradation": round(edge_deg, 3),
                    "surface_roughness": round(rough, 3),
                },
                "note": (
                    "If not surface_study: treat material_condition/organic_integration as background-only metrics; "
                    "do not center interpretation on 'weathered stone / reclamation'."
                ),
            }
            if visual_evidence is not None:
                visual_evidence["scene_gate"] = scene_gate
                result["visual_evidence"] = visual_evidence

            # Hard guardrail: if it's clearly a scene depiction, disable organic_integration claims
            # (These morphological heuristics are only meaningful for true surface studies.)
            if (visual_evidence is not None) and scene_is_depiction:
                oi2 = (visual_evidence.get("organic_integration", {}) or {}).copy()
                if oi2.get("relationship") not in [None, "none"]:
                    oi2["relationship"] = "none"
                    oi2["integration_level"] = "none"
                    oi2["confidence"] = min(float(oi2.get("confidence", 0.0) or 0.0), 0.2)
                    ev = oi2.get("evidence", []) or []
                    ev = list(ev) + [f"scene_gate={scene_type}: organic_integration disabled for non-surface scenes"]
                    oi2["evidence"] = ev[-8:]
                    visual_evidence["organic_integration"] = oi2
                    result["visual_evidence"] = visual_evidence
        except Exception as e:
            logger.warning(f"Scene gate failed (non-fatal): {e}")
        
        # === INTERPRETIVE REASONER (THE BRAIN) ===
        # Phase 3: Reasoning-first architecture - interpret evidence before critique
        # NOTE: This is the OLD reasoner. When intelligence core is active, this is optional/fallback.
        # NOTE: interpret_scene.py and interpretive_memory.py are legacy fallbacks only.
        # They must never run when FRAMED_USE_INTELLIGENCE_CORE is enabled.
        # Feature flag: FRAMED_USE_INTELLIGENCE_CORE (default: true) controls whether to skip this.
        USE_INTELLIGENCE_CORE = os.getenv("FRAMED_USE_INTELLIGENCE_CORE", "true").lower() == "true"
        
        if not USE_INTELLIGENCE_CORE:
            # Old reasoner (backward compatibility mode)
            try:
                from .interpret_scene import interpret_scene
                from .interpretive_memory import create_pattern_signature, query_memory_patterns, store_interpretation
                
                # Prepare semantic signals
                # Handle new CLIP inventory format (list of dicts with confidence/source) or legacy format (list of strings)
                if isinstance(clip_inventory, list) and len(clip_inventory) > 0:
                    if isinstance(clip_inventory[0], dict):
                        # New format: list of dicts with item, confidence, source
                        clip_inventory_items = [item["item"] for item in clip_inventory]
                    else:
                        # Legacy format: list of strings
                        clip_inventory_items = clip_inventory
                else:
                    clip_inventory_items = []
                
                semantic_signals = {
                    "clip_inventory": clip_inventory_items,  # Use items only for reasoner
                    "clip_inventory_full": clip_inventory if isinstance(clip_inventory, list) and len(clip_inventory) > 0 and isinstance(clip_inventory[0], dict) else [],  # Full metadata for other uses
                    "clip_caption": clip_data.get("caption", ""),
                    "yolo_objects": object_data.get("objects", [])
                }
                
                # Prepare technical stats
                technical_stats = {
                    "brightness": result["perception"]["technical"].get("brightness"),
                    "contrast": result["perception"]["technical"].get("contrast"),
                    "sharpness": result["perception"]["technical"].get("sharpness"),
                    "color_mood": result["perception"]["color"].get("mood")
                }
                
                # Query interpretive memory for similar patterns
                pattern_signature = create_pattern_signature(visual_evidence, semantic_signals)
                memory_patterns = query_memory_patterns(pattern_signature, limit=5)
                
                # Call interpretive reasoner
                interpretive_conclusions = interpret_scene(
                    visual_evidence=visual_evidence,
                    semantic_signals=semantic_signals,
                    technical_stats=technical_stats,
                    interpretive_memory_patterns=memory_patterns
                )
                
                # Store conclusions in result
                result["interpretive_conclusions"] = interpretive_conclusions
                
                # Store interpretation in memory for learning
                primary = interpretive_conclusions.get("primary_interpretation", {})
                chosen_interp = primary.get("conclusion", "unclear_interpretation")
                confidence = primary.get("confidence", 0.5)
                store_interpretation(pattern_signature, chosen_interp, confidence)
                
                logger.info(f"Interpretive reasoner: {chosen_interp} (confidence: {confidence:.2f})")
                
            except Exception as e:
                logger.warning(f"Interpretive reasoner failed (non-fatal): {e}")
                # Don't add error - reasoner is optional for now (backward compatibility)
                result["interpretive_conclusions"] = {}
        
        # === SCENE UNDERSTANDING SYNTHESIS ===
        # Synthesize contextual understanding of "what is happening here"
        # Called after all perception, before semantic anchors
        # NOW USES VISUAL EVIDENCE as primary source (ground truth from pixels)
        # NOTE: When intelligence core is active, this is optional/fallback.
        # Intelligence core Layer 1-4 provides scene understanding via LLM reasoning.
        # This rule-based synthesis is kept for backward compatibility and as fallback.
        # Temporarily store clip_inventory and image_path for Scene Understanding access
        result["_clip_inventory"] = clip_inventory if isinstance(clip_inventory, list) else []
        result["_image_path"] = path  # For visual evidence extraction in Scene Understanding
        
        # Only run rule-based scene understanding if intelligence core is disabled
        # Otherwise, intelligence core will provide scene understanding
        if not USE_INTELLIGENCE_CORE:
            try:
                scene_understanding = synthesize_scene_understanding(result)
                # Only add understanding if any elements were synthesized (sparse by default)
                if scene_understanding:
                    result["scene_understanding"] = scene_understanding
            except Exception as e:
                logger.warning(f"Scene understanding synthesis failed (non-fatal): {e}")
                # Don't add error - understanding is optional
        
        # Clean up temporary storage
        result.pop("_clip_inventory", None)
        result.pop("_image_path", None)
        result.pop("_visual_evidence", None)
        
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
        
        # === INTELLIGENCE CORE (7-LAYER REASONING) ===
        # Phase 1: Intelligence Core - FRAMED's brain
        # This replaces rule-based synthesis with LLM-based reasoning
        #
        # Feature flag: used by regression tests to avoid API spend.
        ENABLE_INTELLIGENCE_CORE = os.getenv("FRAMED_ENABLE_INTELLIGENCE_CORE", "true").lower() == "true"
        if ENABLE_INTELLIGENCE_CORE:
            try:
                from .intelligence_core import framed_intelligence
                from .temporal_memory import (
                    create_pattern_signature as create_temporal_signature,
                    format_temporal_memory_for_intelligence,
                    store_interpretation as store_temporal_interpretation,
                    track_user_trajectory,
                )

                # Prepare semantic signals for intelligence core
                semantic_signals_for_intelligence = {
                    "objects": object_data.get("objects", []),
                    "tags": clip_data.get("tags", []),
                    "caption_keywords": clip_data.get("caption", "").split()[:20] if clip_data.get("caption") else [],
                }

                # Create pattern signature for temporal memory
                temporal_signature = create_temporal_signature(visual_evidence, semantic_signals_for_intelligence)

                # Query temporal memory
                temporal_memory_data = format_temporal_memory_for_intelligence(temporal_signature, user_id=photo_id)

                # Get user history (from temporal memory)
                user_history = temporal_memory_data.get("user_trajectory", {})

                # Run intelligence core (7-layer reasoning)
                intelligence_output = framed_intelligence(
                    visual_evidence=visual_evidence,
                    analysis_result=result,
                    temporal_memory=temporal_memory_data,
                    user_history=user_history,
                    pattern_signature=temporal_signature,
                )

                # Store intelligence output in result
                result["intelligence"] = intelligence_output
                result["pattern_signature"] = temporal_signature  # For HITL feedback injection

                # Store interpretation in temporal memory for learning
                confidence = intelligence_output.get("meta_cognition", {}).get("confidence", 0.85)
                store_temporal_interpretation(
                    signature=temporal_signature,
                    interpretation=intelligence_output,
                    confidence=confidence,
                )

                # Track user trajectory
                track_user_trajectory(
                    analysis_result=result,
                    intelligence_output=intelligence_output,
                    user_id=photo_id,
                )

                # Implicit learning (Phase 4)
                try:
                    from .learning_system import learn_implicitly
                    learn_implicitly(
                        analysis_result=result,
                        intelligence_output=intelligence_output,
                        user_history=user_history,
                    )
                except Exception as e:
                    logger.warning(f"Implicit learning failed (non-fatal): {e}")

                logger.info(f"Intelligence core completed: confidence={confidence:.2f}")
            except Exception as e:
                logger.warning(f"Intelligence core failed (non-fatal): {e}")
                # Don't add error - intelligence core is optional for backward compatibility
                result["intelligence"] = {}
        else:
            # Useful for regression tests that must not spend API credits.
            result["intelligence"] = {}
        
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

def generate_vocabulary_locks(scene_understanding, visual_evidence=None):
    """
    Generate universal vocabulary locks based on visual evidence and scene understanding.
    Works for ANY image type (architecture, nature, portraits, street, interiors).
    
    Universal patterns:
    - High coverage + structural salience → forbid "minimal"
    - Organic growth + weathering → forbid "cold/sterile"
    - No humans + organic integration + slow pace → forbid "alienation"
    
    Args:
        scene_understanding: Dict from synthesize_scene_understanding()
        visual_evidence: Optional dict from extract_visual_features()
    
    Returns:
        Dict with:
            - forbidden_words: list of words/phrases that must not be used
            - required_words: list of words/phrases that should be used
            - vocabulary_rules: list of explainable rules
    """
    locks = {
        "forbidden_words": [],
        "required_words": [],
        "vocabulary_rules": []
    }
    
    # Extract visual evidence if not provided
    if visual_evidence is None:
        # Try to extract from scene understanding
        material_condition = scene_understanding.get("material_condition", {})
        organic_interaction = scene_understanding.get("organic_interaction", {})
        
        # Extract from material condition if it has visual source
        if material_condition.get("source") == "visual_analysis":
            green_coverage = material_condition.get("organic_growth_coverage", 0.0)
            salience = material_condition.get("organic_growth_salience", "minimal")
            condition = material_condition.get("surface_state", "unknown")
        else:
            green_coverage = 0.0
            salience = "minimal"
            condition = "unknown"
        
        relationship = organic_interaction.get("relationship", "none")
        integration_level = organic_interaction.get("integration_level", "none")
    else:
        organic_growth = visual_evidence.get("organic_growth", {})
        material_condition_vis = visual_evidence.get("material_condition", {})
        organic_integration_vis = visual_evidence.get("organic_integration", {})
        
        green_coverage = organic_growth.get("green_coverage", 0.0)
        salience = organic_growth.get("salience", "minimal")
        condition = material_condition_vis.get("condition", "unknown")
        relationship = organic_integration_vis.get("relationship", "none")
        integration_level = organic_integration_vis.get("integration_level", "none")
    
    # Extract scene understanding components
    material_condition = scene_understanding.get("material_condition", {})
    organic_interaction = scene_understanding.get("organic_interaction", {})
    temporal_context = scene_understanding.get("temporal_context", {})
    negative_evidence = scene_understanding.get("negative_evidence", {})
    emotional_substrate = scene_understanding.get("emotional_substrate", {})
    
    # === UNIVERSAL PATTERN 1: Coverage + Salience → Vocabulary Lock ===
    # Works for ANY image with organic growth (ivy, moss, vegetation, plants, etc.)
    if green_coverage > 0.25 and salience in ["structural", "incidental"]:
        # High coverage + structural salience → forbid diminutive words
        locks["forbidden_words"].extend([
            "minimal", "slight", "incidental", "subtle", "trace", "hint", "faint", "barely"
        ])
        locks["vocabulary_rules"].append(
            f"green_coverage={green_coverage:.3f} AND salience={salience} → FORBIDDEN: minimal/slight/incidental"
        )
    
    if green_coverage > 0.35:
        # Very high coverage → require strong descriptors
        locks["required_words"].extend([
            "extensive", "significant", "substantial", "dominant", "prominent"
        ])
        locks["vocabulary_rules"].append(
            f"green_coverage={green_coverage:.3f} → REQUIRED: extensive/significant/substantial"
        )
    
    if salience == "structural":
        # Structural salience → require structural descriptors
        locks["required_words"].extend([
            "structural", "integrated", "reclaiming", "climbing", "on the structure"
        ])
        locks["vocabulary_rules"].append(
            f"salience=structural → REQUIRED: structural/integrated/reclaiming"
        )
    
    # === UNIVERSAL PATTERN 2: Organic Growth + Weathering → Emotional Lock ===
    # Works for ANY image showing organic growth and material weathering
    has_organic = green_coverage > 0.1 or material_condition.get("organic_growth") not in [None, "none"]
    has_weathering = condition in ["weathered", "degraded"] or material_condition.get("surface_state") in ["weathered", "degraded"]
    
    if has_organic and has_weathering:
        # Organic + weathering → forbid cold/sterile emotional descriptors
        locks["forbidden_words"].extend([
            "cold", "sterile", "clinical", "alienating", "distant", "remote", 
            "unfeeling", "barren", "lifeless", "harsh", "uninviting"
        ])
        locks["required_words"].extend([
            "warm", "patient", "enduring", "reclaimed", "lived-in", "softened"
        ])
        locks["vocabulary_rules"].append(
            "organic_growth + weathering → FORBIDDEN: cold/sterile/clinical, REQUIRED: warm/patient/enduring"
        )
    
    # === UNIVERSAL PATTERN 3: Integration Level → Relationship Lock ===
    # Works for ANY image showing organic-inorganic interaction
    if relationship in ["reclamation", "integration"] and integration_level in ["high", "moderate"]:
        # High integration → forbid separation/alienation descriptors
        locks["forbidden_words"].extend([
            "separate", "isolated", "disconnected", "apart", "divorced from"
        ])
        locks["required_words"].extend([
            "integrated", "in dialogue", "intertwined", "unified", "coexisting"
        ])
        locks["vocabulary_rules"].append(
            f"relationship={relationship}, integration={integration_level} → FORBIDDEN: separate/isolated, REQUIRED: integrated/in dialogue"
        )
    
    # === UNIVERSAL PATTERN 4: Negative Evidence + Context → Interpretation Lock ===
    # Works for ANY image with absence of elements
    no_humans = negative_evidence.get("no_human_presence", False)
    temporal_direction = temporal_context.get("temporal_direction", "static")
    temporal_pace = temporal_context.get("pace", "static")
    
    if no_humans and relationship in ["reclamation", "integration"] and temporal_pace in ["slow", "static"]:
        # No humans + organic integration + slow pace → forbid alienation, require stillness
        locks["forbidden_words"].extend([
            "alienation", "isolation", "abandonment", "deserted", "empty", "void",
            "lonely", "forsaken", "neglected", "uninhabited"
        ])
        locks["required_words"].extend([
            "stillness", "patience", "continuity", "endurance", "intentional absence",
            "quiet presence", "temporal continuity"
        ])
        locks["vocabulary_rules"].append(
            "no_human_presence + organic_integration + slow_pace → FORBIDDEN: alienation/isolation, REQUIRED: stillness/patience/continuity"
        )
    
    # === UNIVERSAL PATTERN 5: Temporal Direction → Change Vocabulary ===
    # Works for ANY image showing temporal change
    if temporal_direction == "accreting":
        # Accreting (growing) → require growth vocabulary
        locks["required_words"].extend([
            "growing", "expanding", "emerging", "developing", "accumulating"
        ])
        locks["forbidden_words"].extend([
            "static", "unchanging", "frozen", "stagnant"
        ])
        locks["vocabulary_rules"].append(
            "temporal_direction=accreting → REQUIRED: growing/expanding, FORBIDDEN: static/unchanging"
        )
    elif temporal_direction == "decaying":
        # Decaying → require decay vocabulary
        locks["required_words"].extend([
            "decaying", "eroding", "breaking down", "deteriorating", "fading"
        ])
        locks["forbidden_words"].extend([
            "pristine", "new", "fresh", "untouched"
        ])
        locks["vocabulary_rules"].append(
            "temporal_direction=decaying → REQUIRED: decaying/eroding, FORBIDDEN: pristine/new"
        )
    
    # === UNIVERSAL PATTERN 6: Emotional Substrate Corrective Signals ===
    # Works for ANY image with emotional corrective signals
    if emotional_substrate:
        temperature = emotional_substrate.get("temperature", {})
        if isinstance(temperature, dict) and temperature.get("contradictions"):
            forbidden = temperature["contradictions"].get("forbidden", [])
            if forbidden:
                locks["forbidden_words"].extend(forbidden)
                locks["vocabulary_rules"].append(
                    f"emotional_substrate.temperature → FORBIDDEN: {', '.join(forbidden)}"
                )
        
        presence = emotional_substrate.get("presence", {})
        if isinstance(presence, dict) and presence.get("contradictions"):
            forbidden = presence["contradictions"].get("forbidden", [])
            if forbidden:
                locks["forbidden_words"].extend(forbidden)
                locks["vocabulary_rules"].append(
                    f"emotional_substrate.presence → FORBIDDEN: {', '.join(forbidden)}"
                )
    
    # Deduplicate lists
    locks["forbidden_words"] = sorted(list(set(locks["forbidden_words"])))
    locks["required_words"] = sorted(list(set(locks["required_words"])))
    
    return locks


def generate_resolved_contradictions(scene_understanding, visual_evidence=None):
    """
    Generate resolved contradictions based on visual evidence and scene understanding.
    Universal: works for ANY image type.
    
    Once visual evidence resolves a contradiction, it cannot be used as a tension point.
    
    Universal patterns:
    - Organic growth + weathering → warm vs cold is RESOLVED (warm wins)
    - Organic integration → organic vs sterile is RESOLVED (organic wins)
    - High integration → integrated vs alienated is RESOLVED (integrated wins)
    
    Args:
        scene_understanding: Dict from synthesize_scene_understanding()
        visual_evidence: Optional dict from extract_visual_features()
    
    Returns:
        Dict with:
            - resolved: list of resolved contradictions (cannot be used as tension)
            - valid_tensions: list of valid tension points (can be used)
    """
    resolved = []
    valid_tensions = []
    
    # Extract visual evidence if not provided
    if visual_evidence is None:
        material_condition = scene_understanding.get("material_condition", {})
        organic_interaction = scene_understanding.get("organic_interaction", {})
        
        if material_condition.get("source") == "visual_analysis":
            green_coverage = material_condition.get("organic_growth_coverage", 0.0)
            condition = material_condition.get("surface_state", "unknown")
        else:
            green_coverage = 0.0
            condition = "unknown"
        
        relationship = organic_interaction.get("relationship", "none")
        integration_level = organic_interaction.get("integration_level", "none")
    else:
        organic_growth = visual_evidence.get("organic_growth", {})
        material_condition_vis = visual_evidence.get("material_condition", {})
        organic_integration_vis = visual_evidence.get("organic_integration", {})
        
        green_coverage = organic_growth.get("green_coverage", 0.0)
        condition = material_condition_vis.get("condition", "unknown")
        relationship = organic_integration_vis.get("relationship", "none")
        integration_level = organic_integration_vis.get("integration_level", "none")
    
    # Extract scene understanding components
    material_condition = scene_understanding.get("material_condition", {})
    organic_interaction = scene_understanding.get("organic_interaction", {})
    temporal_context = scene_understanding.get("temporal_context", {})
    negative_evidence = scene_understanding.get("negative_evidence", {})
    
    # === UNIVERSAL PATTERN 1: Organic Growth + Weathering → Warm vs Cold RESOLVED ===
    # Works for ANY image showing organic growth and weathering
    has_organic = green_coverage > 0.1 or material_condition.get("organic_growth") not in [None, "none"]
    has_weathering = condition in ["weathered", "degraded"] or material_condition.get("surface_state") in ["weathered", "degraded"]
    
    if has_organic and has_weathering:
        resolved.append({
            "contradiction": "warm vs cold",
            "resolution": "warm (organic growth + weathering = warmth of time)",
            "reason": "Visual evidence proves organic growth and weathering, which indicates warmth of time, not coldness"
        })
        resolved.append({
            "contradiction": "organic vs sterile",
            "resolution": "organic (organic growth present = organic)",
            "reason": "Visual evidence proves organic growth, which resolves organic vs sterile in favor of organic"
        })
    
    # === UNIVERSAL PATTERN 2: Integration Level → Integrated vs Alienated RESOLVED ===
    # Works for ANY image showing organic-inorganic interaction
    if relationship in ["reclamation", "integration"] and integration_level in ["high", "moderate"]:
        resolved.append({
            "contradiction": "integrated vs alienated",
            "resolution": "integrated (organic integration present = integrated)",
            "reason": "Visual evidence proves organic integration, which resolves integrated vs alienated in favor of integrated"
        })
    
    # === UNIVERSAL PATTERN 3: Negative Evidence + Context → Alienation RESOLVED ===
    # Works for ANY image with absence of elements
    no_humans = negative_evidence.get("no_human_presence", False)
    temporal_pace = temporal_context.get("pace", "static")
    
    if no_humans and relationship in ["reclamation", "integration"] and temporal_pace in ["slow", "static"]:
        resolved.append({
            "contradiction": "stillness vs alienation",
            "resolution": "stillness (no humans + organic integration + slow pace = intentional stillness, not alienation)",
            "reason": "Negative evidence (no humans) combined with organic integration and slow temporal pace indicates intentional stillness, patience, and continuity, not alienation"
        })
    
    # === UNIVERSAL PATTERN 4: Temporal Direction → Change State RESOLVED ===
    # Works for ANY image showing temporal change
    temporal_direction = temporal_context.get("temporal_direction", "static")
    
    if temporal_direction == "accreting":
        resolved.append({
            "contradiction": "static vs changing",
            "resolution": "changing (temporal direction = accreting = growth/change)",
            "reason": "Visual evidence shows temporal direction is accreting (growing), which resolves static vs changing in favor of changing"
        })
    elif temporal_direction == "decaying":
        resolved.append({
            "contradiction": "pristine vs aged",
            "resolution": "aged (temporal direction = decaying = deterioration)",
            "reason": "Visual evidence shows temporal direction is decaying (deteriorating), which resolves pristine vs aged in favor of aged"
        })
    
    # === VALID TENSION POINTS (Universal) ===
    # These are always valid, regardless of image type
    valid_tensions = [
        "permanence vs change",
        "endurance vs use",
        "stillness vs memory",
        "monumentality vs intimacy",
        "presence vs absence",
        "revelation vs concealment",
        "control vs surrender",
        "intention vs accident",
        "precision vs ambiguity",
        "distance vs proximity"
    ]
    
    return {
        "resolved": resolved,
        "valid_tensions": valid_tensions
    }


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
    
    # === EXTRACT INTERPRETIVE CONCLUSIONS (PRIMARY SOURCE) ===
    # Phase 6: Critique receives interpreted conclusions, not raw evidence
    interpretive_conclusions = photo_data.get("interpretive_conclusions", {})
    
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
        
        # Extract scene understanding (if present)
        scene_understanding = photo_data.get("scene_understanding", {})
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
        
        # Extract scene understanding (if present) - legacy format may not have it
        scene_understanding = photo_data.get("scene_understanding", {})

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
    # Phase 6: NEW ORDER: Interpretive Conclusions (AUTHORITATIVE) → Uncertainty Flags → Scene Understanding (fallback) → Semantic Anchors → Observations → Task → Governance
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

You are given VERIFIED OBSERVATIONS and SYNTHESIZED UNDERSTANDING about a photograph.
These are not opinions. They are measured facts and contextual synthesis.

---
"""
    
    # === STEP 1: INTERPRETIVE CONCLUSIONS (AUTHORITATIVE - PRIMARY) ===
    # Phase 6: Critique receives interpreted conclusions, not raw evidence
    interpretive_conclusions_section = ""
    uncertainty_flags_section = ""
    
    if interpretive_conclusions:
        primary = interpretive_conclusions.get("primary_interpretation", {})
        alternatives = interpretive_conclusions.get("alternatives", [])
        uncertainty = interpretive_conclusions.get("uncertainty", {})
        emotional_reading = interpretive_conclusions.get("emotional_reading", {})
        
        # Format primary interpretation
        conclusion = primary.get("conclusion", "unclear_interpretation")
        confidence = primary.get("confidence", 0.5)
        evidence_chain = primary.get("evidence_chain", [])
        reasoning = primary.get("reasoning", "")
        
        interpretive_conclusions_section = f"""
INTERPRETED SCENE CONCLUSIONS (AUTHORITATIVE):
Primary Interpretation: {conclusion}
Confidence: {confidence:.2f}
Evidence Chain: {', '.join(evidence_chain[:5]) if evidence_chain else 'N/A'}
Reasoning: {reasoning}

Alternatives Considered:
"""
        if alternatives:
            for alt in alternatives[:3]:  # Top 3 alternatives
                alt_interp = alt.get("interpretation", "")
                alt_conf = alt.get("confidence", 0.0)
                alt_reason = alt.get("reason_rejected", "")
                interpretive_conclusions_section += f"- {alt_interp} (confidence: {alt_conf:.2f}, rejected: {alt_reason[:100]})\n"
        else:
            interpretive_conclusions_section += "- None\n"
        
        # Format emotional reading
        if emotional_reading:
            primary_emotion = emotional_reading.get("primary", "")
            secondary_emotion = emotional_reading.get("secondary", "")
            emotion_reasoning = emotional_reading.get("reasoning", "")
            interpretive_conclusions_section += f"""
Emotional Reading: {primary_emotion} (secondary: {secondary_emotion})
Emotional Reasoning: {emotion_reasoning}
"""
        
        # Format uncertainty flags
        requires_uncertainty = uncertainty.get("requires_uncertainty_acknowledgment", False)
        uncertainty_reason = uncertainty.get("reason", "")
        
        if requires_uncertainty:
            uncertainty_flags_section = f"""
UNCERTAINTY FLAGS (MANDATORY ACKNOWLEDGMENT):
Confidence is below threshold ({confidence:.2f} < 0.65). You MUST acknowledge ambiguity.
Reason: {uncertainty_reason}

You must use uncertainty language (e.g., "perhaps", "might", "suggests", "appears", "uncertain").
Do not speak with false authority when confidence is low.
"""
    
    # === STEP 2: SCENE UNDERSTANDING (FALLBACK/LEGACY) ===
    # Keep for backward compatibility, but interpretive conclusions take precedence
    scene_understanding_section = ""
    if scene_understanding and not interpretive_conclusions:
        understanding_lines = []
        
        # Material condition
        material = scene_understanding.get("material_condition", {})
        if material:
            material_parts = []
            if "surface_state" in material:
                material_parts.append(f"Surface: {material['surface_state']}")
            if "organic_growth" in material and material["organic_growth"] != "none":
                material_parts.append(f"Organic growth: {material['organic_growth']}")
            if "growth_types" in material:
                material_parts.append(f"Growth types: {', '.join(material['growth_types'])}")
            if "erosion_level" in material:
                material_parts.append(f"Erosion: {material['erosion_level']}")
            if material_parts:
                understanding_lines.append(f"Material Condition: {', '.join(material_parts)}")
        
        # Temporal context
        temporal = scene_understanding.get("temporal_context", {})
        if temporal:
            temporal_parts = []
            if "time_scale" in temporal:
                temporal_parts.append(f"Time scale: {temporal['time_scale']}")
            if "pace" in temporal:
                temporal_parts.append(f"Pace: {temporal['pace']}")
            if "endurance" in temporal:
                temporal_parts.append(f"Endurance: {temporal['endurance']}")
            if temporal_parts:
                understanding_lines.append(f"Temporal Context: {', '.join(temporal_parts)}")
        
        # Organic interaction
        organic = scene_understanding.get("organic_interaction", {})
        if organic:
            organic_parts = []
            if "relationship" in organic and organic["relationship"] != "none":
                organic_parts.append(f"Relationship: {organic['relationship']}")
            if "integration_level" in organic and organic["integration_level"] != "none":
                organic_parts.append(f"Integration: {organic['integration_level']}")
            if organic_parts:
                understanding_lines.append(f"Organic Interaction: {', '.join(organic_parts)}")
        
        # Emotional substrate
        emotional_sub = scene_understanding.get("emotional_substrate", {})
        if emotional_sub:
            emotional_parts = []
            if "temperature" in emotional_sub:
                emotional_parts.append(f"Temperature: {emotional_sub['temperature']}")
            if "pace" in emotional_sub:
                emotional_parts.append(f"Pace: {emotional_sub['pace']}")
            if "presence" in emotional_sub:
                emotional_parts.append(f"Presence: {emotional_sub['presence']}")
            if "quality" in emotional_sub:
                emotional_parts.append(f"Quality: {emotional_sub['quality']}")
            if emotional_parts:
                understanding_lines.append(f"Emotional Substrate: {', '.join(emotional_parts)}")
            
            # Corrective signals
            corrective = emotional_sub.get("corrective_signals", {})
            if corrective:
                corrective_lines = []
                for key, override in corrective.items():
                    if isinstance(override, dict) and "from" in override and "to" in override:
                        corrective_lines.append(f"{override['from']} → {override['to']}: {override.get('reason', '')}")
                if corrective_lines:
                    understanding_lines.append(f"Corrective Signals: {'; '.join(corrective_lines)}")
        
        # Contextual relationships
        relationships = scene_understanding.get("contextual_relationships", {})
        if relationships:
            rel_parts = []
            if "subject_vs_environment" in relationships:
                rel_parts.append(f"Subject-Environment: {relationships['subject_vs_environment']}")
            if "time_vs_subject" in relationships:
                rel_parts.append(f"Time-Subject: {relationships['time_vs_subject']}")
            if "human_vs_space" in relationships:
                rel_parts.append(f"Human-Space: {relationships['human_vs_space']}")
            if rel_parts:
                understanding_lines.append(f"Contextual Relationships: {', '.join(rel_parts)}")
        
        if understanding_lines:
            # Check if visual evidence was used (ground truth from pixels)
            # Look for evidence fields that indicate visual analysis
            visual_evidence_used = False
            for line in understanding_lines:
                if "(visual)" in line or "visual_analysis" in line or "proven from pixels" in line:
                    visual_evidence_used = True
                    break
            
            # Also check material_condition and organic_interaction for visual source
            if not visual_evidence_used:
                material = scene_understanding.get("material_condition", {})
                organic = scene_understanding.get("organic_interaction", {})
                if material.get("source") == "visual_analysis" or organic.get("source") == "visual_analysis":
                    visual_evidence_used = True
            
            if visual_evidence_used:
                scene_understanding_section = f"""
SCENE UNDERSTANDING (AUTHORITATIVE - GROUND TRUTH FROM PIXELS):
{chr(10).join(understanding_lines)}

This understanding synthesizes material condition, temporal context, organic interaction, and emotional substrate.
Elements marked "(visual)" or sourced from "visual_analysis" are PROVEN FROM PIXELS using deterministic computer vision.
These are ground truth measurements, not text inference. Examples:
- Green pixel coverage (organic growth) is measured from HSV color thresholds
- Surface roughness (weathering) is measured from texture variance
- Integration level (organic-structure relationship) is measured from morphological operations

This is AUTHORITATIVE CONTEXT. You must not contradict it. Ground your critique in what is actually happening in the image.
If visual evidence indicates organic growth, weathering, or integration, you MUST reference it explicitly.
If visual evidence forbids "cold" or "clinical" (due to organic warmth), you MUST NOT use those terms.
"""
            else:
                scene_understanding_section = f"""
SCENE UNDERSTANDING (AUTHORITATIVE):
{chr(10).join(understanding_lines)}

This understanding synthesizes material condition, temporal context, organic interaction, and emotional substrate.
This is AUTHORITATIVE CONTEXT. You must not contradict it. Ground your critique in what is actually happening in the image.
"""
        else:
            scene_understanding_section = ""
    
    # === STEP 3: SEMANTIC ANCHORS (NAMING PERMISSION) ===
    # Phase 7: Removed vocabulary locks and resolved contradictions (replaced with conclusion enforcement)
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
SEMANTIC ANCHORS (NAMING PERMISSION):
{chr(10).join(anchors_lines)}

These anchors are safe to reference explicitly.
If structure_elements include specific structures, you must name them.
Do not invent elements beyond these anchors.
"""
    
    # === STEP 5: VERIFIED OBSERVATIONS (TECHNICAL) ===
    observations_section = f"""
VERIFIED OBSERVATIONS (TECHNICAL):
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
    
    # === STEP 5: GOVERNANCE RULES (SIMPLIFIED & STRONG) ===
    # Phase 7: Replace vocabulary locks with conclusion enforcement
    if interpretive_conclusions:
        governance_rules = """
RULES (NON-NEGOTIABLE):
- You must not contradict the Interpretive Conclusions. They are the result of reasoning, not raw evidence.
- You must ground your critique in the primary interpretation and its evidence chain.
- If alternatives were considered and rejected, you must not use those rejected interpretations.
- If uncertainty is flagged, you MUST acknowledge ambiguity explicitly.
- You must not invent facts not in the evidence chain.
- Do NOT describe the image literally.
- Do NOT list tips.
- Do NOT sound instructional.
- Do NOT flatter.

CONCLUSION CONSISTENCY ENFORCEMENT (ABSOLUTE):
- The Interpretive Conclusions are AUTHORITATIVE. You cannot contradict them.
- If the reasoner concluded "ivy on structure", you cannot say "green building".
- If the reasoner rejected "painted surface", you cannot use that interpretation.
- You must work with the conclusions, not against them.

UNCERTAINTY ACKNOWLEDGMENT (MANDATORY):
- If requires_uncertainty_acknowledgment=true, you MUST use uncertainty language.
- Do not speak with false authority when confidence is low.
- Acknowledge ambiguity explicitly (e.g., "perhaps", "might", "suggests", "appears", "uncertain").
"""
    elif scene_understanding:
        # Fallback to scene understanding if interpretive conclusions not available
        governance_rules = """
RULES (NON-NEGOTIABLE):
- You must not contradict Scene Understanding.
- If organic growth or weathering is present in Scene Understanding, you must reference it explicitly.
- If human_presence is 'none detected' in Semantic Anchors, you must not imply or invent human subjects.
- Do not describe the image as cold, sterile, or clinical if Scene Understanding indicates warmth or organic integration.
- Every interpretive claim must be grounded in Scene Understanding, Anchors, or Measured Evidence.
- Do NOT describe the image literally.
- Do NOT list tips.
- Do NOT sound instructional.
- Do NOT flatter.
"""
    else:
        governance_rules = """
RULES (NON-NEGOTIABLE):
- You must ground your critique in verified observations.
- Do not invent facts not in evidence.
- Do NOT describe the image literally.
- Do NOT list tips.
- Do NOT sound instructional.
- Do NOT flatter.
"""
    
    # Assemble prompt in correct order: Interpretive Conclusions → Uncertainty Flags → Scene Understanding (fallback) → Semantic Anchors → Observations → Task → Governance
    # Phase 7: Removed vocabulary_locks_section, resolved_contradictions_section, corrective_signals_section (replaced with conclusion enforcement)
    prompt += interpretive_conclusions_section + uncertainty_flags_section + scene_understanding_section + anchors_section + observations_section + f"""
Your task:

1. Interpret what these choices reveal about the photographer's intent.
2. Identify where the image is honest — and where it is safe.
3. Speak to the photograph as a serious work, not a draft.
4. Surface a tension, contradiction, or unanswered question.
5. End with a provocation that suggests evolution — not instruction.

{governance_rules}

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




