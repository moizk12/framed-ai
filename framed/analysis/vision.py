import os, uuid, json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from .schema import create_empty_analysis_result, normalize_to_schema, validate_schema
from . import prompts_clip

logger = logging.getLogger(__name__)

# REF:D1 — runtime paths (side effect: ensure_directories on import)
from .runtime_paths import (
    BASE_DATA_DIR,
    CACHE_VERSION,
    CACHE_DIR,
    DATA_ROOT,
    DEFAULT_BASE_DATA_DIR,
    EXPRESSION_CACHE_DIR,
    ECHO_MEMORY_PATH,
    HF_HOME,
    HUGGINGFACE_HUB_CACHE,
    MODELS_DIR,
    MODEL_DIR,
    PERCEPTION_MAX_WORKERS,
    RESULTS_FOLDER,
    TMP_FOLDER,
    TRANSFORMERS_CACHE,
    TORCH_HOME,
    ULTRALYTICS_CFG,
    UPLOAD_DIR,
    UPLOAD_FOLDER,
    XDG_CACHE_HOME,
    YOLO_CONFIG_DIR,
    ensure_directories,
)
# REF:D2 — file-hash analysis cache
from .analysis_cache import compute_file_hash, get_cached_analysis, save_cached_analysis

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

# Heavy models (YOLO, CLIP, NIMA): lazy-loaded on first use, not at import.

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", os.path.join(MODEL_DIR, "yolov8n.pt"))
_yolo_model = None

def get_yolo_model():
    from .models import get_yolo_model as _get_yolo_model

    return _get_yolo_model()

# CLIP model - lazy loaded
_clip_model = None
_clip_processor = None
_device = None

def get_clip_model():
    from .models import get_clip_model as _get_clip_model

    return _get_clip_model()


from .visual_evidence import (
    detect_organic_growth,
    detect_material_condition,
    detect_organic_integration,
    extract_places365_signals,
    validate_visual_evidence,
    detect_contradictions,
    extract_visual_features,
)
# REF:C3 — scene anchors + substrate + scene understanding
from .scene_and_anchors import (
    generate_semantic_anchors,
    synthesize_emotional_substrate_constrained,
    synthesize_scene_understanding,
)

# NIMA model - lazy loaded
_nima_model = None

def get_nima_model():
    from .models import get_nima_model as _get_nima_model

    return _get_nima_model()

# OpenAI: no client at import time (/health must stay cheap). Do not cache None;
# re-read OPENAI_API_KEY each call (delayed injection on HF/Gunicorn).
def get_openai_client():
    """Return OpenAI client or None if no API key."""
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
    """Lazy OpenAI client; first attribute access constructs the real client."""
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

client = _ClientProxy()  # legacy alias; prefer get_openai_client() in new code

# REF:C4 — vocab locks / critique / describe_stat (uses lazy vision.get_openai_client inside critique)
from .critique import (
    describe_stat,
    generate_merged_critique,
    generate_resolved_contradictions,
    generate_vocabulary_locks,
)


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

    candidate_captions = prompts_clip.CLIP_CAPTION_CANDIDATES

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
    
    structural_candidates = prompts_clip.CLIP_STRUCTURAL_INVENTORY
    material_condition_candidates = prompts_clip.CLIP_MATERIAL_CONDITION_INVENTORY
    atmosphere_candidates = prompts_clip.CLIP_ATMOSPHERE_INVENTORY

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
    
    return inventory_with_metadata



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
    

def run_full_analysis(image_path, photo_id: str = "", filename: str = ""):
    from .pipeline import run_full_analysis as _run_full_analysis

    return _run_full_analysis(image_path, photo_id=photo_id, filename=filename)


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
    from .pipeline import analyze_image as _analyze_image

    return _analyze_image(path, photo_id=photo_id, filename=filename, disable_cache=disable_cache)
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
            print("Warning: Corrupted echo_memory.json — resetting.")
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

def _get_genre_pair(photo):
    g = photo.get("genre")
    if isinstance(g, dict):
        return g.get("genre", "General"), g.get("subgenre", "General")
    # string + separate subgenre field
    return g or "General", photo.get("subgenre", "General")




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

    mood_poetry = poetic_mood_description(fingerprint["dominant_moods"])
    genre_trend = genre_trajectory(fingerprint["dominant_genres"], fingerprint["dominant_subgenres"])
    style_tech = style_summary(fingerprint["tech_signature"])
    subject_behavior = subject_motif(fingerprint["subject_styles"])
    emotional_theme = join_phrases(fingerprint["emotional_motifs"], "emotional themes")
    lighting = join_phrases(fingerprint["lighting_styles"], "lighting moods", 2)
    color_poetry = join_phrases(fingerprint["color_moods"], "color moods", 2)
    composition_trend = join_phrases(fingerprint["composition_positions"], "framing positions", 3)
    tags = join_phrases(fingerprint["clip_themes"], "CLIP tags", 6)

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

    openai_client = get_openai_client()
    if openai_client is None:
        return "ECHO requires Cloud Enhance. (Host has no OPENAI_API_KEY set.)"
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()




