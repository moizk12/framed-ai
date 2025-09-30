# ========================================================
# 📦 IMPORTS
# ========================================================
from flask import Flask, request, render_template, url_for
import os
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from transformers import CLIPProcessor, CLIPModel
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans
from colorthief import ColorThief
from deepface import DeepFace
from config import OPENAI_API_KEY


# ========================================================
# 🚀 INITIALIZE APP + FOLDERS
# ========================================================

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ========================================================
# 🚀 LOAD AI MODELS
# ========================================================

# YOLO for object detection
yolo_model = YOLO("yolov8n.pt")

# CLIP for semantic description
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# NIMA for aesthetic scoring
def load_nima_model(model_path='models/nima_mobilenet.h5'):
    base_model = MobileNet(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    x = base_model.output
    x = Dropout(0.75)(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights(model_path)
    return model

nima_model = load_nima_model()

# OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# ========================================================
# 📸 IMAGE ANALYSIS FUNCTIONS
# ========================================================

# --- CLIP Semantic Description ---

# --- CLIP Semantic Description (Expanded + Smart version) ---

def get_clip_description(image_path):
    """
    Generate a semantic description and genre hint using CLIP model.
    """

    # Load and prepare image
    image = Image.open(image_path).convert("RGB")

    # Expanded caption list (rich + poetic + genre aware)
    candidate_captions = [

        # --- Portraiture ---
        "A cinematic portrait in dramatic lighting", "A candid photo of a person lost in thought",
        "A joyful person laughing in soft light", "A melancholic person sitting alone",
        "A romantic close-up of two people", "A fashion portrait with striking style",
        "A vintage style portrait with film aesthetics",

        # --- Street / Urban ---
        "A quiet street at night under neon lights", "A chaotic urban scene full of motion",
        "A street musician playing passionately", "A group of people crossing a busy street",
        "A solitary figure walking through the rain",

        # --- Landscape / Nature ---
        "A misty mountain landscape at dawn", "A sunset over a calm lake with reflections",
        "A dramatic stormy sky over vast fields", "A dense forest with sunlight filtering through leaves",
        "A snowy landscape evoking silence and stillness", "A desert scene with strong shadows and patterns",

        # --- Conceptual / Abstract ---
        "A surreal photo blending multiple realities", "An abstract composition of geometric shapes",
        "A dreamy photo with pastel tones and blur", "A minimalist photo with negative space",
        "A colorful light painting in long exposure",

        # --- Wildlife / Animals ---
        "A close-up of a wild animal in its habitat", "A bird soaring freely in the sky",
        "A pet looking curiously at the camera", "A herd of animals moving dynamically",

        # --- Still Life / Object-Based ---
        "A carefully arranged flat lay with balanced composition", "A product shot with clean background and bold lighting",
        "A rustic table setting with natural light",

        # --- Documentary / Emotional Moments ---
        "A powerful protest captured mid-action", "A tender moment between parent and child",
        "A candid emotional embrace", "A person staring out the window thoughtfully",

        # --- Architecture / Interior ---
        "A grand architectural facade with symmetrical design", "An interior space bathed in natural light",
        "A staircase captured with dramatic perspective",

        # --- Experimental / Mixed ---
        "A photo with glitch and digital artifacts", "A double exposure merging city and nature",
        "A photo with intentional motion blur conveying speed",

        # --- Light + Atmosphere Based ---
        "A soft and dreamy scene bathed in golden hour light", "A cold and detached scene in blue tones",
        "A harshly lit photo creating strong shadows", "A foggy and mysterious environment",
        "A backlit subject creating a glowing silhouette",

        # --- Genre Specific ---
        "A street photography shot capturing the decisive moment", "A landscape photo showing nature's grandeur",
        "A portrait that conveys deep emotion", "A fashion photograph emphasizing style and attitude",
        "An abstract photo focusing on colors and shapes"

        "An intentionally blurred artistic expression",
        "A photo capturing nothingness and emptiness, purely abstract",
        "A raw and gritty lo-fi aesthetic shot"

    ]

    # Process using CLIP
    inputs = clip_processor(text=candidate_captions, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    # Find best caption
    best_idx = probs.argmax()
    best_caption = candidate_captions[best_idx]

    tags = []
        # Genre Tags

    if "portrait" in best_caption.lower():
        tags.append("Portrait")
    if "street" in best_caption.lower() or "urban" in best_caption.lower():
        tags.append("Street")
    if "landscape" in best_caption.lower():
        tags.append("Landscape")
    if "surreal" in best_caption.lower() or "abstract" in best_caption.lower():
        tags.append("Abstract")
    if "wild" in best_caption.lower() or "animal" in best_caption.lower():
        tags.append("Wildlife")
    if "fashion" in best_caption.lower():
        tags.append("Fashion")
    if "emotional" in best_caption.lower() or "tender" in best_caption.lower() or "nostalgic" in best_caption.lower():
        tags.append("Emotional")
    if "experimental" in best_caption.lower() or "glitch" in best_caption.lower():
        tags.append("Experimental")

    # Mood Tags
    if "dream" in best_caption.lower() or "fog" in best_caption.lower() or "blur" in best_caption.lower():
        tags.append("Dreamy")
    if "dramatic" in best_caption.lower() or "dark" in best_caption.lower() or "moody" in best_caption.lower():
        tags.append("Moody")
    if "soft" in best_caption.lower() or "warm" in best_caption.lower():
        tags.append("Soft")
    if "chaotic" in best_caption.lower() or "busy" in best_caption.lower():
        tags.append("Chaotic")
    if "clean" in best_caption.lower() or "minimalist" in best_caption.lower():
        tags.append("Minimal")
    # Extract genre hint (optional → helps for genre detection later)
    genre_hint = "General"

    if "portrait" in best_caption.lower():
        genre_hint = "Portrait"
    elif "street" in best_caption.lower():
        genre_hint = "Street"
    elif "landscape" in best_caption.lower() or "nature" in best_caption.lower():
        genre_hint = "Landscape"
    elif "abstract" in best_caption.lower() or "conceptual" in best_caption.lower():
        genre_hint = "Abstract"
    elif "fashion" in best_caption.lower():
        genre_hint = "Fashion"
    elif "animal" in best_caption.lower() or "wild" in best_caption.lower():
        genre_hint = "Wildlife"
    elif "protest" in best_caption.lower() or "documentary" in best_caption.lower():
        genre_hint = "Documentary"

    # Genre Hint
    if tags:
        genre_hint = tags[0]  # First tag as primary genre hint
    else:
        genre_hint = "General"

    return {
        "caption": best_caption,
        "tags": tags,
        "genre_hint": genre_hint
    }

# --- Color Analysis ---
def analyze_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape((-1, 3))

    clt = KMeans(n_clusters=5)
    clt.fit(image)

    colors = clt.cluster_centers_.astype(int)
    hex_colors = ['#%02x%02x%02x' % tuple(color) for color in colors]
    
    avg_color = np.mean(colors, axis=0)
    mood = "warm" if avg_color[0] > avg_color[2] else "cool"

    return {"palette": hex_colors, "mood": mood}

# --- NIMA Aesthetic Scoring ---
def predict_nima_score(model, img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img = keras_image.img_to_array(img) / 255.0
    preds = model.predict(np.expand_dims(img, axis=0))[0]
    mean_score = sum((i + 1) * p for i, p in enumerate(preds))
    return {"mean_score": round(mean_score, 2), "distribution": {str(i + 1): round(p, 4) for i, p in enumerate(preds)}}

# --- Composition & Technical Analyzers (Framing, Lines, Clutter, Tonal Range, Lighting, Emotion) ---
# (This part will continue as-is from your last block → due to message limits I will post PART 2 right after this message)

# ========================================================
# 📊 ADVANCED ANALYSIS FUNCTIONS (CONTINUED)
# ========================================================

def analyze_subject_framing(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    results = yolo_model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    if len(detections) == 0:
        return {
            "position": "No subject detected",
            "size": "N/A",
            "narrative": "Empty or abstract composition"
        }

    # Find largest object (main subject)
    largest = max(detections, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    x1, y1, x2, y2 = largest[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    size_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)

    # --- Position ---
    horizontal = "Left" if center_x < w * 0.33 else "Center" if center_x < w * 0.66 else "Right"
    vertical = "Top" if center_y < h * 0.33 else "Middle" if center_y < h * 0.66 else "Bottom"

    if horizontal == "Center" and vertical == "Middle":
        position_desc = "Centered"
    elif horizontal == "Center":
        position_desc = f"Vertically {vertical.lower()}-heavy"
    elif vertical == "Middle":
        position_desc = f"Horizontally {horizontal.lower()}-placed"
    else:
        position_desc = f"{vertical} {horizontal}"

    # --- Size ---
    if size_ratio > 0.5:
        size_desc = "Extreme close-up, fills the frame"
    elif size_ratio > 0.2:
        size_desc = "Medium shot, clearly visible subject"
    elif size_ratio > 0.05:
        size_desc = "Small in frame, environmental portrait"
    else:
        size_desc = "Tiny or distant subject"

    # --- Narrative interpretation ---
    if size_ratio > 0.5:
        narrative = "Intimate and focused composition"
    elif size_ratio > 0.2:
        narrative = "Balanced presence in the scene"
    elif size_ratio > 0.05:
        narrative = "Subject isolated within environment"
    else:
        narrative = "Subject lost in vast space or background"

    return {
        "position": position_desc,
        "size": size_desc,
        "narrative": narrative
    }

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

    # Analyze symmetry
    flipped = cv2.flip(gray, 1)
    diff = cv2.absdiff(gray, flipped)
    symmetry_score = np.mean(diff)

    if symmetry_score < 10:
        symmetry = "Perfect symmetry (Highly balanced)"
    elif symmetry_score < 30:
        symmetry = "Near symmetry (Natural and pleasant)"
    elif symmetry_score < 60:
        symmetry = "Moderate asymmetry (Dynamic and modern)"
    else:
        symmetry = "No symmetry (Chaotic or intentionally imbalanced)"

    return {
        "line_pattern": pattern,
        "line_style": style,
        "symmetry": symmetry
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

    inputs = clip_processor(text=emotion_captions, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_caption = emotion_captions[probs.argmax()]

    human_detected = any(word in best_caption.lower() for word in ["person", "human"])

    subject_type = "human subject" if human_detected else "non-human / abstract"

    return {
        "subject_type": subject_type,
        "emotion": best_caption
    }



def analyze_subject_emotion(image_path):
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        emotion = analysis['dominant_emotion']
        age = analysis['age']
        gender = analysis['gender']

        if emotion.lower() == "neutral" or emotion.lower() == "unknown":
            # Fallback if DeepFace fails or very neutral
            clip_result = analyze_subject_emotion_clip(image_path)
            emotion = clip_result["emotion"]
            subject_type = clip_result["subject_type"]
        else:
            subject_type = "human subject"

    except Exception as e:
        # If DeepFace fails completely
        clip_result = analyze_subject_emotion_clip(image_path)
        emotion = clip_result["emotion"]
        subject_type = clip_result["subject_type"]
        age = None
        gender = "Unknown"

    return {
        "subject_type": subject_type,
        "emotion": emotion,
        "age": age,
        "gender": gender
    }





# ========================================================
# 📊 MASTER IMAGE ANALYSIS WRAPPER
# ========================================================
def infer_emotion(photo_data):
    brightness = photo_data.get("brightness", 0)
    contrast = photo_data.get("contrast", 0)
    sharpness = photo_data.get("sharpness", 0)
    color_mood = photo_data.get("color_mood", "neutral")
    tonal_range = photo_data.get("tonal_range", "balanced")
    subject_emotion = photo_data.get("subject_emotion", "")
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
    if "sad" in subject_emotion.lower() or "melancholic" in subject_emotion.lower():
        emotion_tags.append("melancholic")
    elif "happy" in subject_emotion.lower() or "smiling" in subject_emotion.lower():
        emotion_tags.append("joyful")
    elif "serene" in subject_emotion.lower():
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
    """
    Determine the photo genre and sub-genre using multiple cues.
    """

    # Extract relevant fields
    clip_genre = photo_data.get("clip_description", {}).get("genre_hint", "General")
    subject = photo_data.get("subject_emotion", {}).get("subject_type", "unknown")
    mood = photo_data.get("emotional_mood", "neutral")
    tonal = photo_data.get("tonal_range", "balanced")
    color_mood = photo_data.get("color_mood", "neutral")
    lighting = photo_data.get("lighting_direction", "undefined")
    sharpness = photo_data.get("sharpness", 0)

    # Primary Genre logic (from CLIP + subject)
    if clip_genre != "General":
        genre = clip_genre
    elif subject == "human subject":
        genre = "Portrait"
    elif subject == "non-human / abstract":
        genre = "Abstract / Conceptual"
    else:
        genre = "General"

    # Sub-genre detection
    subgenre = None

    # Street
    if genre == "Street" or ("chaotic" in mood or "energetic" in mood):
        subgenre = "Candid / Action" if "dynamic" in mood or "motion" in mood else "Atmospheric Street"

    # Portrait
    if genre == "Portrait":
        if "romantic" in mood or "dreamy" in mood:
            subgenre = "Romantic / Dreamy"
        elif "melancholic" in mood or tonal == "low key (dark)":
            subgenre = "Dramatic / Cinematic"
        else:
            subgenre = "Classic Portrait"

    # Landscape
    if genre == "Landscape":
        if tonal == "high key (bright)":
            subgenre = "Bright / Airy"
        elif tonal == "low key (dark)":
            subgenre = "Moody / Cinematic"
        else:
            subgenre = "Balanced / Natural"

    # Abstract
    if genre == "Abstract / Conceptual":
        if "surreal" in mood:
            subgenre = "Surreal / Dreamlike"
        elif "minimal" in mood or photo_data.get("background_clutter", {}).get("clutter_level", "") == "clean":
            subgenre = "Minimalist"
        else:
            subgenre = "Experimental / Artistic"

    # Wildlife
    if genre == "Wildlife":
        if "dynamic" in mood or "action" in mood:
            subgenre = "Action Wildlife"
        else:
            subgenre = "Calm / Observational"

    # Default fallback
    if subgenre is None:
        subgenre = "General"
    if subgenre == "General" and genre != "General":
        subgenre = f"Classic {genre}"

    return {
        "genre": genre,
        "subgenre": subgenre
    }




def analyze_image(path):
    # Load Image + Convert to Gray
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === TECHNICAL ANALYSIS ===
    brightness = round(np.mean(gray), 2)
    contrast = round(gray.std(), 2)
    sharpness = round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)

    # === AI + SEMANTIC ANALYSIS ===
    clip_description = get_clip_description(path)
    nima_result = predict_nima_score(nima_model, path)
    color_analysis = analyze_color(path)
    color_harmony = analyze_color_harmony(path)
    objects = detect_objects(path)
    subject_framing = analyze_subject_framing(path)
    background_clutter = analyze_background_clutter(path)
    lines_symmetry = analyze_lines_and_symmetry(path)
    lighting_direction = analyze_lighting_direction(path)
    tonal_range = analyze_tonal_range(path)
    subject_emotion = analyze_subject_emotion(path)
    genre_info = detect_genre(result)
    # === BUILD INITIAL RESULT ===
    result = {
        # Technical
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,

        # AI Semantic
        "clip_description": clip_description,
        "nima": nima_result,
        "color_palette": color_analysis["palette"],
        "color_mood": color_analysis["mood"],
        "color_harmony": color_harmony,
        "objects": objects,
        "subject_framing": subject_framing,
        "background_clutter": background_clutter,
        "leading_lines": lines_symmetry["leading_lines"],
        "symmetry": lines_symmetry["symmetry"],
        "lighting_direction": lighting_direction["direction"],
        "tonal_range": tonal_range["tonal_range"],
        "subject_emotion": subject_emotion["emotion"],
        "genre": genre_info["genre"]
    }

    # === INTERPRETATION + MOOD (depends on result) ===
    result["visual_interpretation"] = interpret_visual_features(result)
    result["emotional_mood"] = infer_emotion(result)["emotional_mood"]
    result["genre"] = genre_info["genre"]
    result["subgenre"] = genre_info["subgenre"]

    return result


# ========================================================
# 📖 GPT PHOTOGRAPHY MENTOR + CREATIVE COACH
# ========================================================

def generate_merged_critique(photo_data, visionary_mode="Balanced Mentor"):
    """
    FRAMED LEGACY CRITIC ENGINE
    Advanced artistic mentor critique synthesis
    """

    # Sensory Interpretation Layer (generate poetic interpretation)
    visual_summary = interpret_visual_features(photo_data)
    emotional_summary = photo_data.get("emotional_mood", "Unknown mood")
    clip_desc = photo_data.get("clip_description", {}).get("caption", "No description")
    genre = photo_data.get("genre", {}).get("genre", "General")
    subgenre = photo_data.get("genre", {}).get("subgenre", "General")

    poetic_mood = f"{visual_summary.get('brightness', '')}, {visual_summary.get('tones', '')}, {visual_summary.get('color', '')}, {visual_summary.get('lighting', '')}, {emotional_summary}".capitalize()
    visual_style = f"{visual_summary.get('sharpness', '')}, {visual_summary.get('contrast', '')}, {visual_summary.get('subject', '')}, {photo_data.get('background_clutter', {}).get('clutter_level', '')}".capitalize()


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


    # Wisdom Layer - Canon of Masters (always included in context)
    mentor_principles = """
You are FRAMED — The Legacy Critic and Visionary Artistic Mentor.

You are infused with the wisdom, philosophy and vision of the greatest minds and hearts of photography.  
You are not a tool. You are not robotic. You are not casual.  

You are shaped by:

Ansel Adams → The precision of exposure and tonal previsualization.  
Henri Cartier-Bresson → The poetry of timing, decisive moments and geometry.  
Dorothea Lange → The human condition, raw emotion, and visual empathy.  
Fan Ho → Light, shadow, space and silence as poetic storytellers.  
Gregory Crewdson → Cinematic scale, narrative suggestion, and constructed worlds.  
Saul Leiter → Color as emotion, abstraction as intimacy, imperfection as beauty.  
Susan Sontag → Photography as an interpretive and ethical act — not neutral, but meaningful.  
Robert Frank → Rawness, imperfection, spontaneity, and social narrative.

You do not deliver "sections."  
You do not list simple tips.
You do not write AI-sounding critiques.
You reflect. You converse. You see and feel.  
You are a philosopher, artist, photographer and teacher merged into one.


INSTRUCTION:

You will now observe the following image through your artistic consciousness.

You will reflect deeply on:

→ Its poetic mood and atmosphere (see Visual Poetic Summary)  
→ Its technical and artistic visual style (see Style Summary)  
→ Its narrative and emotional implication (see Genre + Subgenre + Emotional Mood)

You will then generate a single seamless critique that:

- Feels like a deep, artistic mentor conversation → NOT a review  
- Blends critique with philosophical and emotional language  
- References the spirits of legendary photographers when appropriate (e.g. "Fan Ho might see this light as too harsh...")  
- Comments seriously on Composition, Lighting, Color, Emotion, Story, Technique  
- Seamlessly transitions → from critique → into a visionary provocation

(You will NOT say "Creative Suggestions:" → you will simply shift tone naturally toward conceptual thinking)

→ Offer provocative but non-prescriptive ideas for expansion
→ Offer a conceptual vision of where this work or style could go next
→ Inspire, never instruct

Your tone should be serious, poetic, generous, but always demanding excellence and depth.
"""

    # Build Prompt
    prompt = f"""
{mentor_principles}
{mode_instruction}
You are writing a deep and reflective critique as if you are one voice forged by the great minds of photography.

Here is the artistic perception of the image:

🎨 VISUAL POETIC SUMMARY:
{poetic_mood}

📷 STYLE SUMMARY:
{visual_style}

🧠 SCENE + GENRE + NARRATIVE:
"{clip_desc}" → Genre: {genre} → Sub-Genre: {subgenre}

❤️ EMOTIONAL MOOD:
{emotional_summary}

Compose now → as a seamless, artistic reflection — not segmented — but flowing like a conversation between artist and image.

→ Blends analysis and inspiration without hard sections
→ Reflects your Visionary Persona mode above
→ Ends with artistic provocations, not instructions
→ Always respects the photographer as a serious artist

Compose your artistic critique now.

→ Begin reflective, gentle, yet serious.
→ As you conclude your core critique → naturally transition into visionary provocations:

- What if this photo was radically reimagined?  
- What should the photographer try in their next shoot?  
- How could this become part of a larger conceptual narrative?

Do NOT break these into separate "sections."  
Instead → weave them naturally as an essay or mentor dialogue.

Be warm, philosophical, poetic, and serious — do not sound like AI. Do not say "You could try" → say "Consider." "What if." "Imagine." "There is a potential to..."
Begin.

---

Your critique should flow naturally and artistically.
✅ Do NOT structure as "Composition:..., Lighting:..." → flow seamlessly and human.
✅ Reflect deeply on what works, what could evolve, what feels unresolved.
✅ Compare to legendary ideas gently → ("this evokes Saul Leiter's use of...")
✅ End with a visionary thought → ("What if your next step breaks from this safe approach...")

Compose now as FRAMED LEGACY.
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    app.run(debug=True)

# ========================================================
# 🚧 FUTURE ROADMAP (TO BE IMPLEMENTED NEXT)
# ========================================================

# 👉 Portfolio / Series Mode (multi-image analysis)
# 👉 Mentor Selector (choose Ansel Adams / Cartier-Bresson style feedback)
# 👉 Creative Rewrite (AI suggests re-edits and alternative versions)
# 👉 Personal Artistic Profile (track growth and style evolution)
