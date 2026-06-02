"""Perception helpers (pure image-path analyzers)."""

from collections import Counter
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
from colorthief import ColorThief
from sklearn.cluster import KMeans

from . import prompts_clip
from .models import DeepFace, NIMA_AVAILABLE, _import_tf_keras, get_clip_model, get_yolo_model


def detect_objects(image_path: str) -> List[str]:
    yolo_model = get_yolo_model()
    res = yolo_model(image_path)
    names = res[0].names
    objects: List[str] = []
    for b in res[0].boxes:
        cls_idx = int(b.cls.item())
        objects.append(names.get(cls_idx, str(cls_idx)))
    return objects if objects else ["No objects detected"]


def get_clip_description(image_path: str) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    candidate_captions = prompts_clip.CLIP_CAPTION_CANDIDATES

    clip_model, clip_processor, device = get_clip_model()
    inputs = clip_processor(text=candidate_captions, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = int(probs.argmax().item() if hasattr(probs.argmax(), "item") else probs.argmax())
    best_caption = candidate_captions[best_idx]

    caption_lower = best_caption.lower()
    tags: List[str] = []

    genre_map = {
        "portrait": "Portrait",
        "street": "Street",
        "urban": "Street",
        "landscape": "Landscape",
        "nature": "Landscape",
        "abstract": "Abstract",
        "conceptual": "Abstract",
        "fashion": "Fashion",
        "animal": "Wildlife",
        "wild": "Wildlife",
        "protest": "Documentary",
        "documentary": "Documentary",
    }
    mood_map = {
        "dream": "Dreamy",
        "fog": "Dreamy",
        "blur": "Dreamy",
        "dramatic": "Moody",
        "dark": "Moody",
        "moody": "Moody",
        "soft": "Soft",
        "warm": "Soft",
        "chaotic": "Chaotic",
        "busy": "Chaotic",
        "clean": "Minimal",
        "minimalist": "Minimal",
    }

    for keyword, genre in genre_map.items():
        if keyword in caption_lower:
            tags.append(genre)
    for keyword, mood in mood_map.items():
        if keyword in caption_lower:
            tags.append(mood)

    return {"caption": best_caption, "tags": tags, "genre_hint": tags[0] if tags else "General"}


def get_clip_inventory(image_path: str) -> List[Dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    clip_model, clip_processor, device = get_clip_model()

    structural_candidates = prompts_clip.CLIP_STRUCTURAL_INVENTORY
    material_condition_candidates = prompts_clip.CLIP_MATERIAL_CONDITION_INVENTORY
    atmosphere_candidates = prompts_clip.CLIP_ATMOSPHERE_INVENTORY

    all_candidates = structural_candidates + material_condition_candidates + atmosphere_candidates
    material_start = len(structural_candidates)
    atmosphere_start = material_start + len(material_condition_candidates)

    inputs = clip_processor(text=all_candidates, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    top_indices = probs.topk(15).indices[0].cpu().tolist()
    inventory_with_metadata: List[Dict[str, Any]] = []
    seen_lower = set()

    for idx in top_indices:
        confidence = float(probs[0][idx].item())
        if confidence <= 0.05:
            continue
        item = all_candidates[idx]
        item_lower = item.lower()
        if item_lower in seen_lower:
            continue
        seen_lower.add(item_lower)

        if idx < material_start:
            source = "structural_prompt"
        elif idx < atmosphere_start:
            source = "material_condition_prompt"
        else:
            source = "atmosphere_prompt"

        inventory_with_metadata.append({"item": item, "confidence": confidence, "source": source})

    return inventory_with_metadata


def analyze_color(image_path: str) -> Dict[str, Any]:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    hex_colors = ["#%02x%02x%02x" % (c[0], c[1], c[2]) for c in colors]

    avg_color = np.mean(colors, axis=0)
    mood = "warm" if avg_color[0] > avg_color[2] else "cool"
    return {"palette": hex_colors, "mood": mood}


def analyze_color_harmony(image_path: str) -> Dict[str, Any]:
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    r, g, b = dominant_color

    diff_rg = abs(r - g)
    diff_gb = abs(g - b)
    diff_rb = abs(r - b)

    if diff_rg < 15 and diff_gb < 15:
        harmony = "Monochromatic — serene and unified"
    elif diff_rb > 100 and diff_gb < 50:
        harmony = "Complementary — bold contrast"
    elif 50 < diff_rb < 100 or 50 < diff_gb < 100:
        harmony = "Split complementary — dynamic, controlled tension"
    elif diff_rg < 50 and diff_gb < 50 and diff_rb < 50:
        harmony = "Analogous — gentle continuity"
    elif diff_rb > 75 and diff_gb > 75:
        harmony = "Triadic — playful balance"
    else:
        harmony = "Unclassified"

    return {"dominant_color": f"#{r:02x}{g:02x}{b:02x}", "harmony": harmony}


def analyze_lines_and_symmetry(image_path: str) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)

    if lines is not None:
        num_lines = len(lines)
        if num_lines > 50:
            pattern = "Strong lines"
        elif num_lines > 20:
            pattern = "Prominent lines"
        elif num_lines > 5:
            pattern = "Some lines"
        else:
            pattern = "Minimal lines"
    else:
        pattern = "No significant lines"

    if "Strong" in pattern:
        style = "Chaotic or highly structured"
    elif "Prominent" in pattern:
        style = "Intentional leading lines"
    elif "Some" in pattern:
        style = "Subtle structure"
    else:
        style = "Soft or abstract"

    height, width = gray.shape
    if width > 1:
        left_half = gray[:, : width // 2]
        right_half = gray[:, width // 2 :]
        right_half_flipped = cv2.flip(right_half, 1)
        h1, w1 = left_half.shape
        h2, w2 = right_half_flipped.shape
        if w1 != w2:
            min_w = min(w1, w2)
            left_half = left_half[:, :min_w]
            right_half_flipped = right_half_flipped[:, :min_w]

        diff = cv2.absdiff(left_half, right_half_flipped)
        symmetry_score = float(np.mean(diff))
        if symmetry_score < 10:
            symmetry_desc = "Highly symmetrical"
        elif symmetry_score < 30:
            symmetry_desc = "Mostly symmetrical"
        elif symmetry_score < 60:
            symmetry_desc = "Noticeably asymmetrical"
        else:
            symmetry_desc = "Asymmetrical"
    else:
        symmetry_desc = "Too narrow"

    return {"line_pattern": pattern, "line_style": style, "symmetry": symmetry_desc}


def analyze_lighting_direction(image_path: str) -> Dict[str, Any]:
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    _, _, _, max_loc = cv2.minMaxLoc(gray)
    w = gray.shape[1]
    direction = (
        "light from left"
        if max_loc[0] < w * 0.33
        else "light from right"
        if max_loc[0] > w * 0.66
        else "light from center/top"
    )
    return {"direction": direction}


def analyze_tonal_range(image_path: str) -> Dict[str, Any]:
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    black, mid, white = float(np.sum(hist[:50])), float(np.sum(hist[50:200])), float(np.sum(hist[200:]))
    tone = "high key" if white > black and white > mid else "low key" if black > white and black > mid else "balanced"
    return {"tonal_range": tone}


def analyze_background_clutter(image_path: str) -> Dict[str, Any]:
    yolo_model = get_yolo_model()
    results = yolo_model(image_path)
    detections = results[0].boxes.data.cpu().numpy()
    num_objects = len(detections)

    if num_objects == 0:
        clutter_level = "Minimal"
        impact = "Subject isolation"
    elif num_objects <= 2:
        clutter_level = "Low"
        impact = "Clean context"
    elif num_objects <= 5:
        clutter_level = "Moderate"
        impact = "Some competition"
    elif num_objects <= 10:
        clutter_level = "High"
        impact = "Background competes"
    else:
        clutter_level = "Chaotic"
        impact = "Strong distraction"

    if num_objects == 0:
        narrative = "Minimalist"
    elif num_objects <= 2:
        narrative = "Controlled"
    elif num_objects <= 5:
        narrative = "Balanced"
    elif num_objects <= 10:
        narrative = "Visually active"
    else:
        narrative = "Fragmented attention"

    return {"num_objects": num_objects, "clutter_level": clutter_level, "impact": impact, "narrative": narrative}


def analyze_subject_emotion(image_path: str) -> Dict[str, Any]:
    if DeepFace is not None:
        try:
            res = DeepFace.analyze(img_path=image_path, actions=["emotion"], enforce_detection=False)
            emo = res[0]["dominant_emotion"] if isinstance(res, list) else res["dominant_emotion"]
            return {"subject_type": "human subject", "emotion": emo, "age": None, "gender": "Unknown"}
        except Exception:
            pass
    return analyze_subject_emotion_clip(image_path)


def analyze_subject_emotion_clip(image_path: str) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    emotion_captions = [
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
        "An empty and lonely landscape",
        "A peaceful and quiet environment",
        "A chaotic and energetic scene",
        "A surreal and dreamy conceptual scene",
        "A nostalgic vintage-style photo",
        "A dark and tense atmosphere",
        "A light and playful composition",
        "An abstract and ambiguous image",
    ]

    clip_model, clip_processor, device = get_clip_model()
    inputs = clip_processor(text=emotion_captions, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = int(probs.argmax().item() if hasattr(probs.argmax(), "item") else probs.argmax())
    best_caption = emotion_captions[best_idx]

    human_detected = any(word in best_caption.lower() for word in ["person", "human"])
    return {"subject_type": "human subject" if human_detected else "non-human / abstract", "emotion": best_caption}


def predict_nima_score(model, img_path: str) -> Dict[str, Any]:
    if model is None or not NIMA_AVAILABLE:
        return {"mean_score": None, "distribution": {}}
    try:
        _, _, _, _, preprocess_input, keras_image = _import_tf_keras()
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img = keras_image.img_to_array(img)
        img = preprocess_input(img)
        preds = model.predict(np.expand_dims(img, axis=0))[0]
        mean_score = sum((i + 1) * float(p) for i, p in enumerate(preds))
        return {
            "mean_score": round(mean_score, 2),
            "distribution": {str(i + 1): float(f"{p:.4f}") for i, p in enumerate(preds)},
        }
    except Exception:
        return {"mean_score": None, "distribution": {}}


def detect_objects_and_framing(image_path: str, confidence_threshold: float = 0.3) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    yolo_model = get_yolo_model()
    results = yolo_model(image_path)
    detections = results[0].boxes.data.cpu().numpy() if results and results[0].boxes is not None else []

    if len(detections) == 0:
        return {
            "objects": ["No objects detected"],
            "object_narrative": "No identifiable subject.",
            "subject_position": "Undefined",
            "subject_size": "N/A",
            "framing_description": "No dominant subject",
            "spatial_interpretation": "Open, undefined space.",
        }

    labels = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_names = results[0].names

    detected_objects = [class_names[int(lbl)] for lbl, conf in zip(labels, confidences) if conf >= confidence_threshold]
    if not detected_objects:
        return {
            "objects": ["No confident objects detected"],
            "object_narrative": "Subjects are too ambiguous to identify.",
            "subject_position": "Unknown",
            "subject_size": "N/A",
            "framing_description": "Undefined composition",
            "spatial_interpretation": "Ambiguous subject; open reading.",
        }

    object_counts = Counter(detected_objects)
    summarized_objects = [f"{count}x {obj}" if count > 1 else obj for obj, count in object_counts.items()]

    if len(summarized_objects) == 1:
        narrative = f"Primary subject: {summarized_objects[0]}."
    elif len(summarized_objects) <= 3:
        narrative = f"Subjects: {', '.join(summarized_objects)}."
    elif len(summarized_objects) <= 6:
        narrative = f"Busy frame: {', '.join(summarized_objects)}."
    else:
        narrative = "Densely populated frame."

    largest = max(detections, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    x1, y1, x2, y2 = largest[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    size_ratio = float(((x2 - x1) * (y2 - y1)) / (w * h))

    horizontal = "Left" if center_x < w * 0.33 else "Center" if center_x < w * 0.66 else "Right"
    vertical = "Top" if center_y < h * 0.33 else "Middle" if center_y < h * 0.66 else "Bottom"
    position = f"{vertical} {horizontal}"

    if size_ratio > 0.5:
        size_desc = "Extreme close-up"
    elif size_ratio > 0.2:
        size_desc = "Medium subject"
    elif size_ratio > 0.05:
        size_desc = "Small in frame"
    else:
        size_desc = "Tiny or distant"

    if horizontal == "Center" and vertical == "Middle":
        framing_desc = "Centered"
    elif horizontal == "Center":
        framing_desc = f"Vertically {vertical.lower()}"
    elif vertical == "Middle":
        framing_desc = f"Horizontally {horizontal.lower()}"
    else:
        framing_desc = f"Asymmetrical ({position.lower()})"

    if size_ratio > 0.5:
        spatial_interpretation = "High intimacy."
    elif size_ratio > 0.2:
        spatial_interpretation = "Subject-space balance."
    elif size_ratio > 0.05:
        spatial_interpretation = "Environmental context."
    else:
        spatial_interpretation = "Distance dominates."

    return {
        "objects": detected_objects,
        "object_narrative": narrative,
        "subject_position": position,
        "subject_size": size_desc,
        "framing_description": framing_desc,
        "spatial_interpretation": spatial_interpretation,
    }

