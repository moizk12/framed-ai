"""Derived fields from legacy perception dicts (genre, emotion, feature interpretation)."""

from typing import Any, Dict


def infer_emotion(photo_data: Dict[str, Any]) -> Dict[str, Any]:
    brightness = photo_data.get("brightness", 0)
    contrast = photo_data.get("contrast", 0)
    sharpness = photo_data.get("sharpness", 0)
    color_mood = photo_data.get("color_mood", "neutral")
    tonal_range = photo_data.get("tonal_range", "balanced")
    _subject_emotion_val = photo_data.get("subject_emotion", "")
    if isinstance(_subject_emotion_val, dict):
        subject_emotion = str(_subject_emotion_val.get("emotion", "") or "")
    else:
        subject_emotion = str(_subject_emotion_val or "")
    clutter = photo_data.get("background_clutter", {}).get("clutter_level", "minimal clutter")
    symmetry = photo_data.get("symmetry", "moderate")

    emotion_tags = []

    if brightness < 40 or tonal_range == "low key (dark)":
        emotion_tags.append("moody")
    elif brightness > 180 or tonal_range == "high key (bright)":
        emotion_tags.append("uplifting and airy")

    if contrast < 20:
        emotion_tags.append("soft and calm")
    elif contrast > 100:
        emotion_tags.append("bold and intense")

    if sharpness < 20:
        emotion_tags.append("dreamy or hazy")
    elif sharpness > 150:
        emotion_tags.append("razor sharp and clear")

    if color_mood == "warm":
        emotion_tags.append("warm and inviting")
    elif color_mood == "cool":
        emotion_tags.append("cold and distant")
    elif color_mood == "neutral":
        emotion_tags.append("quiet and subtle")

    subj_lower = subject_emotion.lower()
    if "sad" in subj_lower or "melancholic" in subj_lower:
        emotion_tags.append("melancholic")
    elif "happy" in subj_lower or "smiling" in subj_lower:
        emotion_tags.append("joyful")
    elif "serene" in subj_lower:
        emotion_tags.append("peaceful")

    if "high clutter" in clutter:
        emotion_tags.append("chaotic or busy")
    elif "minimal clutter" in clutter:
        emotion_tags.append("clean and minimal")

    if symmetry == "high":
        emotion_tags.append("harmonious and balanced")
    elif symmetry == "low":
        emotion_tags.append("asymmetrical and dynamic")

    mood = "neutral and open-ended" if not emotion_tags else ", ".join(emotion_tags)
    return {"emotional_mood": mood}


def interpret_visual_features(photo_data: Dict[str, Any]) -> Dict[str, Any]:
    interpretation: Dict[str, Any] = {}

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

    tonal = photo_data.get("tonal_range", "balanced")
    if tonal == "high key (bright)":
        interpretation["tones"] = "High key tones, light and airy with delicate charm"
    elif tonal == "low key (dark)":
        interpretation["tones"] = "Low key, dominated by darkness and cinematic melancholy"
    else:
        interpretation["tones"] = "Balanced tones, harmonious and versatile"

    color_mood = photo_data.get("color_mood", "neutral")
    if color_mood == "warm":
        interpretation["color"] = "Warm hues, inviting and emotionally resonant"
    elif color_mood == "cool":
        interpretation["color"] = "Cool palette, calm, distant and tranquil"
    else:
        interpretation["color"] = "Neutral tones, understated and sophisticated"

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


def detect_genre(photo_data: Dict[str, Any]) -> Dict[str, Any]:
    clip_info = photo_data.get("clip_description", {}) or {}
    clip_genre = clip_info.get("genre_hint", "General")

    subject_type = photo_data.get("subject_type", "unknown")
    mood = photo_data.get("emotional_mood", "neutral")
    tonal = photo_data.get("tonal_range", "balanced")

    background_info = photo_data.get("background_clutter", {}) or {}
    clutter_level = background_info.get("clutter_level", "")

    if clip_genre != "General":
        genre = clip_genre
    elif subject_type == "human subject":
        genre = "Portrait"
    elif subject_type == "non-human / abstract":
        genre = "Abstract / Conceptual"
    else:
        genre = "General"

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
        subgenre = "Action Wildlife" if "dynamic" in mood or "action" in mood else "Calm / Observational"

    if subgenre is None:
        subgenre = f"Classic {genre}" if genre != "General" else "General"

    return {"genre": genre, "subgenre": subgenre}

