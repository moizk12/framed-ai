"""
FRAMED Negative Evidence Detection

This module detects what is NOT present in the image to prevent false interpretations.
Examples: no humans, no motion, no artificial uniformity.

Key Principle: Absence informs reasoning (stillness ≠ alienation).
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def detect_negative_evidence(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect negative evidence (what is NOT present) to prevent false interpretations.
    
    Universal: works for any image type.
    
    Args:
        analysis_result: Canonical schema analysis result
    
    Returns:
        Dict with negative evidence:
        {
            "no_human_presence": bool,
            "no_motion_detected": bool,
            "no_artificial_surface_uniformity": bool,
            "evidence": str - explainable evidence string
        }
    """
    negative_evidence = {}
    evidence_parts = []
    
    # Extract perception data
    perception = analysis_result.get("perception", {})
    semantics = perception.get("semantics", {})
    emotion = perception.get("emotion", {})
    
    # Extract YOLO objects (from composition or direct)
    composition = perception.get("composition", {})
    yolo_objects = []
    if "objects" in analysis_result:
        yolo_objects = analysis_result.get("objects", [])
    elif composition.get("subject_framing"):
        # Try to extract from legacy format
        pass
    
    # Extract visual evidence
    visual_evidence = analysis_result.get("visual_evidence", {})
    material_condition = visual_evidence.get("material_condition", {})
    
    # === CHECK 1: No Human Presence ===
    # Check YOLO objects
    yolo_has_person = any(
        obj.get("name", "").lower() in ["person", "people", "human", "man", "woman", "child"]
        for obj in yolo_objects
    )
    
    # Check CLIP caption and tags
    clip_caption = semantics.get("caption", "").lower() if semantics.get("available") else ""
    clip_tags = semantics.get("tags", [])
    clip_text = f"{clip_caption} {' '.join(clip_tags)}".lower()
    
    human_terms = ["person", "people", "human", "man", "woman", "child", "crowd", "figure", "face"]
    clip_has_human = any(term in clip_text for term in human_terms)
    
    # Check emotion detection
    emotion_has_human = emotion.get("available", False) and emotion.get("subject_type") == "human subject"
    
    if not yolo_has_person and not clip_has_human and not emotion_has_human:
        negative_evidence["no_human_presence"] = True
        evidence_parts.append("YOLO: no person, CLIP: no human terms, emotion: no human subject")
    else:
        negative_evidence["no_human_presence"] = False
    
    # === CHECK 2: No Motion Detected ===
    # Check temporal context from scene understanding
    scene_understanding = analysis_result.get("scene_understanding", {})
    temporal_context = scene_understanding.get("temporal_context", {})
    temporal_pace = temporal_context.get("pace", "static")
    
    # Check CLIP for motion terms
    motion_terms = ["motion", "movement", "dynamic", "action", "running", "walking", "moving", "busy", "chaotic"]
    clip_has_motion = any(term in clip_text for term in motion_terms)
    
    if temporal_pace in ["static", "slow"] and not clip_has_motion:
        negative_evidence["no_motion_detected"] = True
        evidence_parts.append(f"temporal: {temporal_pace}, CLIP: no motion terms")
    else:
        negative_evidence["no_motion_detected"] = False
    
    # === CHECK 3: No Artificial Surface Uniformity ===
    # Check material condition
    condition = material_condition.get("condition", "unknown")
    color_uniformity = material_condition.get("color_uniformity", 0.0)
    green_coverage = visual_evidence.get("organic_growth", {}).get("green_coverage", 0.0)
    surface_roughness = material_condition.get("surface_roughness", 0.0)
    
    # Artificial uniformity = pristine condition + high color uniformity + minimal organic growth + low texture variance
    if (condition == "pristine" and 
        color_uniformity > 0.8 and 
        green_coverage < 0.05 and 
        surface_roughness < 0.05):
        negative_evidence["no_artificial_surface_uniformity"] = True
        evidence_parts.append("pristine condition + high uniformity + minimal organic + low texture = artificial uniformity")
    else:
        negative_evidence["no_artificial_surface_uniformity"] = False
    
    # Combine evidence string
    if evidence_parts:
        negative_evidence["evidence"] = "; ".join(evidence_parts)
    else:
        negative_evidence["evidence"] = "No negative evidence detected"
    
    return negative_evidence
