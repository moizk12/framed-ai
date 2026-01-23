"""
FRAMED Canonical Analysis Schema

This module defines the authoritative structure for all image analysis results.
Every analysis must conform to this schema for consistency, caching, and downstream processing.

Version: 2.0
Status: Authoritative
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json


def create_empty_analysis_result() -> Dict[str, Any]:
    """
    Creates an empty AnalysisResult conforming to the canonical schema.
    All fields are initialized to safe defaults.
    """
    return {
        "metadata": {
            "photo_id": "",
            "filename": "",
            "file_hash": "",
            "timestamp": ""
        },
        "perception": {
            "technical": {
                "available": False,
                "brightness": None,
                "contrast": None,
                "sharpness": None
            },
            "composition": {
                "available": False,
                "line_pattern": None,
                "line_style": None,
                "symmetry": None,
                "subject_framing": {
                    "position": None,
                    "size": None,
                    "style": None,
                    "interpretation": None
                }
            },
            "color": {
                "available": False,
                "palette": [],
                "mood": None,
                "harmony": {
                    "dominant_color": None,
                    "harmony_type": None
                }
            },
            "lighting": {
                "available": False,
                "direction": None,
                "quality": None
            },
            "aesthetics": {
                "available": False,
                "mean_score": None,
                "distribution": {}
            },
            "semantics": {
                "available": False,
                "caption": None,
                "tags": [],
                "genre_hint": None,
                "embeddings": None
            },
            "emotion": {
                "available": False,
                "subject_type": None,
                "emotion": None
            }
        },
        "derived": {
            "emotional_mood": None,
            "genre": {
                "genre": None,
                "subgenre": None
            },
            "visual_interpretation": {}
        },
        "confidence": {
            "clip": False,
            "yolo": False,
            "nima": False,
            "deepface": False
        },
        "errors": {},
        # Semantic Anchors: High-confidence, low-risk labels derived from multiple signals
        # Only keys that pass confidence thresholds are included (sparse by default)
        # Missing key = not permitted, present key = safe to reference
        "semantic_anchors": {}
    }


def validate_schema(result: Dict[str, Any]) -> bool:
    """
    Validates that a result conforms to the canonical schema structure.
    Returns True if valid, False otherwise.
    
    Note: semantic_anchors is optional (for backward compatibility).
    """
    required_keys = ["metadata", "perception", "derived", "confidence", "errors"]
    
    if not isinstance(result, dict):
        return False
    
    for key in required_keys:
        if key not in result:
            return False
    
    # Validate perception structure
    perception = result.get("perception", {})
    required_perception_keys = [
        "technical", "composition", "color", "lighting",
        "aesthetics", "semantics", "emotion"
    ]
    
    for key in required_perception_keys:
        if key not in perception:
            return False
        if not isinstance(perception[key], dict):
            return False
        if "available" not in perception[key]:
            return False
    
    return True


def normalize_to_schema(legacy_result: Dict[str, Any], 
                        photo_id: str = "",
                        filename: str = "",
                        file_hash: str = "",
                        timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Converts a legacy analysis result to the canonical schema format.
    This function handles migration from old flat structures to the new schema.
    """
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Start with empty schema
    result = create_empty_analysis_result()
    
    # Set metadata
    result["metadata"] = {
        "photo_id": photo_id,
        "filename": filename,
        "file_hash": file_hash,
        "timestamp": timestamp
    }
    
    # Map technical analysis
    if "brightness" in legacy_result:
        result["perception"]["technical"]["available"] = True
        result["perception"]["technical"]["brightness"] = legacy_result.get("brightness")
        result["perception"]["technical"]["contrast"] = legacy_result.get("contrast")
        result["perception"]["technical"]["sharpness"] = legacy_result.get("sharpness")
    
    # Map composition
    if "line_pattern" in legacy_result or "subject_framing" in legacy_result:
        result["perception"]["composition"]["available"] = True
        result["perception"]["composition"]["line_pattern"] = legacy_result.get("line_pattern")
        result["perception"]["composition"]["line_style"] = legacy_result.get("line_style")
        result["perception"]["composition"]["symmetry"] = legacy_result.get("symmetry")
        
        if "subject_framing" in legacy_result:
            result["perception"]["composition"]["subject_framing"] = legacy_result["subject_framing"]
    
    # Map color
    if "color_palette" in legacy_result or "color_mood" in legacy_result:
        result["perception"]["color"]["available"] = True
        result["perception"]["color"]["palette"] = legacy_result.get("color_palette", [])
        result["perception"]["color"]["mood"] = legacy_result.get("color_mood")
        
        if "color_harmony" in legacy_result:
            harmony = legacy_result["color_harmony"]
            if isinstance(harmony, dict):
                result["perception"]["color"]["harmony"]["dominant_color"] = harmony.get("dominant_color")
                result["perception"]["color"]["harmony"]["harmony_type"] = harmony.get("harmony")
    
    # Map lighting
    if "lighting_direction" in legacy_result:
        result["perception"]["lighting"]["available"] = True
        result["perception"]["lighting"]["direction"] = legacy_result.get("lighting_direction")
    
    # Map aesthetics (NIMA)
    if "nima" in legacy_result:
        nima = legacy_result["nima"]
        if isinstance(nima, dict) and nima.get("mean_score") is not None:
            result["perception"]["aesthetics"]["available"] = True
            result["perception"]["aesthetics"]["mean_score"] = nima.get("mean_score")
            result["perception"]["aesthetics"]["distribution"] = nima.get("distribution", {})
    
    # Map semantics (CLIP)
    if "clip_description" in legacy_result:
        clip = legacy_result["clip_description"]
        if isinstance(clip, dict):
            result["perception"]["semantics"]["available"] = True
            result["perception"]["semantics"]["caption"] = clip.get("caption")
            result["perception"]["semantics"]["tags"] = clip.get("tags", [])
            result["perception"]["semantics"]["genre_hint"] = clip.get("genre_hint")
    
    # Map emotion
    if "subject_emotion" in legacy_result:
        emotion = legacy_result["subject_emotion"]
        if isinstance(emotion, dict):
            result["perception"]["emotion"]["available"] = True
            result["perception"]["emotion"]["subject_type"] = emotion.get("subject_type")
            result["perception"]["emotion"]["emotion"] = emotion.get("emotion")
    
    # Map derived fields
    if "emotional_mood" in legacy_result:
        result["derived"]["emotional_mood"] = legacy_result.get("emotional_mood")
    
    if "genre" in legacy_result or "subgenre" in legacy_result:
        result["derived"]["genre"]["genre"] = legacy_result.get("genre")
        result["derived"]["genre"]["subgenre"] = legacy_result.get("subgenre")
    
    if "visual_interpretation" in legacy_result:
        result["derived"]["visual_interpretation"] = legacy_result.get("visual_interpretation", {})
    
    # Map confidence flags
    result["confidence"]["clip"] = result["perception"]["semantics"]["available"]
    result["confidence"]["yolo"] = "objects" in legacy_result and len(legacy_result.get("objects", [])) > 0
    result["confidence"]["nima"] = result["perception"]["aesthetics"]["available"]
    result["confidence"]["deepface"] = False  # DeepFace is toggleable, default to False
    
    # Map errors
    if "errors" in legacy_result:
        result["errors"] = legacy_result["errors"]
    
    return result
