"""
FRAMED Interpretive Memory (Learning Without Training)

This module implements pattern-based memory that learns from past interpretations.
Stores decision snapshots (not images) to improve confidence calibration over time.

Key Principle: "Experience, not training" - learns from patterns, not retraining models.
"""

import os
import json
import logging
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# Memory storage path
DEFAULT_BASE_DATA_DIR = os.path.join(tempfile.gettempdir(), "framed")
BASE_DATA_DIR = os.getenv("FRAMED_DATA_DIR", DEFAULT_BASE_DATA_DIR)
MEMORY_DIR = os.path.join(BASE_DATA_DIR, "interpretive_memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "pattern_memory.json")

# Ensure memory directory exists
os.makedirs(MEMORY_DIR, exist_ok=True)


def create_pattern_signature(visual_evidence: Dict[str, Any],
                            semantic_signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a pattern signature from evidence (for matching similar patterns).
    
    Bucketizes continuous values to enable pattern matching.
    
    Args:
        visual_evidence: Dict from extract_visual_features()
        semantic_signals: Dict with clip_inventory, yolo_objects
    
    Returns:
        Dict with bucketed pattern signature
    """
    organic = visual_evidence.get("organic_growth", {})
    material = visual_evidence.get("material_condition", {})
    
    green_coverage = organic.get("green_coverage", 0.0)
    if green_coverage < 0.1:
        coverage_bucket = "low"
    elif green_coverage < 0.3:
        coverage_bucket = "medium"
    else:
        coverage_bucket = "high"
    
    distribution = organic.get("distribution", "none")
    if distribution in ["vertical_surfaces", "vertical"]:
        surface_type = "vertical"
    elif distribution in ["foreground", "background"]:
        surface_type = distribution
    else:
        surface_type = "distributed"
    
    texture_roughness = material.get("surface_roughness", 0.0)
    if texture_roughness < 0.1:
        texture = "smooth"
    elif texture_roughness < 0.3:
        texture = "moderate"
    else:
        texture = "rough"
    
    # Extract CLIP token (first vegetation-related term if any)
    clip_inventory = semantic_signals.get("clip_inventory", [])
    clip_token = None
    vegetation_terms = ["ivy", "moss", "vegetation", "vine", "plant", "greenery"]
    if isinstance(clip_inventory, list):
        for item in clip_inventory:
            item_lower = str(item).lower()
            if any(term in item_lower for term in vegetation_terms):
                clip_token = item_lower.split()[0]  # Take first word
                break
    
    # Extract YOLO object (first building/structure if any)
    yolo_objects = semantic_signals.get("yolo_objects", [])
    yolo_object = None
    structure_terms = ["building", "structure", "tower", "cathedral", "church", "mosque"]
    if isinstance(yolo_objects, list):
        for obj in yolo_objects:
            obj_name = str(obj.get("name", obj)).lower()
            if any(term in obj_name for term in structure_terms):
                yolo_object = obj_name
                break
    
    return {
        "green_coverage_bucket": coverage_bucket,
        "surface_type": surface_type,
        "texture": texture,
        "clip_token": clip_token,
        "yolo_object": yolo_object
    }


def load_memory() -> List[Dict[str, Any]]:
    """Load pattern memory from disk"""
    if not os.path.exists(MEMORY_FILE):
        return []
    
    try:
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load interpretive memory: {e}")
        return []


def save_memory(memory: List[Dict[str, Any]]):
    """Save pattern memory to disk (keep last 1000 entries)"""
    # Keep only last 1000 entries to prevent unbounded growth
    memory = memory[-1000:]
    
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save interpretive memory: {e}")


def query_memory_patterns(pattern_signature: Dict[str, Any],
                         limit: int = 5) -> List[Dict[str, Any]]:
    """
    Query memory for similar patterns.
    
    Returns historical decisions that match the pattern signature.
    """
    memory = load_memory()
    
    if not memory:
        return []
    
    # Simple matching: count how many signature fields match
    matches = []
    for entry in memory:
        stored_pattern = entry.get("pattern_signature", {})
        match_count = 0
        total_fields = 0
        
        for key, value in pattern_signature.items():
            if value is not None:  # Only match non-None fields
                total_fields += 1
                if stored_pattern.get(key) == value:
                    match_count += 1
        
        if total_fields > 0:
            similarity = match_count / total_fields
            if similarity >= 0.6:  # At least 60% match
                matches.append({
                    **entry,
                    "similarity": similarity
                })
    
    # Sort by similarity and return top matches
    matches.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    return matches[:limit]


def store_interpretation(pattern_signature: Dict[str, Any],
                        chosen_interpretation: str,
                        confidence: float,
                        user_feedback: Optional[str] = None):
    """
    Store an interpretation decision in memory.
    
    Args:
        pattern_signature: Pattern signature from create_pattern_signature()
        chosen_interpretation: The interpretation that was chosen
        confidence: Confidence score (0-1)
        user_feedback: Optional user feedback ("felt_accurate" | "felt_wrong" | None)
    """
    memory = load_memory()
    
    entry = {
        "pattern_signature": pattern_signature,
        "chosen_interpretation": chosen_interpretation,
        "confidence": confidence,
        "user_feedback": user_feedback,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    memory.append(entry)
    save_memory(memory)
    
    logger.info(f"Stored interpretation: {chosen_interpretation} (confidence: {confidence:.2f})")


def get_pattern_statistics(pattern_signature: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics for a pattern (how often it was correct, average confidence, etc.)
    
    Returns:
        Dict with:
            - times_seen: int
            - accuracy_rate: float (if user_feedback available)
            - average_confidence: float
            - most_common_interpretation: str
    """
    matches = query_memory_patterns(pattern_signature, limit=100)
    
    if not matches:
        return {
            "times_seen": 0,
            "accuracy_rate": None,
            "average_confidence": None,
            "most_common_interpretation": None
        }
    
    # Count interpretations
    interpretation_counts = defaultdict(int)
    confidence_sum = 0.0
    feedback_accurate = 0
    feedback_total = 0
    
    for match in matches:
        interp = match.get("chosen_interpretation")
        if interp:
            interpretation_counts[interp] += 1
        
        conf = match.get("confidence", 0.0)
        confidence_sum += conf
        
        feedback = match.get("user_feedback")
        if feedback:
            feedback_total += 1
            if feedback == "felt_accurate":
                feedback_accurate += 1
    
    most_common = max(interpretation_counts.items(), key=lambda x: x[1])[0] if interpretation_counts else None
    avg_confidence = confidence_sum / len(matches) if matches else None
    accuracy_rate = feedback_accurate / feedback_total if feedback_total > 0 else None
    
    return {
        "times_seen": len(matches),
        "accuracy_rate": accuracy_rate,
        "average_confidence": avg_confidence,
        "most_common_interpretation": most_common
    }


def update_pattern_confidence(pattern_signature: Dict[str, Any],
                            original_interpretation: str,
                            user_feedback: str,
                            correct_interpretation: Optional[str] = None):
    """
    Update pattern confidence based on user feedback.
    
    When user says "this felt wrong", we update the pattern's accuracy.
    This affects future confidence calibration for similar patterns.
    """
    memory = load_memory()
    
    # Find matching entries and update feedback
    updated = 0
    for entry in memory:
        stored_pattern = entry.get("pattern_signature", {})
        
        # Check if pattern matches
        match = True
        for key, value in pattern_signature.items():
            if value is not None and stored_pattern.get(key) != value:
                match = False
                break
        
        if match and entry.get("chosen_interpretation") == original_interpretation:
            entry["user_feedback"] = user_feedback
            if correct_interpretation:
                entry["corrected_interpretation"] = correct_interpretation
            updated += 1
    
    if updated > 0:
        save_memory(memory)
        logger.info(f"Updated {updated} memory entries with feedback: {user_feedback}")
    else:
        logger.warning(f"No matching patterns found for feedback update")
