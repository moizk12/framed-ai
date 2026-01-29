"""
FRAMED Temporal Memory System

Memory that learns and evolves over time.
Stores reasoning patterns, not just results.
Tracks evolution of both FRAMED and the user.

🔒 CRITICAL INVARIANT:
Learning must NEVER happen inside the LLM.
All learning, memory, and evolution must land in this memory layer.

This ensures:
- Models are swappable
- Progress is permanent
- Evolution is cumulative

Key Functions:
- create_pattern_signature: Create hashable signature from evidence
- store_interpretation: Store interpretation in temporal memory (with confidence decay, evolution tracking, correction ingestion)
- query_memory_patterns: Find similar past interpretations
- get_evolution_history: Get "I used to think X, now I think Y" records
- format_evolution_history_for_prompt: Format evolution for LLM prompts
- track_user_trajectory: Track user's themes, patterns, evolution
"""

import os
import json
import hashlib
import logging
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# ========================================================
# MEMORY STORAGE CONFIGURATION
# ========================================================

# Memory file path (under centralized runtime directory)
DEFAULT_BASE_DATA_DIR = os.path.join(tempfile.gettempdir(), "framed")
BASE_DATA_DIR = os.getenv("FRAMED_DATA_DIR", DEFAULT_BASE_DATA_DIR)
TEMPORAL_MEMORY_PATH = os.path.join(BASE_DATA_DIR, "temporal_memory.json")
USER_TRAJECTORY_PATH = os.path.join(BASE_DATA_DIR, "user_trajectory.json")

# Maximum entries per pattern (to prevent unbounded growth)
MAX_INTERPRETATIONS_PER_PATTERN = 1000

# Maximum patterns to keep in memory
MAX_PATTERNS = 10000


# ========================================================
# PATTERN SIGNATURE CREATION
# ========================================================

def create_pattern_signature(
    visual_evidence: Dict[str, Any],
    semantic_signals: Dict[str, Any],
) -> str:
    """
    Create a hashable signature from evidence.
    
    Used to find similar past interpretations.
    Normalizes evidence to create consistent signatures.
    
    Args:
        visual_evidence: Visual analysis results (from extract_visual_features())
        semantic_signals: Semantic signals (CLIP, YOLO, composition)
    
    Returns:
        str: 16-character hex signature
    """
    try:
        # Normalize visual evidence
        organic_growth = visual_evidence.get("organic_growth", {})
        material_condition = visual_evidence.get("material_condition", {})
        organic_integration = visual_evidence.get("organic_integration", {})
        
        normalized = {
            "visual": {
                # Round to 2 decimal places for consistency
                "green_coverage": round(organic_growth.get("green_coverage", 0.0), 2),
                "condition": material_condition.get("condition", "unknown"),
                "surface_roughness": round(material_condition.get("surface_roughness", 0.0), 2),
                "relationship": organic_integration.get("relationship", "none"),
                "integration_level": organic_integration.get("integration_level", "none"),
                "salience": organic_growth.get("salience", "minimal"),
            },
            "semantic": {
                # Normalize semantic signals
                "objects": sorted([str(obj).lower() for obj in semantic_signals.get("objects", [])]),
                "tags": sorted([str(tag).lower() for tag in semantic_signals.get("tags", [])]),
                "caption_keywords": sorted([str(kw).lower() for kw in semantic_signals.get("caption_keywords", [])]),
            }
        }
        
        # Create hash
        signature_str = json.dumps(normalized, sort_keys=True)
        signature_hash = hashlib.sha256(signature_str.encode()).hexdigest()
        
        # Return first 16 characters (sufficient for uniqueness)
        return signature_hash[:16]
    
    except Exception as e:
        logger.error(f"Pattern signature creation failed: {e}", exc_info=True)
        # Return a fallback signature based on timestamp
        return hashlib.sha256(str(datetime.now().isoformat()).encode()).hexdigest()[:16]


# ========================================================
# MEMORY STORAGE AND LOADING
# ========================================================

def load_temporal_memory() -> Dict[str, Any]:
    """
    Load temporal memory from disk.
    
    Returns:
        Dict with structure:
        {
            "patterns": {
                "signature": {
                    "first_seen": "ISO date",
                    "interpretations": [
                        {
                            "date": "ISO date",
                            "interpretation": {...},
                            "confidence": 0.92,
                            "user_feedback": {...}  # optional
                        }
                    ]
                }
            },
            "last_updated": "ISO date"
        }
    """
    if not os.path.exists(TEMPORAL_MEMORY_PATH):
        return {
            "patterns": {},
            "last_updated": datetime.now().isoformat()
        }
    
    try:
        with open(TEMPORAL_MEMORY_PATH, 'r') as f:
            memory = json.load(f)
        
        # Ensure structure is valid
        if "patterns" not in memory:
            memory["patterns"] = {}
        if "last_updated" not in memory:
            memory["last_updated"] = datetime.now().isoformat()
        
        return memory
    
    except Exception as e:
        logger.error(f"Failed to load temporal memory: {e}", exc_info=True)
        return {
            "patterns": {},
            "last_updated": datetime.now().isoformat()
        }


def save_temporal_memory(memory: Dict[str, Any]) -> bool:
    """
    Save temporal memory to disk.
    
    Args:
        memory: Memory dictionary to save
    
    Returns:
        bool: True on success, False on error
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(TEMPORAL_MEMORY_PATH), exist_ok=True)
        
        # Update last_updated timestamp
        memory["last_updated"] = datetime.now().isoformat()
        
        # Limit patterns if too many
        if len(memory.get("patterns", {})) > MAX_PATTERNS:
            # Keep most recent patterns (by first_seen date)
            patterns = memory["patterns"]
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1].get("first_seen", ""),
                reverse=True
            )
            memory["patterns"] = dict(sorted_patterns[:MAX_PATTERNS])
            logger.info(f"Limited temporal memory to {MAX_PATTERNS} patterns")
        
        # Save to disk
        with open(TEMPORAL_MEMORY_PATH, 'w') as f:
            json.dump(memory, f, indent=2, default=str)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to save temporal memory: {e}", exc_info=True)
        return False


# ========================================================
# INTERPRETATION STORAGE
# ========================================================

def store_interpretation(
    signature: str,
    interpretation: Dict[str, Any],
    confidence: float,
    user_feedback: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Store interpretation in temporal memory.
    
    Tracks evolution over time by storing multiple interpretations
    for the same pattern signature.
    
    Features:
    - Confidence decay: Old interpretations lose confidence over time
    - Evolution tracking: "I used to think X, now I think Y" records
    - Correction ingestion: User feedback recalibrates confidence
    
    Args:
        signature: Pattern signature (from create_pattern_signature())
        interpretation: Intelligence output (from intelligence_core.py)
        confidence: Confidence score (0.0-1.0)
        user_feedback: Optional user feedback dict
    
    Returns:
        bool: True on success, False on error
    """
    try:
        memory = load_temporal_memory()
        
        # Initialize pattern if not exists
        if signature not in memory["patterns"]:
            memory["patterns"][signature] = {
                "first_seen": datetime.now().isoformat(),
                "interpretations": [],
                "evolution_history": []  # Track "I used to think X, now I think Y"
            }
        
        pattern = memory["patterns"][signature]
        past_interpretations = pattern.get("interpretations", [])
        
        # === EVOLUTION TRACKING: "I Used to Think X, Now I Think Y" ===
        if past_interpretations:
            # Get the most recent interpretation
            latest = past_interpretations[-1]
            old_interpretation = latest.get("interpretation", {})
            old_recognition = old_interpretation.get("recognition", {})
            old_what_i_see = old_recognition.get("what_i_see", "")
            
            # Get new interpretation
            new_recognition = interpretation.get("recognition", {})
            new_what_i_see = new_recognition.get("what_i_see", "")
            
            # If interpretation changed, record evolution
            if old_what_i_see and new_what_i_see and old_what_i_see != new_what_i_see:
                evolution_entry = {
                    "date": datetime.now().isoformat(),
                    "old_interpretation": old_what_i_see[:200],  # First 200 chars
                    "new_interpretation": new_what_i_see[:200],
                    "old_confidence": latest.get("confidence", 0.0),
                    "new_confidence": float(confidence),
                    "reason": "interpretation_evolved"
                }
                
                # Add to evolution history
                if "evolution_history" not in pattern:
                    pattern["evolution_history"] = []
                pattern["evolution_history"].append(evolution_entry)
                
                # Limit evolution history (keep last 50)
                if len(pattern["evolution_history"]) > 50:
                    pattern["evolution_history"] = pattern["evolution_history"][-50:]
                
                logger.info(f"Evolution recorded: '{old_what_i_see[:50]}...' -> '{new_what_i_see[:50]}...'")
        
        # === CONFIDENCE DECAY: Apply decay to old interpretations ===
        # Decay factor: 0.95 per month (5% decay per month)
        # Applied when storing new interpretation
        current_date = datetime.now()
        for entry in past_interpretations:
            entry_date_str = entry.get("date", "")
            if entry_date_str:
                try:
                    entry_date = datetime.fromisoformat(entry_date_str)
                    months_old = (current_date - entry_date).days / 30.0
                    
                    # Apply decay (only if > 1 month old)
                    if months_old > 1.0:
                        decay_factor = 0.95 ** months_old
                        original_confidence = entry.get("original_confidence", entry.get("confidence", 0.0))
                        if "original_confidence" not in entry:
                            entry["original_confidence"] = entry.get("confidence", 0.0)
                        entry["confidence"] = max(0.0, original_confidence * decay_factor)
                except (ValueError, TypeError):
                    # Invalid date, skip decay
                    pass
        
        # === CORRECTION INGESTION: Apply user feedback if present ===
        if user_feedback:
            # User feedback recalibrates confidence
            if user_feedback.get("missed_the_point"):
                # Decrease confidence
                confidence = max(0.0, confidence - 0.1)
                logger.info(f"Correction ingested: confidence decreased to {confidence:.2f}")
            elif user_feedback.get("felt_exactly_right"):
                # Increase confidence
                confidence = min(1.0, confidence + 0.05)
                logger.info(f"Correction ingested: confidence increased to {confidence:.2f}")
        
        # Create entry
        entry = {
            "date": datetime.now().isoformat(),
            "interpretation": interpretation,
            "confidence": float(confidence),
            "original_confidence": float(confidence),  # Store original for decay calculation
        }
        
        if user_feedback:
            entry["user_feedback"] = user_feedback
        
        # Add to interpretations
        pattern["interpretations"].append(entry)
        
        # Limit interpretations per pattern
        if len(pattern["interpretations"]) > MAX_INTERPRETATIONS_PER_PATTERN:
            pattern["interpretations"] = pattern["interpretations"][-MAX_INTERPRETATIONS_PER_PATTERN:]
            logger.info(f"Limited interpretations for pattern {signature} to {MAX_INTERPRETATIONS_PER_PATTERN}")
        
        # Save to disk
        return save_temporal_memory(memory)
    
    except Exception as e:
        logger.error(f"Failed to store interpretation: {e}", exc_info=True)
        return False


# ========================================================
# EVOLUTION HISTORY RETRIEVAL
# ========================================================

def get_evolution_history(signature: str) -> List[Dict[str, Any]]:
    """
    Get evolution history for a pattern signature.
    
    Returns "I used to think X, now I think Y" records.
    
    Args:
        signature: Pattern signature
    
    Returns:
        List of evolution entries, most recent first
    """
    try:
        memory = load_temporal_memory()
        
        if signature not in memory.get("patterns", {}):
            return []
        
        pattern = memory["patterns"][signature]
        evolution_history = pattern.get("evolution_history", [])
        
        # Return most recent first
        return list(reversed(evolution_history[-10:]))  # Last 10 entries
    
    except Exception as e:
        logger.error(f"Failed to get evolution history: {e}", exc_info=True)
        return []


def format_evolution_history_for_prompt(signature: str) -> str:
    """
    Format evolution history for LLM prompts.
    
    Returns a string like:
    "I used to interpret this pattern as 'weathered stone' (confidence: 0.85),
    but now I see it as 'organic reclamation of structure' (confidence: 0.90).
    This evolution occurred because..."
    
    Args:
        signature: Pattern signature
    
    Returns:
        Formatted string for prompt inclusion
    """
    evolution_history = get_evolution_history(signature)
    
    if not evolution_history:
        return "No evolution history for this pattern (first interpretation)."
    
    lines = []
    lines.append("EVOLUTION HISTORY (I used to think X, now I think Y):")
    
    for i, entry in enumerate(evolution_history[:5], 1):  # Last 5 evolutions
        old_interp = entry.get("old_interpretation", "N/A")
        new_interp = entry.get("new_interpretation", "N/A")
        old_conf = entry.get("old_confidence", 0.0)
        new_conf = entry.get("new_confidence", 0.0)
        date = entry.get("date", "unknown")
        
        lines.append(
            f"  {i}. {date}: "
            f"I used to interpret this as '{old_interp[:80]}...' (confidence: {old_conf:.2f}), "
            f"but now I see it as '{new_interp[:80]}...' (confidence: {new_conf:.2f})."
        )
    
    return "\n".join(lines)


# ========================================================
# MEMORY QUERY
# ========================================================

def query_memory_patterns(
    signature: str,
    similarity_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Find similar past interpretations.
    
    Currently uses exact signature matching.
    Can be enhanced with similarity scoring in the future.
    
    Args:
        signature: Pattern signature to search for
        similarity_threshold: Similarity threshold (0.0-1.0) - currently unused (exact match)
    
    Returns:
        List of pattern data dictionaries with evolution history
    """
    try:
        memory = load_temporal_memory()
        
        # Exact match (for now)
        # TODO: Enhance with similarity scoring (e.g., Hamming distance, cosine similarity)
        if signature in memory.get("patterns", {}):
            pattern_data = memory["patterns"][signature]
            return [pattern_data]
        
        # No match found
        return []
    
    except Exception as e:
        logger.error(f"Failed to query memory patterns: {e}", exc_info=True)
        return []


def get_pattern_statistics(signature: str) -> Dict[str, Any]:
    """
    Get statistics for a pattern.
    
    Args:
        signature: Pattern signature
    
    Returns:
        Dict with statistics:
        {
            "count": number of interpretations,
            "first_seen": ISO date,
            "last_seen": ISO date,
            "avg_confidence": average confidence,
            "evolution_count": number of distinct interpretations
        }
    """
    try:
        memory = load_temporal_memory()
        
        if signature not in memory.get("patterns", {}):
            return {
                "count": 0,
                "first_seen": None,
                "last_seen": None,
                "avg_confidence": 0.0,
                "evolution_count": 0
            }
        
        pattern = memory["patterns"][signature]
        interpretations = pattern.get("interpretations", [])
        
        if not interpretations:
            return {
                "count": 0,
                "first_seen": pattern.get("first_seen"),
                "last_seen": None,
                "avg_confidence": 0.0,
                "evolution_count": 0
            }
        
        # Calculate statistics
        confidences = [entry.get("confidence", 0.0) for entry in interpretations]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Count distinct interpretations (by what_i_see)
        distinct_interpretations = set()
        for entry in interpretations:
            interpretation = entry.get("interpretation", {})
            recognition = interpretation.get("recognition", {})
            what_i_see = recognition.get("what_i_see", "")
            if what_i_see:
                distinct_interpretations.add(what_i_see[:100])  # First 100 chars
        
        return {
            "count": len(interpretations),
            "first_seen": pattern.get("first_seen"),
            "last_seen": interpretations[-1].get("date") if interpretations else None,
            "avg_confidence": avg_confidence,
            "evolution_count": len(distinct_interpretations)
        }
    
    except Exception as e:
        logger.error(f"Failed to get pattern statistics: {e}", exc_info=True)
        return {
            "count": 0,
            "first_seen": None,
            "last_seen": None,
            "avg_confidence": 0.0,
            "evolution_count": 0
        }


# ========================================================
# USER TRAJECTORY TRACKING
# ========================================================

def load_user_trajectory(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load user trajectory from disk.
    
    Args:
        user_id: Optional user ID (for multi-user support in future)
    
    Returns:
        Dict with structure:
        {
            "themes": ["time", "decay", "organic integration"],
            "patterns": ["minimal compositions", "night photography"],
            "evolution": [
                {"date": "ISO date", "state": "exploring minimalism"},
                {"date": "ISO date", "state": "committing to organic integration"}
            ],
            "last_updated": "ISO date"
        }
    """
    # For now, single user trajectory (can be extended for multi-user)
    trajectory_file = USER_TRAJECTORY_PATH
    
    if not os.path.exists(trajectory_file):
        return {
            "themes": [],
            "patterns": [],
            "evolution": [],
            "last_updated": datetime.now().isoformat()
        }
    
    try:
        with open(trajectory_file, 'r') as f:
            trajectory = json.load(f)
        
        # Ensure structure is valid
        if "themes" not in trajectory:
            trajectory["themes"] = []
        if "patterns" not in trajectory:
            trajectory["patterns"] = []
        if "evolution" not in trajectory:
            trajectory["evolution"] = []
        if "last_updated" not in trajectory:
            trajectory["last_updated"] = datetime.now().isoformat()
        
        return trajectory
    
    except Exception as e:
        logger.error(f"Failed to load user trajectory: {e}", exc_info=True)
        return {
            "themes": [],
            "patterns": [],
            "evolution": [],
            "last_updated": datetime.now().isoformat()
        }


def save_user_trajectory(trajectory: Dict[str, Any], user_id: Optional[str] = None) -> bool:
    """
    Save user trajectory to disk.
    
    Args:
        trajectory: Trajectory dictionary to save
        user_id: Optional user ID (for multi-user support in future)
    
    Returns:
        bool: True on success, False on error
    """
    # For now, single user trajectory
    trajectory_file = USER_TRAJECTORY_PATH
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(trajectory_file), exist_ok=True)
        
        # Update last_updated timestamp
        trajectory["last_updated"] = datetime.now().isoformat()
        
        # Limit evolution entries (keep last 1000)
        if len(trajectory.get("evolution", [])) > 1000:
            trajectory["evolution"] = trajectory["evolution"][-1000:]
        
        # Save to disk
        with open(trajectory_file, 'w') as f:
            json.dump(trajectory, f, indent=2, default=str)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to save user trajectory: {e}", exc_info=True)
        return False


def track_user_trajectory(
    analysis_result: Dict[str, Any],
    intelligence_output: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Track user's themes, patterns, and evolution.
    
    Extracts themes and patterns from analysis and intelligence output,
    and updates user trajectory.
    
    Args:
        analysis_result: Full analysis result (canonical schema)
        intelligence_output: Intelligence output (from intelligence_core.py)
        user_id: Optional user ID
    
    Returns:
        Dict with trajectory summary
    """
    try:
        trajectory = load_user_trajectory(user_id)
        
        # Extract themes from intelligence output
        continuity = intelligence_output.get("continuity", {})
        patterns_learned = continuity.get("patterns_learned", [])
        user_pattern = continuity.get("user_pattern", "")
        
        # Extract patterns from analysis result
        composition = analysis_result.get("perception", {}).get("composition", {})
        semantics = analysis_result.get("perception", {}).get("semantics", {})
        
        # Update themes (from patterns_learned)
        themes = trajectory.get("themes", [])
        for pattern in patterns_learned:
            # Extract theme keywords (simplified - can be enhanced)
            if "time" in pattern.lower() and "time" not in themes:
                themes.append("time")
            if "decay" in pattern.lower() and "decay" not in themes:
                themes.append("decay")
            if "organic" in pattern.lower() and "organic integration" not in themes:
                themes.append("organic integration")
            if "minimal" in pattern.lower() and "minimalism" not in themes:
                themes.append("minimalism")
        
        # Update patterns (from composition and semantics)
        patterns = trajectory.get("patterns", [])
        
        # Composition patterns
        if composition.get("available"):
            framing_style = composition.get("subject_framing", {}).get("style", "")
            if framing_style and framing_style not in patterns:
                patterns.append(framing_style)
        
        # Semantic patterns
        if semantics.get("available"):
            genre_hint = semantics.get("genre_hint", "")
            if genre_hint and genre_hint not in patterns:
                patterns.append(genre_hint)
        
        # Update evolution
        evolution = trajectory.get("evolution", [])
        current_state = continuity.get("trajectory", "")
        if current_state:
            # Check if state has changed
            if not evolution or evolution[-1].get("state") != current_state:
                evolution.append({
                    "date": datetime.now().isoformat(),
                    "state": current_state
                })
        
        # Update trajectory
        trajectory["themes"] = themes[:20]  # Limit to top 20
        trajectory["patterns"] = patterns[:20]  # Limit to top 20
        trajectory["evolution"] = evolution
        
        # Save to disk
        save_user_trajectory(trajectory, user_id)
        
        return trajectory
    
    except Exception as e:
        logger.error(f"Failed to track user trajectory: {e}", exc_info=True)
        return {
            "themes": [],
            "patterns": [],
            "evolution": [],
            "last_updated": datetime.now().isoformat()
        }


# ========================================================
# MEMORY FORMATTING FOR INTELLIGENCE CORE
# ========================================================

def format_temporal_memory_for_intelligence(
    signature: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format temporal memory for intelligence core consumption.
    
    Combines pattern memory and user trajectory into a single dict
    that can be passed to intelligence_core.py.
    
    Args:
        signature: Pattern signature
        user_id: Optional user ID
    
    Returns:
        Dict with structure:
        {
            "patterns": [...],  # Similar patterns from memory
            "user_trajectory": {...}  # User trajectory
        }
    """
    try:
        # Query memory patterns
        patterns = query_memory_patterns(signature)
        
        # Load user trajectory
        user_trajectory = load_user_trajectory(user_id)
        
        return {
            "patterns": patterns,
            "user_trajectory": user_trajectory
        }
    
    except Exception as e:
        logger.error(f"Failed to format temporal memory: {e}", exc_info=True)
        return {
            "patterns": [],
            "user_trajectory": {}
        }
