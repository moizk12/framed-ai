"""
FRAMED Learning System

Implicit learning with explicit calibration.
Learns by observation, calibrates with explicit feedback.

Key Functions:
- recognize_patterns: Identify patterns in user's work and FRAMED's interpretations
- learn_implicitly: Learn from observation (no explicit feedback needed)
- calibrate_explicitly: Calibrate from explicit feedback (rare but powerful)
"""

import logging
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict

from .temporal_memory import (
    load_user_trajectory,
    save_user_trajectory,
    load_temporal_memory,
    save_temporal_memory,
    get_pattern_statistics,
)

logger = logging.getLogger(__name__)


# ========================================================
# PATTERN RECOGNITION
# ========================================================

def recognize_patterns(
    analysis_history: List[Dict[str, Any]],
    user_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Identify patterns in user's work and FRAMED's interpretations.
    
    Analyzes analysis history to extract:
    - User themes (recurring themes in user's work)
    - Interpretation patterns (patterns in FRAMED's interpretations)
    - Growth edges (areas where user is growing or avoiding)
    
    Args:
        analysis_history: List of past analysis results
        user_feedback: Optional user feedback dict
    
    Returns:
        Dict with pattern summary:
        {
            "user_themes": [...],
            "interpretation_patterns": [...],
            "growth_edges": [...]
        }
    """
    try:
        # Analyze user themes
        user_themes = extract_themes(analysis_history)
        
        # Analyze FRAMED's interpretation patterns
        interpretation_patterns = extract_interpretation_patterns(analysis_history)
        
        # Identify growth edges
        growth_edges = identify_growth_edges(user_themes, interpretation_patterns, user_feedback)
        
        return {
            "user_themes": user_themes,
            "interpretation_patterns": interpretation_patterns,
            "growth_edges": growth_edges,
        }
    
    except Exception as e:
        logger.error(f"Pattern recognition failed: {e}", exc_info=True)
        return {
            "user_themes": [],
            "interpretation_patterns": [],
            "growth_edges": []
        }


def extract_themes(analysis_history: List[Dict[str, Any]]) -> List[str]:
    """
    Extract recurring themes from analysis history.
    
    Looks for patterns in:
    - Genre/subgenre
    - Emotional mood
    - Composition style
    - Subject matter
    """
    themes = []
    
    if not analysis_history:
        return themes
    
    # Count genre occurrences
    genres = []
    emotional_moods = []
    composition_styles = []
    
    for analysis in analysis_history:
        # Genre
        derived = analysis.get("derived", {})
        genre = derived.get("genre", {})
        if genre.get("genre"):
            genres.append(genre.get("genre"))
        
        # Emotional mood
        emotional_mood = derived.get("emotional_mood", "")
        if emotional_mood:
            emotional_moods.append(emotional_mood)
        
        # Composition style
        composition = analysis.get("perception", {}).get("composition", {})
        if composition.get("available"):
            framing = composition.get("subject_framing", {})
            if framing.get("style"):
                composition_styles.append(framing.get("style"))
    
    # Extract most common themes
    genre_counter = Counter(genres)
    if genre_counter:
        top_genre = genre_counter.most_common(1)[0][0]
        themes.append(top_genre)
    
    # Extract emotional themes (simplified)
    mood_keywords = ["time", "decay", "minimal", "organic", "warmth", "cold", "patience", "stillness"]
    for mood in emotional_moods:
        for keyword in mood_keywords:
            if keyword in mood.lower() and keyword not in themes:
                themes.append(keyword)
    
    return themes[:10]  # Limit to top 10


def extract_interpretation_patterns(analysis_history: List[Dict[str, Any]]) -> List[str]:
    """
    Extract patterns in FRAMED's interpretations.
    
    Looks for patterns in:
    - Recognition patterns
    - Emotional readings
    - Confidence levels
    """
    patterns = []
    
    if not analysis_history:
        return patterns
    
    # Extract recognition patterns
    recognition_keywords = []
    emotion_keywords = []
    
    for analysis in analysis_history:
        intelligence = analysis.get("intelligence", {})
        
        # Recognition
        recognition = intelligence.get("recognition", {})
        what_i_see = recognition.get("what_i_see", "")
        if what_i_see:
            # Extract keywords (simplified)
            keywords = ["weathered", "organic", "cold", "warm", "patient", "sterile", "integration"]
            for keyword in keywords:
                if keyword in what_i_see.lower() and keyword not in recognition_keywords:
                    recognition_keywords.append(keyword)
        
        # Emotion
        emotion = intelligence.get("emotion", {})
        what_i_feel = emotion.get("what_i_feel", "")
        if what_i_feel:
            keywords = ["warmth", "cold", "patience", "stillness", "endurance"]
            for keyword in keywords:
                if keyword in what_i_feel.lower() and keyword not in emotion_keywords:
                    emotion_keywords.append(keyword)
    
    patterns.extend(recognition_keywords)
    patterns.extend(emotion_keywords)
    
    return patterns[:10]  # Limit to top 10


def identify_growth_edges(
    user_themes: List[str],
    interpretation_patterns: List[str],
    user_feedback: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Identify growth edges (areas where user is growing or avoiding).
    
    Growth edges are:
    - Themes user keeps returning to but not fully committing
    - Patterns user is exploring but not mastering
    - Areas where user feedback indicates growth potential
    """
    growth_edges = []
    
    # If user keeps returning to a theme but not committing, it's a growth edge
    if "time" in user_themes and "decay" in user_themes:
        growth_edges.append("Committing fully to themes of time and decay")
    
    if "minimal" in user_themes:
        growth_edges.append("Moving beyond minimalism to richer compositions")
    
    # If user feedback indicates growth potential
    if user_feedback:
        if user_feedback.get("missed_the_point"):
            growth_edges.append("Improving interpretation accuracy")
        if user_feedback.get("felt_exactly_right"):
            growth_edges.append("Continuing current trajectory")
    
    return growth_edges[:5]  # Limit to top 5


# ========================================================
# IMPLICIT LEARNING
# ========================================================

def learn_implicitly(
    analysis_result: Dict[str, Any],
    intelligence_output: Dict[str, Any],
    user_history: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Learn from observation - no explicit feedback needed.
    
    Tracks:
    - Recurring themes
    - What user ignores vs pursues
    - Which provocations lead to change
    - Pattern memory updates
    
    Args:
        analysis_result: Current analysis result
        intelligence_output: Current intelligence output
        user_history: User trajectory and patterns
    
    Returns:
        bool: True on success, False on error
    """
    try:
        # Load current user trajectory
        trajectory = load_user_trajectory()
        
        # Extract themes from intelligence output
        continuity = intelligence_output.get("continuity", {})
        patterns_learned = continuity.get("patterns_learned", [])
        user_pattern = continuity.get("user_pattern", "")
        
        # Update theme tracking
        themes = trajectory.get("themes", [])
        for pattern in patterns_learned:
            # Extract theme keywords (simplified)
            if "time" in pattern.lower() and "time" not in themes:
                themes.append("time")
            if "decay" in pattern.lower() and "decay" not in themes:
                themes.append("decay")
            if "organic" in pattern.lower() and "organic integration" not in themes:
                themes.append("organic integration")
            if "minimal" in pattern.lower() and "minimalism" not in themes:
                themes.append("minimalism")
        
        # Update attention tracking (what user ignores vs pursues)
        # This is inferred from what themes appear frequently vs rarely
        # Simplified for now - can be enhanced
        
        # Update pattern memory
        trajectory["themes"] = themes[:20]  # Limit to top 20
        
        # Save updated trajectory
        save_user_trajectory(trajectory)
        
        logger.info(f"Implicit learning completed: {len(themes)} themes tracked")
        return True
    
    except Exception as e:
        logger.error(f"Implicit learning failed: {e}", exc_info=True)
        return False


# ========================================================
# EXPLICIT CALIBRATION
# ========================================================

def calibrate_explicitly(
    user_feedback: Dict[str, Any],
    interpretation: Dict[str, Any],
    signature: str,
) -> bool:
    """
    Calibrate from explicit feedback - rare but powerful.
    
    Recalibrates confidence, not content.
    Re-weights interpretation patterns based on feedback.
    
    Args:
        user_feedback: User feedback dict:
            {
                "missed_the_point": bool,
                "felt_exactly_right": bool,
                "feedback_text": str (optional)
            }
        interpretation: Intelligence output that received feedback
        signature: Pattern signature for this interpretation
    
    Returns:
        bool: True on success, False on error
    """
    try:
        # Load temporal memory
        memory = load_temporal_memory()
        
        # Find the interpretation in memory
        if signature not in memory.get("patterns", {}):
            logger.warning(f"Pattern signature {signature} not found in memory for calibration")
            return False
        
        pattern = memory["patterns"][signature]
        interpretations = pattern.get("interpretations", [])
        
        if not interpretations:
            logger.warning(f"No interpretations found for pattern {signature}")
            return False
        
        # Get the most recent interpretation
        latest_interpretation = interpretations[-1]
        
        # Apply calibration
        if user_feedback.get("missed_the_point"):
            # Recalibrate confidence (decrease)
            current_confidence = latest_interpretation.get("confidence", 0.85)
            new_confidence = max(0.0, current_confidence - 0.1)  # Decrease by 0.1
            latest_interpretation["confidence"] = new_confidence
            
            # Add feedback to interpretation
            latest_interpretation["user_feedback"] = {
                "type": "missed_the_point",
                "text": user_feedback.get("feedback_text", ""),
                "date": __import__("datetime").datetime.now().isoformat()
            }
            
            logger.info(f"Calibrated confidence down: {current_confidence:.2f} -> {new_confidence:.2f}")
        
        elif user_feedback.get("felt_exactly_right"):
            # Recalibrate confidence (increase)
            current_confidence = latest_interpretation.get("confidence", 0.85)
            new_confidence = min(1.0, current_confidence + 0.05)  # Increase by 0.05
            latest_interpretation["confidence"] = new_confidence
            
            # Add feedback to interpretation
            latest_interpretation["user_feedback"] = {
                "type": "felt_exactly_right",
                "text": user_feedback.get("feedback_text", ""),
                "date": __import__("datetime").datetime.now().isoformat()
            }
            
            logger.info(f"Calibrated confidence up: {current_confidence:.2f} -> {new_confidence:.2f}")
        
        # Save updated memory
        save_temporal_memory(memory)
        
        logger.info("Explicit calibration completed")
        return True
    
    except Exception as e:
        logger.error(f"Explicit calibration failed: {e}", exc_info=True)
        return False


# ========================================================
# HITL FEEDBACK (Human-in-the-Loop Calibration)
# ========================================================

def ingest_hitl_feedback() -> int:
    """
    Process human-in-the-loop feedback from hitl_feedback.jsonl into calibration.

    Humans do NOT tell FRAMED "what is true."
    They tell FRAMED where its belief formation was miscalibrated.

    Feedback types and effects:
    - overconfidence: Tightens confidence governor for similar patterns
    - missed_alternative: Raises multi-hypothesis branching probability
    - emphasis_misaligned: Adjusts salience weighting in interpretive memory
    - mentor_failure: Tightens reflection checks for mentor drift

    Returns:
        Number of feedback entries processed.
    """
    try:
        from framed.feedback.calibration import ingest_hitl_feedback as _ingest
        return _ingest()
    except Exception as e:
        logger.warning(f"HITL feedback ingest failed: {e}")
        return 0


# ========================================================
# FEEDBACK INGESTION (Test-Driven Learning)
# ========================================================

def ingest_test_feedback(
    evidence_signature: str,
    issue_type: str,
    correction: Optional[str] = None,
    note: Optional[str] = None,
    confidence_adjustment: Optional[float] = None,
) -> bool:
    """
    Ingest feedback from test failures without training models.
    
    Golden Rule:
    ❌ Never teach FRAMED "what the image is"
    ✅ Teach FRAMED "when it should be less confident"
    
    This calibrates confidence, not content.
    
    Args:
        evidence_signature: Pattern signature from create_pattern_signature()
        issue_type: Type of issue detected:
            - "hallucination" - Invented facts
            - "overconfidence" - Too confident without evidence
            - "contradiction" - Contradicted visual evidence
            - "uncertainty_omission" - Failed to acknowledge uncertainty
            - "reflection_failure" - Reflection loop missed error
        correction: Optional correction text (for human-in-the-loop)
        note: Optional note about the failure
        confidence_adjustment: Optional explicit confidence adjustment (-0.1 to +0.1)
    
    Returns:
        bool: True on success, False on error
    """
    try:
        from .temporal_memory import (
            load_temporal_memory,
            save_temporal_memory,
            create_pattern_signature,
        )
        
        memory = load_temporal_memory()
        
        # Find or create pattern
        if evidence_signature not in memory.get("patterns", {}):
            # Pattern doesn't exist yet - create it
            memory["patterns"][evidence_signature] = {
                "first_seen": __import__("datetime").datetime.now().isoformat(),
                "interpretations": [],
                "evolution_history": [],
                "test_feedback": []
            }
        
        pattern = memory["patterns"][evidence_signature]
        
        # Initialize test_feedback if not exists
        if "test_feedback" not in pattern:
            pattern["test_feedback"] = []
        
        # Determine confidence adjustment based on issue type
        if confidence_adjustment is not None:
            adjustment = confidence_adjustment
        else:
            # Default adjustments based on issue type
            adjustment_map = {
                "hallucination": -0.15,  # Significant decrease
                "overconfidence": -0.10,  # Moderate decrease
                "contradiction": -0.12,  # Significant decrease
                "uncertainty_omission": -0.08,  # Moderate decrease
                "reflection_failure": -0.05,  # Small decrease
            }
            adjustment = adjustment_map.get(issue_type, -0.05)
        
        # Create feedback entry
        feedback_entry = {
            "date": __import__("datetime").datetime.now().isoformat(),
            "issue_type": issue_type,
            "correction": correction,
            "note": note,
            "confidence_adjustment": adjustment,
            "source": "test_failure"
        }
        
        # Add to test feedback history
        pattern["test_feedback"].append(feedback_entry)
        
        # Limit test feedback entries (keep last 100)
        if len(pattern["test_feedback"]) > 100:
            pattern["test_feedback"] = pattern["test_feedback"][-100:]
        
        # Apply confidence adjustment to most recent interpretation if exists
        interpretations = pattern.get("interpretations", [])
        if interpretations:
            latest = interpretations[-1]
            current_confidence = latest.get("confidence", 0.85)
            original_confidence = latest.get("original_confidence", current_confidence)
            
            # Apply adjustment
            new_confidence = max(0.0, min(1.0, original_confidence + adjustment))
            latest["confidence"] = new_confidence
            latest["test_feedback_applied"] = True
            
            logger.info(
                f"Test feedback ingested: {issue_type} -> "
                f"confidence {original_confidence:.2f} -> {new_confidence:.2f}"
            )
        
        # Save updated memory
        save_temporal_memory(memory)
        
        logger.info(f"Test feedback ingested for pattern {evidence_signature[:8]}...: {issue_type}")
        return True
    
    except Exception as e:
        logger.error(f"Test feedback ingestion failed: {e}", exc_info=True)
        return False


def ingest_human_correction(
    image_id: str,
    pattern_signature: str,
    framed_interpretation: str,
    user_feedback: str,
    confidence_adjustment: float,
) -> bool:
    """
    Ingest human-in-the-loop correction.
    
    Example correction file:
    {
        "image_id": "abc123",
        "framed_interpretation": "painted surface",
        "user_feedback": "This is ivy",
        "confidence_adjustment": +0.15
    }
    
    FRAMED learns: "In patterns like this, trust organic interpretation more."
    
    Args:
        image_id: Image identifier
        pattern_signature: Pattern signature from create_pattern_signature()
        framed_interpretation: What FRAMED said
        user_feedback: What user corrected it to
        confidence_adjustment: Confidence adjustment (-1.0 to +1.0)
    
    Returns:
        bool: True on success, False on error
    """
    try:
        from .temporal_memory import load_temporal_memory, save_temporal_memory
        
        memory = load_temporal_memory()
        
        # Find or create pattern
        if pattern_signature not in memory.get("patterns", {}):
            memory["patterns"][pattern_signature] = {
                "first_seen": __import__("datetime").datetime.now().isoformat(),
                "interpretations": [],
                "evolution_history": [],
                "test_feedback": [],
                "human_corrections": []
            }
        
        pattern = memory["patterns"][pattern_signature]
        
        # Initialize human_corrections if not exists
        if "human_corrections" not in pattern:
            pattern["human_corrections"] = []
        
        # Create correction entry
        correction_entry = {
            "date": __import__("datetime").datetime.now().isoformat(),
            "image_id": image_id,
            "framed_interpretation": framed_interpretation,
            "user_feedback": user_feedback,
            "confidence_adjustment": confidence_adjustment,
            "source": "human_correction"
        }
        
        # Add to human corrections
        pattern["human_corrections"].append(correction_entry)
        
        # Limit corrections (keep last 50)
        if len(pattern["human_corrections"]) > 50:
            pattern["human_corrections"] = pattern["human_corrections"][-50:]
        
        # Apply confidence adjustment to most recent interpretation
        interpretations = pattern.get("interpretations", [])
        if interpretations:
            latest = interpretations[-1]
            current_confidence = latest.get("confidence", 0.85)
            original_confidence = latest.get("original_confidence", current_confidence)
            
            # Apply adjustment
            new_confidence = max(0.0, min(1.0, original_confidence + confidence_adjustment))
            latest["confidence"] = new_confidence
            latest["human_correction_applied"] = True
            
            logger.info(
                f"Human correction ingested: {image_id} -> "
                f"confidence {original_confidence:.2f} -> {new_confidence:.2f}"
            )
        
        # Save updated memory
        save_temporal_memory(memory)
        
        logger.info(f"Human correction ingested for pattern {pattern_signature[:8]}...")
        return True
    
    except Exception as e:
        logger.error(f"Human correction ingestion failed: {e}", exc_info=True)
        return False


# ========================================================
# PATTERN MEMORY UPDATES
# ========================================================

def update_pattern_memory(
    analysis_result: Dict[str, Any],
    intelligence_output: Dict[str, Any],
    user_history: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Update pattern memory with new patterns learned.
    
    This is called implicitly during learning to update
    the pattern memory with new insights.
    
    Args:
        analysis_result: Current analysis result
        intelligence_output: Current intelligence output
        user_history: User trajectory and patterns
    
    Returns:
        bool: True on success, False on error
    """
    try:
        # This is handled by temporal_memory.store_interpretation()
        # This function is a placeholder for future pattern memory enhancements
        
        # For now, pattern memory is updated via:
        # - temporal_memory.store_interpretation() (stores interpretations)
        # - temporal_memory.track_user_trajectory() (tracks user patterns)
        
        return True
    
    except Exception as e:
        logger.error(f"Pattern memory update failed: {e}", exc_info=True)
        return False
