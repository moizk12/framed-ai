"""
FRAMED Self-Assessment Memory (Option 2: Self-reported uncertainty calibration)

Stores self-reported calibration signals:
- "I was too confident here"
- "I missed alternatives here"

These bias future governors slightly (e.g., reduce confidence ceiling if we've been overconfident).
"""

import os
import json
import logging
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# ========================================================
# CONFIGURATION
# ========================================================

DEFAULT_BASE_DATA_DIR = os.path.join(tempfile.gettempdir(), "framed")
BASE_DATA_DIR = os.getenv("FRAMED_DATA_DIR", DEFAULT_BASE_DATA_DIR)
SELF_ASSESSMENT_PATH = os.path.join(BASE_DATA_DIR, "self_assessment.json")

MAX_ENTRIES = 50
WINDOW_FOR_BIAS = 10  # Use last N assessments for governor bias


# ========================================================
# LOAD / SAVE
# ========================================================

def load_self_assessments() -> List[Dict[str, Any]]:
    """Load self-assessment history from disk."""
    if not os.path.exists(SELF_ASSESSMENT_PATH):
        return []
    try:
        with open(SELF_ASSESSMENT_PATH, "r") as f:
            data = json.load(f)
        entries = data.get("entries", [])
        return entries if isinstance(entries, list) else []
    except Exception as e:
        logger.warning(f"Failed to load self-assessments: {e}")
        return []


def save_self_assessments(entries: List[Dict[str, Any]]) -> bool:
    """Save self-assessment history to disk."""
    try:
        os.makedirs(os.path.dirname(SELF_ASSESSMENT_PATH), exist_ok=True)
        data = {"entries": entries[-MAX_ENTRIES:], "last_updated": datetime.now().isoformat()}
        with open(SELF_ASSESSMENT_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.warning(f"Failed to save self-assessments: {e}")
        return False


# ========================================================
# EXTRACT & STORE
# ========================================================

def extract_self_assessment_from_intelligence(
    intelligence_output: Dict[str, Any],
    reflection: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract self-assessment signals from intelligence output and reflection.

    Sources:
    - self_critique.evolution, past_errors: "too confident", "missed alternatives"
    - meta_cognition.what_i_might_be_missing
    - reflection: quality low + overconfidence = calibration issue

    Returns:
        {"confidence_calibration": "too_high"|"appropriate"|"too_low", "missed_alternatives": bool, "note": str}
        or None if nothing to store
    """
    signals = []
    confidence_calibration = "appropriate"
    missed_alternatives = False
    note = ""

    self_critique = intelligence_output.get("self_critique", {}) or {}
    evolution = (self_critique.get("evolution") or "").lower()
    past_errors = self_critique.get("past_errors", []) or []
    past_errors_text = " ".join(str(e) for e in past_errors).lower()

    meta = intelligence_output.get("meta_cognition", {}) or {}
    what_missing = (meta.get("what_i_might_be_missing") or "").lower()

    combined = f"{evolution} {past_errors_text} {what_missing}"

    # Detect "too confident"
    too_confident_phrases = [
        "too confident", "overconfident", "overly confident", "confident when i shouldn't",
        "was wrong to be certain", "certainty was misplaced", "assumed too much",
    ]
    if any(p in combined for p in too_confident_phrases):
        confidence_calibration = "too_high"
        signals.append("too_confident")

    # Detect "missed alternatives"
    missed_phrases = [
        "missed alternatives", "didn't consider", "should have considered",
        "another interpretation", "alternative reading", "other possibilities",
        "too quick to conclude", "collapsed to one interpretation", "single hypothesis",
    ]
    if any(p in combined for p in missed_phrases):
        missed_alternatives = True
        signals.append("missed_alternatives")

    # Reflection overconfidence (expression used definitive language when confidence was low)
    if reflection and reflection.get("overconfidence_score", 1.0) < 0.6:
        if confidence_calibration == "appropriate":
            confidence_calibration = "too_high"
        signals.append("reflection_overconfidence")

    # Reflection confidence-language mismatch
    if reflection and reflection.get("confidence_language_score", 1.0) < 0.6:
        if confidence_calibration == "appropriate":
            confidence_calibration = "too_high"
        signals.append("reflection_confidence_language")

    # Reflection hypothesis suppression (required multi-hyp but expression had only one)
    if reflection and reflection.get("hypothesis_suppression_score", 1.0) < 0.6:
        if not missed_alternatives:
            missed_alternatives = True
        signals.append("reflection_hypothesis_suppression")

    if confidence_calibration != "appropriate" or missed_alternatives:
        note = "; ".join(signals) if signals else "extracted from self-critique"
        return {
            "confidence_calibration": confidence_calibration,
            "missed_alternatives": missed_alternatives,
            "note": note,
            "date": datetime.now().isoformat(),
            "signals": signals,
        }
    return None


def store_self_assessment(
    intelligence_output: Dict[str, Any],
    reflection: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Extract and store self-assessment if applicable.
    Call after each run (intelligence + reflection available).
    """
    entry = extract_self_assessment_from_intelligence(intelligence_output, reflection)
    if entry:
        entries = load_self_assessments()
        entries.append(entry)
        return save_self_assessments(entries)
    return False


# ========================================================
# GOVERNOR BIAS
# ========================================================

def get_governor_bias(signature: Optional[str] = None) -> Dict[str, float]:
    """
    Compute bias for future governors from:
    1. Recent self-assessments (automatic)
    2. HITL feedback (human-in-the-loop calibration)

    Returns:
        confidence_ceiling_adjustment: -0.05 to 0 (lower ceiling if too confident)
        confidence_floor_adjustment: 0 to 0.05 (higher floor if we missed alternatives)
        multi_hypothesis_bias: 0 to 0.05 (slightly more likely to require multi-hyp when we missed alts)
    """
    # 1. Self-assessment bias
    entries = load_self_assessments()
    recent = entries[-WINDOW_FOR_BIAS:] if entries else []

    too_high_count = sum(1 for e in recent if e.get("confidence_calibration") == "too_high")
    missed_count = sum(1 for e in recent if e.get("missed_alternatives"))
    n = len(recent)

    rationale = []
    ceiling_adj = 0.0
    floor_adj = 0.0
    multi_bias = 0.0

    if n > 0:
        if too_high_count >= 2:
            ratio = too_high_count / n
            ceiling_adj = -0.03 * min(ratio, 1.0)
            rationale.append(f"too_confident {too_high_count}/{n} → ceiling -{abs(ceiling_adj):.2f}")
        if missed_count >= 2:
            ratio = missed_count / n
            floor_adj = 0.02 * min(ratio, 1.0)
            multi_bias = 0.02 * min(ratio, 1.0)
            rationale.append(f"missed_alternatives {missed_count}/{n} → floor +{floor_adj:.2f}, multi_hyp +{multi_bias:.2f}")

    # 2. HITL feedback calibration (human-in-the-loop)
    try:
        from framed.feedback.calibration import get_hitl_calibration
        hitl = get_hitl_calibration(signature)
        hitl_ceiling = hitl.get("confidence_ceiling_adjustment", 0.0)
        hitl_multi = hitl.get("multi_hypothesis_bias", 0.0)
        if hitl_ceiling < 0:
            ceiling_adj = min(ceiling_adj, hitl_ceiling)
            rationale.extend(hitl.get("rationale", [])[:1])
        if hitl_multi > 0:
            multi_bias = max(multi_bias, hitl_multi)
    except Exception:
        pass

    return {
        "confidence_ceiling_adjustment": round(ceiling_adj, 3),
        "confidence_floor_adjustment": round(floor_adj, 3),
        "multi_hypothesis_bias": round(multi_bias, 3),
        "rationale": rationale,
    }
