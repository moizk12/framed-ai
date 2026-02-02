"""
HITL Calibration Store

Derived state from human feedback. Used by:
- Governor bias (overconfidence)
- Ambiguity sensitivity (overconfidence)
- Hypothesis branching (missed_alternative)
- Salience weighting (emphasis_misaligned)
- Mentor tone filters (mentor_failure)

Learning happens in memory, not in prompts.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict
from datetime import datetime

from .storage import load_feedback, HITL_FEEDBACK_PATH

logger = logging.getLogger(__name__)

try:
    import tempfile
    DEFAULT_BASE = os.path.join(tempfile.gettempdir(), "framed")
except Exception:
    DEFAULT_BASE = os.path.expanduser("~/.framed")
BASE_DIR = os.getenv("FRAMED_DATA_DIR", DEFAULT_BASE)
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
HITL_CALIBRATION_PATH = os.path.join(FEEDBACK_DIR, "hitl_calibration.json")

# Max entries per signature for calibration lookback
MAX_OVERCONFIDENCE_PER_SIG = 20
MAX_MISSED_ALT_PER_SIG = 20
MAX_EMPHASIS_PER_SIG = 15
MAX_MENTOR_FAILURE = 30


def _load_calibration() -> Dict[str, Any]:
    if not os.path.exists(HITL_CALIBRATION_PATH):
        return {"last_processed_line": 0, "overconfidence": {}, "missed_alternative": {}, "emphasis_misaligned": {}, "mentor_failure": {}}
    try:
        with open(HITL_CALIBRATION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load HITL calibration: {e}")
        return {"last_processed_line": 0, "overconfidence": {}, "missed_alternative": {}, "emphasis_misaligned": {}, "mentor_failure": {}}


def _save_calibration(cal: Dict[str, Any]) -> bool:
    try:
        os.makedirs(os.path.dirname(HITL_CALIBRATION_PATH), exist_ok=True)
        with open(HITL_CALIBRATION_PATH, "w", encoding="utf-8") as f:
            json.dump(cal, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Failed to save HITL calibration: {e}")
        return False


def ingest_hitl_feedback() -> int:
    """
    Process unprocessed HITL feedback from JSONL into calibration store.

    Returns:
        Number of feedback entries processed.
    """
    cal = _load_calibration()
    last_line = cal.get("last_processed_line", 0)

    if not os.path.exists(HITL_FEEDBACK_PATH):
        return 0

    processed = 0
    line_idx = 0

    with open(HITL_FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_idx += 1
            if line_idx <= last_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Fail fast: pattern_signature is mandatory at ingestion
            sig = entry.get("pattern_signature", "") or entry.get("signature", "")
            sig = (sig or "").strip()
            if not sig:
                logger.warning("HITL feedback skipped: pattern_signature missing (keeps calibration localized)")
                continue

            image_id = entry.get("image_id", "")
            fb = entry.get("feedback", {})
            fb_type = fb.get("type", "")

            if fb_type == "overconfidence":
                scope = fb.get("scope", "belief_calibration")
                delta_hint = fb.get("confidence_delta_hint")
                if "overconfidence" not in cal:
                    cal["overconfidence"] = {}
                if sig not in cal["overconfidence"]:
                    cal["overconfidence"][sig] = []
                cal["overconfidence"][sig].append({
                    "image_id": image_id,
                    "scope": scope,
                    "confidence_delta_hint": delta_hint,
                    "timestamp": entry.get("timestamp", ""),
                })
                cal["overconfidence"][sig] = cal["overconfidence"][sig][-MAX_OVERCONFIDENCE_PER_SIG:]
                processed += 1

            elif fb_type == "missed_alternative":
                hint = fb.get("alternative_hint", "")
                if "missed_alternative" not in cal:
                    cal["missed_alternative"] = {}
                if sig not in cal["missed_alternative"]:
                    cal["missed_alternative"][sig] = []
                cal["missed_alternative"][sig].append({
                    "image_id": image_id,
                    "alternative_hint": hint,
                    "timestamp": entry.get("timestamp", ""),
                })
                cal["missed_alternative"][sig] = cal["missed_alternative"][sig][-MAX_MISSED_ALT_PER_SIG:]
                processed += 1

            elif fb_type == "emphasis_misaligned":
                dim = fb.get("dimension", "general")
                if "emphasis_misaligned" not in cal:
                    cal["emphasis_misaligned"] = {}
                if sig not in cal["emphasis_misaligned"]:
                    cal["emphasis_misaligned"][sig] = []
                cal["emphasis_misaligned"][sig].append({
                    "image_id": image_id,
                    "dimension": dim,
                    "timestamp": entry.get("timestamp", ""),
                })
                cal["emphasis_misaligned"][sig] = cal["emphasis_misaligned"][sig][-MAX_EMPHASIS_PER_SIG:]
                processed += 1

            elif fb_type == "mentor_failure":
                reason = fb.get("reason", "generic")
                if "mentor_failure" not in cal:
                    cal["mentor_failure"] = {}
                if sig not in cal["mentor_failure"]:
                    cal["mentor_failure"][sig] = []
                cal["mentor_failure"][sig].append({
                    "image_id": image_id,
                    "reason": reason,
                    "timestamp": entry.get("timestamp", ""),
                })
                cal["mentor_failure"][sig] = cal["mentor_failure"][sig][-MAX_MENTOR_FAILURE:]
                processed += 1

    cal["last_processed_line"] = line_idx
    if processed > 0:
        cal["last_ingested"] = datetime.utcnow().isoformat()
        _save_calibration(cal)
        logger.info(f"HITL feedback ingested: {processed} entries")
    return processed


def get_hitl_calibration(signature: Optional[str] = None) -> Dict[str, Any]:
    """
    Get HITL-derived calibration for a pattern signature (or global).

    Used by: governor bias, ambiguity thresholds, hypothesis branching, reflection.

    Returns:
        {
            "confidence_ceiling_adjustment": float (negative = tighter),
            "ambiguity_sensitivity_bump": float (positive = more sensitive),
            "multi_hypothesis_bias": float (positive = more likely to branch),
            "mentor_drift_penalty": float (positive = stricter reflection),
            "emphasis_adjustments": { dimension: float },
            "rejected_hypothesis_hints": [str],  # for missed_alternative
        }
    """
    ingest_hitl_feedback()  # Process any new feedback first
    cal = _load_calibration()

    sig = signature or "_global"
    result = {
        "confidence_ceiling_adjustment": 0.0,
        "ambiguity_sensitivity_bump": 0.0,
        "multi_hypothesis_bias": 0.0,
        "mentor_drift_penalty": 0.0,
        "emphasis_adjustments": {},
        "rejected_hypothesis_hints": [],
        "rationale": [],
    }

    # GUARDRAIL: No HITL signal may push confidence up.
    # HITL is for humility, breadth, alignment — not assertiveness.
    def _clamp_no_confidence_increase(val: float) -> float:
        return min(val, 0.0)

    # Overconfidence: tighten governor for this signature
    overconf = cal.get("overconfidence", {})
    entries = overconf.get(sig, [])
    if entries:
        # Use confidence_delta_hint if present (magnitude from human), else heuristic
        delta_hints = [e.get("confidence_delta_hint") for e in entries if e.get("confidence_delta_hint") is not None]
        if delta_hints:
            avg_delta = sum(delta_hints) / len(delta_hints)
            adj = _clamp_no_confidence_increase(avg_delta)  # Ignore positive hints
        else:
            n = len(entries)
            adj = -0.02 * min(n, 3)
        sens = 0.02 * min(len(entries), 3)
        result["confidence_ceiling_adjustment"] = min(result["confidence_ceiling_adjustment"], adj)
        result["ambiguity_sensitivity_bump"] = max(result["ambiguity_sensitivity_bump"], sens)
        result["rationale"].append(f"overconfidence {len(entries)} for {sig}")

    # Missed alternative: raise multi-hypothesis pressure
    missed = cal.get("missed_alternative", {})
    entries = missed.get(sig, [])
    if entries:
        n = len(entries)
        result["multi_hypothesis_bias"] = max(result["multi_hypothesis_bias"], 0.03 * min(n, 3))
        for e in entries[-5:]:  # Last 5 hints
            h = e.get("alternative_hint", "").strip()
            if h and h not in result["rejected_hypothesis_hints"]:
                result["rejected_hypothesis_hints"].append(h)
        result["rationale"].append(f"missed_alternative {n} for {sig}")

    # Emphasis misaligned: dimension-specific salience adjustments
    emphasis = cal.get("emphasis_misaligned", {})
    entries = emphasis.get(sig, [])
    if entries:
        dims = defaultdict(int)
        for e in entries:
            dims[e.get("dimension", "general")] += 1
        for dim, count in dims.items():
            result["emphasis_adjustments"][dim] = result["emphasis_adjustments"].get(dim, 0) - 0.1 * min(count, 2)
        result["rationale"].append(f"emphasis_misaligned {len(entries)} for {sig}")

    # Mentor failure: tighten reflection drift checks (per-signature)
    mentor = cal.get("mentor_failure", {})
    entries = mentor.get(sig, [])
    if entries:
        n = len(entries)
        result["mentor_drift_penalty"] = 0.05 * min(n, 4)  # Up to 0.2 stricter
        result["rationale"].append(f"mentor_failure {n} for {sig}")

    # GUARDRAIL: No HITL signal may push confidence up.
    # HITL is for humility, breadth, alignment — not assertiveness.
    result["confidence_ceiling_adjustment"] = min(result["confidence_ceiling_adjustment"], 0.0)
    for dim in result["emphasis_adjustments"]:
        result["emphasis_adjustments"][dim] = min(result["emphasis_adjustments"][dim], 0.0)

    return result
