"""
HITL Feedback Storage

Append-only JSONL storage for human feedback.
Path: framed/feedback/hitl_feedback.jsonl (or FRAMED_DATA_DIR/feedback/hitl_feedback.jsonl)
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Use FRAMED_DATA_DIR if set, else package-relative
try:
    import tempfile
    DEFAULT_BASE = os.path.join(tempfile.gettempdir(), "framed")
except Exception:
    DEFAULT_BASE = os.path.expanduser("~/.framed")
BASE_DIR = os.getenv("FRAMED_DATA_DIR", DEFAULT_BASE)
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
HITL_FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "hitl_feedback.jsonl")

VALID_FEEDBACK_TYPES = frozenset(["overconfidence", "missed_alternative", "emphasis_misaligned", "mentor_failure"])


def _ensure_dir():
    os.makedirs(FEEDBACK_DIR, exist_ok=True)


def append_feedback(
    image_id: str,
    feedback: Dict[str, Any],
    signature: str,
) -> bool:
    """
    Append a single feedback entry to the JSONL file.

    Args:
        image_id: Image identifier (e.g. "architecture_042")
        feedback: Feedback payload, e.g.:
            {"type": "overconfidence", "scope": "belief_calibration", "confidence_delta_hint": -0.15}
            {"type": "missed_alternative", "alternative_hint": "green could be painted facade"}
            {"type": "emphasis_misaligned", "dimension": "emotional_weighting"}
            {"type": "mentor_failure", "reason": "generic guidance"}
        signature: Pattern signature (REQUIRED). Prevents global drift from one-off feedback.

    Returns:
        bool: True on success. False if validation fails (e.g. missing signature).
    """
    # Fail fast: pattern_signature is mandatory
    sig = (signature or "").strip()
    if not sig:
        logger.warning("HITL feedback rejected: pattern_signature is required (keeps calibration localized)")
        return False

    entry = {
        "image_id": image_id,
        "pattern_signature": sig,
        "feedback": feedback,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Validate feedback type
    fb_type = feedback.get("type", "")
    if fb_type not in VALID_FEEDBACK_TYPES:
        logger.warning(f"Invalid HITL feedback type: {fb_type}")
        return False
    try:
        _ensure_dir()
        with open(HITL_FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"HITL feedback appended: {image_id} {fb_type}")
        return True
    except Exception as e:
        logger.error(f"Failed to append HITL feedback: {e}", exc_info=True)
        return False


def load_feedback(include_processed: bool = False) -> List[Dict[str, Any]]:
    """
    Load all feedback entries from the JSONL file.

    Args:
        include_processed: If True, include entries marked as processed.
            By default returns only unprocessed (no "processed_at" key).

    Returns:
        List of feedback entries
    """
    if not os.path.exists(HITL_FEEDBACK_PATH):
        return []
    entries = []
    try:
        with open(HITL_FEEDBACK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if include_processed or "processed_at" not in entry:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries
    except Exception as e:
        logger.error(f"Failed to load HITL feedback: {e}", exc_info=True)
        return []
