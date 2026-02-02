"""
FRAMED Human-in-the-Loop (HITL) Feedback

Humans do not tell FRAMED "what is true."
They tell FRAMED where its belief formation was miscalibrated.

That's the difference between training a model and evolving an intelligence.

Feedback types:
- overconfidence: FRAMED sounded certain when evidence was ambiguous
- missed_alternative: Another reading was clearly viable
- emphasis_misaligned: FRAMED focused on the wrong thing
- mentor_failure: Tone felt generic, advice obvious, voice drifted
"""

from .storage import append_feedback, load_feedback, HITL_FEEDBACK_PATH
from .calibration import get_hitl_calibration, ingest_hitl_feedback

__all__ = [
    "append_feedback",
    "load_feedback",
    "HITL_FEEDBACK_PATH",
    "get_hitl_calibration",
    "ingest_hitl_feedback",
]
