"""Shared critique reflection, regeneration, and kill-switch downgrade."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _reflect(critique: str, intelligence_output: Dict[str, Any], interpretive_conclusions: Optional[Dict[str, Any]], hitl_penalty: float):
    from framed.analysis.reflection import reflect_on_critique

    if intelligence_output.get("recognition", {}).get("what_i_see"):
        return reflect_on_critique(critique, intelligence_output, hitl_mentor_drift_penalty=hitl_penalty)
    if interpretive_conclusions:
        return reflect_on_critique(critique, interpretive_conclusions)
    return None


def finalize_critique_with_reflection(
    critique: str,
    intelligence_output: Dict[str, Any],
    *,
    interpretive_conclusions: Optional[Dict[str, Any]] = None,
    analysis_result: Optional[Dict[str, Any]] = None,
    mentor_mode: str = "Balanced Mentor",
    hitl_mentor_drift_penalty: float = 0.0,
    regenerate_fn: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """Apply reflection, optional regeneration, and tentative downgrade."""
    intelligence_output = intelligence_output or {}
    interpretive_conclusions = interpretive_conclusions or {}

    reflection = _reflect(critique, intelligence_output, interpretive_conclusions, hitl_mentor_drift_penalty)
    if not reflection:
        return {
            "critique": critique,
            "reflection_report": None,
            "regen_count": 0,
            "downgraded_to_tentative": False,
        }

    if intelligence_output.get("recognition", {}).get("what_i_see"):
        try:
            from framed.analysis.self_assessment import store_self_assessment
            store_self_assessment(intelligence_output, reflection)
        except Exception:
            pass

    max_regenerations = int(os.environ.get("FRAMED_MAX_REGENERATIONS", "1"))
    regen_count = 0

    while reflection.get("requires_regeneration", False) and regen_count < max_regenerations:
        logger.warning(
            "Reflection: Regenerating critique (attempt %s/%s, quality: %.2f)",
            regen_count + 1,
            max_regenerations,
            reflection.get("quality_score", 0.0),
        )
        if regenerate_fn is not None:
            critique = regenerate_fn()
        elif intelligence_output.get("recognition", {}).get("what_i_see"):
            from framed.analysis.expression_layer import generate_poetic_critique, integrate_self_correction
            critique = generate_poetic_critique(intelligence_output=intelligence_output, mentor_mode=mentor_mode)
            critique = integrate_self_correction(critique, intelligence_output.get("self_critique", {}))
        elif analysis_result is not None:
            from framed.analysis.vision import generate_merged_critique
            critique = generate_merged_critique(analysis_result, mentor_mode)
        reflection = _reflect(critique, intelligence_output, interpretive_conclusions, hitl_mentor_drift_penalty)
        if intelligence_output.get("recognition", {}).get("what_i_see") and reflection:
            try:
                from framed.analysis.self_assessment import store_self_assessment
                store_self_assessment(intelligence_output, reflection)
            except Exception:
                pass
        regen_count += 1

    downgraded_to_tentative = False
    if reflection.get("requires_regeneration", False) and regen_count >= max_regenerations:
        primary = (
            intelligence_output.get("recognition", {}).get("what_i_see")
            or interpretive_conclusions.get("primary_interpretation", {}).get("conclusion", "")
        )
        critique = (
            f"One plausible reading is: {primary[:200]}... "
            "This interpretation remains tentative; the evidence supports multiple readings."
        )
        reflection = {**reflection, "requires_regeneration": False, "downgraded_to_tentative": True}
        downgraded_to_tentative = True

    return {
        "critique": critique,
        "reflection_report": reflection,
        "regen_count": regen_count,
        "downgraded_to_tentative": downgraded_to_tentative,
    }
