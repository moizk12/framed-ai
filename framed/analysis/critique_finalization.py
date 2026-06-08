"""Shared critique reflection, regeneration, and kill-switch downgrade."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BANNED_OVER_POETIC = re.compile(
    r"\b(whisper(?:s|ing|ed)?|tapestry|symphony|soul|ethereal|silent conversation)\b",
    re.I,
)


def _active_correction_rules() -> List[str]:
    try:
        from framed.analysis.interpretive_memory import get_active_rules
        return list(get_active_rules() or [])
    except Exception as exc:
        logger.debug("correction rules unavailable: %s", exc)
        return []


def _rules_want_anti_poetic(rules: List[str]) -> bool:
    for rule in rules:
        lower = rule.lower()
        if "over-poetic" in lower or "over_poetic" in lower or "over poetic" in lower:
            return True
    return False


def check_vocab_guard(critique: str, rules: Optional[List[str]] = None) -> bool:
    """Return True if critique violates anti-over-poetic guard when rules are active."""
    rules = rules if rules is not None else _active_correction_rules()
    if not _rules_want_anti_poetic(rules):
        return False
    return bool(_BANNED_OVER_POETIC.search(critique))


def _is_inside_quotes(text: str, start_idx: int) -> bool:
    # Straight quotes: inside if odd number of `"` before the match.
    straight_inside = (text[:start_idx].count('"') % 2) == 1
    # Curly quotes: inside if we've seen an opening “ without its closing ”.
    curly_inside = text[:start_idx].count("“") > text[:start_idx].count("”")
    return straight_inside or curly_inside


def sanitize_banned_vocab(critique: str) -> Tuple[str, bool]:
    """Replace banned terms with non-banned alternatives.

    Returns: (sanitized_critique, changed)
    """
    replacements = {
        "tapestry": "pattern",
        "symphony": "composition",
        "soul": "inner presence",
        "ethereal": "delicate",
        "silent conversation": "quiet exchange",
    }

    changed = False

    def _repl(match: re.Match[str]) -> str:
        nonlocal changed
        if _is_inside_quotes(critique, match.start()):
            return match.group(0)

        term = match.group(0).lower()
        if term.startswith("whisper"):
            changed = True
            return "subtle suggestion"
        if term in replacements:
            changed = True
            return replacements[term]
        return match.group(0)

    out = _BANNED_OVER_POETIC.sub(_repl, critique)
    return out, changed


def _tentative_critique(
    intelligence_output: Dict[str, Any],
    interpretive_conclusions: Dict[str, Any],
) -> str:
    primary = (
        intelligence_output.get("recognition", {}).get("what_i_see")
        or interpretive_conclusions.get("primary_interpretation", {}).get("conclusion", "")
    )
    return (
        f"One plausible reading is: {primary[:200]}... "
        "This interpretation remains tentative; the evidence supports multiple readings."
    )


def _apply_downgrade(
    intelligence_output: Dict[str, Any],
    interpretive_conclusions: Dict[str, Any],
    reflection: Dict[str, Any],
    *,
    vocab_guard: bool = False,
) -> Tuple[str, Dict[str, Any], bool]:
    critique = _tentative_critique(intelligence_output, interpretive_conclusions)
    report = {**reflection, "requires_regeneration": False, "downgraded_to_tentative": True}
    return critique, report, True


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
            "vocab_guard_triggered": False,
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
    vocab_guard_triggered = False
    if reflection.get("requires_regeneration", False) and regen_count >= max_regenerations:
        critique, reflection, downgraded_to_tentative = _apply_downgrade(
            intelligence_output, interpretive_conclusions, reflection
        )
    elif check_vocab_guard(critique):
        vocab_guard_triggered = True
        critique, changed = sanitize_banned_vocab(critique)
        # If terms were quoted and left intact, fall back to downgrade.
        if not changed or _BANNED_OVER_POETIC.search(critique):
            critique, reflection, downgraded_to_tentative = _apply_downgrade(
                intelligence_output, interpretive_conclusions, reflection
            )

    return {
        "critique": critique,
        "reflection_report": reflection,
        "regen_count": regen_count,
        "downgraded_to_tentative": downgraded_to_tentative,
        "vocab_guard_triggered": vocab_guard_triggered,
    }
