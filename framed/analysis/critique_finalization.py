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

# IC_0027a — structured runtime failure (never accept provider dumps as critique)
_RUNTIME_ERROR_RE = re.compile(
    r"(?i)(critique generation unavailable|error generating critique|"
    r"error code:\s*\d+|insufficient_quota|exceeded your current quota|"
    r"rate[_ ]?limit|api[_ ]?error|openai\.com/docs/guides/error-codes)",
)

STABLE_ERROR_CODES = frozenset(
    {
        "insufficient_quota",
        "rate_limit",
        "api_error",
        "critique_unavailable",
        "empty_critique",
    }
)


class CritiqueRuntimeError(RuntimeError):
    """Provider/runtime failure during critique generation (IC_0027a)."""

    def __init__(self, message: str, *, error_code: str = "critique_unavailable"):
        code = error_code if error_code in STABLE_ERROR_CODES else "critique_unavailable"
        self.error_code = code
        self.stable_message = message
        super().__init__(message)


def classify_critique_runtime_failure(exc_or_text: Any) -> str:
    """Map exception/text → stable error_code (prefer structured signals)."""
    text = str(exc_or_text or "")
    lower = text.lower()
    code = None
    if isinstance(exc_or_text, BaseException):
        code = getattr(exc_or_text, "code", None) or getattr(exc_or_text, "error_code", None)
        body = getattr(exc_or_text, "body", None)
        if isinstance(body, dict):
            err = body.get("error") or {}
            if isinstance(err, dict):
                code = code or err.get("code") or err.get("type")
        err_obj = getattr(exc_or_text, "error", None)
        if isinstance(err_obj, dict):
            code = code or err_obj.get("code") or err_obj.get("type")
    if isinstance(exc_or_text, dict):
        code = exc_or_text.get("code") or (exc_or_text.get("error") or {}).get("code")
    code_s = str(code or "").lower()
    if "insufficient_quota" in code_s or "insufficient_quota" in lower:
        return "insufficient_quota"
    if "rate_limit" in code_s or "rate limit" in lower or "429" in lower:
        return "rate_limit"
    if code_s in STABLE_ERROR_CODES:
        return code_s
    if _RUNTIME_ERROR_RE.search(text):
        return "api_error" if "error" in lower else "critique_unavailable"
    return "critique_unavailable"


def is_runtime_failure_critique(critique: Optional[str]) -> bool:
    """True when critique text is a provider/runtime error dump (legacy or new)."""
    if critique is None:
        return False
    text = str(critique).strip()
    if not text:
        return False
    return bool(_RUNTIME_ERROR_RE.search(text))


def runtime_failure_result(error_code: str, message: str = "") -> Dict[str, Any]:
    code = error_code if error_code in STABLE_ERROR_CODES else "critique_unavailable"
    return {
        "critique": "",
        "failed": True,
        "error_code": code,
        "error": message or code,
        "reflection_report": None,
        "regen_count": 0,
        "downgraded_to_tentative": False,
        "vocab_guard_triggered": False,
        "learning_impact": {"memory_updated": False, "new_pattern_stored": False},
    }


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
    public_safe: bool = False,
) -> Dict[str, Any]:
    """Apply reflection, optional regeneration, and tentative downgrade."""
    intelligence_output = intelligence_output or {}
    interpretive_conclusions = interpretive_conclusions or {}

    # IC_0027a: never reflect/store on runtime error dumps or empty required critique
    if is_runtime_failure_critique(critique):
        code = classify_critique_runtime_failure(critique)
        logger.warning("IC_0027a: short-circuit finalization for runtime failure (%s)", code)
        return runtime_failure_result(code, "critique_runtime_failure")
    if critique is None or not str(critique).strip():
        return runtime_failure_result("empty_critique", "empty_critique")

    reflection = _reflect(critique, intelligence_output, interpretive_conclusions, hitl_mentor_drift_penalty)
    if not reflection:
        return {
            "critique": critique,
            "reflection_report": None,
            "regen_count": 0,
            "downgraded_to_tentative": False,
            "vocab_guard_triggered": False,
        }

    if intelligence_output.get("recognition", {}).get("what_i_see") and not public_safe:
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
            critique = generate_poetic_critique(
                intelligence_output=intelligence_output,
                mentor_mode=mentor_mode,
                public_safe=public_safe,
            )
            critique = integrate_self_correction(
                critique,
                {} if public_safe else intelligence_output.get("self_critique", {}),
            )
        elif analysis_result is not None:
            from framed.analysis.vision import generate_merged_critique
            critique = generate_merged_critique(analysis_result, mentor_mode)
        reflection = _reflect(critique, intelligence_output, interpretive_conclusions, hitl_mentor_drift_penalty)
        if intelligence_output.get("recognition", {}).get("what_i_see") and reflection and not public_safe:
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
