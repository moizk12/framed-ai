"""Versioned, allowlisted public API adapter for Track A."""

from __future__ import annotations

import math
import time
import uuid
from typing import Any


class PublicAnalysisUnavailable(RuntimeError):
    """The analysis core did not produce a complete public critique."""


def run_public_analysis(image_path: str, filename: str) -> tuple[dict, int]:
    """Run the existing analysis core without research persistence."""
    from framed.analysis.critique_finalization import CritiqueRuntimeError, finalize_critique_with_reflection
    from framed.analysis.expression_layer import generate_poetic_critique, integrate_self_correction
    from framed.analysis.vision import run_full_analysis

    started = time.perf_counter()
    internal = run_full_analysis(
        image_path,
        photo_id=f"public-{uuid.uuid4().hex}",
        filename=filename,
        public_safe=True,
    )
    errors = internal.get("errors") if isinstance(internal, dict) else None
    if not isinstance(internal, dict) or (isinstance(errors, dict) and (errors.get("critical") or errors.get("image_load") or errors.get("pipeline"))):
        raise PublicAnalysisUnavailable("analysis_core_failed")

    intelligence = internal.get("intelligence") if isinstance(internal.get("intelligence"), dict) else {}
    recognition = intelligence.get("recognition") if isinstance(intelligence.get("recognition"), dict) else {}
    if not recognition.get("what_i_see"):
        raise PublicAnalysisUnavailable("recognition_unavailable")

    try:
        critique = generate_poetic_critique(
            intelligence,
            mentor_mode="Balanced Mentor",
            public_safe=True,
        )
        critique = integrate_self_correction(critique, {})
        finalized = finalize_critique_with_reflection(
            critique,
            intelligence,
            analysis_result=internal,
            mentor_mode="Balanced Mentor",
            hitl_mentor_drift_penalty=0.0,
            public_safe=True,
        )
    except CritiqueRuntimeError as exc:
        raise PublicAnalysisUnavailable("critique_provider_unavailable") from exc
    if finalized.get("failed") or not str(finalized.get("critique") or "").strip():
        raise PublicAnalysisUnavailable("critique_unavailable")
    internal["critique"] = finalized["critique"]
    duration_ms = max(0, round((time.perf_counter() - started) * 1000))
    return internal, duration_ms


def build_public_analysis_dto(
    internal: dict,
    *,
    request_id: str,
    analysis_id: str,
    duration_ms: int,
) -> dict:
    """Construct the v1 DTO from an explicit public field allowlist."""
    intelligence = _mapping(internal.get("intelligence"))
    recognition = _mapping(intelligence.get("recognition"))
    perception = _mapping(internal.get("perception"))
    semantics = _mapping(_mapping(perception.get("semantics")))
    visual = _mapping(internal.get("visual_evidence"))
    scene_gate = _mapping(visual.get("scene_gate"))

    recognition_text = _text(recognition.get("what_i_see")) or _text(semantics.get("caption")) or "Recognition was unavailable."
    scene_type = _text(scene_gate.get("scene_type")) or "unclassified photograph"
    confidence = _number(recognition.get("confidence"))
    grounding = visual.get("grounding")
    boxes = [_public_box(item) for item in grounding] if isinstance(grounding, list) else []
    boxes = [item for item in boxes if item]
    grounding_state = "disabled"
    if _grounding_enabled():
        grounding_state = "available" if boxes else "empty"
    else:
        boxes = []

    claim_traces = _claim_traces(visual)
    limitations = ["FRAMED offers one evidence-based interpretation, not a final judgment."]
    if grounding_state == "disabled":
        limitations.append("Localized grounding was disabled for this analysis.")
    elif grounding_state == "empty":
        limitations.append("No localized grounding regions were available; claims rely on scene-level evidence.")

    public_recognition: dict[str, Any] = {"text": recognition_text}
    if confidence is not None:
        public_recognition["confidence"] = max(0.0, min(1.0, confidence))
    return {
        "request_id": request_id,
        "analysis_id": analysis_id,
        "status": "complete",
        "critique": str(internal.get("critique") or "").strip(),
        "evidence": {
            "recognition": public_recognition,
            "scene": {"type": scene_type},
            "grounding": {"state": grounding_state, "boxes": boxes},
            "claim_traces": claim_traces,
        },
        "limitations": limitations,
        "meta": {"duration_ms": duration_ms, "cached": False, "contract_version": "1"},
    }


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _grounding_enabled() -> bool:
    import os

    return os.getenv("ENABLE_DENSE_GROUNDING_PROBE", "false").lower() == "true"


def _public_box(value: Any) -> dict | None:
    box = _mapping(value)
    coordinates = {name: _number(box.get(name)) for name in ("x", "y", "w", "h")}
    if any(value is None for value in coordinates.values()):
        return None
    x, y, width, height = (coordinates[name] for name in ("x", "y", "w", "h"))
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1 and x + width <= 1.000001 and y + height <= 1.000001):
        return None
    result: dict[str, Any] = coordinates
    label = _text(box.get("label"))
    confidence = _number(box.get("confidence"))
    if label:
        result["label"] = label
    if confidence is not None:
        result["confidence"] = max(0.0, min(1.0, confidence))
    return result


def _claim_traces(visual: dict) -> list[dict]:
    license_data = _mapping(visual.get("theme_claim_license"))
    reasons = [item for item in license_data.get("reasons", []) if isinstance(item, str)]
    if not reasons:
        return []
    tier = (_text(license_data.get("tier")) or "").lower()
    public_tier = tier if tier in {"licensed", "cautious", "restricted"} else "limited"
    return [{"claim": "Theme interpretation", "support": f"Evidence was evaluated at the {public_tier} confidence tier."}]
