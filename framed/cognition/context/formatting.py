"""Format typed cognition context for model prompts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from framed.cognition.contracts.memory import MemoryReference

COGNITION_CONTEXT_HEADER = (
    "PRIOR EXPERIENCE — RETRIEVED, PROVISIONAL, NON-AUTHORITATIVE\n"
    "Use as one piece of evidence. Ground recognition in the CURRENT image.\n"
    "Provisional memory must NOT increase confidence solely because a prior exists."
)


def memory_reference_to_dict(ref: MemoryReference) -> Dict[str, Any]:
    return {
        "memory_ref_id": ref.memory_ref_id,
        "source_episode_id": ref.source_episode_id,
        "source_event_id": ref.source_event_id,
        "scene_signature": ref.scene_signature,
        "category_signature": ref.category_signature,
        "hypothesis_summary": ref.hypothesis_summary,
        "confidence_at_source": ref.confidence_at_source,
        "epistemic_status": ref.epistemic_status,
        "lifecycle_status": ref.lifecycle_status,
        "trust_level": ref.trust_level,
        "memory_role": ref.memory_role,
        "match_reason": ref.match_reason,
        "contamination_flags": list(ref.scores.contamination_flags),
    }


def build_cognition_context(
    references: List[MemoryReference],
) -> Optional[Dict[str, Any]]:
    if not references:
        return None
    return {
        "schema": "cognition_context_v1",
        "label": COGNITION_CONTEXT_HEADER,
        "retrieved_experiences": [memory_reference_to_dict(r) for r in references],
        "memory_reference_ids": [r.memory_ref_id for r in references],
        "confidence_delta_cap": 0.0,
    }


def format_cognition_context_for_prompt(cognition_context: Optional[Dict[str, Any]]) -> str:
    if not cognition_context or not cognition_context.get("retrieved_experiences"):
        return ""
    lines = [COGNITION_CONTEXT_HEADER, ""]
    for i, exp in enumerate(cognition_context["retrieved_experiences"], 1):
        lines.append(f"Prior experience {i}:")
        lines.append(f"  memory_ref_id: {exp.get('memory_ref_id')}")
        lines.append(f"  source_episode_id: {exp.get('source_episode_id')}")
        lines.append(f"  hypothesis_summary: {exp.get('hypothesis_summary')}")
        lines.append(f"  epistemic_status: {exp.get('epistemic_status')}")
        lines.append(f"  trust_level: {exp.get('trust_level')}")
        lines.append(f"  match_reason: {exp.get('match_reason')}")
        flags = exp.get("contamination_flags") or []
        if flags:
            lines.append(f"  contamination_flags: {', '.join(flags)}")
        lines.append("")
    lines.append(
        "You may introduce alternatives, request evidence, change strategy, branch, or reduce confidence. "
        "Do NOT increase confidence solely because prior experience exists."
    )
    return "\n".join(lines)
