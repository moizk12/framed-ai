"""Deliberation snapshot contract for baseline/memory comparison."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from framed.cognition.constants import FROZEN_DELIBERATION_INPUT_SCHEMA, GOVERNANCE_POLICY_VERSION


@dataclass
class DeliberationSnapshot:
    primary_hypothesis: str
    confidence: float
    strategy: str
    requested_evidence: List[str] = field(default_factory=list)
    alternative_hypotheses: List[str] = field(default_factory=list)
    hypothesis_ranking: List[str] = field(default_factory=list)
    branch_abstain: Optional[str] = None
    scene_signature: str = ""
    category_signature: str = ""
    memory_reference_ids: List[str] = field(default_factory=list)
    run_id: str = ""
    state_version_id: str = ""
    perception_artefact_hash: str = ""
    context_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GovernedSnapshotResult:
    snapshot: DeliberationSnapshot
    snapshot_dict: Dict[str, Any]
    confidence_provenance: Dict[str, Any]
    raw_confidence: float
    baseline_for_compare: Optional[DeliberationSnapshot] = None


def snapshot_from_intelligence(
    intelligence_output: Dict[str, Any],
    *,
    run_id: str,
    state_version_id: str,
    perception_artefact_hash: str,
    context_fingerprint: str,
    memory_reference_ids: List[str],
    scene_signature: str,
    category_signature: str,
    strategy: str,
    requested_evidence: List[str],
) -> DeliberationSnapshot:
    recognition = intelligence_output.get("recognition") or {}
    primary = str(recognition.get("what_i_see") or "")
    alts = recognition.get("alternatives") or recognition.get("rejected_alternatives") or []
    alt_texts = [str(a.get("conclusion", a) if isinstance(a, dict) else a) for a in alts]
    ranking = [primary] + [a for a in alt_texts if a and a != primary]
    abstain = None
    if recognition.get("abstain"):
        abstain = "abstain"
    return DeliberationSnapshot(
        primary_hypothesis=primary,
        confidence=float(recognition.get("confidence", 0.5)),
        strategy=strategy,
        requested_evidence=list(requested_evidence),
        alternative_hypotheses=alt_texts,
        hypothesis_ranking=ranking,
        branch_abstain=abstain,
        scene_signature=scene_signature,
        category_signature=category_signature,
        memory_reference_ids=list(memory_reference_ids),
        run_id=run_id,
        state_version_id=state_version_id,
        perception_artefact_hash=perception_artefact_hash,
        context_fingerprint=context_fingerprint,
    )


def build_frozen_deliberation_input(
    intelligence_output: Dict[str, Any],
    *,
    run_id: str,
    state_version_id: str,
    perception_artefact_hash: str,
    context_fingerprint: str,
    memory_reference_ids: List[str],
    scene_signature: str,
    category_signature: str,
    strategy: str,
    requested_evidence: List[str],
    prompt_policy_version: str = "slice_a_v1",
    model_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical pre-governance deliberation input for deterministic replay."""
    recognition = intelligence_output.get("recognition") or {}
    primary = str(recognition.get("what_i_see") or "")
    alts = recognition.get("alternatives") or recognition.get("rejected_alternatives") or []
    alt_texts = [str(a.get("conclusion", a) if isinstance(a, dict) else a) for a in alts]
    ranking = [primary] + [a for a in alt_texts if a and a != primary]
    abstain = "abstain" if recognition.get("abstain") else None
    raw_confidence = float(recognition.get("confidence", 0.5))
    # Strip non-durable / secret-bearing keys from intelligence for storage.
    safe_intelligence = {
        "recognition": {
            "what_i_see": recognition.get("what_i_see"),
            "confidence": recognition.get("confidence"),
            "alternatives": recognition.get("alternatives"),
            "rejected_alternatives": recognition.get("rejected_alternatives"),
            "abstain": recognition.get("abstain"),
        }
    }
    return {
        "schema": FROZEN_DELIBERATION_INPUT_SCHEMA,
        "intelligence_output": safe_intelligence,
        "primary_hypothesis": primary,
        "alternative_hypotheses": alt_texts,
        "hypothesis_ranking": ranking,
        "raw_confidence": raw_confidence,
        "strategy": strategy,
        "requested_evidence": list(requested_evidence),
        "branch_abstain": abstain,
        "scene_signature": scene_signature,
        "category_signature": category_signature,
        "perception_artefact_hash": perception_artefact_hash,
        "context_fingerprint": context_fingerprint,
        "state_version_id": state_version_id,
        "run_id": run_id,
        "memory_reference_ids": list(memory_reference_ids),
        "prompt_policy_version": prompt_policy_version,
        "deterministic_seed": os.getenv("FRAMED_DETERMINISTIC_SEED"),
        "model_provenance": model_provenance or {
            "model": os.getenv("FRAMED_MODEL_A", "default"),
            "seed": os.getenv("FRAMED_DETERMINISTIC_SEED"),
        },
        "governance_policy_version": GOVERNANCE_POLICY_VERSION,
    }


def build_governed_deliberation_snapshot(
    frozen_input: Dict[str, Any],
    baseline_snapshot: Optional[DeliberationSnapshot],
    memory_reference_ids: List[str],
    policy: Optional[Dict[str, Any]] = None,
) -> GovernedSnapshotResult:
    """
    Production snapshot/governance path shared by finalize and replay.

    Applies provisional confidence clamp when provisional memories + compatible baseline exist.
    """
    _ = policy  # reserved for future policy knobs; clamp policy is versioned in frozen input
    intelligence = frozen_input.get("intelligence_output") or {}
    snap_obj = snapshot_from_intelligence(
        intelligence,
        run_id=str(frozen_input.get("run_id") or ""),
        state_version_id=str(frozen_input.get("state_version_id") or ""),
        perception_artefact_hash=str(frozen_input.get("perception_artefact_hash") or ""),
        context_fingerprint=str(frozen_input.get("context_fingerprint") or ""),
        memory_reference_ids=list(memory_reference_ids),
        scene_signature=str(frozen_input.get("scene_signature") or ""),
        category_signature=str(frozen_input.get("category_signature") or ""),
        strategy=str(frozen_input.get("strategy") or "standard"),
        requested_evidence=list(frozen_input.get("requested_evidence") or []),
    )
    raw_confidence = float(frozen_input.get("raw_confidence", snap_obj.confidence))
    baseline_for_compare = baseline_snapshot
    baseline_confidence = float(baseline_for_compare.confidence) if baseline_for_compare else raw_confidence
    clamp_applied = False
    final_confidence = raw_confidence
    comparison_status = "no_compatible_baseline"
    if memory_reference_ids and baseline_for_compare:
        comparison_status = "compatible_baseline"
        if raw_confidence > baseline_confidence:
            final_confidence = baseline_confidence
            clamp_applied = True
    elif memory_reference_ids:
        comparison_status = "missing_baseline"

    snap_obj = DeliberationSnapshot(
        **{
            **snap_obj.to_dict(),
            "confidence": final_confidence,
            "memory_reference_ids": list(memory_reference_ids),
        }
    )
    confidence_provenance = {
        "raw_confidence": raw_confidence,
        "baseline_confidence": baseline_confidence if baseline_for_compare else None,
        "final_confidence": final_confidence,
        "clamp_applied": clamp_applied,
        "comparison_status": comparison_status,
        "governing_policy_version": frozen_input.get("governance_policy_version") or GOVERNANCE_POLICY_VERSION,
        "source_memory_reference_ids": list(memory_reference_ids),
    }
    snap = snap_obj.to_dict()
    snap["perception_artefact_hash"] = frozen_input.get("perception_artefact_hash")
    snap["confidence_provenance"] = confidence_provenance
    return GovernedSnapshotResult(
        snapshot=snap_obj,
        snapshot_dict=snap,
        confidence_provenance=confidence_provenance,
        raw_confidence=raw_confidence,
        baseline_for_compare=baseline_for_compare,
    )
