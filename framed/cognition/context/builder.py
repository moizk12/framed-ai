"""Build deliberation context and compare snapshots."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from framed.cognition.contracts.delta import DeliberationDelta
from framed.cognition.contracts.memory import MemoryReference
from framed.cognition.contracts.snapshot import DeliberationSnapshot


class DeliberationContext:
    """Runtime deliberation context built from retrieved memories."""

    def __init__(
        self,
        memory_references: Optional[List[MemoryReference]] = None,
        prior_hypothesis: Optional[str] = None,
        prior_confidence: Optional[float] = None,
        confidence_delta_cap: float = 0.0,
        requested_evidence: Optional[List[str]] = None,
        strategy_hint: Optional[str] = None,
        prompt_block: str = "",
    ) -> None:
        self.memory_references = memory_references or []
        self.prior_hypothesis = prior_hypothesis
        self.prior_confidence = prior_confidence
        self.confidence_delta_cap = confidence_delta_cap
        self.requested_evidence = requested_evidence or []
        self.strategy_hint = strategy_hint
        self.prompt_block = prompt_block


def build_deliberation_context(
    references: List[MemoryReference],
    baseline_hypothesis: str,
    baseline_confidence: float,
) -> DeliberationContext:
    from framed.cognition.context.formatting import format_cognition_context_for_prompt

    ctx = DeliberationContext(memory_references=list(references))
    if not references:
        ctx.prior_hypothesis = baseline_hypothesis
        ctx.prior_confidence = baseline_confidence
        return ctx
    primary = references[0]
    ctx.prior_hypothesis = primary.hypothesis_summary or baseline_hypothesis
    ctx.prior_confidence = baseline_confidence
    ctx.confidence_delta_cap = 0.0
    ctx.requested_evidence = ["verify_scene_consistency", "check_prior_failure_mode"]
    ctx.strategy_hint = "consider_prior_provisional_experience"
    from framed.cognition.context.formatting import build_cognition_context

    cog = build_cognition_context(references)
    ctx.prompt_block = format_cognition_context_for_prompt(cog)
    return ctx


def snapshots_compatible(baseline: DeliberationSnapshot, memory: DeliberationSnapshot) -> Tuple[bool, str]:
    if baseline.perception_artefact_hash != memory.perception_artefact_hash:
        return False, "perception_artefact_hash mismatch"
    if baseline.scene_signature != memory.scene_signature:
        return False, "scene_signature mismatch"
    if baseline.category_signature != memory.category_signature:
        return False, "category_signature mismatch"
    return True, "ok"


def compare_deliberation_snapshots(
    baseline: DeliberationSnapshot,
    memory: DeliberationSnapshot,
    source_refs: List[str],
) -> List[DeliberationDelta]:
    ok, reason = snapshots_compatible(baseline, memory)
    if not ok:
        return [
            DeliberationDelta(
                field_changed="_compatibility",
                baseline_value=baseline.perception_artefact_hash,
                memory_condition_value=memory.perception_artefact_hash,
                source_memory_refs=source_refs,
                mechanism="reject_incompatible",
                reason=reason,
            )
        ]
    b = baseline.to_dict()
    m = memory.to_dict()
    return compute_deliberation_delta(b, m, source_refs)


def compute_deliberation_delta(
    baseline: Dict[str, Any],
    with_memory: Dict[str, Any],
    source_refs: List[str],
) -> List[DeliberationDelta]:
    deltas: List[DeliberationDelta] = []
    if baseline.get("primary_hypothesis") != with_memory.get("primary_hypothesis"):
        deltas.append(
            DeliberationDelta(
                field_changed="primary_hypothesis",
                baseline_value=baseline.get("primary_hypothesis"),
                memory_condition_value=with_memory.get("primary_hypothesis"),
                source_memory_refs=source_refs,
                mechanism="introduce_alternative",
                reason="Retrieved provisional hypothesis altered primary hypothesis",
            )
        )
    b_alts = set(baseline.get("alternative_hypotheses") or [])
    m_alts = set(with_memory.get("alternative_hypotheses") or [])
    if b_alts != m_alts:
        deltas.append(
            DeliberationDelta(
                field_changed="candidate_set",
                baseline_value=sorted(b_alts),
                memory_condition_value=sorted(m_alts),
                source_memory_refs=source_refs,
                mechanism="introduce_alternative",
                reason="Retrieved experience changed candidate hypothesis set",
            )
        )
    if baseline.get("hypothesis_ranking") != with_memory.get("hypothesis_ranking"):
        deltas.append(
            DeliberationDelta(
                field_changed="ranking",
                baseline_value=baseline.get("hypothesis_ranking"),
                memory_condition_value=with_memory.get("hypothesis_ranking"),
                source_memory_refs=source_refs,
                mechanism="reprioritize",
                reason="Retrieved experience changed hypothesis ranking",
            )
        )
    if with_memory.get("requested_evidence") and with_memory.get("requested_evidence") != baseline.get("requested_evidence"):
        deltas.append(
            DeliberationDelta(
                field_changed="requested_evidence",
                baseline_value=baseline.get("requested_evidence", []),
                memory_condition_value=with_memory.get("requested_evidence"),
                source_memory_refs=source_refs,
                mechanism="request_evidence",
                reason="Retrieved failure mode triggered evidence request",
            )
        )
    b_conf = float(baseline.get("confidence", 0.5))
    m_conf = float(with_memory.get("confidence", 0.5))
    if m_conf < b_conf:
        deltas.append(
            DeliberationDelta(
                field_changed="confidence",
                baseline_value=b_conf,
                memory_condition_value=m_conf,
                source_memory_refs=source_refs,
                mechanism="reduce_confidence",
                reason="Provisional memory reduced confidence",
            )
        )
    elif m_conf > b_conf and source_refs:
        deltas.append(
            DeliberationDelta(
                field_changed="confidence",
                baseline_value=b_conf,
                memory_condition_value=m_conf,
                source_memory_refs=source_refs,
                mechanism="reject_confidence_increase",
                reason="Provisional memory cannot increase confidence by policy",
            )
        )
    if with_memory.get("strategy") != baseline.get("strategy"):
        deltas.append(
            DeliberationDelta(
                field_changed="strategy",
                baseline_value=baseline.get("strategy"),
                memory_condition_value=with_memory.get("strategy"),
                source_memory_refs=source_refs,
                mechanism="change_strategy",
                reason="Retrieved experience changed critique strategy",
            )
        )
    if with_memory.get("branch_abstain") != baseline.get("branch_abstain"):
        deltas.append(
            DeliberationDelta(
                field_changed="abstention",
                baseline_value=baseline.get("branch_abstain"),
                memory_condition_value=with_memory.get("branch_abstain"),
                source_memory_refs=source_refs,
                mechanism="branch_or_abstain",
                reason="Retrieved experience changed abstention/branch decision",
            )
        )
    return deltas


def snapshot_to_legacy_dict(snap: DeliberationSnapshot) -> Dict[str, Any]:
    return snap.to_dict()
