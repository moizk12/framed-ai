"""Deliberation snapshot contract for baseline/memory comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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
