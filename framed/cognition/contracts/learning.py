"""Slice B controlled-learning contracts. Additive; does not replace Slice A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Outcome:
    outcome_id: str
    workspace_id: str
    source_episode_id: str
    source_run_id: str
    kind: str
    verdict: str
    created_by: str
    artefact_hash: str
    created_at: str
    note: str = ""


@dataclass(frozen=True)
class UpdateProposal:
    proposal_id: str
    workspace_id: str
    base_state_version_id: str
    outcome_id: str
    kind: str
    created_by: str
    artefact_hash: str
    created_at: str
    source_episode_id: str
    proposed_snapshot: Dict[str, Any] = field(default_factory=dict)
    proposed_label: str = ""


@dataclass(frozen=True)
class ProposalEvaluation:
    evaluation_id: str
    proposal_id: str
    status: str
    artefact_hash: str
    created_at: str
    replay_status: str = ""
    checks: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PromotionDecision:
    decision_id: str
    proposal_id: str
    action: str
    authority_kind: str
    actor_id: str
    artefact_hash: str
    created_at: str
    evaluation_id: Optional[str] = None
    resulting_state_version_id: Optional[str] = None


def promoted_episode_ids_from_snapshot(snapshot: Optional[Dict[str, Any]]) -> List[str]:
    if not snapshot:
        return []
    raw = snapshot.get("promoted_episode_ids") or []
    return [str(x) for x in raw]
