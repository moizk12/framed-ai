"""Retrieval contracts — amended Slice A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from framed.cognition.contracts.runs import SameAssetPolicy


@dataclass(frozen=True)
class ScoreComponents:
    category_score: float
    scene_score: float
    goal_score: float
    relation_score: float
    recency_score: float
    final_score: float
    contamination_flags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalQuery:
    workspace_id: str
    actor_id: str
    asset_id: str
    goal_type: str
    goal_instance_id: Optional[str]
    scene_signature: str
    category_signature: str
    exclude_episode_ids: Tuple[str, ...] = ()
    exclude_run_ids: Tuple[str, ...] = ()
    exclude_asset_ids: Tuple[str, ...] = ()
    comparison_group_id: Optional[str] = None
    max_results: int = 5
    same_asset_policy: SameAssetPolicy = SameAssetPolicy.EXCLUDE
    as_of_visibility: Optional[str] = None


@dataclass(frozen=True)
class MemoryReference:
    memory_ref_id: str
    source_episode_id: str
    source_run_id: str
    source_event_id: str
    source_asset_id: str
    source_run_purpose: str
    epistemic_status: str
    lifecycle_status: str
    memory_role: str
    trust_level: str
    artefact_hash: str
    scene_signature: str
    category_signature: str
    hypothesis_summary: str
    confidence_at_source: Optional[float]
    scores: ScoreComponents
    match_reason: str
    eligibility_decision: str = "selected"


@dataclass
class RetrievalResult:
    query: RetrievalQuery
    references: list[MemoryReference] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
