"""Amended retrieval scoring and candidate selection."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from framed.cognition.contracts.memory import MemoryReference, RetrievalQuery, RetrievalResult, ScoreComponents
from framed.cognition.contracts.runs import RETRIEVAL_ELIGIBLE_PURPOSES, SameAssetPolicy
from framed.cognition.ledger.sqlite_store import CognitionLedger, get_ledger


def _row_val(row: Any, key: str, default: Any = "") -> Any:
    try:
        val = row[key]
    except (KeyError, IndexError):
        return default
    return default if val is None else val


def _reject(
    rejected: List[Dict[str, Any]],
    row: Any,
    reason: str,
    **extra: Any,
) -> None:
    rejected.append(
        {
            "episode_id": row["episode_id"],
            "source_run_id": _row_val(row, "source_run_id"),
            "run_purpose": _row_val(row, "run_purpose", "live"),
            "asset_id": _row_val(row, "asset_id"),
            "rejection_reason": reason,
            **extra,
        }
    )


def _score_candidate(row: Any, query: RetrievalQuery) -> ScoreComponents:
    cat = 1.0 if row["category_signature"] == query.category_signature else 0.0
    scene = 1.0 if row["scene_signature"] == query.scene_signature else 0.0
    goal = 1.0 if row["goal_type"] == query.goal_type else 0.0
    relation = 0.0
    if query.goal_instance_id and row["goal_instance_id"] == query.goal_instance_id:
        relation = 1.0
    recency = 0.1
    final = min(1.0, cat * 0.4 + scene * 0.25 + goal * 0.2 + relation * 0.1 + recency * 0.05)
    return ScoreComponents(
        category_score=cat,
        scene_score=scene,
        goal_score=goal,
        relation_score=relation,
        recency_score=recency,
        final_score=final,
        contamination_flags=(),
    )


def retrieve_memories(
    query: RetrievalQuery,
    ledger: Optional[CognitionLedger] = None,
    state_snapshot: Optional[Dict[str, Any]] = None,
) -> RetrievalResult:
    ledger = ledger or get_ledger()
    result = RetrievalResult(query=query)
    if state_snapshot is not None and not state_snapshot.get("retrieval_enabled", True):
        return result
    horizon = (state_snapshot or {}).get("memory_visibility_horizon")
    cutoff = float((state_snapshot or {}).get("cutoff_score", 0.7))
    candidates = ledger.list_retrieval_candidates(query.workspace_id, query.actor_id)
    scored: List[tuple[Any, ScoreComponents]] = []
    for row in candidates:
        ep_id = row["episode_id"]
        run_id = _row_val(row, "source_run_id")
        purpose = _row_val(row, "run_purpose", "live")
        asset = _row_val(row, "asset_id")

        if ep_id in query.exclude_episode_ids:
            _reject(result.rejected, row, "excluded_by_experiment", excluded_by_experiment=True)
            continue
        if run_id and run_id in query.exclude_run_ids:
            _reject(result.rejected, row, "excluded_by_experiment", excluded_by_experiment=True)
            continue
        if asset in query.exclude_asset_ids:
            _reject(result.rejected, row, "excluded_asset", excluded_by_experiment=True)
            continue
        if purpose not in RETRIEVAL_ELIGIBLE_PURPOSES:
            _reject(result.rejected, row, "ineligible_run_purpose", ineligible_run_purpose=purpose)
            continue
        if asset == query.asset_id:
            if query.same_asset_policy == SameAssetPolicy.EXCLUDE:
                _reject(result.rejected, row, "same_asset", same_asset=True)
                continue
            elif query.same_asset_policy not in (
                SameAssetPolicy.ALLOW_RELATED_REVISION,
                SameAssetPolicy.ALLOW_REPLAY,
            ):
                _reject(result.rejected, row, "same_asset", same_asset=True)
                continue
        if horizon and row["closed_at"] and row["closed_at"] > horizon:
            _reject(result.rejected, row, "outside_visibility_horizon")
            continue

        scores = _score_candidate(row, query)
        if scores.category_score <= 0:
            _reject(result.rejected, row, "category_mismatch", below_threshold=True)
            continue
        if not (scores.scene_score > 0 or scores.goal_score > 0 or scores.relation_score > 0):
            _reject(result.rejected, row, "no_category_signal")
            continue
        if scores.final_score < cutoff:
            _reject(result.rejected, row, "below_threshold", below_threshold=True, final_score=scores.final_score)
            continue
        scored.append((row, scores))

    scored.sort(key=lambda x: x[1].final_score, reverse=True)
    for row, scores in scored[: query.max_results]:
        event_id = _latest_deliberation_event(ledger, row["episode_id"])
        hypothesis = _hypothesis_from_episode(ledger, row["episode_id"])
        ref = MemoryReference(
            memory_ref_id=str(uuid.uuid4()),
            source_episode_id=row["episode_id"],
            source_run_id=_row_val(row, "source_run_id"),
            source_event_id=event_id,
            source_asset_id=_row_val(row, "asset_id"),
            source_run_purpose=_row_val(row, "run_purpose", "live"),
            epistemic_status="provisional",
            lifecycle_status="closed",
            memory_role="prior_experience",
            trust_level="low",
            artefact_hash="",
            scene_signature=row["scene_signature"] or "",
            category_signature=row["category_signature"] or "",
            hypothesis_summary=hypothesis,
            confidence_at_source=None,
            scores=scores,
            match_reason=f"category+signal final={scores.final_score:.2f}",
            eligibility_decision="selected",
        )
        result.references.append(ref)
        result.candidates.append(
            {
                "episode_id": row["episode_id"],
                "source_run_id": _row_val(row, "source_run_id"),
                "run_purpose": _row_val(row, "run_purpose", "live"),
                "scores": scores.__dict__,
            }
        )
    return result


def _latest_deliberation_event(ledger: CognitionLedger, episode_id: str) -> str:
    for ev in reversed(ledger.get_episode_events(episode_id)):
        if ev["event_type"] == "deliberation_snapshot":
            return ev["event_id"]
    return ""


def _hypothesis_from_episode(ledger: CognitionLedger, episode_id: str) -> str:
    for ev in reversed(ledger.get_episode_events(episode_id)):
        if ev["event_type"] == "deliberation_snapshot":
            import json

            payload = json.loads(ev["payload_json"])
            return str(payload.get("primary_hypothesis", ""))
    return ""
