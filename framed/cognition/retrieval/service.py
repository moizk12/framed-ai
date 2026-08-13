"""Amended retrieval scoring and candidate selection."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from framed.cognition.constants import MAX_MEMORY_REFS
from framed.cognition.contracts.learning import promoted_episode_ids_from_snapshot
from framed.cognition.contracts.memory import MemoryReference, RetrievalQuery, RetrievalResult, ScoreComponents
from framed.cognition.contracts.runs import RETRIEVAL_ELIGIBLE_PURPOSES, SameAssetPolicy
from framed.cognition.ledger.sqlite_store import CognitionLedger, get_ledger
from framed.cognition.ledger.artefact_store import artefact_hash as compute_artefact_hash


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
        if query.as_of_visibility and row["closed_at"] and row["closed_at"] > query.as_of_visibility:
            _reject(result.rejected, row, "outside_retrieval_as_of")
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
    max_results = min(query.max_results, MAX_MEMORY_REFS)
    for row, scores in scored[:max_results]:
        event_id, source_artefact_hash, hypothesis, source_confidence = _source_deliberation_provenance(
            ledger, row["episode_id"]
        )
        if not event_id or not source_artefact_hash:
            _reject(
                result.rejected,
                row,
                "incomplete_source_provenance",
                missing_event_id=not bool(event_id),
                missing_artefact_hash=not bool(source_artefact_hash),
            )
            continue
        if not _row_val(row, "source_run_id"):
            _reject(result.rejected, row, "incomplete_source_provenance", missing_source_run_id=True)
            continue
        promoted_ids = set(promoted_episode_ids_from_snapshot(state_snapshot))
        if row["episode_id"] in promoted_ids:
            epistemic_status = "accepted"
            trust_level = "medium"
            memory_role = "promoted_belief"
        else:
            epistemic_status = "provisional"
            trust_level = "low"
            memory_role = "prior_experience"
        allowed_epistemic = (state_snapshot or {}).get("allowed_epistemic_states")
        if allowed_epistemic and epistemic_status not in allowed_epistemic:
            _reject(result.rejected, row, "epistemic_status_not_allowed", epistemic_status=epistemic_status)
            continue
        deterministic_memory_ref_id = _deterministic_memory_ref_id(
            query=query,
            source_episode_id=row["episode_id"],
            source_event_id=event_id,
            source_run_id=_row_val(row, "source_run_id"),
        )
        ref = MemoryReference(
            memory_ref_id=deterministic_memory_ref_id,
            source_episode_id=row["episode_id"],
            source_run_id=_row_val(row, "source_run_id"),
            source_event_id=event_id,
            source_asset_id=_row_val(row, "asset_id"),
            source_run_purpose=_row_val(row, "run_purpose", "live"),
            epistemic_status=epistemic_status,
            lifecycle_status="closed",
            memory_role=memory_role,
            trust_level=trust_level,
            artefact_hash=source_artefact_hash,
            scene_signature=row["scene_signature"] or "",
            category_signature=row["category_signature"] or "",
            hypothesis_summary=hypothesis,
            confidence_at_source=source_confidence,
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


def _deterministic_memory_ref_id(
    *,
    query: RetrievalQuery,
    source_episode_id: str,
    source_event_id: str,
    source_run_id: str,
) -> str:
    # Deterministic provenance-based ID so replay can reproduce deltas exactly.
    payload = {
        "memory_ref_id_seed": 1,
        "workspace_id": query.workspace_id,
        "actor_id": query.actor_id,
        "asset_id": query.asset_id,
        "goal_type": query.goal_type,
        "goal_instance_id": query.goal_instance_id,
        "scene_signature": query.scene_signature,
        "category_signature": query.category_signature,
        "exclude_episode_ids": query.exclude_episode_ids,
        "exclude_run_ids": query.exclude_run_ids,
        "comparison_group_id": query.comparison_group_id,
        "same_asset_policy": query.same_asset_policy.value,
        "source_episode_id": source_episode_id,
        "source_event_id": source_event_id,
        "source_run_id": source_run_id,
    }
    return compute_artefact_hash(payload)


def _source_deliberation_provenance(
    ledger: CognitionLedger, episode_id: str
) -> tuple[str, str, str, Optional[float]]:
    for ev in reversed(ledger.get_episode_events(episode_id)):
        if ev["event_type"] != "deliberation_snapshot":
            continue
        payload = json.loads(ev["payload_json"])
        event_id = ev.get("event_id") or ""
        artefact = ev.get("artefact_hash") or ""
        hypothesis = str(payload.get("primary_hypothesis", ""))
        confidence = payload.get("confidence")
        try:
            confidence_val = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_val = None
        return event_id, artefact, hypothesis, confidence_val
    return "", "", "", None


def _latest_deliberation_event(ledger: CognitionLedger, episode_id: str) -> str:
    event_id, _, _, _ = _source_deliberation_provenance(ledger, episode_id)
    return event_id


def _hypothesis_from_episode(ledger: CognitionLedger, episode_id: str) -> str:
    _, _, hypothesis, _ = _source_deliberation_provenance(ledger, episode_id)
    return hypothesis
