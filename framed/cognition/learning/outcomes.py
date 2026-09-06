"""Record typed outcomes/feedback. Does not generate proposals or promote state."""

from __future__ import annotations

from typing import Any, Dict, Optional

from framed.cognition.constants import ALLOWED_PROMOTION_AUTHORITIES, OUTCOME_SCHEMA
from framed.cognition.ledger.artefact_store import ArtefactStore, artefact_hash
from framed.cognition.ledger.sqlite_store import CognitionLedger, get_ledger
from framed.cognition.learning.errors import PromotionAuthorityError


def record_outcome(
    *,
    workspace_id: str,
    source_episode_id: str,
    source_run_id: str,
    kind: str,
    verdict: str,
    created_by: str,
    note: str = "",
    ledger: Optional[CognitionLedger] = None,
) -> Dict[str, Any]:
    """Persist an immutable outcome. Model output cannot record outcomes."""
    if created_by not in ALLOWED_PROMOTION_AUTHORITIES:
        raise PromotionAuthorityError(
            f"Outcome created_by {created_by!r} is not an external authority"
        )
    if kind not in ("human_feedback", "testdaemon_eval"):
        raise ValueError(f"Unsupported outcome kind: {kind!r}")
    if verdict not in ("useful", "not_useful", "correction"):
        raise ValueError(f"Unsupported outcome verdict: {verdict!r}")

    ledger = ledger or get_ledger()
    episode = ledger.get_episode(source_episode_id)
    if episode is None:
        raise ValueError(f"Unknown source episode: {source_episode_id}")
    if episode.get("status") != "closed":
        raise ValueError("Outcomes may only attach to closed episodes")
    if episode.get("workspace_id") != workspace_id:
        raise ValueError("Outcome workspace does not match source episode")
    run = ledger.get_run(source_run_id)
    if run is None or run.get("episode_id") != source_episode_id:
        raise ValueError("source_run_id does not belong to source_episode_id")

    payload = {
        "schema": OUTCOME_SCHEMA,
        "workspace_id": workspace_id,
        "source_episode_id": source_episode_id,
        "source_run_id": source_run_id,
        "kind": kind,
        "verdict": verdict,
        "created_by": created_by,
        "note": note,
        "episode_state_version_id": episode.get("state_version_id"),
        "episode_final_fingerprint": episode.get("final_fingerprint"),
        "run_purpose": run.get("run_purpose"),
    }
    digest = artefact_hash(payload)
    existing = ledger.get_outcome(digest)
    if existing is not None:
        existing_payload = ledger.artefacts.get(existing["artefact_hash"])
        return {"outcome_id": existing["outcome_id"], **existing, "payload": existing_payload}

    created_at = ArtefactStore.utc_now()
    stored_hash = ledger.put_artefact("outcome", "v1", {**payload, "created_at": created_at})
    ledger.insert_outcome(
        outcome_id=digest,
        workspace_id=workspace_id,
        source_episode_id=source_episode_id,
        source_run_id=source_run_id,
        kind=kind,
        verdict=verdict,
        created_by=created_by,
        artefact_hash=stored_hash,
        created_at=created_at,
    )
    return {
        "outcome_id": digest,
        "workspace_id": workspace_id,
        "source_episode_id": source_episode_id,
        "source_run_id": source_run_id,
        "kind": kind,
        "verdict": verdict,
        "created_by": created_by,
        "artefact_hash": stored_hash,
        "created_at": created_at,
        "payload": {**payload, "created_at": created_at},
    }
