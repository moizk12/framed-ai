"""Rollback a Slice B promoted state to its parent. Requires external authority."""

from __future__ import annotations

from typing import Any, Dict, Optional

from framed.cognition.constants import ALLOWED_PROMOTION_AUTHORITIES, FORBIDDEN_PROMOTION_AUTHORITIES, ROLLBACK_RECORD_SCHEMA
from framed.cognition.ledger.artefact_store import ArtefactStore, artefact_hash
from framed.cognition.ledger.sqlite_store import CognitionLedger, get_ledger
from framed.cognition.learning.errors import PromotionAuthorityError


def rollback_promoted_state(
    *,
    workspace_id: str,
    authority_kind: str,
    actor_id: str,
    ledger: Optional[CognitionLedger] = None,
) -> Dict[str, Any]:
    """Restore the parent of the current promoted active state."""
    if authority_kind in FORBIDDEN_PROMOTION_AUTHORITIES or authority_kind not in ALLOWED_PROMOTION_AUTHORITIES:
        raise PromotionAuthorityError(
            f"Rollback requires explicit human or TestDaemon authority; got {authority_kind!r}"
        )
    if not actor_id:
        raise PromotionAuthorityError("Rollback requires an explicit actor_id")

    ledger = ledger or get_ledger()
    active = ledger.get_active_state(workspace_id)
    from_id = active["state_version_id"]
    current = ledger.get_state_version(from_id)
    if current is None:
        raise ValueError("Active state version is missing")
    decision = ledger.get_decision_by_resulting_state(from_id)
    if decision is None:
        raise ValueError("Active state was not created by an accepted Slice B proposal")
    parent_id = current.get("parent_version_id")
    if not parent_id:
        raise ValueError("Promoted state has no parent to restore")

    ledger.activate_state_version(workspace_id, parent_id)
    created_at = ArtefactStore.utc_now()
    payload = {
        "schema": ROLLBACK_RECORD_SCHEMA,
        "workspace_id": workspace_id,
        "from_state_version_id": from_id,
        "to_state_version_id": parent_id,
        "authority_kind": authority_kind,
        "actor_id": actor_id,
        "source_decision_id": decision["decision_id"],
        "created_at": created_at,
    }
    rollback_id = artefact_hash(
        {
            "workspace_id": workspace_id,
            "from_state_version_id": from_id,
            "to_state_version_id": parent_id,
            "authority_kind": authority_kind,
            "actor_id": actor_id,
        }
    )
    stored_hash = ledger.put_artefact("rollback_record", "v1", payload)
    ledger.insert_rollback(
        rollback_id=rollback_id,
        workspace_id=workspace_id,
        from_state_version_id=from_id,
        to_state_version_id=parent_id,
        authority_kind=authority_kind,
        actor_id=actor_id,
        artefact_hash=stored_hash,
        created_at=created_at,
    )
    restored = ledger.get_active_state(workspace_id)
    return {
        "rollback_id": rollback_id,
        "from_state_version_id": from_id,
        "to_state_version_id": parent_id,
        "active_state_version_id": restored["state_version_id"],
        "artefact_hash": stored_hash,
        "payload": payload,
    }
