"""Explicit external promotion authority. Separate from proposal generation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from framed.cognition.constants import (
    ALLOWED_PROMOTION_AUTHORITIES,
    FORBIDDEN_PROMOTION_AUTHORITIES,
    PROMOTION_DECISION_SCHEMA,
)
from framed.cognition.ledger.artefact_store import ArtefactStore, artefact_hash
from framed.cognition.ledger.sqlite_store import CognitionLedger, get_ledger
from framed.cognition.learning.errors import (
    PromotionAuthorityError,
    PromotionBlockedError,
    ProposalImmutableError,
)
from framed.cognition.learning.proposals import proposal_payload


def _require_external_authority(authority_kind: str) -> None:
    if authority_kind in FORBIDDEN_PROMOTION_AUTHORITIES or authority_kind not in ALLOWED_PROMOTION_AUTHORITIES:
        raise PromotionAuthorityError(
            f"Promotion requires explicit human or TestDaemon authority; got {authority_kind!r}"
        )


def accept_proposal(
    *,
    proposal_id: str,
    authority_kind: str,
    actor_id: str,
    ledger: Optional[CognitionLedger] = None,
) -> Dict[str, Any]:
    return decide_proposal(
        proposal_id=proposal_id,
        action="accept",
        authority_kind=authority_kind,
        actor_id=actor_id,
        ledger=ledger,
    )


def reject_proposal(
    *,
    proposal_id: str,
    authority_kind: str,
    actor_id: str,
    ledger: Optional[CognitionLedger] = None,
) -> Dict[str, Any]:
    return decide_proposal(
        proposal_id=proposal_id,
        action="reject",
        authority_kind=authority_kind,
        actor_id=actor_id,
        ledger=ledger,
    )


def decide_proposal(
    *,
    proposal_id: str,
    action: str,
    authority_kind: str,
    actor_id: str,
    ledger: Optional[CognitionLedger] = None,
) -> Dict[str, Any]:
    """Accept or reject a proposal. Model output cannot reach this path."""
    _require_external_authority(authority_kind)
    if action not in ("accept", "reject"):
        raise ValueError(f"Unsupported decision action: {action!r}")
    if not actor_id:
        raise PromotionAuthorityError("Decision requires an explicit actor_id")

    ledger = ledger or get_ledger()
    proposal = ledger.get_proposal(proposal_id)
    if proposal is None:
        raise ValueError(f"Unknown proposal: {proposal_id}")
    existing = ledger.get_decision_for_proposal(proposal_id)
    if existing is not None:
        raise ProposalImmutableError("Proposal already has a recorded decision")

    evaluation = ledger.get_latest_evaluation(proposal_id)
    resulting_state_version_id = None
    if action == "accept":
        if evaluation is None or evaluation.get("status") != "pass":
            raise PromotionBlockedError(
                "Accept requires a passing replay/regression evaluation recorded before promotion"
            )
        payload = proposal_payload(proposal)
        active = ledger.get_active_state(proposal["workspace_id"])
        if active["state_version_id"] != proposal["base_state_version_id"]:
            raise PromotionBlockedError("Active state no longer matches the proposal base state")
        resulting_state_version_id = ledger.create_state_version(
            workspace_id=proposal["workspace_id"],
            parent_version_id=proposal["base_state_version_id"],
            label=payload["proposed_label"],
            snapshot=payload["proposed_snapshot"],
        )
        ledger.activate_state_version(proposal["workspace_id"], resulting_state_version_id)

    created_at = ArtefactStore.utc_now()
    decision_payload = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "proposal_id": proposal_id,
        "proposal_artefact_hash": proposal["artefact_hash"],
        "evaluation_id": None if evaluation is None else evaluation["evaluation_id"],
        "evaluation_artefact_hash": None if evaluation is None else evaluation["artefact_hash"],
        "action": action,
        "authority_kind": authority_kind,
        "actor_id": actor_id,
        "resulting_state_version_id": resulting_state_version_id,
        "created_at": created_at,
    }
    decision_id = artefact_hash(
        {
            "proposal_id": proposal_id,
            "action": action,
            "authority_kind": authority_kind,
            "actor_id": actor_id,
            "evaluation_id": decision_payload["evaluation_id"],
            "resulting_state_version_id": resulting_state_version_id,
        }
    )
    stored_hash = ledger.put_artefact("promotion_decision", "v1", decision_payload)
    ledger.insert_decision(
        decision_id=decision_id,
        proposal_id=proposal_id,
        evaluation_id=None if evaluation is None else evaluation["evaluation_id"],
        action=action,
        authority_kind=authority_kind,
        actor_id=actor_id,
        resulting_state_version_id=resulting_state_version_id,
        artefact_hash=stored_hash,
        created_at=created_at,
    )
    return ledger.get_decision_for_proposal(proposal_id)
