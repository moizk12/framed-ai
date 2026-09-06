"""Deterministic update-proposal generation. Cannot activate or promote state."""

from __future__ import annotations

from typing import Any, Dict, Optional

from framed.cognition.constants import (
    BELIEF_POLICY_VERSION,
    PROMOTABLE_VERDICTS,
    PROPOSAL_GENERATOR_ID,
    UPDATE_PROPOSAL_SCHEMA,
)
from framed.cognition.contracts.learning import promoted_episode_ids_from_snapshot
from framed.cognition.ledger.artefact_store import ArtefactStore, artefact_hash
from framed.cognition.ledger.sqlite_store import CognitionLedger, get_ledger


def build_promoted_snapshot(parent_snapshot: Dict[str, Any], episode_id: str, label: str) -> Dict[str, Any]:
    promoted = list(promoted_episode_ids_from_snapshot(parent_snapshot))
    if episode_id not in promoted:
        promoted.append(episode_id)
    allowed = list(parent_snapshot.get("allowed_epistemic_states") or ["inferred", "provisional"])
    if "accepted" not in allowed:
        allowed.append("accepted")
    snapshot = dict(parent_snapshot)
    snapshot.update(
        {
            "schema": "state_snapshot_v1",
            "label": label,
            "retrieval_enabled": True,
            "promoted_episode_ids": sorted(promoted),
            "allowed_epistemic_states": allowed,
            "belief_policy_version": BELIEF_POLICY_VERSION,
        }
    )
    return snapshot


def generate_proposal(
    *,
    outcome_id: str,
    ledger: Optional[CognitionLedger] = None,
) -> Dict[str, Any]:
    """Create an immutable typed proposal from an outcome. Does not change active state."""
    ledger = ledger or get_ledger()
    outcome = ledger.get_outcome(outcome_id)
    if outcome is None:
        raise ValueError(f"Unknown outcome: {outcome_id}")
    if outcome["verdict"] not in PROMOTABLE_VERDICTS:
        raise ValueError(
            f"Outcome verdict {outcome['verdict']!r} does not generate a promotion proposal"
        )

    episode = ledger.get_episode(outcome["source_episode_id"])
    if episode is None or episode.get("status") != "closed":
        raise ValueError("Proposal requires a closed source episode")
    indexed = ledger.get_indexed_episode(outcome["source_episode_id"])
    if indexed is None:
        raise ValueError("Source episode is not retrieval-indexed; cannot propose belief promotion")

    workspace_id = outcome["workspace_id"]
    active = ledger.get_active_state(workspace_id)
    base_state_version_id = active["state_version_id"]
    if not active["snapshot"].get("retrieval_enabled"):
        raise ValueError("Cannot propose belief promotion while retrieval is disabled on the active state")

    identity_payload = {
        "schema": UPDATE_PROPOSAL_SCHEMA,
        "kind": "promote_episode_belief",
        "workspace_id": workspace_id,
        "base_state_version_id": base_state_version_id,
        "outcome_id": outcome_id,
        "source_episode_id": outcome["source_episode_id"],
        "source_run_id": outcome["source_run_id"],
        "created_by": PROPOSAL_GENERATOR_ID,
        "belief_policy_version": BELIEF_POLICY_VERSION,
        "parent_snapshot_hash": artefact_hash(active["snapshot"]),
    }
    proposal_id = artefact_hash(identity_payload)
    label = f"state_belief_{proposal_id[:12]}"
    proposed_snapshot = build_promoted_snapshot(
        active["snapshot"], outcome["source_episode_id"], label
    )
    payload = {
        **identity_payload,
        "proposed_label": label,
        "proposed_snapshot": proposed_snapshot,
        "source_scene_signature": indexed.get("scene_signature"),
        "source_category_signature": indexed.get("category_signature"),
    }

    existing = ledger.get_proposal(proposal_id)
    if existing is not None:
        return existing

    created_at = ArtefactStore.utc_now()
    stored_hash = ledger.put_artefact("update_proposal", "v1", {**payload, "created_at": created_at})
    ledger.insert_proposal(
        proposal_id=proposal_id,
        workspace_id=workspace_id,
        base_state_version_id=base_state_version_id,
        outcome_id=outcome_id,
        kind="promote_episode_belief",
        created_by=PROPOSAL_GENERATOR_ID,
        artefact_hash=stored_hash,
        created_at=created_at,
    )
    return ledger.get_proposal(proposal_id)


def proposal_payload(proposal: Dict[str, Any]) -> Dict[str, Any]:
    payload = proposal.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Proposal artefact payload missing")
    return payload
