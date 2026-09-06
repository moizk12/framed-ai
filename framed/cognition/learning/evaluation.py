"""Replay/regression evaluation for update proposals. Does not promote."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from framed.cognition.constants import PROPOSAL_EVALUATION_SCHEMA
from framed.cognition.contracts.learning import promoted_episode_ids_from_snapshot
from framed.cognition.contracts.memory import RetrievalQuery
from framed.cognition.ledger.artefact_store import ArtefactStore, artefact_hash, canonical_json_dumps
from framed.cognition.ledger.sqlite_store import CognitionLedger, get_ledger, reset_ledger
from framed.cognition.learning.proposals import proposal_payload
from framed.cognition.replay.engine import execute_replay
from framed.cognition.retrieval.service import retrieve_memories


def evaluate_proposal(
    *,
    proposal_id: str,
    ledger: Optional[CognitionLedger] = None,
    replay_bundle_path: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run replay + retrieval regression. Never activates a new state."""
    ledger = ledger or get_ledger()
    proposal = ledger.get_proposal(proposal_id)
    if proposal is None:
        raise ValueError(f"Unknown proposal: {proposal_id}")
    if ledger.get_decision_for_proposal(proposal_id) is not None:
        raise ValueError("Proposal already decided; evaluation is closed")

    payload = proposal_payload(proposal)
    checks: List[Dict[str, Any]] = []
    workspace_id = proposal["workspace_id"]
    source_episode_id = payload["source_episode_id"]
    proposed_snapshot = payload["proposed_snapshot"]
    base_state_version_id = proposal["base_state_version_id"]

    active = ledger.get_active_state(workspace_id)
    checks.append(
        {
            "name": "base_state_is_active",
            "pass": active["state_version_id"] == base_state_version_id,
            "active_state_version_id": active["state_version_id"],
            "base_state_version_id": base_state_version_id,
        }
    )
    checks.append(
        {
            "name": "experience_not_automatically_belief",
            "pass": source_episode_id not in promoted_episode_ids_from_snapshot(active["snapshot"]),
            "source_episode_id": source_episode_id,
        }
    )
    checks.append(
        {
            "name": "proposed_snapshot_promotes_source",
            "pass": source_episode_id in promoted_episode_ids_from_snapshot(proposed_snapshot),
        }
    )
    checks.append(
        {
            "name": "proposed_snapshot_keeps_retrieval_enabled",
            "pass": bool(proposed_snapshot.get("retrieval_enabled")),
        }
    )
    checks.append(
        {
            "name": "proposed_cutoff_unchanged",
            "pass": float(proposed_snapshot.get("cutoff_score", 0.7))
            == float(active["snapshot"].get("cutoff_score", 0.7)),
        }
    )
    checks.append(
        {
            "name": "accepted_epistemic_allowed",
            "pass": "accepted" in list(proposed_snapshot.get("allowed_epistemic_states") or []),
        }
    )

    indexed = ledger.get_indexed_episode(source_episode_id)
    if indexed is None:
        checks.append({"name": "source_still_indexed", "pass": False})
        regression_ok = False
        parent_refs: List[str] = []
        proposed_refs: List[str] = []
        proposed_status = None
        parent_status = None
    else:
        query = RetrievalQuery(
            workspace_id=indexed["workspace_id"],
            actor_id=indexed["actor_id"],
            asset_id="slice-b-regression-later-asset",
            goal_type=indexed.get("goal_type") or "critique",
            goal_instance_id=indexed.get("goal_instance_id"),
            scene_signature=str(indexed.get("scene_signature") or ""),
            category_signature=str(indexed.get("category_signature") or ""),
        )
        parent_result = retrieve_memories(query, ledger=ledger, state_snapshot=active["snapshot"])
        proposed_result = retrieve_memories(query, ledger=ledger, state_snapshot=proposed_snapshot)
        parent_match = next(
            (r for r in parent_result.references if r.source_episode_id == source_episode_id),
            None,
        )
        proposed_match = next(
            (r for r in proposed_result.references if r.source_episode_id == source_episode_id),
            None,
        )
        parent_status = parent_match.epistemic_status if parent_match else None
        proposed_status = proposed_match.epistemic_status if proposed_match else None
        parent_refs = [r.source_episode_id for r in parent_result.references]
        proposed_refs = [r.source_episode_id for r in proposed_result.references]
        checks.append(
            {
                "name": "parent_retrieves_as_provisional",
                "pass": parent_match is not None and parent_match.epistemic_status == "provisional",
                "parent_status": parent_status,
            }
        )
        checks.append(
            {
                "name": "proposed_retrieves_as_accepted",
                "pass": proposed_match is not None
                and proposed_match.epistemic_status == "accepted"
                and proposed_match.memory_role == "promoted_belief",
                "proposed_status": proposed_status,
            }
        )
        regression_ok = all(
            c["pass"]
            for c in checks
            if c["name"] in ("parent_retrieves_as_provisional", "proposed_retrieves_as_accepted")
        )

    replay_status, replay_report, used_bundle = _run_replay(
        ledger=ledger,
        source_episode_id=source_episode_id,
        replay_bundle_path=replay_bundle_path,
        evidence_dir=evidence_dir,
    )
    checks.append(
        {
            "name": "replay_pass",
            "pass": replay_status == "PASS",
            "replay_status": replay_status,
        }
    )

    status = "pass" if all(c.get("pass") for c in checks) else "fail"
    created_at = ArtefactStore.utc_now()
    eval_payload = {
        "schema": PROPOSAL_EVALUATION_SCHEMA,
        "proposal_id": proposal_id,
        "proposal_artefact_hash": proposal["artefact_hash"],
        "status": status,
        "replay_status": replay_status,
        "replay_bundle_path": str(used_bundle) if used_bundle else None,
        "checks": checks,
        "parent_retrieved_episode_ids": parent_refs,
        "proposed_retrieved_episode_ids": proposed_refs,
        "created_at": created_at,
        "regression_ok": regression_ok,
    }
    evaluation_id = artefact_hash(
        {
            "proposal_id": proposal_id,
            "status": status,
            "checks": checks,
            "replay_status": replay_status,
        }
    )
    stored_hash = ledger.put_artefact("proposal_evaluation", "v1", eval_payload)
    ledger.insert_evaluation(
        evaluation_id=evaluation_id,
        proposal_id=proposal_id,
        status=status,
        artefact_hash=stored_hash,
        created_at=created_at,
    )
    record = ledger.get_latest_evaluation(proposal_id)
    assert record is not None
    record["replay_report"] = replay_report
    return record


def _run_replay(
    *,
    ledger: CognitionLedger,
    source_episode_id: str,
    replay_bundle_path: Optional[Path],
    evidence_dir: Optional[Path],
) -> tuple[str, Dict[str, Any], Optional[Path]]:
    bundle_path = replay_bundle_path
    if bundle_path is None:
        unique_ids = _collect_eval_run_ids(ledger, source_episode_id)
        if not unique_ids:
            return "FAIL", {"status": "FAIL", "reason": "no_runs_to_replay"}, None
        bundle = ledger.export_replay_bundle(unique_ids)
        target_dir = Path(evidence_dir) if evidence_dir is not None else Path(tempfile.mkdtemp(prefix="framed_slice_b_eval_"))
        target_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = target_dir / "slice_b_eval_replay_bundle.json"
        bundle_path.write_text(canonical_json_dumps(bundle), encoding="utf-8")
    try:
        report = execute_replay(bundle_path)
        status = str(report.get("status") or "FAIL")
    except Exception as exc:
        report = {"status": "FAIL", "error": str(exc), "bundle_path": str(bundle_path)}
        status = "FAIL"
    finally:
        reset_ledger(ledger.db_path, artefact_root=ledger.artefacts.root)
    return status, report, bundle_path


def _collect_eval_run_ids(ledger: CognitionLedger, source_episode_id: str) -> List[str]:
    runs = ledger.list_runs_for_episode(source_episode_id) + ledger.list_runs_retrieving_episode(source_episode_id)
    pending: List[str] = []
    for run in runs:
        pending.append(run["run_id"])
        if run.get("baseline_run_id"):
            pending.append(run["baseline_run_id"])
    unique_ids: List[str] = []
    seen = set()
    for run_id in pending:
        if run_id and run_id not in seen:
            seen.add(run_id)
            unique_ids.append(run_id)
    return unique_ids
