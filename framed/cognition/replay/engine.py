"""Deterministic replay execution and bundle validation."""

from __future__ import annotations

import json
import gc
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from framed.cognition.constants import REPLAY_BUNDLE_SCHEMA
from framed.cognition.contracts.memory import MemoryReference, RetrievalQuery
from framed.cognition.context.builder import compare_deliberation_snapshots
from framed.cognition.contracts.runs import SameAssetPolicy
from framed.cognition.contracts.snapshot import (
    DeliberationSnapshot,
    build_governed_deliberation_snapshot,
)
from framed.cognition.ledger.artefact_store import ArtefactStore, artefact_hash, canonical_json_dumps
from framed.cognition.ledger.sqlite_store import CognitionLedger, clear_ledger, release_ledger_store, reset_ledger
from framed.cognition.retrieval.service import retrieve_memories


REQUIRED_BUNDLE_KEYS = frozenset(
    {"schema", "runs", "episodes", "events", "memory_references", "artefact_manifest"}
)


def validate_bundle_schema(bundle: Dict[str, Any]) -> None:
    if bundle.get("schema") != REPLAY_BUNDLE_SCHEMA:
        raise ValueError(f"Invalid bundle schema: {bundle.get('schema')!r}")
    missing = REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    if missing:
        raise ValueError(f"Bundle missing required keys: {sorted(missing)}")
    if not isinstance(bundle["runs"], list):
        raise ValueError("Bundle runs must be a list")


def validate_bundle_integrity(
    bundle: Dict[str, Any],
    *,
    artefacts: Optional[ArtefactStore] = None,
) -> None:
    """Validate artefact manifest entries and expected snapshot hashes."""
    validate_bundle_schema(bundle)
    store = artefacts or _artefacts_from_manifest(bundle)
    manifest = bundle.get("artefact_manifest") or []
    reachable = _reachable_artefact_hashes(bundle)
    manifest_hashes = {row["artefact_hash"] for row in manifest}
    missing_from_manifest = reachable - manifest_hashes
    if missing_from_manifest:
        raise ValueError(f"Reachable artefacts missing from manifest: {sorted(missing_from_manifest)[:5]}")
    for row in manifest:
        digest = row["artefact_hash"]
        if digest not in reachable:
            continue
        payload = store.get(digest)
        computed = artefact_hash(payload)
        if computed != digest:
            raise ValueError(f"Artefact content hash mismatch for {digest}")
    expected = bundle.get("expected_hashes") or {}
    for run in bundle.get("runs", []):
        run_id = run["run_id"]
        run_expected = (expected.get("runs") or {}).get(run_id) or {}
        expected_snapshot_hash = run_expected.get("deliberation_snapshot_hash")
        enforce_expected_deltas = "deliberation_delta_hashes" in run_expected
        expected_delta_hashes = sorted(run_expected.get("deliberation_delta_hashes") or [])

        actual_snapshot_hash: Optional[str] = None
        actual_delta_hashes: List[str] = []
        for ev in bundle.get("events", []):
            if ev.get("run_id") != run_id:
                continue
            if ev.get("event_type") == "deliberation_snapshot":
                ev_hash = ev.get("artefact_hash")
                if ev_hash:
                    payload = json.loads(ev["payload_json"])
                    payload_hash = artefact_hash(payload)
                    if payload_hash != ev_hash:
                        raise ValueError(f"Deliberation snapshot payload hash mismatch for run {run_id}")
                    actual_snapshot_hash = ev_hash
            elif ev.get("event_type") == "deliberation_delta":
                ev_hash = ev.get("artefact_hash")
                if ev_hash:
                    payload = json.loads(ev["payload_json"])
                    payload_hash = artefact_hash(payload)
                    if payload_hash != ev_hash:
                        raise ValueError(f"Deliberation delta payload hash mismatch for run {run_id}")
                    actual_delta_hashes.append(ev_hash)

        if expected_snapshot_hash is not None and actual_snapshot_hash != expected_snapshot_hash:
            raise ValueError(
                f"Expected deliberation_snapshot hash mismatch for run {run_id}: "
                f"expected={expected_snapshot_hash}, actual={actual_snapshot_hash}"
            )
        if enforce_expected_deltas and sorted(actual_delta_hashes) != expected_delta_hashes:
            raise ValueError(
                f"Expected deliberation_delta hash mismatch for run {run_id}: "
                f"expected={expected_delta_hashes}, actual={sorted(actual_delta_hashes)}"
            )


def execute_replay(
    bundle_path: Path,
    *,
    cognition_dir: Optional[Path] = None,
    mutate: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Replay a bundle in an isolated cognition store.

    Re-imports ledger state, re-runs retrieval for memory-enabled runs, and
    compares outputs against recorded references and expected hashes.
    """
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    validate_bundle_schema(bundle)

    original_dir = os.environ.get("FRAMED_COGNITION_DIR")
    original_flag = os.environ.get("FRAMED_COGNITION_V1")
    os.environ["FRAMED_COGNITION_V1"] = "true"

    temp_dir_ctx = None
    resolved_dir: Path
    if cognition_dir is not None:
        resolved_dir = cognition_dir.resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="framed_replay_store_", ignore_cleanup_errors=True)
        resolved_dir = Path(temp_dir_ctx.name)

    try:
        os.environ["FRAMED_COGNITION_DIR"] = str(resolved_dir)
        clear_ledger()
        ledger = reset_ledger(resolved_dir / "cognition_ledger.sqlite3")

        if mutate:
            bundle = _apply_mutation(bundle, mutate)

        _import_bundle(ledger, bundle)
        try:
            validate_bundle_integrity(bundle, artefacts=ledger.artefacts)
        except Exception as exc:
            return {
                "status": "FAIL",
                "bundle_path": str(bundle_path),
                "cognition_dir": str(resolved_dir),
                "replay_checks": [{"match": False, "reason": "bundle_integrity_validation_failed"}],
                "mutated": bool(mutate),
                "error": str(exc),
            }

        replay_checks: List[Dict[str, Any]] = []
        for run in bundle.get("runs", []):
            check = _replay_retrieval_for_run(ledger, bundle, run)
            replay_checks.append(check)

        all_pass = all(c.get("match") for c in replay_checks) if replay_checks else True
        return {
            "status": "PASS" if all_pass else "FAIL",
            "bundle_path": str(bundle_path),
            "cognition_dir": str(resolved_dir),
            "replay_checks": replay_checks,
            "mutated": bool(mutate),
        }
    finally:
        _cleanup_replay_store(resolved_dir)
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()
        if original_dir is None:
            os.environ.pop("FRAMED_COGNITION_DIR", None)
        else:
            os.environ["FRAMED_COGNITION_DIR"] = original_dir
        if original_flag is None:
            os.environ.pop("FRAMED_COGNITION_V1", None)
        else:
            os.environ["FRAMED_COGNITION_V1"] = original_flag


def _cleanup_replay_store(resolved_dir: Path) -> None:
    release_ledger_store(resolved_dir / "cognition_ledger.sqlite3")
    clear_ledger()
    gc.collect()
    for _ in range(20):
        try:
            shutil.rmtree(resolved_dir)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.1)
    shutil.rmtree(resolved_dir, ignore_errors=True)


def _artefacts_from_manifest(bundle: Dict[str, Any]) -> ArtefactStore:
    store = ArtefactStore(root=Path(tempfile.mkdtemp(prefix="framed_replay_artefacts_")))
    for row in bundle.get("artefact_payloads") or []:
        digest = row["artefact_hash"]
        payload = row["payload"]
        rel = f"{digest[:2]}/{digest}.json"
        path = store.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json_dumps(payload), encoding="utf-8")
    return store


def _reachable_artefact_hashes(bundle: Dict[str, Any]) -> Set[str]:
    hashes: Set[str] = set()
    for ev in bundle.get("events", []):
        if ev.get("artefact_hash"):
            hashes.add(ev["artefact_hash"])
    for ep in bundle.get("episodes", []):
        if ep.get("perception_artefact_hash"):
            hashes.add(ep["perception_artefact_hash"])
    for ref in bundle.get("memory_references", []):
        if ref.get("artefact_hash"):
            hashes.add(ref["artefact_hash"])
    for row in bundle.get("state_versions") or []:
        if row.get("snapshot_artefact_hash"):
            hashes.add(row["snapshot_artefact_hash"])
    return hashes


def _import_bundle(ledger: CognitionLedger, bundle: Dict[str, Any]) -> None:
    payloads_by_hash: Dict[str, Any] = {}
    for item in bundle.get("artefact_payloads") or []:
        if isinstance(item, dict) and item.get("artefact_hash"):
            payloads_by_hash[item["artefact_hash"]] = item.get("payload")

    for row in bundle.get("artefact_manifest") or []:
        digest = row["artefact_hash"]
        try:
            ledger.artefacts.get(digest)
            continue
        except (FileNotFoundError, OSError):
            payload = payloads_by_hash.get(digest)
            if payload is not None:
                ledger.put_artefact(row["schema_name"], row["schema_version"], payload)

    with ledger._connect() as conn:
        for row in bundle.get("state_versions") or []:
            conn.execute(
                """
                INSERT OR IGNORE INTO cognitive_state_versions
                (state_version_id, workspace_id, parent_version_id, label, created_at, is_active, snapshot_artefact_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["state_version_id"],
                    row["workspace_id"],
                    row.get("parent_version_id"),
                    row["label"],
                    row.get("created_at", ArtefactStore.utc_now()),
                    row.get("is_active", 0),
                    row["snapshot_artefact_hash"],
                ),
            )
        for row in bundle.get("episodes", []):
            conn.execute(
                """
                INSERT OR IGNORE INTO episodes
                (episode_id, workspace_id, actor_id, asset_id, goal_type, goal_instance_id,
                 status, source_kind, asset_filename, created_at, closed_at, perception_artefact_hash,
                 state_version_id, final_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["episode_id"],
                    row["workspace_id"],
                    row["actor_id"],
                    row["asset_id"],
                    row["goal_type"],
                    row.get("goal_instance_id"),
                    row.get("status", "closed"),
                    row.get("source_kind", "live"),
                    row.get("asset_filename"),
                    row["created_at"],
                    row.get("closed_at"),
                    row.get("perception_artefact_hash"),
                    row["state_version_id"],
                    row.get("final_fingerprint"),
                ),
            )
        for row in bundle.get("runs", []):
            conn.execute(
                """
                INSERT OR IGNORE INTO cognitive_runs
                (run_id, episode_id, mode, run_purpose, baseline_run_id, comparison_group_id,
                 state_version_id, context_fingerprint, retrieval_enabled, retrieval_eligible,
                 model_provenance_json, prompt_provenance_json, started_at, completed_at,
                 provenance_manifest_json, failure_code, failure_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["run_id"],
                    row["episode_id"],
                    row["mode"],
                    row.get("run_purpose", "migration"),
                    row.get("baseline_run_id"),
                    row.get("comparison_group_id"),
                    row["state_version_id"],
                    row.get("context_fingerprint"),
                    row.get("retrieval_enabled", 0),
                    row.get("retrieval_eligible", 0),
                    row.get("model_provenance_json") or json.dumps(row.get("model_provenance") or {}),
                    row.get("prompt_provenance_json") or json.dumps(row.get("prompt_provenance") or {}),
                    row["started_at"],
                    row.get("completed_at"),
                    row.get("provenance_manifest_json"),
                    row.get("failure_code"),
                    row.get("failure_stage"),
                ),
            )
        for row in bundle.get("events", []):
            conn.execute(
                """
                INSERT OR IGNORE INTO episode_events
                (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["event_id"],
                    row["episode_id"],
                    row["run_id"],
                    row["event_type"],
                    row["sequence_num"],
                    row["recorded_at"],
                    row.get("artefact_hash"),
                    row["payload_json"],
                ),
            )
        for row in bundle.get("memory_references", []):
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_references
                (memory_ref_id, run_id, target_episode_id, source_episode_id, source_event_id,
                 source_run_id, source_asset_id, source_run_purpose, eligibility_decision,
                 ref_type, epistemic_status, lifecycle_status, memory_role, trust_level,
                 category_score, scene_score, goal_score, relation_score, recency_score, final_score,
                 contamination_flags_json, match_reason, artefact_hash, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["memory_ref_id"],
                    row["run_id"],
                    row["target_episode_id"],
                    row["source_episode_id"],
                    row.get("source_event_id", ""),
                    row.get("source_run_id", ""),
                    row.get("source_asset_id", ""),
                    row.get("source_run_purpose", "live"),
                    row.get("eligibility_decision", "selected"),
                    row.get("ref_type") or row.get("memory_role", "prior_experience"),
                    row["epistemic_status"],
                    row["lifecycle_status"],
                    row["memory_role"],
                    row["trust_level"],
                    row["category_score"],
                    row["scene_score"],
                    row["goal_score"],
                    row["relation_score"],
                    row["recency_score"],
                    row["final_score"],
                    row.get("contamination_flags_json") or "[]",
                    row["match_reason"],
                    row.get("artefact_hash"),
                    row.get("retrieved_at", ArtefactStore.utc_now()),
                ),
            )
        for row in bundle.get("retrieval_index") or []:
            conn.execute(
                """
                INSERT OR REPLACE INTO retrieval_index
                (episode_id, workspace_id, actor_id, asset_id, scene_signature, category_signature,
                 goal_type, goal_instance_id, recorded_at, closed_at, source_run_id, run_purpose)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["episode_id"],
                    row["workspace_id"],
                    row["actor_id"],
                    row["asset_id"],
                    row.get("scene_signature"),
                    row.get("category_signature"),
                    row.get("goal_type"),
                    row.get("goal_instance_id"),
                    row["recorded_at"],
                    row["closed_at"],
                    row.get("source_run_id"),
                    row.get("run_purpose"),
                ),
            )


def _replay_retrieval_for_run(
    ledger: CognitionLedger,
    bundle: Dict[str, Any],
    run: Dict[str, Any],
) -> Dict[str, Any]:
    def _normalize_scores(*, ref: Dict[str, Any]) -> Dict[str, float]:
        return {
            "category_score": float(ref.get("category_score", 0.0)),
            "scene_score": float(ref.get("scene_score", 0.0)),
            "goal_score": float(ref.get("goal_score", 0.0)),
            "relation_score": float(ref.get("relation_score", 0.0)),
            "recency_score": float(ref.get("recency_score", 0.0)),
            "final_score": float(ref.get("final_score", 0.0)),
        }

    def _normalize_expected_ref(ref: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory_ref_id": ref.get("memory_ref_id"),
            "source_episode_id": ref.get("source_episode_id"),
            "source_event_id": ref.get("source_event_id"),
            "source_run_id": ref.get("source_run_id"),
            "source_asset_id": ref.get("source_asset_id"),
            "source_run_purpose": ref.get("source_run_purpose"),
            "epistemic_status": ref.get("epistemic_status"),
            "lifecycle_status": ref.get("lifecycle_status"),
            "memory_role": ref.get("memory_role"),
            "trust_level": ref.get("trust_level"),
            "artefact_hash": ref.get("artefact_hash"),
            "scene_signature": ref.get("scene_signature"),
            "category_signature": ref.get("category_signature"),
            "hypothesis_summary": ref.get("hypothesis_summary"),
            "confidence_at_source": ref.get("confidence_at_source"),
            "match_reason": ref.get("match_reason"),
            "eligibility_decision": ref.get("eligibility_decision"),
            "scores": _normalize_scores(ref=ref),
        }

    def _normalize_actual_ref(ref: MemoryReference) -> Dict[str, Any]:
        return {
            "memory_ref_id": ref.memory_ref_id,
            "source_episode_id": ref.source_episode_id,
            "source_event_id": ref.source_event_id,
            "source_run_id": ref.source_run_id,
            "source_asset_id": ref.source_asset_id,
            "source_run_purpose": ref.source_run_purpose,
            "epistemic_status": ref.epistemic_status,
            "lifecycle_status": ref.lifecycle_status,
            "memory_role": ref.memory_role,
            "trust_level": ref.trust_level,
            "artefact_hash": ref.artefact_hash,
            "scene_signature": ref.scene_signature,
            "category_signature": ref.category_signature,
            "hypothesis_summary": ref.hypothesis_summary,
            "confidence_at_source": ref.confidence_at_source,
            "match_reason": ref.match_reason,
            "eligibility_decision": ref.eligibility_decision,
            "scores": {
                "category_score": float(ref.scores.category_score),
                "scene_score": float(ref.scores.scene_score),
                "goal_score": float(ref.scores.goal_score),
                "relation_score": float(ref.scores.relation_score),
                "recency_score": float(ref.scores.recency_score),
                "final_score": float(ref.scores.final_score),
            },
        }

    def _frozen_input_for_run(run_id: str) -> Optional[Dict[str, Any]]:
        for ev in bundle.get("events", []):
            if ev.get("run_id") != run_id or ev.get("event_type") != "frozen_deliberation_input_recorded":
                continue
            digest = ev.get("artefact_hash")
            if not digest:
                return None
            return ledger.artefacts.get(digest)
        return None

    def _baseline_payload_for_run_id(run_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not run_id:
            return None
        for ev in bundle.get("events", []):
            if ev.get("run_id") != run_id or ev.get("event_type") != "deliberation_snapshot":
                continue
            return json.loads(ev["payload_json"])
        return None

    run_id = run["run_id"]
    episode_id = run["episode_id"]
    episode = next(e for e in bundle["episodes"] if e["episode_id"] == episode_id)
    expected = (bundle.get("expected_hashes") or {}).get("runs") or {}
    run_expected = expected.get(run_id) or {}
    expected_delta_hashes = sorted(run_expected.get("deliberation_delta_hashes") or [])
    expected_snapshot_hash = run_expected.get("deliberation_snapshot_hash")
    expected_frozen_hash = run_expected.get("frozen_deliberation_input_hash")
    expected_confidence_hash = run_expected.get("confidence_provenance_hash")
    expected_context_fp = run_expected.get("context_fingerprint") or run.get("context_fingerprint")
    expected_state_version = run_expected.get("state_version_id") or run.get("state_version_id")
    expected_perception = run_expected.get("perception_artefact_hash")
    retrieval_as_of = run_expected.get("retrieval_as_of") or run.get("started_at")

    expected_refs = [r for r in bundle.get("memory_references", []) if r.get("run_id") == run_id]
    expected_ref_norm = sorted(
        [_normalize_expected_ref(r) for r in expected_refs],
        key=lambda x: (x["source_episode_id"], x["source_event_id"], x["memory_ref_id"]),
    )

    state_version_id = run.get("state_version_id")
    sv = next((s for s in bundle.get("state_versions", []) if s.get("state_version_id") == state_version_id), None)
    state_snapshot = ledger.artefacts.get(sv["snapshot_artefact_hash"]) if sv else {}

    frozen_input = _frozen_input_for_run(run_id)
    if frozen_input is None:
        return {
            "run_id": run_id,
            "match": False,
            "reason": "missing_frozen_deliberation_input",
            "expected_ref_count": len(expected_refs),
            "actual_ref_count": 0,
            "expected_delta_hashes": expected_delta_hashes,
            "actual_delta_hashes": [],
        }
    frozen_hash = artefact_hash(frozen_input)
    if expected_frozen_hash and frozen_hash != expected_frozen_hash:
        return {
            "run_id": run_id,
            "match": False,
            "reason": "frozen_input_hash_mismatch",
            "expected_frozen_hash": expected_frozen_hash,
            "actual_frozen_hash": frozen_hash,
            "expected_ref_count": len(expected_refs),
            "actual_ref_count": 0,
            "expected_delta_hashes": expected_delta_hashes,
            "actual_delta_hashes": [],
        }

    expected_retrieval_enabled = bool(run.get("retrieval_enabled"))
    actual_refs: List[MemoryReference] = []
    if expected_retrieval_enabled:
        policy = state_snapshot.get("same_asset_policy", "exclude")
        try:
            same_asset_policy = SameAssetPolicy(policy)
        except ValueError:
            same_asset_policy = SameAssetPolicy.EXCLUDE

        exclude_episode_ids: tuple[str, ...] = ()
        exclude_run_ids: tuple[str, ...] = ()
        baseline_run_id = run.get("baseline_run_id")
        if baseline_run_id:
            baseline_run = next((r for r in bundle.get("runs", []) if r.get("run_id") == baseline_run_id), None)
            if baseline_run:
                exclude_episode_ids = exclude_episode_ids + (baseline_run.get("episode_id"),)
                exclude_run_ids = exclude_run_ids + (baseline_run_id,)
        exclude_episode_ids = exclude_episode_ids + (episode_id,)

        q = RetrievalQuery(
            workspace_id=episode["workspace_id"],
            actor_id=episode["actor_id"],
            asset_id=episode["asset_id"],
            goal_type=episode["goal_type"],
            goal_instance_id=episode.get("goal_instance_id"),
            scene_signature=str(frozen_input.get("scene_signature") or ""),
            category_signature=str(frozen_input.get("category_signature") or ""),
            exclude_episode_ids=exclude_episode_ids,
            exclude_run_ids=exclude_run_ids,
            comparison_group_id=run.get("comparison_group_id"),
            same_asset_policy=same_asset_policy,
            as_of_visibility=retrieval_as_of,
        )
        replay_result = retrieve_memories(q, ledger=ledger, state_snapshot=state_snapshot)
        actual_refs = replay_result.references

    actual_ref_norm = sorted(
        [_normalize_actual_ref(r) for r in actual_refs],
        key=lambda x: (x["source_episode_id"], x["source_event_id"], x["memory_ref_id"]),
    )
    match_refs = actual_ref_norm == expected_ref_norm
    actual_memory_ref_ids = [r.memory_ref_id for r in actual_refs]

    baseline_run_id = run.get("baseline_run_id")
    baseline_snap_payload = _baseline_payload_for_run_id(baseline_run_id)
    baseline_ds = None
    if baseline_snap_payload is not None:
        baseline_ds = DeliberationSnapshot(
            primary_hypothesis=str(baseline_snap_payload.get("primary_hypothesis", "")),
            confidence=float(baseline_snap_payload.get("confidence", 0.5)),
            strategy=str(baseline_snap_payload.get("strategy", "standard")),
            requested_evidence=list(baseline_snap_payload.get("requested_evidence") or []),
            perception_artefact_hash=str(frozen_input.get("perception_artefact_hash") or ""),
            scene_signature=str(frozen_input.get("scene_signature") or ""),
            category_signature=str(frozen_input.get("category_signature") or ""),
            run_id=str(baseline_run_id or ""),
        )

    # Use recorded memory_reference_ids from frozen input for snapshot regeneration when refs match;
    # otherwise use replayed IDs so mismatches surface in snapshot/delta hashes.
    memory_ids_for_gov = list(frozen_input.get("memory_reference_ids") or [])
    if match_refs:
        memory_ids_for_gov = actual_memory_ref_ids or memory_ids_for_gov

    governed = build_governed_deliberation_snapshot(
        frozen_input,
        baseline_ds,
        memory_ids_for_gov,
    )
    regenerated_snapshot_hash = artefact_hash(governed.snapshot_dict)
    regenerated_confidence_hash = artefact_hash(governed.confidence_provenance)

    actual_delta_hashes: List[str] = []
    if baseline_ds is not None:
        delta_objs = compare_deliberation_snapshots(
            baseline_ds, governed.snapshot, memory_ids_for_gov
        )
        deltas = [d.__dict__ for d in delta_objs if d.field_changed != "_compatibility"]
        if deltas:
            delta_payload = {
                "deltas": deltas,
                "baseline_run_id": baseline_ds.run_id or baseline_run_id,
            }
            actual_delta_hashes = [artefact_hash(delta_payload)]

    match_snapshot = regenerated_snapshot_hash == expected_snapshot_hash
    match_deltas = sorted(actual_delta_hashes) == expected_delta_hashes
    match_confidence = (
        expected_confidence_hash is None or regenerated_confidence_hash == expected_confidence_hash
    )
    match_context = (
        expected_context_fp is None
        or expected_context_fp == frozen_input.get("context_fingerprint")
    )
    match_state = expected_state_version is None or expected_state_version == run.get("state_version_id")
    match_perception = (
        expected_perception is None
        or expected_perception == frozen_input.get("perception_artefact_hash")
    )

    return {
        "run_id": run_id,
        "match": bool(
            match_refs
            and match_snapshot
            and match_deltas
            and match_confidence
            and match_context
            and match_state
            and match_perception
        ),
        "expected_ref_count": len(expected_refs),
        "actual_ref_count": len(actual_refs),
        "expected_delta_hashes": expected_delta_hashes,
        "actual_delta_hashes": actual_delta_hashes,
        "expected_snapshot_hash": expected_snapshot_hash,
        "actual_snapshot_hash": regenerated_snapshot_hash,
        "expected_confidence_provenance_hash": expected_confidence_hash,
        "actual_confidence_provenance_hash": regenerated_confidence_hash,
        "retrieval_as_of": retrieval_as_of,
    }


def _apply_mutation(bundle: Dict[str, Any], kind: str) -> Dict[str, Any]:
    import copy

    mutated = copy.deepcopy(bundle)
    runs = mutated.get("runs") or []
    run_ids = [r.get("run_id") for r in runs if r.get("run_id")]
    # Prefer a memory-enabled run as mutation target when present.
    target_run_id = None
    for r in runs:
        if r.get("run_purpose") in ("memory_enabled", "live", "demo_seed") and r.get("retrieval_enabled"):
            target_run_id = r.get("run_id")
            break
    if target_run_id is None and run_ids:
        target_run_id = run_ids[0]

    if kind in ("deliberation_hash", "expected_snapshot_hash"):
        if mutated.get("expected_hashes") and target_run_id:
            mutated["expected_hashes"]["runs"][target_run_id]["deliberation_snapshot_hash"] = "0" * 64
        else:
            for ev in mutated.get("events", []):
                if ev.get("event_type") == "deliberation_snapshot":
                    ev["artefact_hash"] = "0" * 64
                    break
    elif kind == "memory_ref":
        refs = mutated.get("memory_references") or []
        if refs:
            refs[0]["source_episode_id"] = "mutated-episode-id"
    elif kind == "e1_hypothesis":
        run_purpose_map = {r["run_id"]: r.get("run_purpose") for r in runs if r.get("run_id")}
        for ev in mutated.get("events", []):
            if ev.get("event_type") == "deliberation_snapshot" and run_purpose_map.get(ev.get("run_id")) == "live":
                payload = json.loads(ev["payload_json"])
                payload["primary_hypothesis"] = "MUTATED_E1_HYPOTHESIS"
                ev["payload_json"] = json.dumps(payload, sort_keys=True)
                break
    elif kind == "frozen_hypothesis":
        for item in mutated.get("artefact_payloads", []):
            payload = item.get("payload") or {}
            if payload.get("schema") == "frozen_deliberation_input_v1":
                payload["primary_hypothesis"] = "MUTATED_FROZEN_HYPOTHESIS"
                if isinstance(payload.get("intelligence_output"), dict):
                    rec = payload["intelligence_output"].setdefault("recognition", {})
                    rec["what_i_see"] = "MUTATED_FROZEN_HYPOTHESIS"
                item["payload"] = payload
                break
    elif kind == "raw_confidence":
        for item in mutated.get("artefact_payloads", []):
            payload = item.get("payload") or {}
            if payload.get("schema") == "frozen_deliberation_input_v1":
                payload["raw_confidence"] = 0.99
                if isinstance(payload.get("intelligence_output"), dict):
                    rec = payload["intelligence_output"].setdefault("recognition", {})
                    rec["confidence"] = 0.99
                item["payload"] = payload
                break
    elif kind == "context_fingerprint":
        if mutated.get("expected_hashes") and target_run_id:
            mutated["expected_hashes"]["runs"][target_run_id]["context_fingerprint"] = "0" * 64
    elif kind == "policy_version":
        for item in mutated.get("artefact_payloads", []):
            payload = item.get("payload") or {}
            if payload.get("schema") == "frozen_deliberation_input_v1":
                payload["governance_policy_version"] = "mutated_policy_v0"
                payload["prompt_policy_version"] = "mutated_policy_v0"
                item["payload"] = payload
                break
    elif kind == "state_cutoff":
        for item in mutated.get("artefact_payloads", []):
            payload = item.get("payload") or {}
            if payload.get("schema") == "state_snapshot_v1":
                payload["cutoff_score"] = 0.99
                item["payload"] = payload
                break
    elif kind == "removed_source_event":
        refs = mutated.get("memory_references") or []
        if refs:
            source_event_id = refs[0].get("source_event_id")
            mutated["events"] = [e for e in mutated.get("events", []) if e.get("event_id") != source_event_id]
    elif kind == "perception_snapshot":
        for item in mutated.get("artefact_payloads", []):
            if isinstance(item, dict) and (item.get("payload") or {}).get("schema") == "perception_snapshot_v1":
                payload = item["payload"]
                payload["scene_type"] = "MUTATED_SCENE_TYPE"
                item["payload"] = payload
                break
    elif kind == "baseline_confidence":
        run_purpose_map = {r["run_id"]: r.get("run_purpose") for r in runs if r.get("run_id")}
        for ev in mutated.get("events", []):
            if ev.get("event_type") == "deliberation_snapshot" and run_purpose_map.get(ev.get("run_id")) == "baseline":
                payload = json.loads(ev["payload_json"])
                payload["confidence"] = 0.01
                ev["payload_json"] = json.dumps(payload, sort_keys=True)
                break
    elif kind == "changed_memory_reference":
        refs = mutated.get("memory_references") or []
        if refs:
            refs[0]["artefact_hash"] = "0" * 64
    elif kind == "expected_delta_hash":
        if mutated.get("expected_hashes") and target_run_id:
            mutated["expected_hashes"]["runs"][target_run_id]["deliberation_delta_hashes"] = ["0" * 64]
    else:
        raise ValueError(f"Unknown mutation kind: {kind}")
    return mutated
