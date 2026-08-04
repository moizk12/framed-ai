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
from framed.cognition.contracts.memory import RetrievalQuery
from framed.cognition.contracts.runs import SameAssetPolicy
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
        for ev in bundle.get("events", []):
            if ev.get("run_id") != run_id:
                continue
            if ev.get("event_type") != "deliberation_snapshot":
                continue
            ev_hash = ev.get("artefact_hash")
            if ev_hash and run_expected.get("deliberation_snapshot_hash") == ev_hash:
                payload = json.loads(ev["payload_json"])
                if artefact_hash(payload) != ev_hash:
                    raise ValueError(f"Deliberation snapshot payload hash mismatch for run {run_id}")


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
        if mutate == "deliberation_hash":
            try:
                validate_bundle_integrity(bundle, artefacts=ledger.artefacts)
                return {
                    "status": "FAIL",
                    "bundle_path": str(bundle_path),
                    "cognition_dir": str(resolved_dir),
                    "replay_checks": [{"match": False, "reason": "mutation_not_rejected"}],
                    "mutated": True,
                }
            except ValueError:
                return {
                    "status": "FAIL",
                    "bundle_path": str(bundle_path),
                    "cognition_dir": str(resolved_dir),
                    "replay_checks": [{"match": True, "reason": "mutation_rejected_at_integrity"}],
                    "mutated": True,
                }

        validate_bundle_integrity(bundle, artefacts=ledger.artefacts)

        replay_checks: List[Dict[str, Any]] = []
        for run in bundle.get("runs", []):
            purpose = run.get("run_purpose")
            if purpose not in ("memory_enabled", "live", "demo_seed"):
                continue
            if not run.get("retrieval_enabled"):
                continue
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
    run_id = run["run_id"]
    episode_id = run["episode_id"]
    episode = next(e for e in bundle["episodes"] if e["episode_id"] == episode_id)

    scene_sig = ""
    cat_sig = ""
    for ev in bundle.get("events", []):
        if ev.get("run_id") == run_id and ev.get("event_type") == "deliberation_snapshot":
            payload = json.loads(ev["payload_json"])
            scene_sig = payload.get("scene_signature") or ""
            cat_sig = payload.get("category_signature") or ""
            break
    if not scene_sig:
        for row in bundle.get("retrieval_index") or []:
            if row["episode_id"] == episode_id:
                scene_sig = row.get("scene_signature") or ""
                cat_sig = row.get("category_signature") or ""

    state = ledger.get_active_state(episode["workspace_id"])
    policy = state["snapshot"].get("same_asset_policy", "exclude")
    try:
        same_asset_policy = SameAssetPolicy(policy)
    except ValueError:
        same_asset_policy = SameAssetPolicy.EXCLUDE

    recorded_refs = [
        r for r in bundle.get("memory_references", []) if r.get("run_id") == run_id
    ]
    recorded_source_eps = sorted({r["source_episode_id"] for r in recorded_refs})

    q = RetrievalQuery(
        workspace_id=episode["workspace_id"],
        actor_id=episode["actor_id"],
        asset_id=episode["asset_id"],
        goal_type=episode["goal_type"],
        goal_instance_id=episode.get("goal_instance_id"),
        scene_signature=scene_sig or "interior_scene",
        category_signature=cat_sig or "cluttered_room_weak_composition",
        exclude_episode_ids=(episode_id,),
        exclude_run_ids=(run_id,),
        comparison_group_id=run.get("comparison_group_id"),
        same_asset_policy=same_asset_policy,
    )
    if run.get("baseline_run_id"):
        baseline = next((r for r in bundle["runs"] if r["run_id"] == run["baseline_run_id"]), None)
        if baseline:
            q = RetrievalQuery(
                workspace_id=q.workspace_id,
                actor_id=q.actor_id,
                asset_id=q.asset_id,
                goal_type=q.goal_type,
                goal_instance_id=q.goal_instance_id,
                scene_signature=q.scene_signature,
                category_signature=q.category_signature,
                exclude_episode_ids=q.exclude_episode_ids + (baseline["episode_id"],),
                exclude_run_ids=q.exclude_run_ids + (run["baseline_run_id"],),
                comparison_group_id=q.comparison_group_id,
                same_asset_policy=q.same_asset_policy,
            )

    replay_result = retrieve_memories(q, ledger=ledger, state_snapshot=state["snapshot"])
    replay_source_eps = sorted({r.source_episode_id for r in replay_result.references})
    match = replay_source_eps == recorded_source_eps
    return {
        "run_id": run_id,
        "recorded_source_episodes": recorded_source_eps,
        "replay_source_episodes": replay_source_eps,
        "match": match,
    }


def _apply_mutation(bundle: Dict[str, Any], kind: str) -> Dict[str, Any]:
    import copy

    mutated = copy.deepcopy(bundle)
    if kind == "deliberation_hash":
        for ev in mutated.get("events", []):
            if ev.get("event_type") == "deliberation_snapshot":
                ev["artefact_hash"] = "0" * 64
                break
    elif kind == "memory_ref":
        refs = mutated.get("memory_references") or []
        if refs:
            refs[0]["source_episode_id"] = "mutated-episode-id"
    else:
        raise ValueError(f"Unknown mutation kind: {kind}")
    return mutated
