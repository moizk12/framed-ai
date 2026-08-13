"""Slice A E1→E2 A/B/C/D demonstration with replay and rollback proof."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from framed.cognition.context.builder import compute_deliberation_delta
from framed.cognition.contracts.memory import RetrievalQuery
from framed.cognition.contracts.runs import RunMode, RunPurpose
from framed.cognition.constants import PERCEPTION_SNAPSHOT_SCHEMA
from framed.cognition.integration.pipeline_hook import (
    asset_id_from_path,
    begin_cognition_run,
    build_perception_snapshot_v1,
    finalize_cognition_run,
)
from framed.cognition.ledger.artefact_store import artefact_hash, canonical_json_dumps
from framed.cognition.ledger.sqlite_store import clear_ledger, get_ledger, release_ledger_store, reset_ledger
from framed.cognition.replay.engine import execute_replay, validate_bundle_integrity, validate_bundle_schema
from framed.cognition.retrieval.service import retrieve_memories


def _synthetic_result(scene_type: str = "interior_scene") -> Dict[str, Any]:
    return {
        "visual_evidence": {
            "scene_gate": {
                "scene_type": scene_type,
                "signals": {"places_scene_category": scene_type},
            }
        },
        "semantic_anchors": {"scene_type": scene_type},
        "perception": {"technical": {"available": True}},
    }


def _synthetic_intelligence(hypothesis: str, confidence: float = 0.55) -> Dict[str, Any]:
    return {
        "recognition": {"what_i_see": hypothesis, "confidence": confidence},
        "meta_cognition": {"confidence": confidence},
    }


def _write_temp_image(content: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    Path(path).write_bytes(content)
    return path


def _run_episode(
    *,
    image_path: str,
    result: Dict[str, Any],
    run_mode: RunMode,
    run_purpose: RunPurpose,
    state_label: Optional[str],
    intelligence: Dict[str, Any],
    baseline_run_id: Optional[str] = None,
    baseline_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session = begin_cognition_run(
        result=result,
        image_path=image_path,
        asset_filename=os.path.basename(image_path),
        run_mode=run_mode,
        run_purpose=run_purpose,
        state_label=state_label,
        baseline_run_id=baseline_run_id,
        comparison_group_id="slice_a_demo",
    )
    if session is None:
        raise RuntimeError("Cognition session failed — is FRAMED_COGNITION_V1 set?")
    out = finalize_cognition_run(session, result, intelligence, baseline_snapshot=baseline_snapshot)
    return {
        "session": session,
        "result": out,
        "deliberation": {
            "primary_hypothesis": intelligence["recognition"]["what_i_see"],
            "confidence": intelligence["recognition"]["confidence"],
            "strategy": session.deliberation_context.strategy_hint or "standard",
            "requested_evidence": session.deliberation_context.requested_evidence,
        },
    }


def validate_bundle_payloads(bundle: Dict[str, Any]) -> bool:
    """Validate bundle has deliberation snapshot payload hashes."""
    runs = bundle.get("runs") or []
    if not runs:
        return False
    for run in runs:
        events = [e for e in bundle.get("events", []) if e.get("run_id") == run["run_id"]]
        payload_hashes = [
            artefact_hash(json.loads(e["payload_json"]))
            for e in events
            if e.get("payload_json") and e.get("event_type") == "deliberation_snapshot"
        ]
        if run.get("run_purpose") in ("memory_enabled", "live", "demo_seed") and not payload_hashes:
            return False
    return True


def _dir_is_non_empty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _release_demo_store(cognition_dir: Path) -> None:
    release_ledger_store(cognition_dir / "cognition_ledger.sqlite3")


def _remove_tree_with_retries(path: Path) -> None:
    for _ in range(20):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.1)
    shutil.rmtree(path)


@contextmanager
def _demo_store(
    *,
    cognition_dir: Optional[Path],
    keep_store: bool,
    reset_store: bool,
    reuse_store: bool,
) -> Iterator[tuple[Path, Optional[Path]]]:
    retained_store_path: Optional[Path] = None
    if cognition_dir is not None:
        resolved = cognition_dir.resolve()
        if _dir_is_non_empty(resolved) and not (reset_store or reuse_store):
            raise ValueError(
                "Explicit cognition dir is non-empty. Use --reset-store to clear it or --reuse-store to allow reuse."
            )
        if reset_store and resolved.exists():
            _release_demo_store(resolved)
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True, exist_ok=True)
        try:
            yield resolved, retained_store_path
        finally:
            _release_demo_store(resolved)
        return

    temp_ctx = tempfile.TemporaryDirectory(prefix="framed_slice_a_demo_store_", ignore_cleanup_errors=True)
    resolved = Path(temp_ctx.name)
    try:
        if keep_store:
            retained_store_path = Path(tempfile.mkdtemp(prefix="framed_slice_a_demo_store_kept_"))
        try:
            yield resolved, retained_store_path
            if retained_store_path is not None:
                shutil.copytree(resolved, retained_store_path, dirs_exist_ok=True)
        finally:
            _release_demo_store(resolved)
            _remove_tree_with_retries(resolved)
    finally:
        temp_ctx.cleanup()


def _run_slice_a_demo_once(*, cognition_dir: Path, evidence_dir: Path) -> Dict[str, Any]:
    os.environ["FRAMED_COGNITION_DIR"] = str(cognition_dir)
    reset_ledger(cognition_dir / "cognition_ledger.sqlite3")

    e1_bytes = b"slice_a_e1_cluttered_room"
    e2_bytes = b"slice_a_e2_related_interior_same_category"
    control_bytes = b"slice_a_control_unrelated_street"

    e1_path = _write_temp_image(e1_bytes)
    e2_path = _write_temp_image(e2_bytes)
    control_path = _write_temp_image(control_bytes)
    try:
        shared_result = _synthetic_result("interior_scene")
        perception_payload = build_perception_snapshot_v1(shared_result)
        perception_hash = artefact_hash(perception_payload)
        assert perception_payload["schema"] == PERCEPTION_SNAPSHOT_SCHEMA

        e1_intel = _synthetic_intelligence(
            "Cluttered interior with weak composition — prior failure mode noted.",
            0.52,
        )
        e1_out = _run_episode(
            image_path=e1_path,
            result=shared_result,
            run_mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.LIVE,
            state_label="state_memory_enabled",
            intelligence=e1_intel,
        )
        e1_episode_id = e1_out["session"].episode_id
        e1_run_id = e1_out["session"].run_id

        e2a_intel = _synthetic_intelligence("Interior scene with visible clutter.", 0.58)
        e2a_out = _run_episode(
            image_path=e2_path,
            result=shared_result,
            run_mode=RunMode.BASELINE,
            run_purpose=RunPurpose.BASELINE,
            state_label="state_baseline",
            intelligence=e2a_intel,
        )
        baseline_snapshot = e2a_out["deliberation"]
        e2a_episode_id = e2a_out["session"].episode_id
        assert not e2a_out["session"].memory_reference_ids, "E2-A must not retrieve E1"

        e2b_intel = _synthetic_intelligence(
            "Interior clutter — reconsider composition failure from prior experience.",
            0.50,
        )
        e2b_out = _run_episode(
            image_path=e2_path,
            result=shared_result,
            run_mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.MEMORY_ENABLED,
            state_label="state_memory_enabled",
            intelligence=e2b_intel,
            baseline_run_id=e2a_out["session"].run_id,
            baseline_snapshot=baseline_snapshot,
        )
        assert len(e2b_out["session"].memory_reference_ids) > 0, "E2-B should retrieve memory refs"
        ledger = get_ledger()
        refs = ledger.export_replay_bundle([e2b_out["session"].run_id])["memory_references"]
        source_eps = {r["source_episode_id"] for r in refs}
        assert e1_episode_id in source_eps, "E2-B must retrieve E1 episode"
        assert e2a_episode_id not in source_eps, "E2-B must not retrieve E2 baseline episode"
        selected_ref_ids = [r["memory_ref_id"] for r in refs]

        rejected = e2b_out["session"].rejected_candidates
        baseline_rejects = [
            r for r in rejected if r.get("episode_id") == e2a_episode_id or r.get("source_run_id") == e2a_out["session"].run_id
        ]
        assert baseline_rejects, "E2 baseline must appear in rejected candidates"
        assert any(r.get("same_asset") for r in baseline_rejects), "E2 baseline rejected for same_asset"
        assert any(r.get("ineligible_run_purpose") == "baseline" for r in baseline_rejects), "E2 baseline rejected for purpose"

        deltas = compute_deliberation_delta(
            baseline_snapshot,
            e2b_out["deliberation"],
            e2b_out["session"].memory_reference_ids,
        )
        assert len(deltas) >= 1, "Meaningful deliberation delta required"

        control_result = _synthetic_result("people_scene")
        e2c_intel = _synthetic_intelligence("Layered street composition with pedestrian flow.", 0.60)
        e2c_out = _run_episode(
            image_path=control_path,
            result=control_result,
            run_mode=RunMode.CONTROL,
            run_purpose=RunPurpose.CONTROL,
            state_label="state_memory_enabled",
            intelligence=e2c_intel,
        )
        c_refs = get_ledger().export_replay_bundle([e2c_out["session"].run_id])["memory_references"]
        c_sources = {r["source_episode_id"] for r in c_refs}
        assert e1_episode_id not in c_sources, "Control must not retrieve E1"

        run_ids = [e1_run_id, e2a_out["session"].run_id, e2b_out["session"].run_id, e2c_out["session"].run_id]
        bundle = ledger.export_replay_bundle(run_ids)
        bundle["perception_hash"] = perception_hash
        bundle["code_commit"] = os.getenv("FRAMED_CODE_COMMIT", "local")
        bundle["retrieval_policy"] = {"version": "v1", "cutoff": 0.7}
        bundle_hash = artefact_hash({k: bundle[k] for k in ("schema", "runs", "episodes", "events") if k in bundle})
        validate_bundle_schema(bundle)
        validate_bundle_integrity(bundle, artefacts=ledger.artefacts)
        assert validate_bundle_payloads(bundle), "Bundle payload validation failed"

        archive_path = evidence_dir / "slice_a_replay_bundle.json"
        archive_path.write_text(canonical_json_dumps(bundle), encoding="utf-8")
        replay_report = execute_replay(archive_path)
        assert replay_report.get("status") == "PASS", f"Replay execution failed: {replay_report}"

        ledger.activate_state(e2a_out["session"].workspace_id, "state_baseline")
        ident = e2b_out["session"]
        q = RetrievalQuery(
            workspace_id=ident.workspace_id,
            actor_id=ident.actor_id,
            asset_id=asset_id_from_path(e2_path),
            goal_type="critique",
            goal_instance_id=None,
            scene_signature="interior_scene",
            category_signature="cluttered_room_weak_composition",
        )
        state = ledger.get_active_state(ident.workspace_id)
        rollback_result = retrieve_memories(q, ledger=ledger, state_snapshot=state["snapshot"])
        assert not rollback_result.references, "Rollback must make E1 unavailable"

        report = {
            "status": "PASS",
            "cognition_dir": str(cognition_dir),
            "evidence_dir": str(evidence_dir),
            "e1_episode_id": e1_episode_id,
            "e1_run_id": e1_run_id,
            "e2a_episode_id": e2a_episode_id,
            "e2a_run_id": e2a_out["session"].run_id,
            "e2b_episode_id": e2b_out["session"].episode_id,
            "e2b_run_id": e2b_out["session"].run_id,
            "e2c_episode_id": e2c_out["session"].episode_id,
            "e2c_run_id": e2c_out["session"].run_id,
            "delta_count": len(deltas),
            "deltas": [d.__dict__ for d in deltas],
            "bundle_hash": bundle_hash,
            "rollback_retrieval_count": len(rollback_result.references),
            "perception_hash_shared": perception_hash,
            "selected_memory_reference_ids": selected_ref_ids,
            "selected_source_episode_ids": sorted(source_eps),
            "baseline_rejections": baseline_rejects,
            "replay_report": replay_report,
        }
        report_path = evidence_dir / "slice_a_demo_report.json"
        report_path.write_text(canonical_json_dumps(report), encoding="utf-8")
        return report
    finally:
        for p in (e1_path, e2_path, control_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def run_slice_a_demo(
    cognition_dir: Optional[Path] = None,
    *,
    keep_store: bool = False,
    reset_store: bool = False,
    reuse_store: bool = False,
    evidence_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute E1 → E2-A/B/C/D → rollback proof."""
    if cognition_dir is None and data_dir is not None:
        cognition_dir = data_dir

    original_cognition_flag = os.environ.get("FRAMED_COGNITION_V1")
    original_cognition_dir = os.environ.get("FRAMED_COGNITION_DIR")
    os.environ["FRAMED_COGNITION_V1"] = "true"

    resolved_evidence_dir = evidence_dir.resolve() if evidence_dir else Path(
        tempfile.mkdtemp(prefix="framed_slice_a_demo_evidence_")
    )
    resolved_evidence_dir.mkdir(parents=True, exist_ok=True)

    try:
        with _demo_store(
            cognition_dir=cognition_dir,
            keep_store=keep_store,
            reset_store=reset_store,
            reuse_store=reuse_store,
        ) as (resolved_cognition_dir, kept_store_path):
            report = _run_slice_a_demo_once(
                cognition_dir=resolved_cognition_dir,
                evidence_dir=resolved_evidence_dir,
            )
            report["temporary_store"] = cognition_dir is None
            report["kept_store_path"] = str(kept_store_path) if kept_store_path else None
            report_path = resolved_evidence_dir / "slice_a_demo_report.json"
            report_path.write_text(canonical_json_dumps(report), encoding="utf-8")
            return report
    finally:
        clear_ledger()
        if original_cognition_flag is None:
            os.environ.pop("FRAMED_COGNITION_V1", None)
        else:
            os.environ["FRAMED_COGNITION_V1"] = original_cognition_flag
        if original_cognition_dir is None:
            os.environ.pop("FRAMED_COGNITION_DIR", None)
        else:
            os.environ["FRAMED_COGNITION_DIR"] = original_cognition_dir


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Slice A deterministic cognition demo.")
    parser.add_argument("--keep-store", action="store_true", help="Retain the temporary cognition store for debugging.")
    parser.add_argument("--cognition-dir", type=Path, help="Explicit cognition store directory to use.")
    parser.add_argument("--evidence-dir", type=Path, help="Directory for replay bundle and report outputs.")
    parser.add_argument("--reset-store", action="store_true", help="Delete and recreate the explicit cognition directory.")
    parser.add_argument("--reuse-store", action="store_true", help="Allow reuse of a non-empty explicit cognition directory.")
    parser.add_argument("--replay", type=Path, help="Replay an exported bundle in an isolated cognition store.")
    parser.add_argument(
        "--replay-mutate",
        choices=(
            "deliberation_hash",
            "expected_snapshot_hash",
            "memory_ref",
            "e1_hypothesis",
            "frozen_hypothesis",
            "raw_confidence",
            "baseline_confidence",
            "changed_memory_reference",
            "state_cutoff",
            "removed_source_event",
            "perception_snapshot",
            "context_fingerprint",
            "expected_delta_hash",
            "policy_version",
        ),
        help="Apply a mutation to the bundle before replay (expects replay failure).",
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        args = _parse_args()
        if args.replay is not None:
            report = execute_replay(args.replay, mutate=args.replay_mutate)
            print(json.dumps(report, indent=2))
            if args.replay_mutate:
                return 0 if report.get("status") == "FAIL" else 1
            return 0 if report.get("status") == "PASS" else 1
        report = run_slice_a_demo(
            cognition_dir=args.cognition_dir,
            keep_store=args.keep_store,
            reset_store=args.reset_store,
            reuse_store=args.reuse_store,
            evidence_dir=args.evidence_dir,
        )
        print(json.dumps(report, indent=2))
        return 0 if report.get("status") == "PASS" else 1
    except AssertionError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
