"""Slice A integrity hardening tests."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path

import pytest

from framed.cognition.constants import MAX_TOTAL_COGNITION_BLOCK_CHARS, PERCEPTION_SNAPSHOT_SCHEMA
from framed.cognition.context.formatting import build_cognition_context, format_cognition_context_for_prompt
from framed.cognition.contracts.memory import MemoryReference, RetrievalQuery, ScoreComponents
from framed.cognition.contracts.runs import RunMode, RunPurpose, validate_mode_purpose
from framed.cognition.contracts.snapshot import DeliberationSnapshot
from framed.cognition.integration.pipeline_hook import (
    build_perception_snapshot_v1,
    fail_cognition_run,
    perception_artefact_from_result,
)
from framed.cognition.ledger.artefact_store import artefact_hash
from framed.cognition.ledger.sqlite_store import CognitionLedger, reset_ledger
from framed.cognition.replay.engine import execute_replay, validate_bundle_integrity, validate_bundle_schema


@pytest.fixture
def integrity_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(tmp_path))
    ledger = reset_ledger(tmp_path / "integrity.sqlite3")
    yield ledger, tmp_path
    reset_ledger()


def _ref(**overrides) -> MemoryReference:
    base = dict(
        memory_ref_id=str(uuid.uuid4()),
        source_episode_id=str(uuid.uuid4()),
        source_run_id=str(uuid.uuid4()),
        source_event_id=str(uuid.uuid4()),
        source_asset_id="asset-a",
        source_run_purpose="live",
        epistemic_status="provisional",
        lifecycle_status="closed",
        memory_role="prior_experience",
        trust_level="low",
        artefact_hash="abc123",
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
        hypothesis_summary="Prior hypothesis",
        confidence_at_source=0.5,
        scores=ScoreComponents(1, 1, 1, 0, 0.1, 0.9),
        match_reason="test",
    )
    base.update(overrides)
    return MemoryReference(**base)


def test_artefact_root_isolated_from_db(integrity_env):
    _, tmp_path = integrity_env
    db_path = tmp_path / "nested" / "ledger.sqlite3"
    ledger = reset_ledger(db_path)
    assert ledger.artefacts.root == db_path.parent / "artefacts"
    digest = ledger.put_artefact("test", "v1", {"x": 1})
    assert (ledger.artefacts.root / digest[:2] / f"{digest}.json").exists()


def test_migration_003_applies(integrity_env):
    ledger, tmp_path = integrity_env
    with ledger._connect() as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 3
        cols = {row[1] for row in conn.execute("PRAGMA table_info(cognitive_runs)").fetchall()}
        assert "provenance_manifest_json" in cols
        assert "failure_code" in cols


def test_append_event_concurrency(integrity_env):
    ledger, _ = integrity_env
    ws = str(uuid.uuid4())
    baseline_id, _ = ledger.ensure_initial_states(ws)
    episode_id = ledger.open_episode(
        workspace_id=ws,
        actor_id="actor",
        asset_id="asset",
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run_id = str(uuid.uuid4())
    from framed.cognition.contracts.runs import CognitiveRun
    from framed.cognition.integration.pipeline_hook import CognitionSession

    ledger.create_run(
        CognitiveRun(
            run_id=run_id,
            episode_id=episode_id,
            mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.LIVE,
            state_version_id=baseline_id,
            context_fingerprint=None,
            retrieval_enabled=True,
            model_provenance={},
            prompt_provenance={},
            started_at="2026-01-01T00:00:00Z",
        )
    )

    errors = []

    def _writer(n: int) -> None:
        try:
            ledger.append_event(
                episode_id=episode_id,
                run_id=run_id,
                event_type="concurrency_probe",
                payload={"n": n},
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    events = ledger.get_episode_events(episode_id)
    probe_events = [e for e in events if e["event_type"] == "concurrency_probe"]
    assert len(probe_events) == 100
    seqs = [e["sequence_num"] for e in probe_events]
    assert len(set(seqs)) == 100


def test_fail_cognition_run_marks_failed_without_index(integrity_env):
    ledger, _ = integrity_env
    ws = str(uuid.uuid4())
    baseline_id, _ = ledger.ensure_initial_states(ws)
    episode_id = ledger.open_episode(
        workspace_id=ws,
        actor_id="actor",
        asset_id="asset",
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run_id = str(uuid.uuid4())
    from framed.cognition.contracts.runs import CognitiveRun
    from framed.cognition.integration.pipeline_hook import CognitionSession

    ledger.create_run(
        CognitiveRun(
            run_id=run_id,
            episode_id=episode_id,
            mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.LIVE,
            state_version_id=baseline_id,
            context_fingerprint=None,
            retrieval_enabled=True,
            model_provenance={},
            prompt_provenance={},
            started_at="2026-01-01T00:00:00Z",
        )
    )
    session = CognitionSession(
        episode_id=episode_id,
        run_id=run_id,
        actor_id="actor",
        workspace_id=ws,
        asset_id="asset",
        state_version_id=baseline_id,
        run_mode=RunMode.MEMORY_ENABLED,
        run_purpose=RunPurpose.LIVE,
        retrieval_enabled=True,
    )
    fail_cognition_run(
        session,
        error_code="test_failure",
        safe_message="Injected failure",
        stage="test",
        internal_exception_type="RuntimeError",
    )
    with ledger._connect() as conn:
        ep = conn.execute("SELECT status, failure_code FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        assert ep["status"] == "failed"
        assert ep["failure_code"] == "test_failure"
        run = conn.execute("SELECT completed_at, failure_code, failure_stage FROM cognitive_runs WHERE run_id=?", (run_id,)).fetchone()
        assert run["completed_at"] is not None
        assert run["failure_code"] == "test_failure"
        assert run["failure_stage"] == "test"
        indexed = conn.execute(
            "SELECT COUNT(*) AS c FROM retrieval_index WHERE episode_id=?",
            (episode_id,),
        ).fetchone()["c"]
        assert indexed == 0
        failed_events = conn.execute(
            "SELECT COUNT(*) AS c FROM episode_events WHERE episode_id=? AND event_type='run_failed'",
            (episode_id,),
        ).fetchone()["c"]
        assert failed_events == 1


@pytest.mark.parametrize(
    "inject_at",
    [
        "before_event",
        "after_event",
        "after_episode_update",
        "before_run_update",
        "before_commit",
    ],
)
def test_fail_run_atomic_injection_rolls_back(integrity_env, inject_at):
    ledger, _ = integrity_env
    ws = str(uuid.uuid4())
    baseline_id, _ = ledger.ensure_initial_states(ws)
    episode_id = ledger.open_episode(
        workspace_id=ws,
        actor_id="actor",
        asset_id="asset",
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run_id = str(uuid.uuid4())
    from framed.cognition.contracts.runs import CognitiveRun

    ledger.create_run(
        CognitiveRun(
            run_id=run_id,
            episode_id=episode_id,
            mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.LIVE,
            state_version_id=baseline_id,
            context_fingerprint=None,
            retrieval_enabled=True,
            model_provenance={},
            prompt_provenance={},
            started_at="2026-01-01T00:00:00Z",
        )
    )
    with pytest.raises(RuntimeError, match="injected_fail"):
        ledger.fail_run_atomic(
            episode_id=episode_id,
            run_id=run_id,
            error_code="x",
            safe_message="y",
            stage="z",
            internal_exception_type="RuntimeError",
            run_purpose="live",
            _fail_inject_at=inject_at,
        )
    with ledger._connect() as conn:
        ep = conn.execute("SELECT status FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        assert ep["status"] == "open"
        run = conn.execute("SELECT completed_at FROM cognitive_runs WHERE run_id=?", (run_id,)).fetchone()
        assert run["completed_at"] is None
        events = conn.execute(
            "SELECT COUNT(*) AS c FROM episode_events WHERE episode_id=? AND event_type='run_failed'",
            (episode_id,),
        ).fetchone()["c"]
        assert events == 0


def test_confidence_clamp_in_finalize(integrity_env, monkeypatch):
    from framed.cognition.integration.pipeline_hook import CognitionSession, finalize_cognition_run

    ledger, _ = integrity_env
    ws = str(uuid.uuid4())
    baseline_id, _ = ledger.ensure_initial_states(ws)
    episode_id = ledger.open_episode(
        workspace_id=ws,
        actor_id="actor",
        asset_id="asset",
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run_id = str(uuid.uuid4())
    from framed.cognition.contracts.runs import CognitiveRun
    from framed.cognition.integration.pipeline_hook import CognitionSession

    ledger.create_run(
        CognitiveRun(
            run_id=run_id,
            episode_id=episode_id,
            mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.MEMORY_ENABLED,
            state_version_id=baseline_id,
            context_fingerprint="ctx",
            retrieval_enabled=True,
            model_provenance={},
            prompt_provenance={},
            started_at="2026-01-01T00:00:00Z",
        )
    )
    session = CognitionSession(
        episode_id=episode_id,
        run_id=run_id,
        actor_id="actor",
        workspace_id=ws,
        asset_id="asset",
        state_version_id=baseline_id,
        run_mode=RunMode.MEMORY_ENABLED,
        run_purpose=RunPurpose.MEMORY_ENABLED,
        retrieval_enabled=True,
        memory_reference_ids=["ref-1"],
        perception_artefact_hash="pe-hash",
        context_fingerprint="ctx",
        baseline_snapshot=DeliberationSnapshot(
            primary_hypothesis="baseline",
            confidence=0.55,
            strategy="standard",
            requested_evidence=[],
            perception_artefact_hash="pe-hash",
            scene_signature="interior_scene",
            category_signature="cluttered_room_weak_composition",
        ),
    )
    result = {"visual_evidence": {"scene_gate": {"scene_type": "interior_scene", "signals": {}}}}
    intelligence = {
        "recognition": {"what_i_see": "changed", "confidence": 0.9},
        "meta_cognition": {"confidence": 0.9},
    }
    out = finalize_cognition_run(session, result, intelligence)
    prov = out["cognition_provenance"]["confidence_provenance"]
    assert prov["clamp_applied"] is True
    assert prov["final_confidence"] == 0.55


def test_context_bounds_and_sanitization():
    long_text = "A" * 1000 + "\x07\n" + "B" * 1000
    refs = [_ref(hypothesis_summary=long_text, match_reason=long_text)]
    ctx = build_cognition_context(refs)
    rendered = format_cognition_context_for_prompt(ctx)
    assert len(rendered) <= MAX_TOTAL_COGNITION_BLOCK_CHARS + 3
    assert "\x07" not in rendered


def test_mode_purpose_validation():
    validate_mode_purpose(RunMode.BASELINE, RunPurpose.BASELINE)
    with pytest.raises(ValueError):
        validate_mode_purpose(RunMode.BASELINE, RunPurpose.LIVE)


def test_canonical_perception_snapshot():
    result = {
        "visual_evidence": {"scene_gate": {"scene_type": "interior_scene", "signals": {"x": 1}}},
        "semantic_anchors": {"scene_type": "interior_scene"},
    }
    payload = build_perception_snapshot_v1(result)
    digest, canonical = perception_artefact_from_result(result)
    assert payload == canonical
    assert payload["schema"] == PERCEPTION_SNAPSHOT_SCHEMA
    assert digest == artefact_hash(payload)


def test_replay_mutation_rejection(integrity_env, tmp_path):
    from framed.cognition.demo.slice_a_e1_e2 import run_slice_a_demo

    report = run_slice_a_demo(evidence_dir=tmp_path / "evidence")
    assert report["status"] == "PASS"
    bundle_path = Path(report["evidence_dir"]) / "slice_a_replay_bundle.json"
    replay_ok = execute_replay(bundle_path)
    assert replay_ok["status"] == "PASS"
    mem_check = next(
        c for c in replay_ok["replay_checks"] if c.get("expected_ref_count", 0) >= 1
    )
    assert mem_check["expected_snapshot_hash"] == mem_check["actual_snapshot_hash"]
    assert mem_check["expected_delta_hashes"] == mem_check["actual_delta_hashes"]
    for kind in (
        "deliberation_hash",
        "frozen_hypothesis",
        "raw_confidence",
        "baseline_confidence",
        "memory_ref",
        "state_cutoff",
        "removed_source_event",
        "perception_snapshot",
        "context_fingerprint",
        "expected_delta_hash",
        "policy_version",
    ):
        replay_bad = execute_replay(bundle_path, mutate=kind)
        assert replay_bad["status"] == "FAIL", f"mutation {kind} should fail"


def test_open_run_atomic_injection_leaves_no_records(integrity_env):
    from framed.cognition.contracts.runs import CognitiveRun
    from framed.cognition.ledger.artefact_store import ArtefactStore

    ledger, _ = integrity_env
    ws = str(uuid.uuid4())
    baseline_id, _ = ledger.ensure_initial_states(ws)
    episode_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    payload = {"schema": PERCEPTION_SNAPSHOT_SCHEMA, "scene_type": "interior_scene"}
    digest, rel, byte_len = ledger.artefacts.put("perception_snapshot", "v1", payload)
    run = CognitiveRun(
        run_id=run_id,
        episode_id=episode_id,
        mode=RunMode.BASELINE,
        run_purpose=RunPurpose.BASELINE,
        state_version_id=baseline_id,
        context_fingerprint="ctx",
        retrieval_enabled=False,
        model_provenance={},
        prompt_provenance={},
        started_at=ArtefactStore.utc_now(),
    )
    with pytest.raises(RuntimeError, match="injected_fail"):
        ledger.open_run_atomic(
            episode_id=episode_id,
            workspace_id=ws,
            actor_id="actor",
            asset_id="asset",
            goal_type="critique",
            goal_instance_id=None,
            state_version_id=baseline_id,
            asset_filename="x.jpg",
            source_kind="live",
            run=run,
            provenance_manifest={},
            perception_hash=digest,
            perception_rel=rel,
            perception_len=byte_len,
            experience_opened_payload={"goal_type": "critique"},
            _fail_inject_at="before_commit",
        )
    with ledger._connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM cognitive_runs WHERE run_id=?", (run_id,)).fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM episode_events WHERE episode_id=?", (episode_id,)).fetchone()["c"] == 0


def test_begin_phase_a_failure_leaves_no_episode(integrity_env, tmp_path, monkeypatch):
    from framed.cognition.integration.pipeline_hook import begin_cognition_run

    ledger, root = integrity_env
    ws = str(uuid.uuid4())
    ledger.ensure_initial_states(ws)
    ledger.activate_state(ws, "state_baseline")
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    result = {"visual_evidence": {"scene_gate": {"scene_type": "interior_scene", "signals": {}}}}
    with pytest.raises(RuntimeError, match="injected_fail:asset_hashing"):
        begin_cognition_run(
            result=result,
            image_path=str(img),
            asset_filename="img.jpg",
            run_mode=RunMode.BASELINE,
            run_purpose=RunPurpose.BASELINE,
            _fail_inject_at="asset_hashing",
        )
    with ledger._connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM episodes").fetchone()["c"] == 0


def test_begin_post_open_failure_marks_failed(integrity_env, tmp_path):
    from framed.cognition.integration.pipeline_hook import begin_cognition_run

    ledger, _ = integrity_env
    ws = "default"  # identity workspace may differ; ensure states on actual workspace
    from framed.cognition.identity import get_identity

    ws = get_identity()["workspace_id"]
    ledger.ensure_initial_states(ws)
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    result = {"visual_evidence": {"scene_gate": {"scene_type": "interior_scene", "signals": {}}}}
    with pytest.raises(RuntimeError, match="injected_fail:before_session_return"):
        begin_cognition_run(
            result=result,
            image_path=str(img),
            asset_filename="img.jpg",
            run_mode=RunMode.BASELINE,
            run_purpose=RunPurpose.BASELINE,
            _fail_inject_at="before_session_return",
        )
    with ledger._connect() as conn:
        rows = conn.execute("SELECT status FROM episodes").fetchall()
        assert rows
        assert all(r["status"] == "failed" for r in rows)
        assert conn.execute("SELECT COUNT(*) AS c FROM retrieval_index").fetchone()["c"] == 0


def test_initial_state_defaults_to_baseline(integrity_env):
    ledger, _ = integrity_env
    ws = str(uuid.uuid4())
    ledger.ensure_initial_states(ws)
    active = ledger.get_active_state(ws)
    assert active["label"] == "state_baseline"
    assert active["snapshot"]["retrieval_enabled"] is False
