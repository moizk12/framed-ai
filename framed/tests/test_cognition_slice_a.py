"""Slice A cognition loop tests."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest

from framed.cognition.config import cognition_enabled
from framed.cognition.context.builder import build_deliberation_context, compute_deliberation_delta
from framed.cognition.contracts.memory import MemoryReference, RetrievalQuery, ScoreComponents
from framed.cognition.contracts.runs import CognitiveRun, RunMode, RunPurpose
from framed.cognition.integration.pipeline_hook import legacy_writes_allowed
from framed.cognition.ledger.artefact_store import artefact_hash, canonical_json_dumps
from framed.cognition.ledger.sqlite_store import CognitionLedger, reset_ledger
from framed.cognition.retrieval.service import retrieve_memories


@pytest.fixture
def cognition_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(tmp_path))
    ledger = reset_ledger(tmp_path / "test.sqlite3")
    yield ledger
    reset_ledger()


def _memory_snapshot(ledger: CognitionLedger, workspace: str) -> dict:
    ledger.activate_state(workspace, "state_memory_enabled")
    return ledger.get_active_state(workspace)["snapshot"]


def _seed_closed_episode(ledger: CognitionLedger, workspace: str, actor: str, asset: str, scene: str, cat: str, hyp: str):
    baseline_id, _ = ledger.ensure_demo_states(workspace)
    eid = ledger.open_episode(
        workspace_id=workspace,
        actor_id=actor,
        asset_id=asset,
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run_id = str(uuid.uuid4())
    purpose = RunPurpose.LIVE
    ledger.create_run(
        CognitiveRun(
            run_id=run_id,
            episode_id=eid,
            mode=RunMode.MEMORY_ENABLED,
            run_purpose=purpose,
            state_version_id=baseline_id,
            context_fingerprint=None,
            retrieval_enabled=True,
            model_provenance={},
            prompt_provenance={},
            started_at="2026-01-01T00:00:00Z",
            retrieval_eligible=True,
        )
    )
    snap = {"primary_hypothesis": hyp, "confidence": 0.5}
    snap_hash = ledger.put_artefact("deliberation_snapshot", "v1", snap)
    ledger.append_event(
        episode_id=eid,
        run_id=run_id,
        event_type="deliberation_snapshot",
        payload=snap,
        artefact_hash=snap_hash,
    )
    ledger.close_episode(
        eid,
        run_id=run_id,
        run_purpose=purpose,
        scene_signature=scene,
        category_signature=cat,
        goal_type="critique",
        goal_instance_id=None,
        final_fingerprint=artefact_hash({"e": eid}),
        perception_artefact_hash=snap_hash,
    )
    ledger.complete_run(run_id)
    return eid


def test_legacy_gate_default_off(monkeypatch):
    monkeypatch.delenv("FRAMED_COGNITION_V1", raising=False)
    assert legacy_writes_allowed() is True
    assert cognition_enabled() is False


def test_legacy_gate_blocks_when_cognition_on(monkeypatch):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    assert legacy_writes_allowed() is False


def test_artefact_canonical_hash_stable():
    obj = {"b": 2, "a": 1}
    h1 = artefact_hash(obj)
    h2 = artefact_hash(json.loads(canonical_json_dumps(obj)))
    assert h1 == h2


def test_append_only_events_reject_update(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    baseline_id, _ = ledger.ensure_demo_states(ws)
    eid = ledger.open_episode(
        workspace_id=ws,
        actor_id=actor,
        asset_id="asset1",
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run_id = str(uuid.uuid4())
    ledger.create_run(
        CognitiveRun(
            run_id=run_id,
            episode_id=eid,
            mode=RunMode.BASELINE,
            run_purpose=RunPurpose.BASELINE,
            state_version_id=baseline_id,
            context_fingerprint=None,
            retrieval_enabled=False,
            model_provenance={},
            prompt_provenance={},
            started_at="2026-01-01T00:00:00Z",
        )
    )
    ledger.append_event(episode_id=eid, run_id=run_id, event_type="test", payload={"x": 1})
    with pytest.raises(sqlite3.IntegrityError):
        with ledger._connect() as conn:
            conn.execute("UPDATE episode_events SET payload_json='{}' WHERE episode_id=?", (eid,))


def test_retrieval_threshold_and_category_signal(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    asset = "aaa"
    _seed_closed_episode(ledger, ws, actor, asset, "interior_scene", "cluttered_room_weak_composition", "clutter hypothesis")
    q = RetrievalQuery(
        workspace_id=ws,
        actor_id=actor,
        asset_id="bbb",
        goal_type="critique",
        goal_instance_id=None,
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
    )
    state = _memory_snapshot(ledger, ws)
    result = retrieve_memories(q, ledger=ledger, state_snapshot=state)
    assert len(result.references) == 1
    assert result.references[0].epistemic_status == "provisional"
    assert result.references[0].trust_level == "low"
    assert result.references[0].scores.final_score >= 0.7


def test_same_asset_excluded(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    asset = "same_asset"
    _seed_closed_episode(ledger, ws, actor, asset, "interior_scene", "cluttered_room_weak_composition", "hyp")
    q = RetrievalQuery(
        workspace_id=ws,
        actor_id=actor,
        asset_id=asset,
        goal_type="critique",
        goal_instance_id=None,
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
    )
    result = retrieve_memories(q, ledger=ledger, state_snapshot=_memory_snapshot(ledger, ws))
    assert len(result.references) == 0


def test_provisional_confidence_cap(cognition_env):
    refs = [
        MemoryReference(
            memory_ref_id="m1",
            source_episode_id="e1",
            source_run_id="r1",
            source_event_id="ev1",
            source_asset_id="a1",
            source_run_purpose="live",
            epistemic_status="provisional",
            lifecycle_status="closed",
            memory_role="prior_experience",
            trust_level="low",
            artefact_hash="",
            scene_signature="s",
            category_signature="c",
            hypothesis_summary="prior fail",
            confidence_at_source=0.9,
            scores=ScoreComponents(1, 1, 1, 0, 0.1, 0.85),
            match_reason="test",
        )
    ]
    ctx = build_deliberation_context(refs, "baseline hyp", 0.6)
    assert ctx.confidence_delta_cap == 0.0
    assert ctx.requested_evidence


def test_deliberation_delta_field_level():
    baseline = {"primary_hypothesis": "A", "confidence": 0.6, "strategy": "standard", "requested_evidence": []}
    memory = {
        "primary_hypothesis": "B",
        "confidence": 0.5,
        "strategy": "consider_prior",
        "requested_evidence": ["verify_scene"],
    }
    deltas = compute_deliberation_delta(baseline, memory, ["ref1"])
    fields = {d.field_changed for d in deltas}
    assert "primary_hypothesis" in fields
    assert "requested_evidence" in fields
    assert any(d.mechanism == "reduce_confidence" for d in deltas if d.field_changed == "confidence")


def test_state_baseline_blocks_retrieval(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    _seed_closed_episode(ledger, ws, actor, "x", "interior_scene", "cluttered_room_weak_composition", "hyp")
    ledger.activate_state(ws, "state_baseline")
    q = RetrievalQuery(
        workspace_id=ws,
        actor_id=actor,
        asset_id="y",
        goal_type="critique",
        goal_instance_id=None,
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
    )
    state = ledger.get_active_state(ws)
    assert state["snapshot"]["retrieval_enabled"] is False
    result = retrieve_memories(q, ledger=ledger, state_snapshot=state["snapshot"])
    assert len(result.references) == 0


def test_slice_a_demo_harness(cognition_env, monkeypatch):
    from framed.cognition.demo.slice_a_e1_e2 import run_slice_a_demo

    report = run_slice_a_demo(cognition_dir=Path(cognition_env.db_path).parent, reset_store=True)
    assert report["status"] == "PASS"
    assert report["delta_count"] >= 1
    assert report["rollback_retrieval_count"] == 0


def test_replay_bundle_export(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    baseline_id, _ = ledger.ensure_demo_states(ws)
    eid = ledger.open_episode(
        workspace_id=ws,
        actor_id=actor,
        asset_id="a",
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run_id = str(uuid.uuid4())
    ledger.create_run(
        CognitiveRun(
            run_id=run_id,
            episode_id=eid,
            mode=RunMode.REPLAY,
            run_purpose=RunPurpose.REPLAY,
            state_version_id=baseline_id,
            context_fingerprint=None,
            retrieval_enabled=False,
            model_provenance={},
            prompt_provenance={},
            started_at="2026-01-01T00:00:00Z",
        )
    )
    ledger.append_event(episode_id=eid, run_id=run_id, event_type="replay", payload={"ok": True})
    bundle = ledger.export_replay_bundle([run_id])
    assert bundle["schema"] == "replay_bundle_v1"
    assert len(bundle["runs"]) == 1
    assert len(bundle["events"]) >= 1
