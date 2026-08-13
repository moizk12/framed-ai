"""Retrieval eligibility and contamination control tests (Slice A Task 8)."""

from __future__ import annotations

import uuid

import pytest

from framed.cognition.identity import get_identity

from framed.cognition.contracts.memory import RetrievalQuery
from framed.cognition.contracts.runs import CognitiveRun, RunMode, RunPurpose, SameAssetPolicy
from framed.cognition.integration.pipeline_hook import begin_cognition_run, finalize_cognition_run
from framed.cognition.ledger.sqlite_store import CognitionLedger, reset_ledger
from framed.cognition.retrieval.service import retrieve_memories


@pytest.fixture
def cognition_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(tmp_path))
    ledger = reset_ledger(tmp_path / "test.sqlite3")
    yield ledger
    reset_ledger()


def _memory_snapshot(ledger: CognitionLedger, ws: str) -> dict:
    ledger.activate_state(ws, "state_memory_enabled")
    return ledger.get_active_state(ws)["snapshot"]


def _result(scene: str = "interior_scene"):
    return {"visual_evidence": {"scene_gate": {"scene_type": scene, "signals": {}}}}


def _intel(hyp: str = "hypothesis"):
    return {"recognition": {"what_i_see": hyp, "confidence": 0.55}}


def _close_eligible(ledger: CognitionLedger, ws: str, actor: str, asset: str, purpose: RunPurpose, scene: str, cat: str, hyp: str):
    baseline_id, _ = ledger.ensure_demo_states(ws)
    eid = ledger.open_episode(
        workspace_id=ws,
        actor_id=actor,
        asset_id=asset,
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run_id = str(uuid.uuid4())
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
            retrieval_eligible=purpose in (RunPurpose.LIVE, RunPurpose.DEMO_SEED),
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
        final_fingerprint="fp",
        perception_artefact_hash=snap_hash,
    )
    ledger.complete_run(run_id)
    return eid, run_id


def _query(ws: str, actor: str, asset: str, **kwargs):
    return RetrievalQuery(
        workspace_id=ws,
        actor_id=actor,
        asset_id=asset,
        goal_type="critique",
        goal_instance_id=None,
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
        **kwargs,
    )


def test_same_asset_excluded_before_ranking(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    asset = "shared_asset"
    _close_eligible(ledger, ws, actor, asset, RunPurpose.LIVE, "interior_scene", "cluttered_room_weak_composition", "prior")
    result = retrieve_memories(_query(ws, actor, asset), ledger=ledger, state_snapshot=_memory_snapshot(ledger, ws))
    assert len(result.references) == 0
    assert any(r.get("rejection_reason") == "same_asset" for r in result.rejected)


def test_baseline_in_ledger_not_retrieved(cognition_env, tmp_path):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    img = tmp_path / "img.jpg"
    img.write_bytes(b"baseline-bytes")
    ledger.ensure_demo_states(ws)
    ledger.activate_state(ws, "state_baseline")
    s = begin_cognition_run(
        result=_result(),
        image_path=str(img),
        asset_filename="img.jpg",
        run_mode=RunMode.BASELINE,
        run_purpose=RunPurpose.BASELINE,
    )
    assert s
    finalize_cognition_run(s, _result(), _intel("baseline hyp"))
    candidates = ledger.list_retrieval_candidates(ws, actor)
    assert not any(c["episode_id"] == s.episode_id for c in candidates)
    result = retrieve_memories(_query(ws, actor, "other"), ledger=ledger, state_snapshot=ledger.get_active_state(ws)["snapshot"])
    assert s.episode_id not in {r.source_episode_id for r in result.references}


def test_control_not_retrieved(cognition_env, tmp_path):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    img = tmp_path / "ctrl.jpg"
    img.write_bytes(b"control-bytes")
    ledger.ensure_demo_states(ws)
    s = begin_cognition_run(
        result=_result("people_scene"),
        image_path=str(img),
        asset_filename="ctrl.jpg",
        run_mode=RunMode.CONTROL,
        run_purpose=RunPurpose.CONTROL,
    )
    assert s
    finalize_cognition_run(s, _result("people_scene"), _intel("control"))
    assert s.episode_id not in {c["episode_id"] for c in ledger.list_retrieval_candidates(ws, actor)}


def test_replay_not_retrieved(cognition_env, tmp_path):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    img = tmp_path / "replay.jpg"
    img.write_bytes(b"replay-bytes")
    ledger.ensure_demo_states(ws)
    s = begin_cognition_run(
        result=_result(),
        image_path=str(img),
        asset_filename="replay.jpg",
        run_mode=RunMode.REPLAY,
        run_purpose=RunPurpose.REPLAY,
    )
    assert s
    finalize_cognition_run(s, _result(), _intel("replay"))
    assert s.episode_id not in {c["episode_id"] for c in ledger.list_retrieval_candidates(ws, actor)}


def test_live_e1_retrieved(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    e1, _ = _close_eligible(ledger, ws, actor, "e1_asset", RunPurpose.LIVE, "interior_scene", "cluttered_room_weak_composition", "E1 hyp")
    result = retrieve_memories(_query(ws, actor, "e2_asset"), ledger=ledger, state_snapshot=_memory_snapshot(ledger, ws))
    assert len(result.references) == 1
    assert result.references[0].source_episode_id == e1


def test_baseline_run_id_available_for_comparison(cognition_env, tmp_path):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    shared = b"same-for-ab"
    img = tmp_path / "ab.jpg"
    img.write_bytes(shared)
    ledger.ensure_demo_states(ws)
    ledger.activate_state(ws, "state_baseline")
    base = begin_cognition_run(
        result=_result(),
        image_path=str(img),
        asset_filename="ab.jpg",
        run_mode=RunMode.BASELINE,
        run_purpose=RunPurpose.BASELINE,
    )
    assert base
    finalize_cognition_run(base, _result(), _intel("baseline line"))
    snap = ledger.get_baseline_snapshot_by_run_id(base.run_id)
    assert snap is not None
    assert snap.get("primary_hypothesis") == "baseline line"


def test_e2_memory_retrieves_e1_not_baseline(cognition_env, tmp_path):
    ledger = cognition_env
    ident = get_identity()
    ws, actor = ident["workspace_id"], ident["actor_id"]
    e2_img = tmp_path / "e2.jpg"
    e2_img.write_bytes(b"e2-shared-asset")
    ledger.ensure_demo_states(ws)
    e1, _ = _close_eligible(ledger, ws, actor, "e1_only", RunPurpose.LIVE, "interior_scene", "cluttered_room_weak_composition", "E1")
    ledger.activate_state(ws, "state_baseline")
    base = begin_cognition_run(
        result=_result(),
        image_path=str(e2_img),
        asset_filename="e2.jpg",
        run_mode=RunMode.BASELINE,
        run_purpose=RunPurpose.BASELINE,
    )
    assert base
    finalize_cognition_run(base, _result(), _intel("E2 baseline"))
    ledger.activate_state(ws, "state_memory_enabled")
    mem = begin_cognition_run(
        result=_result(),
        image_path=str(e2_img),
        asset_filename="e2.jpg",
        run_purpose=RunPurpose.MEMORY_ENABLED,
        baseline_run_id=base.run_id,
    )
    assert mem
    assert base.episode_id not in {r.source_episode_id for r in mem.deliberation_context.memory_references}
    finalize_cognition_run(mem, _result(), _intel("E2 memory"))
    bundle = ledger.export_replay_bundle([mem.run_id])
    source_eps = {r["source_episode_id"] for r in bundle["memory_references"]}
    assert e1 in source_eps
    assert base.episode_id not in source_eps
    assert any(r.get("episode_id") == base.episode_id for r in mem.rejected_candidates)


def test_allow_related_revision_same_asset(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    asset = "revision_asset"
    _close_eligible(ledger, ws, actor, asset, RunPurpose.LIVE, "interior_scene", "cluttered_room_weak_composition", "rev")
    q = _query(ws, actor, asset, same_asset_policy=SameAssetPolicy.ALLOW_RELATED_REVISION)
    result = retrieve_memories(q, ledger=ledger, state_snapshot=_memory_snapshot(ledger, ws))
    assert len(result.references) == 1


def test_default_query_excludes_same_asset(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    asset = "default_exclude"
    _close_eligible(ledger, ws, actor, asset, RunPurpose.LIVE, "interior_scene", "cluttered_room_weak_composition", "x")
    q = RetrievalQuery(
        workspace_id=ws,
        actor_id=actor,
        asset_id=asset,
        goal_type="critique",
        goal_instance_id=None,
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
    )
    assert q.same_asset_policy == SameAssetPolicy.EXCLUDE
    result = retrieve_memories(q, ledger=ledger, state_snapshot=_memory_snapshot(ledger, ws))
    assert len(result.references) == 0


def test_rejected_audit_exact_reason(cognition_env):
    ledger = cognition_env
    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    eid2, _ = _close_eligible(
        ledger, ws, actor, "audit_asset", RunPurpose.LIVE, "interior_scene", "cluttered_room_weak_composition", "live"
    )
    result = retrieve_memories(_query(ws, actor, "audit_asset"), ledger=ledger, state_snapshot=_memory_snapshot(ledger, ws))
    same_asset_rejects = [r for r in result.rejected if r.get("episode_id") == eid2]
    assert same_asset_rejects
    assert same_asset_rejects[0]["rejection_reason"] == "same_asset"
