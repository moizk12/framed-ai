"""Slice B controlled-learning loop tests."""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import pytest

from framed.cognition import learning as learning_mod
from framed.cognition.contracts.memory import RetrievalQuery
from framed.cognition.contracts.runs import RunMode, RunPurpose
from framed.cognition.demo.slice_a_e1_e2 import run_slice_a_demo
from framed.cognition.demo.slice_b_loop import run_slice_b_demo
from framed.cognition.identity import get_identity
from framed.cognition.integration.pipeline_hook import begin_cognition_run, finalize_cognition_run
from framed.cognition.ledger.artefact_store import canonical_json_dumps
from framed.cognition.ledger.sqlite_store import CognitionLedger, reset_ledger
from framed.cognition.learning import (
    PromotionAuthorityError,
    PromotionBlockedError,
    ProposalImmutableError,
    accept_proposal,
    evaluate_proposal,
    generate_proposal,
    record_outcome,
    reject_proposal,
    rollback_promoted_state,
)
from framed.cognition.learning.evaluation import _collect_eval_run_ids
from framed.cognition.retrieval.service import retrieve_memories


@pytest.fixture
def cognition_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(tmp_path))
    ledger = reset_ledger(tmp_path / "test.sqlite3")
    yield ledger, tmp_path
    reset_ledger()


def _result(scene: str = "interior_scene") -> dict:
    return {"visual_evidence": {"scene_gate": {"scene_type": scene, "signals": {}}}}


def _intel(hyp: str, confidence: float = 0.55) -> dict:
    return {"recognition": {"what_i_see": hyp, "confidence": confidence}}


def _write_img(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _identity_workspace(ledger: CognitionLedger) -> tuple[str, str]:
    ident = get_identity()
    ws, actor = ident["workspace_id"], ident["actor_id"]
    ledger.ensure_initial_states(ws)
    ledger.activate_state(ws, "state_memory_enabled")
    return ws, actor


def _run_live(tmp_path: Path, name: str, content: bytes, hyp: str, state_label=None):
    img = _write_img(tmp_path, name, content)
    session = begin_cognition_run(
        result=_result(),
        image_path=str(img),
        asset_filename=name,
        run_mode=RunMode.MEMORY_ENABLED,
        run_purpose=RunPurpose.LIVE,
        state_label=state_label,
    )
    assert session is not None
    finalize_cognition_run(session, _result(), _intel(hyp))
    return session, img


def _run_memory_pair(tmp_path: Path, e1_session, later_name: str, later_bytes: bytes):
    img = _write_img(tmp_path, later_name, later_bytes)
    baseline = begin_cognition_run(
        result=_result(),
        image_path=str(img),
        asset_filename=later_name,
        run_mode=RunMode.BASELINE,
        run_purpose=RunPurpose.BASELINE,
        state_label="state_baseline",
    )
    assert baseline is not None
    finalize_cognition_run(baseline, _result(), _intel("baseline interior"))
    memory = begin_cognition_run(
        result=_result(),
        image_path=str(img),
        asset_filename=later_name,
        run_mode=RunMode.MEMORY_ENABLED,
        run_purpose=RunPurpose.MEMORY_ENABLED,
        state_label="state_memory_enabled",
        baseline_run_id=baseline.run_id,
    )
    assert memory is not None
    finalize_cognition_run(
        memory,
        _result(),
        _intel("reconsider prior clutter"),
        baseline_snapshot={
            "primary_hypothesis": "baseline interior",
            "confidence": 0.55,
            "strategy": "standard",
            "requested_evidence": [],
        },
    )
    return memory


def _promote_ready(ledger: CognitionLedger, tmp_path: Path):
    ws, actor = _identity_workspace(ledger)
    e1, _ = _run_live(tmp_path, "e1.jpg", b"slice-b-e1", "E1 clutter failure", "state_memory_enabled")
    e2 = _run_memory_pair(tmp_path, e1, "e2.jpg", b"slice-b-e2")
    assert e1.episode_id in {
        r["source_episode_id"]
        for r in ledger.export_replay_bundle([e2.run_id])["memory_references"]
    }
    outcome = record_outcome(
        workspace_id=ws,
        source_episode_id=e1.episode_id,
        source_run_id=e1.run_id,
        kind="testdaemon_eval",
        verdict="useful",
        created_by="testdaemon",
        ledger=ledger,
    )
    proposal = generate_proposal(outcome_id=outcome["outcome_id"], ledger=ledger)
    return ws, actor, e1, e2, outcome, proposal


def test_proposal_creation_does_not_promote(cognition_env):
    ledger, tmp_path = cognition_env
    ws, _, e1, _, outcome, proposal = _promote_ready(ledger, tmp_path)
    active = ledger.get_active_state(ws)
    assert proposal["kind"] == "promote_episode_belief"
    assert proposal["created_by"] == "proposal_generator"
    assert proposal["payload"]["source_episode_id"] == e1.episode_id
    assert e1.episode_id in proposal["payload"]["proposed_snapshot"]["promoted_episode_ids"]
    assert e1.episode_id not in (active["snapshot"].get("promoted_episode_ids") or [])
    assert active["state_version_id"] == proposal["base_state_version_id"]
    again = generate_proposal(outcome_id=outcome["outcome_id"], ledger=ledger)
    assert again["proposal_id"] == proposal["proposal_id"]


def test_accept_creates_active_promoted_state(cognition_env):
    ledger, tmp_path = cognition_env
    ws, _, e1, _, _, proposal = _promote_ready(ledger, tmp_path)
    parent_id = ledger.get_active_state(ws)["state_version_id"]
    evaluation = evaluate_proposal(proposal_id=proposal["proposal_id"], ledger=ledger, evidence_dir=tmp_path / "eval")
    assert evaluation["status"] == "pass"
    decision = accept_proposal(
        proposal_id=proposal["proposal_id"],
        authority_kind="testdaemon",
        actor_id="daemon-1",
        ledger=ledger,
    )
    active = ledger.get_active_state(ws)
    assert decision["action"] == "accept"
    assert active["state_version_id"] == decision["resulting_state_version_id"]
    assert active["snapshot"]["promoted_episode_ids"] == [e1.episode_id]
    parent = ledger.get_state_version(active["state_version_id"])
    assert parent is not None
    assert parent["parent_version_id"] == parent_id


def test_reject_keeps_state_and_remains_auditable(cognition_env):
    ledger, tmp_path = cognition_env
    ws, _, e1, _, _, proposal = _promote_ready(ledger, tmp_path)
    before = ledger.get_active_state(ws)["state_version_id"]
    decision = reject_proposal(
        proposal_id=proposal["proposal_id"],
        authority_kind="human",
        actor_id="researcher-1",
        ledger=ledger,
    )
    after = ledger.get_active_state(ws)
    assert decision["action"] == "reject"
    assert decision["resulting_state_version_id"] is None
    assert after["state_version_id"] == before
    assert e1.episode_id not in (after["snapshot"].get("promoted_episode_ids") or [])
    stored = ledger.get_decision_for_proposal(proposal["proposal_id"])
    assert stored is not None
    assert stored["action"] == "reject"
    with ledger._connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM update_proposals WHERE proposal_id=?", (proposal["proposal_id"],))
    with pytest.raises(ProposalImmutableError):
        accept_proposal(
            proposal_id=proposal["proposal_id"],
            authority_kind="testdaemon",
            actor_id="daemon-1",
            ledger=ledger,
        )


def test_rollback_restores_prior_active_state(cognition_env):
    ledger, tmp_path = cognition_env
    ws, actor, e1, _, _, proposal = _promote_ready(ledger, tmp_path)
    parent_id = ledger.get_active_state(ws)["state_version_id"]
    evaluate_proposal(proposal_id=proposal["proposal_id"], ledger=ledger, evidence_dir=tmp_path / "eval")
    accept_proposal(
        proposal_id=proposal["proposal_id"],
        authority_kind="testdaemon",
        actor_id="daemon-1",
        ledger=ledger,
    )
    promoted_id = ledger.get_active_state(ws)["state_version_id"]
    rollback = rollback_promoted_state(
        workspace_id=ws,
        authority_kind="human",
        actor_id="researcher-1",
        ledger=ledger,
    )
    restored = ledger.get_active_state(ws)
    assert rollback["from_state_version_id"] == promoted_id
    assert restored["state_version_id"] == parent_id
    q = RetrievalQuery(
        workspace_id=ws,
        actor_id=actor,
        asset_id="later",
        goal_type="critique",
        goal_instance_id=None,
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
    )
    result = retrieve_memories(q, ledger=ledger, state_snapshot=restored["snapshot"])
    match = next(r for r in result.references if r.source_episode_id == e1.episode_id)
    assert match.epistemic_status == "provisional"


def test_replay_regression_failure_blocks_promotion(cognition_env):
    ledger, tmp_path = cognition_env
    ws, _, e1, e2, _, proposal = _promote_ready(ledger, tmp_path)
    run_ids = _collect_eval_run_ids(ledger, e1.episode_id)
    bundle = ledger.export_replay_bundle(run_ids)
    mutated = copy.deepcopy(bundle)
    mutated["expected_hashes"]["runs"][e2.run_id]["deliberation_snapshot_hash"] = "0" * 64
    mut_path = tmp_path / "mutated_replay.json"
    mut_path.write_text(canonical_json_dumps(mutated), encoding="utf-8")
    evaluation = evaluate_proposal(
        proposal_id=proposal["proposal_id"],
        ledger=ledger,
        replay_bundle_path=mut_path,
        evidence_dir=tmp_path / "eval",
    )
    assert evaluation["status"] == "fail"
    before = ledger.get_active_state(ws)["state_version_id"]
    with pytest.raises(PromotionBlockedError):
        accept_proposal(
            proposal_id=proposal["proposal_id"],
            authority_kind="testdaemon",
            actor_id="daemon-1",
            ledger=ledger,
        )
    assert ledger.get_active_state(ws)["state_version_id"] == before
    with pytest.raises(PromotionBlockedError):
        accept_proposal(
            proposal_id=proposal["proposal_id"],
            authority_kind="testdaemon",
            actor_id="daemon-1",
            ledger=ledger,
        )


def test_accept_without_evaluation_is_blocked(cognition_env):
    ledger, tmp_path = cognition_env
    _, _, _, _, _, proposal = _promote_ready(ledger, tmp_path)
    with pytest.raises(PromotionBlockedError):
        accept_proposal(
            proposal_id=proposal["proposal_id"],
            authority_kind="testdaemon",
            actor_id="daemon-1",
            ledger=ledger,
        )


def test_provenance_closure(cognition_env):
    ledger, tmp_path = cognition_env
    ws, _, e1, _, outcome, proposal = _promote_ready(ledger, tmp_path)
    evaluation = evaluate_proposal(proposal_id=proposal["proposal_id"], ledger=ledger, evidence_dir=tmp_path / "eval")
    decision = accept_proposal(
        proposal_id=proposal["proposal_id"],
        authority_kind="testdaemon",
        actor_id="daemon-1",
        ledger=ledger,
    )
    outcome_payload = ledger.artefacts.get(outcome["artefact_hash"])
    proposal_payload = proposal["payload"]
    eval_payload = evaluation["payload"]
    decision_payload = decision["payload"]
    state = ledger.get_state_version(decision["resulting_state_version_id"])
    assert outcome_payload["source_episode_id"] == e1.episode_id
    assert outcome_payload["source_run_id"] == e1.run_id
    assert proposal_payload["outcome_id"] == outcome["outcome_id"]
    assert proposal_payload["base_state_version_id"] == proposal["base_state_version_id"]
    assert eval_payload["proposal_id"] == proposal["proposal_id"]
    assert eval_payload["proposal_artefact_hash"] == proposal["artefact_hash"]
    assert decision_payload["proposal_id"] == proposal["proposal_id"]
    assert decision_payload["evaluation_id"] == evaluation["evaluation_id"]
    assert state is not None
    assert state["parent_version_id"] == proposal["base_state_version_id"]
    later, _ = _run_live(tmp_path, "e3.jpg", b"slice-b-e3", "later consumes belief")
    assert later.state_version_id == decision["resulting_state_version_id"]
    later_refs = ledger.export_replay_bundle([later.run_id])["memory_references"]
    match = next(r for r in later_refs if r["source_episode_id"] == e1.episode_id)
    assert match["epistemic_status"] == "accepted"
    assert later.deliberation_context.strategy_hint == "consider_promoted_belief"


def test_no_direct_model_self_promotion(cognition_env):
    ledger, tmp_path = cognition_env
    _, _, _, _, _, proposal = _promote_ready(ledger, tmp_path)
    evaluate_proposal(proposal_id=proposal["proposal_id"], ledger=ledger, evidence_dir=tmp_path / "eval")
    for forbidden in ("model", "self", "proposal_generator", "llm"):
        with pytest.raises(PromotionAuthorityError):
            accept_proposal(
                proposal_id=proposal["proposal_id"],
                authority_kind=forbidden,
                actor_id="model-output",
                ledger=ledger,
            )
        with pytest.raises(PromotionAuthorityError):
            reject_proposal(
                proposal_id=proposal["proposal_id"],
                authority_kind=forbidden,
                actor_id="model-output",
                ledger=ledger,
            )
        with pytest.raises(PromotionAuthorityError):
            record_outcome(
                workspace_id=proposal["workspace_id"],
                source_episode_id=proposal["payload"]["source_episode_id"],
                source_run_id=proposal["payload"]["source_run_id"],
                kind="testdaemon_eval",
                verdict="useful",
                created_by=forbidden,
                ledger=ledger,
            )
    assert not hasattr(learning_mod, "promote_from_intelligence")
    assert generate_proposal.__module__.endswith("proposals")
    assert accept_proposal.__module__.endswith("authority")
    assert generate_proposal.__module__ != accept_proposal.__module__


def test_promoted_state_survives_restart(cognition_env):
    ledger, tmp_path = cognition_env
    ws, _, e1, _, _, proposal = _promote_ready(ledger, tmp_path)
    evaluate_proposal(proposal_id=proposal["proposal_id"], ledger=ledger, evidence_dir=tmp_path / "eval")
    decision = accept_proposal(
        proposal_id=proposal["proposal_id"],
        authority_kind="testdaemon",
        actor_id="daemon-1",
        ledger=ledger,
    )
    db_path = ledger.db_path
    artefact_root = ledger.artefacts.root
    reopened = reset_ledger(db_path, artefact_root=artefact_root)
    active = reopened.get_active_state(ws)
    assert active["state_version_id"] == decision["resulting_state_version_id"]
    assert e1.episode_id in active["snapshot"]["promoted_episode_ids"]
    stored = reopened.get_proposal(proposal["proposal_id"])
    assert stored is not None
    assert stored["payload"]["source_episode_id"] == e1.episode_id


def test_slice_a_behavior_remains_intact(cognition_env):
    ledger, tmp_path = cognition_env
    report = run_slice_a_demo(cognition_dir=tmp_path / "slice_a", reset_store=True)
    assert report["status"] == "PASS"
    assert report["delta_count"] >= 1
    assert report["rollback_retrieval_count"] == 0
    with ledger._connect() as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 4


def test_slice_b_vertical_slice_demo(cognition_env):
    _, tmp_path = cognition_env
    report = run_slice_b_demo(cognition_dir=tmp_path / "slice_b", reset_store=True, evidence_dir=tmp_path / "evidence")
    assert report["status"] == "PASS"
    assert report["e2_epistemic_status"] == "provisional"
    assert report["e3_epistemic_status"] == "accepted"
    assert report["e3_strategy_hint"] == "consider_promoted_belief"
    assert report["rollback_epistemic_status"] == "provisional"
    assert report["learned"]["kind"] == "promote_episode_belief"
    assert report["promoted_state_version_id"] != report["parent_state_version_id"]
    assert report["restored_state_version_id"] == report["parent_state_version_id"]
