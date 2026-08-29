"""B4 held-out later-behavior evaluation tests."""

from __future__ import annotations

import json

import pytest

from framed.cognition.demo.slice_b4_later_behavior import (
    ACTION_FINALIZE_STANDARD,
    ACTION_REQUEST_COMPOSITION,
    COMPOSITION_FAILURE_MARKER,
    FROZEN_CASES,
    FROZEN_METRICS,
    FROZEN_ORACLE,
    apply_frozen_recognizer,
    freeze_evaluation_protocol,
    observable_case,
    run_b4_evaluation,
    score_outcome_task,
)
from framed.cognition.ledger.artefact_store import artefact_hash
from framed.cognition.ledger.sqlite_store import reset_ledger


@pytest.fixture
def cognition_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(tmp_path))
    ledger = reset_ledger(tmp_path / "test.sqlite3")
    yield ledger, tmp_path
    reset_ledger()


def test_observable_cases_exclude_oracle_fields():
    assert len(FROZEN_CASES) >= 5
    roles = [c["role"] for c in FROZEN_CASES]
    assert roles.count("transfer") >= 2
    assert roles.count("near_miss") >= 1
    assert roles.count("control") >= 2
    for case in FROZEN_CASES:
        obs = observable_case(case)
        assert "expected_critique_action" not in obs
        assert "expected_transfer_outcome" not in obs
    assert set(FROZEN_ORACLE) == {c["case_id"] for c in FROZEN_CASES}


def test_outcome_scorer_uses_only_task_output_and_oracle():
    assert score_outcome_task("L1", ACTION_REQUEST_COMPOSITION)["outcome_task_score"] == 1
    assert score_outcome_task("L1", ACTION_FINALIZE_STANDARD)["outcome_task_score"] == 0
    assert score_outcome_task("L3", ACTION_FINALIZE_STANDARD)["outcome_task_score"] == 1
    assert score_outcome_task("L3", ACTION_REQUEST_COMPOSITION)["outcome_task_score"] == 0
    assert score_outcome_task("N1", ACTION_FINALIZE_STANDARD)["outcome_task_score"] == 1


def test_frozen_recognizer_never_reads_oracle_and_respects_near_miss():
    class _Ref:
        def __init__(self, status: str):
            self.scene_signature = "interior_scene"
            self.hypothesis_summary = f"prior {COMPOSITION_FAILURE_MARKER}"
            self.epistemic_status = status
            self.source_episode_id = "e1"

    class _Ctx:
        def __init__(self, status: str):
            self.memory_references = [_Ref(status)]

    class _Session:
        def __init__(self, status: str):
            self.deliberation_context = _Ctx(status)

    elevated = observable_case(next(c for c in FROZEN_CASES if c["case_id"] == "L1"))
    near_miss = observable_case(next(c for c in FROZEN_CASES if c["case_id"] == "L3"))
    assert "expected_critique_action" not in elevated
    accepted_elevated = apply_frozen_recognizer(elevated, _Session("accepted"))
    accepted_near_miss = apply_frozen_recognizer(near_miss, _Session("accepted"))
    provisional = apply_frozen_recognizer(elevated, _Session("provisional"))
    none_sess = _Session("accepted")
    none_sess.deliberation_context.memory_references = []
    baseline_out = apply_frozen_recognizer(elevated, none_sess)
    assert accepted_elevated["recognition"]["critique_action"] == ACTION_REQUEST_COMPOSITION
    assert accepted_near_miss["recognition"]["critique_action"] == ACTION_FINALIZE_STANDARD
    assert provisional["recognition"]["critique_action"] == ACTION_FINALIZE_STANDARD
    assert baseline_out["recognition"]["critique_action"] == ACTION_FINALIZE_STANDARD


def test_b4_later_behavior_closeout(cognition_env):
    _, tmp_path = cognition_env
    evidence = tmp_path / "b4_evidence"
    freeze_evaluation_protocol(evidence)
    freeze_before = json.loads((evidence / "freeze_record.json").read_text(encoding="utf-8"))
    report = run_b4_evaluation(
        cognition_dir=tmp_path / "b4_store",
        reset_store=True,
        evidence_dir=evidence,
    )
    freeze_after = json.loads((evidence / "freeze_record.json").read_text(encoding="utf-8"))
    assert freeze_after["cases_hash"] == freeze_before["cases_hash"]
    assert freeze_after["oracle_hash"] == freeze_before["oracle_hash"]
    assert freeze_after["metrics_hash"] == freeze_before["metrics_hash"]
    assert freeze_after["metrics_hash"] == artefact_hash(dict(FROZEN_METRICS))
    assert report["train"]["held_out_absent_from_proposal"] is True
    assert report["train"]["evaluation_status"] == "pass"
    assert (evidence / "frozen_oracle.json").exists()

    a = report["conditions"]["A_baseline"]
    b = report["conditions"]["B_provisional"]
    c = report["conditions"]["C_promoted"]
    a_l1 = report["outputs"]["A_baseline"][0]
    b_l1 = report["outputs"]["B_provisional"][0]
    c_l1 = report["outputs"]["C_promoted"][0]
    l3_c = next(o for o in report["outputs"]["C_promoted"] if o["case_id"] == "L3")
    n1_c = next(o for o in report["outputs"]["C_promoted"] if o["case_id"] == "N1")

    assert a_l1["critique_action"] == ACTION_FINALIZE_STANDARD
    assert b_l1["critique_action"] == ACTION_FINALIZE_STANDARD
    assert c_l1["critique_action"] == ACTION_REQUEST_COMPOSITION
    assert l3_c["critique_action"] == ACTION_FINALIZE_STANDARD
    assert n1_c["critique_action"] == ACTION_FINALIZE_STANDARD
    assert a_l1["scores"]["outcome_task_score"] == 0
    assert b_l1["scores"]["outcome_task_score"] == 0
    assert c_l1["scores"]["outcome_task_score"] == 1
    assert l3_c["scores"]["outcome_task_score"] == 1

    assert report["score_deltas"]["changed_without_outcome_improvement_B_vs_A"] is True
    assert report["score_deltas"]["C_minus_B_outcome_transfer"] > 0
    assert report["score_deltas"]["B_minus_A_outcome_transfer"] == 0
    assert c["mean_outcome_task_score_guard"] == 1.0
    assert report["regressions"]["guard_case_ids"] == []

    rollback = report["rollback"]
    assert rollback["outcome_task_score"] == b_l1["scores"]["outcome_task_score"]
    assert rollback["critique_action"] == ACTION_FINALIZE_STANDARD
    assert rollback["to_state_version_id"] == report["train"]["parent_state_version_id"]

    assert report["verdict"] == "B4 CLOSEOUT PASS — LEAKAGE-FREE INDEPENDENT OUTCOME IMPROVEMENT PROVEN"
    assert report["status"] == "PASS"
    assert (evidence / "outcome_scores.json").exists()


def test_b4_metrics_are_not_weakened_after_results(cognition_env):
    _, tmp_path = cognition_env
    evidence = tmp_path / "b4_metrics_lock"
    report = run_b4_evaluation(
        cognition_dir=tmp_path / "b4_store_lock",
        reset_store=True,
        evidence_dir=evidence,
    )
    stored_metrics = json.loads((evidence / "frozen_metrics.json").read_text(encoding="utf-8"))
    assert stored_metrics == dict(FROZEN_METRICS)
    assert report["freeze"]["metrics_hash"] == artefact_hash(dict(FROZEN_METRICS))
    assert any(r["name"] == "metrics_hash_unchanged" and r["pass"] for r in report["pass_fail_reasons"])
    assert any(r["name"] == "oracle_hash_unchanged" and r["pass"] for r in report["pass_fail_reasons"])
