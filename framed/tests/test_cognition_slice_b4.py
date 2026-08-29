"""B4 held-out later-behavior evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from framed.cognition.demo.slice_b4_later_behavior import (
    COMPOSITION_FAILURE_MARKER,
    FROZEN_CASES,
    FROZEN_METRICS,
    INDEPENDENT_TRANSFER_OUTCOME,
    apply_frozen_recognizer,
    freeze_evaluation_protocol,
    run_b4_evaluation,
    score_observation,
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


def test_frozen_protocol_is_defined_before_results():
    assert len(FROZEN_CASES) >= 4
    roles = [c["role"] for c in FROZEN_CASES]
    assert roles.count("transfer") >= 2
    assert roles.count("control") >= 2
    assert FROZEN_METRICS["defined_before_results"] is True
    assert "mean(C.outcome_task_score transfer) > mean(B.outcome_task_score transfer)" in FROZEN_METRICS[
        "closeout_pass_criteria"
    ]
    assert INDEPENDENT_TRANSFER_OUTCOME in FROZEN_METRICS["later_task"]
    naive = " ".join(c["naive_hypothesis"] for c in FROZEN_CASES)
    assert INDEPENDENT_TRANSFER_OUTCOME not in naive
    assert COMPOSITION_FAILURE_MARKER not in naive


def test_outcome_scorer_uses_only_task_output():
    transfer = next(c for c in FROZEN_CASES if c["role"] == "transfer")
    control = next(c for c in FROZEN_CASES if c["role"] == "control")
    success = score_outcome_task(
        transfer,
        f"{transfer['naive_hypothesis']} {INDEPENDENT_TRANSFER_OUTCOME}",
    )
    fail = score_outcome_task(transfer, transfer["naive_hypothesis"])
    assert success["outcome_task_score"] == 1
    assert fail["outcome_task_score"] == 0
    assert score_outcome_task(control, control["naive_hypothesis"])["outcome_task_score"] == 1
    assert (
        score_outcome_task(
            control,
            f"{control['naive_hypothesis']} {INDEPENDENT_TRANSFER_OUTCOME}",
        )["outcome_task_score"]
        == 0
    )


def test_frozen_recognizer_appends_outcome_only_for_accepted_belief():
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

    case = dict(FROZEN_CASES[0])
    accepted = apply_frozen_recognizer(case, _Session("accepted"))
    provisional = apply_frozen_recognizer(case, _Session("provisional"))
    none_sess = _Session("accepted")
    none_sess.deliberation_context.memory_references = []
    baseline = apply_frozen_recognizer(case, none_sess)
    accepted_hyp = accepted["recognition"]["what_i_see"]
    provisional_hyp = provisional["recognition"]["what_i_see"]
    assert INDEPENDENT_TRANSFER_OUTCOME in accepted_hyp
    assert COMPOSITION_FAILURE_MARKER not in accepted_hyp
    assert INDEPENDENT_TRANSFER_OUTCOME not in provisional_hyp
    assert INDEPENDENT_TRANSFER_OUTCOME not in baseline["recognition"]["what_i_see"]
    assert "Provisional prior noted" in provisional_hyp


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
    assert freeze_after["metrics_hash"] == freeze_before["metrics_hash"]
    assert freeze_after["metrics_hash"] == artefact_hash(dict(FROZEN_METRICS))
    assert report["train"]["held_out_absent_from_proposal"] is True
    assert report["train"]["evaluation_status"] == "pass"

    a = report["conditions"]["A_baseline"]
    b = report["conditions"]["B_provisional"]
    c = report["conditions"]["C_promoted"]
    a_l1 = report["outputs"]["A_baseline"][0]
    b_l1 = report["outputs"]["B_provisional"][0]
    c_l1 = report["outputs"]["C_promoted"][0]
    n1_c = next(o for o in report["outputs"]["C_promoted"] if o["case_id"] == "N1")
    n2_c = next(o for o in report["outputs"]["C_promoted"] if o["case_id"] == "N2")

    assert a_l1["scores"]["outcome_task_score"] == 0
    assert b_l1["scores"]["outcome_task_score"] == 0
    assert c_l1["scores"]["outcome_task_score"] == 1
    assert INDEPENDENT_TRANSFER_OUTCOME in c_l1["primary_hypothesis"]
    assert COMPOSITION_FAILURE_MARKER not in c_l1["primary_hypothesis"]
    assert n1_c["scores"]["outcome_task_score"] == 1
    assert n2_c["scores"]["outcome_task_score"] == 1
    assert n1_c["primary_hypothesis"] == next(x for x in FROZEN_CASES if x["case_id"] == "N1")["naive_hypothesis"]
    assert n2_c["primary_hypothesis"] == next(x for x in FROZEN_CASES if x["case_id"] == "N2")["naive_hypothesis"]

    assert report["score_deltas"]["changed_without_outcome_improvement_B_vs_A"] is True
    assert report["score_deltas"]["C_minus_B_outcome_transfer"] > 0
    assert report["score_deltas"]["B_minus_A_outcome_transfer"] == 0
    assert c["mean_outcome_task_score_control"] == 1.0
    assert report["regressions"]["control_case_ids"] == []

    rollback = report["rollback"]
    assert rollback["outcome_task_score"] == b_l1["scores"]["outcome_task_score"]
    assert rollback["to_state_version_id"] == report["train"]["parent_state_version_id"]

    assert report["outputs"]["C_promoted"][0]["state_version_id"] == report["promoted_state_version_id"]
    assert report["verdict"] == "B4 CLOSEOUT PASS — INDEPENDENT OUTCOME IMPROVEMENT PROVEN"
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
