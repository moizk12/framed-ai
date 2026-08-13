"""B4 held-out later-behavior evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from framed.cognition.demo.slice_b4_later_behavior import (
    COMPOSITION_FAILURE_MARKER,
    FROZEN_CASES,
    FROZEN_METRICS,
    apply_frozen_recognizer,
    freeze_evaluation_protocol,
    run_b4_evaluation,
    score_observation,
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
    assert "mean(C.transfer_task_score) > mean(B.transfer_task_score)" in FROZEN_METRICS["pass_criteria"]
    assert COMPOSITION_FAILURE_MARKER in FROZEN_METRICS["later_task"]
    naive = " ".join(c["naive_hypothesis"] for c in FROZEN_CASES)
    assert COMPOSITION_FAILURE_MARKER not in naive


def test_frozen_recognizer_adopts_only_accepted_belief():
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
    assert COMPOSITION_FAILURE_MARKER in accepted["recognition"]["what_i_see"]
    assert COMPOSITION_FAILURE_MARKER not in provisional["recognition"]["what_i_see"]
    assert COMPOSITION_FAILURE_MARKER not in baseline["recognition"]["what_i_see"]
    assert "Provisional prior noted" in provisional["recognition"]["what_i_see"]


def test_control_score_penalizes_promoted_bias():
    case = next(c for c in FROZEN_CASES if c["role"] == "control")
    clean = score_observation(
        case,
        {
            "primary_hypothesis": case["naive_hypothesis"],
            "retrieved_source": False,
            "strategy_hint": "standard",
        },
    )
    biased = score_observation(
        case,
        {
            "primary_hypothesis": f"{case['naive_hypothesis']} {COMPOSITION_FAILURE_MARKER}",
            "retrieved_source": True,
            "strategy_hint": "consider_promoted_belief",
        },
    )
    assert clean["control_task_score"] == 1
    assert biased["control_task_score"] == 0


def test_b4_later_behavior_evaluation(cognition_env):
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

    assert a_l1["retrieved_source"] is False
    assert a_l1["strategy_hint"] == "standard"
    assert b_l1["retrieved_source"] is True
    assert b_l1["source_epistemic_status"] == "provisional"
    assert b_l1["strategy_hint"] == "consider_prior_provisional_experience"
    assert c_l1["retrieved_source"] is True
    assert c_l1["source_epistemic_status"] == "accepted"
    assert c_l1["source_memory_role"] == "promoted_belief"
    assert c_l1["strategy_hint"] == "consider_promoted_belief"
    assert n1_c["retrieved_source"] is False
    assert n2_c["retrieved_source"] is False
    assert n1_c["strategy_hint"] != "consider_promoted_belief"
    assert n2_c["strategy_hint"] != "consider_promoted_belief"

    assert report["score_deltas"]["changed_without_improvement_B_vs_A"] is True
    assert report["score_deltas"]["C_minus_B_transfer"] > 0
    assert report["score_deltas"]["B_minus_A_transfer"] == 0
    assert c["mean_control_task_score"] == 1.0
    assert report["regressions"]["control_case_ids"] == []

    rollback = report["rollback"]
    assert rollback["transfer_task_score"] == b_l1["scores"]["transfer_task_score"]
    assert rollback["strategy_hint"] == b_l1["strategy_hint"]
    assert rollback["source_epistemic_status"] == "provisional"
    assert rollback["to_state_version_id"] == report["train"]["parent_state_version_id"]

    assert report["outputs"]["C_promoted"][0]["state_version_id"] == report["promoted_state_version_id"]
    assert report["verdict"] == "B4 PASS — MEASURABLE LATER-BEHAVIOR IMPROVEMENT PROVEN"
    assert report["status"] == "PASS"


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
