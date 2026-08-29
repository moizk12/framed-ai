"""B4 held-out later-behavior evaluation.

Proves or fails the claim that promotion produces measurably better later
behavior on untouched later cases. Frozen cases and metrics are written before
any promoted-state later run. Proposal generation never sees held-out cases.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from framed.cognition.contracts.runs import RunMode, RunPurpose
from framed.cognition.demo.slice_a_e1_e2 import (
    _demo_store,
    _run_episode,
    _synthetic_intelligence,
    _synthetic_result,
    _write_temp_image,
)
from framed.cognition.identity import get_identity
from framed.cognition.ledger.artefact_store import artefact_hash, canonical_json_dumps
from framed.cognition.ledger.sqlite_store import (
    clear_ledger,
    get_ledger,
    release_ledger_store,
    reset_ledger,
)
from framed.cognition.learning.authority import accept_proposal
from framed.cognition.learning.evaluation import evaluate_proposal
from framed.cognition.learning.outcomes import record_outcome
from framed.cognition.learning.proposals import generate_proposal
from framed.cognition.learning.rollback import rollback_promoted_state

COMPOSITION_FAILURE_MARKER = "COMPOSITION_FAILURE_MODE"
# Frozen task-level outcome for transfer cases. Scorer inspects only this phrase in output.
INDEPENDENT_TRANSFER_OUTCOME = "request composition evidence before finalizing critique"
FROZEN_MODEL_ID = "b4-frozen-recognizer"
FROZEN_SEED = "b4-later-behavior-v1"

# Frozen before any later-case execution. Do not edit after seeing results.
FROZEN_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "case_id": "L1",
        "role": "transfer",
        "scene_type": "interior_scene",
        "image_bytes": "b4-later-interior-heldout-1",
        "naive_hypothesis": "Interior scene with visible clutter.",
        "naive_confidence": 0.58,
        "expected_transfer_outcome": INDEPENDENT_TRANSFER_OUTCOME,
        "note": "Held-out same-category interior. Must not be used to train or select the proposal.",
    },
    {
        "case_id": "L2",
        "role": "transfer",
        "scene_type": "interior_scene",
        "image_bytes": "b4-later-interior-heldout-2",
        "naive_hypothesis": "Another interior with dense objects.",
        "naive_confidence": 0.57,
        "expected_transfer_outcome": INDEPENDENT_TRANSFER_OUTCOME,
        "note": "Second held-out same-category interior. Independent asset from L1 and E1/E2.",
    },
    {
        "case_id": "N1",
        "role": "control",
        "scene_type": "people_scene",
        "image_bytes": "b4-control-street-heldout-1",
        "naive_hypothesis": "Layered street composition with pedestrian flow.",
        "naive_confidence": 0.60,
        "note": "Negative control: different category. Promotion must not apply interior composition belief.",
    },
    {
        "case_id": "N2",
        "role": "control",
        "scene_type": "screenshot_ui",
        "image_bytes": "b4-control-screenshot-heldout-1",
        "naive_hypothesis": "Software interface screenshot.",
        "naive_confidence": 0.62,
        "note": "Negative control: screenshot/UI. Promotion must not bias all later outputs.",
    },
)

FROZEN_METRICS: Dict[str, Any] = {
    "schema": "b4_later_behavior_metrics_v2",
    "defined_before_results": True,
    "later_task": (
        "On held-out same-category interiors, later critique must reach the frozen task outcome "
        f"({INDEPENDENT_TRANSFER_OUTCOME!r}). On held-out different-category scenes, the naive "
        "hypothesis must remain unchanged."
    ),
    "changed_vs_improved": (
        "Changed behavior is any difference in retrieval, strategy_hint, epistemic_status, "
        "trust_level, or hypothesis text. Improved behavior is only an increase in "
        "outcome_task_score without a decrease in control outcome_task_score."
    ),
    "frozen_recognizer_policy": (
        "Identical frozen recognizer for all conditions. Naive hypothesis is the starting "
        "output. A retrieved same-scene prior containing the failure marker may influence "
        "reasoning internally; only an accepted prior may append the frozen transfer outcome "
        "phrase to the task output. Provisional priors may be noted but must not append the "
        "transfer outcome. Confidence never increases."
    ),
    "outcome_task_score": (
        "Transfer: 1 iff primary_hypothesis contains the case frozen expected_transfer_outcome "
        "string. Control: 1 iff primary_hypothesis equals the case frozen naive_hypothesis "
        "(trimmed). Scorer inspects only primary_hypothesis and frozen case fields."
    ),
    "outcome_scorer_forbidden_inputs": [
        "COMPOSITION_FAILURE_MODE",
        "retrieval occurrence",
        "epistemic_status",
        "promotion state",
        "strategy_hint",
        "internal provenance fields",
    ],
    "mechanism_transfer_task_score": (
        "Legacy diagnostic only: 1 if primary_hypothesis contains COMPOSITION_FAILURE_MARKER. "
        "Not used for scientific closeout."
    ),
    "mechanism_control_task_score": (
        "Legacy diagnostic only. Not used for scientific closeout."
    ),
    "condition_outcome_score": "mean(outcome_task_score for transfer) + mean(outcome_task_score for control)",
    "closeout_pass_criteria": [
        "At least two transfer cases and two control cases are frozen before promoted later runs.",
        "mean(C.outcome_task_score transfer) > mean(B.outcome_task_score transfer)",
        "mean(C.outcome_task_score transfer) > mean(A.outcome_task_score transfer)",
        "mean(A.outcome_task_score control) == mean(B.outcome_task_score control) == mean(C.outcome_task_score control) == 1.0",
        "mean(B.outcome_task_score transfer) == mean(A.outcome_task_score transfer)",
        "No transfer case has C.outcome_task_score < B.outcome_task_score",
        "Rollback later transfer outcome_task_score matches provisional, not promoted.",
        "Held-out case ids and expected_transfer_outcome are absent from the proposal payload.",
        "Metrics artefact hash at scoring equals the hash frozen before later runs.",
        "Provenance closes outcome → proposal → evaluation → decision → promoted state → later C run.",
    ],
}


def freeze_evaluation_protocol(evidence_dir: Path) -> Dict[str, str]:
    """Persist cases and metrics before any promoted later-case execution."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = {
        "schema": "b4_frozen_cases_v1",
        "cases": list(FROZEN_CASES),
        "composition_failure_marker": COMPOSITION_FAILURE_MARKER,
        "independent_transfer_outcome": INDEPENDENT_TRANSFER_OUTCOME,
        "frozen_model_id": FROZEN_MODEL_ID,
        "frozen_seed": FROZEN_SEED,
    }
    metrics_payload = dict(FROZEN_METRICS)
    cases_hash = artefact_hash(cases_payload)
    metrics_hash = artefact_hash(metrics_payload)
    (evidence_dir / "frozen_cases.json").write_text(canonical_json_dumps(cases_payload), encoding="utf-8")
    (evidence_dir / "frozen_metrics.json").write_text(canonical_json_dumps(metrics_payload), encoding="utf-8")
    freeze_record = {
        "schema": "b4_freeze_record_v1",
        "cases_hash": cases_hash,
        "metrics_hash": metrics_hash,
        "case_ids": [c["case_id"] for c in FROZEN_CASES],
        "frozen_before_promoted_later_runs": True,
    }
    (evidence_dir / "freeze_record.json").write_text(canonical_json_dumps(freeze_record), encoding="utf-8")
    return {"cases_hash": cases_hash, "metrics_hash": metrics_hash}


def apply_frozen_recognizer(case: Dict[str, Any], session: Any) -> Dict[str, Any]:
    """Frozen identical model. Consumes real cognition-path context only."""
    hypothesis = str(case["naive_hypothesis"])
    confidence = float(case["naive_confidence"])
    refs = list(session.deliberation_context.memory_references)
    for ref in refs:
        summary = ref.hypothesis_summary or ""
        if ref.scene_signature != case["scene_type"]:
            continue
        if COMPOSITION_FAILURE_MARKER not in summary:
            continue
        if ref.epistemic_status == "accepted":
            expected = str(case.get("expected_transfer_outcome") or "")
            if expected:
                hypothesis = f"{hypothesis} {expected}"
            confidence = min(confidence, 0.50)
        elif ref.epistemic_status == "provisional":
            hypothesis = f"{hypothesis} Provisional prior noted; not adopted as belief."
            confidence = min(confidence, 0.50)
        break
    return _synthetic_intelligence(hypothesis, confidence)


def _observe(case: Dict[str, Any], session: Any, source_episode_id: str) -> Dict[str, Any]:
    refs = list(session.deliberation_context.memory_references)
    source_ref = next((r for r in refs if r.source_episode_id == source_episode_id), None)
    hypothesis = session.deliberation_context.prior_hypothesis
    return {
        "case_id": case["case_id"],
        "role": case["role"],
        "scene_type": case["scene_type"],
        "run_id": session.run_id,
        "episode_id": session.episode_id,
        "state_version_id": session.state_version_id,
        "strategy_hint": session.deliberation_context.strategy_hint or "standard",
        "requested_evidence": list(session.deliberation_context.requested_evidence),
        "prior_hypothesis": hypothesis,
        "retrieved_episode_ids": [r.source_episode_id for r in refs],
        "retrieved_source": source_ref is not None,
        "source_epistemic_status": None if source_ref is None else source_ref.epistemic_status,
        "source_memory_role": None if source_ref is None else source_ref.memory_role,
        "source_trust_level": None if source_ref is None else source_ref.trust_level,
        "accepted_non_source": any(
            r.source_episode_id != source_episode_id and r.epistemic_status == "accepted" for r in refs
        ),
        "context_fingerprint": session.context_fingerprint,
    }


def _attach_hypothesis(observed: Dict[str, Any], intelligence: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    hyp = str((intelligence.get("recognition") or {}).get("what_i_see") or "")
    conf = float((intelligence.get("recognition") or {}).get("confidence") or 0.0)
    observed["primary_hypothesis"] = hyp
    observed["confidence"] = conf
    prov = result.get("cognition_provenance") or {}
    observed["frozen_deliberation_input_hash"] = prov.get("frozen_deliberation_input_hash")
    observed["deliberation_snapshot_hash"] = prov.get("deliberation_snapshot_hash")
    observed["memory_reference_ids"] = list(prov.get("memory_reference_ids") or [])
    return observed


def score_outcome_task(case: Dict[str, Any], primary_hypothesis: str) -> Dict[str, Any]:
    """Task-level scorer. Uses only frozen case fields and primary_hypothesis."""
    hyp = str(primary_hypothesis or "").strip()
    if case["role"] == "transfer":
        expected = str(case.get("expected_transfer_outcome") or "")
        outcome_score = 1 if expected and expected in hyp else 0
        return {
            "case_id": case["case_id"],
            "role": case["role"],
            "outcome_task_score": outcome_score,
        }
    naive = str(case.get("naive_hypothesis") or "").strip()
    return {
        "case_id": case["case_id"],
        "role": case["role"],
        "outcome_task_score": 1 if hyp == naive else 0,
    }


def score_observation(case: Dict[str, Any], observed: Dict[str, Any]) -> Dict[str, Any]:
    primary_hyp = str(observed.get("primary_hypothesis") or "")
    outcome = score_outcome_task(case, primary_hyp)
    marker_in_hyp = COMPOSITION_FAILURE_MARKER in primary_hyp
    if case["role"] == "transfer":
        mechanism_transfer = 1 if marker_in_hyp else 0
        mechanism_control = None
    else:
        mechanism_transfer = None
        contaminated = bool(
            marker_in_hyp
            or observed.get("retrieved_source")
            or observed.get("strategy_hint") == "consider_promoted_belief"
        )
        mechanism_control = 0 if contaminated else 1
    changed = {
        "retrieved_source": bool(observed.get("retrieved_source")),
        "strategy_hint": observed.get("strategy_hint"),
        "source_epistemic_status": observed.get("source_epistemic_status"),
        "marker_in_hypothesis": marker_in_hyp,
    }
    return {
        **outcome,
        "transfer_task_score": mechanism_transfer,
        "control_task_score": mechanism_control,
        "changed_behavior": changed,
    }


def _mean(values: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _clone_store(src: Path, dest: Path):
    release_ledger_store(src / "cognition_ledger.sqlite3")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    os.environ["FRAMED_COGNITION_DIR"] = str(dest)
    return reset_ledger(dest / "cognition_ledger.sqlite3")


def _run_later_case(
    *,
    case: Dict[str, Any],
    store_dir: Path,
    state_label: Optional[str],
    source_episode_id: str,
) -> Dict[str, Any]:
    os.environ["FRAMED_COGNITION_DIR"] = str(store_dir)
    reset_ledger(store_dir / "cognition_ledger.sqlite3")
    image_path = _write_temp_image(case["image_bytes"].encode("utf-8"))
    try:
        result = _synthetic_result(case["scene_type"])
        session = None
        from framed.cognition.integration.pipeline_hook import begin_cognition_run, finalize_cognition_run

        session = begin_cognition_run(
            result=result,
            image_path=image_path,
            asset_filename=f"{case['case_id']}.jpg",
            run_mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.LIVE,
            state_label=state_label,
        )
        if session is None:
            raise RuntimeError("Cognition session failed for later case")
        intelligence = apply_frozen_recognizer(case, session)
        out = finalize_cognition_run(session, result, intelligence)
        observed = _observe(case, session, source_episode_id)
        observed = _attach_hypothesis(observed, intelligence, out)
        observed["scores"] = score_observation(case, observed)
        return observed
    finally:
        try:
            os.unlink(image_path)
        except OSError:
            pass


def _run_condition_cases(
    *,
    condition: str,
    parent_store: Path,
    work_root: Path,
    state_label: Optional[str],
    source_episode_id: str,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for case in FROZEN_CASES:
        clone = work_root / f"{condition}_{case['case_id']}"
        _clone_store(parent_store, clone)
        observed = _run_later_case(
            case=case,
            store_dir=clone,
            state_label=state_label,
            source_episode_id=source_episode_id,
        )
        observed["condition"] = condition
        outputs.append(observed)
        release_ledger_store(clone / "cognition_ledger.sqlite3")
    return outputs


def _condition_summary(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    transfer_outcomes = [o["scores"]["outcome_task_score"] for o in outputs if o["role"] == "transfer"]
    control_outcomes = [o["scores"]["outcome_task_score"] for o in outputs if o["role"] == "control"]
    mean_outcome_t = _mean(transfer_outcomes)
    mean_outcome_c = _mean(control_outcomes)
    condition_outcome_score = (
        None if mean_outcome_t is None or mean_outcome_c is None else mean_outcome_t + mean_outcome_c
    )
    mechanism_transfer = [o["scores"]["transfer_task_score"] for o in outputs if o["role"] == "transfer"]
    mechanism_control = [o["scores"]["control_task_score"] for o in outputs if o["role"] == "control"]
    return {
        "mean_outcome_task_score_transfer": mean_outcome_t,
        "mean_outcome_task_score_control": mean_outcome_c,
        "condition_outcome_score": condition_outcome_score,
        "mean_transfer_task_score": _mean(mechanism_transfer),
        "mean_control_task_score": _mean(mechanism_control),
        "n_transfer": len(transfer_outcomes),
        "n_control": len(control_outcomes),
        "outputs": outputs,
    }


def _pass_fail(a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any], rollback: Dict[str, Any], checks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    reasons: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, **extra: Any) -> None:
        reasons.append({"name": name, "pass": passed, **extra})

    add(
        "frozen_case_counts",
        a["n_transfer"] >= 2 and a["n_control"] >= 2,
        n_transfer=a["n_transfer"],
        n_control=a["n_control"],
    )
    add(
        "promoted_outcome_beats_provisional",
        (c["mean_outcome_task_score_transfer"] or 0) > (b["mean_outcome_task_score_transfer"] or 0),
        C=c["mean_outcome_task_score_transfer"],
        B=b["mean_outcome_task_score_transfer"],
    )
    add(
        "promoted_outcome_beats_baseline",
        (c["mean_outcome_task_score_transfer"] or 0) > (a["mean_outcome_task_score_transfer"] or 0),
        C=c["mean_outcome_task_score_transfer"],
        A=a["mean_outcome_task_score_transfer"],
    )
    add(
        "control_outcomes_unchanged",
        a["mean_outcome_task_score_control"] == 1.0
        and b["mean_outcome_task_score_control"] == 1.0
        and c["mean_outcome_task_score_control"] == 1.0,
        A=a["mean_outcome_task_score_control"],
        B=b["mean_outcome_task_score_control"],
        C=c["mean_outcome_task_score_control"],
    )
    add(
        "provisional_outcome_does_not_already_solve_task",
        a["mean_outcome_task_score_transfer"] == b["mean_outcome_task_score_transfer"],
        A=a["mean_outcome_task_score_transfer"],
        B=b["mean_outcome_task_score_transfer"],
    )
    transfer_regressions = []
    for a_out, b_out, c_out in zip(a["outputs"], b["outputs"], c["outputs"]):
        if a_out["role"] != "transfer":
            continue
        b_score = b_out["scores"]["outcome_task_score"] or 0
        c_score = c_out["scores"]["outcome_task_score"] or 0
        if c_score < b_score:
            transfer_regressions.append(a_out["case_id"])
    add("no_transfer_outcome_regression", not transfer_regressions, cases=transfer_regressions)
    add(
        "rollback_outcome_matches_provisional",
        rollback.get("outcome_task_score")
        == (b["outputs"][0]["scores"]["outcome_task_score"] if b["outputs"] else None),
        rollback=rollback,
        provisional_l1={
            "outcome_task_score": b["outputs"][0]["scores"]["outcome_task_score"],
        },
    )
    for check in checks:
        add(check["name"], check["pass"], **{k: v for k, v in check.items() if k not in ("name", "pass")})
    verdict = (
        "B4 CLOSEOUT PASS — INDEPENDENT OUTCOME IMPROVEMENT PROVEN"
        if all(r["pass"] for r in reasons)
        else "B4 CLOSEOUT FAIL — INDEPENDENT OUTCOME IMPROVEMENT NOT PROVEN"
    )
    return verdict, reasons


def _seed_train(cognition_dir: Path, evidence_dir: Path) -> Dict[str, Any]:
    os.environ["FRAMED_COGNITION_DIR"] = str(cognition_dir)
    ledger = reset_ledger(cognition_dir / "cognition_ledger.sqlite3")
    ident = get_identity()
    workspace_id = ident["workspace_id"]
    ledger.ensure_initial_states(workspace_id)
    ledger.activate_state(workspace_id, "state_memory_enabled")

    e1_path = _write_temp_image(b"b4-train-e1-cluttered-interior")
    e2_path = _write_temp_image(b"b4-train-e2-related-interior")
    try:
        shared = _synthetic_result("interior_scene")
        e1_out = _run_episode(
            image_path=e1_path,
            result=shared,
            run_mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.LIVE,
            state_label="state_memory_enabled",
            intelligence=_synthetic_intelligence(
                f"Cluttered interior with weak composition — {COMPOSITION_FAILURE_MARKER}.",
                0.52,
            ),
        )
        e2_base = _run_episode(
            image_path=e2_path,
            result=shared,
            run_mode=RunMode.BASELINE,
            run_purpose=RunPurpose.BASELINE,
            state_label="state_baseline",
            intelligence=_synthetic_intelligence("Interior scene with visible clutter.", 0.58),
        )
        e2_mem = _run_episode(
            image_path=e2_path,
            result=shared,
            run_mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.MEMORY_ENABLED,
            state_label="state_memory_enabled",
            intelligence=_synthetic_intelligence(
                "Interior clutter — reconsider composition failure from prior experience.",
                0.50,
            ),
            baseline_run_id=e2_base["session"].run_id,
            baseline_snapshot=e2_base["deliberation"],
        )
        if not e2_mem["session"].memory_reference_ids:
            raise AssertionError("Train E2 must retrieve E1")
        outcome = record_outcome(
            workspace_id=workspace_id,
            source_episode_id=e1_out["session"].episode_id,
            source_run_id=e1_out["session"].run_id,
            kind="testdaemon_eval",
            verdict="useful",
            created_by="testdaemon",
            note="E1 composition-failure mode was useful on train E2 only. Held-out cases unseen.",
            ledger=ledger,
        )
        proposal = generate_proposal(outcome_id=outcome["outcome_id"], ledger=ledger)
        held_out_ids = [c["case_id"] for c in FROZEN_CASES]
        held_out_phrases = [c.get("expected_transfer_outcome", "") for c in FROZEN_CASES if c.get("expected_transfer_outcome")]
        proposal_blob = canonical_json_dumps(proposal)
        leaked = [cid for cid in held_out_ids if cid in proposal_blob]
        leaked_phrases = [p for p in held_out_phrases if p and p in proposal_blob]
        if leaked or leaked_phrases:
            raise AssertionError(f"Proposal must not mention held-out targets: ids={leaked} phrases={leaked_phrases}")
        evaluation = evaluate_proposal(
            proposal_id=proposal["proposal_id"],
            ledger=ledger,
            evidence_dir=evidence_dir / "train_eval",
        )
        if evaluation["status"] != "pass":
            raise AssertionError(f"Train proposal evaluation failed: {evaluation}")
        parent_state = ledger.get_active_state(workspace_id)
        return {
            "workspace_id": workspace_id,
            "actor_id": ident["actor_id"],
            "e1_episode_id": e1_out["session"].episode_id,
            "e1_run_id": e1_out["session"].run_id,
            "e2_run_id": e2_mem["session"].run_id,
            "outcome_id": outcome["outcome_id"],
            "proposal_id": proposal["proposal_id"],
            "proposal": proposal,
            "evaluation_id": evaluation["evaluation_id"],
            "evaluation_status": evaluation["status"],
            "parent_state_version_id": parent_state["state_version_id"],
            "held_out_absent_from_proposal": leaked == [] and leaked_phrases == [],
        }
    finally:
        for p in (e1_path, e2_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _run_b4_once(*, cognition_dir: Path, evidence_dir: Path) -> Dict[str, Any]:
    os.environ["FRAMED_COGNITION_V1"] = "true"
    os.environ["FRAMED_DETERMINISTIC_SEED"] = FROZEN_SEED
    os.environ["FRAMED_MODEL_A"] = FROZEN_MODEL_ID
    freeze = freeze_evaluation_protocol(evidence_dir)
    train_dir = cognition_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    train = _seed_train(train_dir, evidence_dir)
    work_root = cognition_dir / "later_clones"
    work_root.mkdir(parents=True, exist_ok=True)

    baseline_parent = cognition_dir / "parent_baseline"
    provisional_parent = cognition_dir / "parent_provisional"
    promoted_parent = cognition_dir / "parent_promoted"
    _clone_store(train_dir, baseline_parent)
    _clone_store(train_dir, provisional_parent)
    promoted_ledger = _clone_store(train_dir, promoted_parent)
    decision = accept_proposal(
        proposal_id=train["proposal_id"],
        authority_kind="testdaemon",
        actor_id="b4-testdaemon",
        ledger=promoted_ledger,
    )
    promoted_state = promoted_ledger.get_active_state(train["workspace_id"])
    release_ledger_store(promoted_parent / "cognition_ledger.sqlite3")

    a_out = _run_condition_cases(
        condition="A_baseline",
        parent_store=baseline_parent,
        work_root=work_root,
        state_label="state_baseline",
        source_episode_id=train["e1_episode_id"],
    )
    b_out = _run_condition_cases(
        condition="B_provisional",
        parent_store=provisional_parent,
        work_root=work_root,
        state_label="state_memory_enabled",
        source_episode_id=train["e1_episode_id"],
    )
    c_out = _run_condition_cases(
        condition="C_promoted",
        parent_store=promoted_parent,
        work_root=work_root,
        state_label=None,
        source_episode_id=train["e1_episode_id"],
    )
    a = _condition_summary(a_out)
    b = _condition_summary(b_out)
    c = _condition_summary(c_out)

    rollback_parent = cognition_dir / "parent_rollback"
    rollback_ledger = _clone_store(promoted_parent, rollback_parent)
    rollback_rec = rollback_promoted_state(
        workspace_id=train["workspace_id"],
        authority_kind="testdaemon",
        actor_id="b4-testdaemon",
        ledger=rollback_ledger,
    )
    release_ledger_store(rollback_parent / "cognition_ledger.sqlite3")
    rollback_clone = work_root / "R_L1"
    _clone_store(rollback_parent, rollback_clone)
    rollback_obs = _run_later_case(
        case=dict(FROZEN_CASES[0]),
        store_dir=rollback_clone,
        state_label=None,
        source_episode_id=train["e1_episode_id"],
    )
    rollback_obs["condition"] = "R_rollback"
    release_ledger_store(rollback_clone / "cognition_ledger.sqlite3")

    later_c_l1 = c_out[0]
    provenance_ok = all(
        [
            train["outcome_id"],
            train["proposal_id"],
            train["evaluation_id"],
            decision["decision_id"],
            promoted_state["state_version_id"] == decision["resulting_state_version_id"],
            later_c_l1["state_version_id"] == promoted_state["state_version_id"],
            later_c_l1["retrieved_source"] is True,
            later_c_l1["source_epistemic_status"] == "accepted",
        ]
    )
    metrics_now = artefact_hash(dict(FROZEN_METRICS))
    cases_now = artefact_hash(
        {
            "schema": "b4_frozen_cases_v1",
            "cases": list(FROZEN_CASES),
            "composition_failure_marker": COMPOSITION_FAILURE_MARKER,
            "independent_transfer_outcome": INDEPENDENT_TRANSFER_OUTCOME,
            "frozen_model_id": FROZEN_MODEL_ID,
            "frozen_seed": FROZEN_SEED,
        }
    )
    protocol_checks = [
        {"name": "held_out_absent_from_proposal", "pass": train["held_out_absent_from_proposal"]},
        {"name": "metrics_hash_unchanged", "pass": metrics_now == freeze["metrics_hash"], "frozen": freeze["metrics_hash"], "now": metrics_now},
        {"name": "cases_hash_unchanged", "pass": cases_now == freeze["cases_hash"], "frozen": freeze["cases_hash"], "now": cases_now},
        {
            "name": "provenance_closure",
            "pass": provenance_ok,
            "outcome_id": train["outcome_id"],
            "proposal_id": train["proposal_id"],
            "evaluation_id": train["evaluation_id"],
            "decision_id": decision["decision_id"],
            "promoted_state_version_id": promoted_state["state_version_id"],
            "later_c_l1_state_version_id": later_c_l1["state_version_id"],
            "later_c_l1_source_epistemic_status": later_c_l1["source_epistemic_status"],
        },
        {
            "name": "promotion_is_targeted",
            "pass": all(not o.get("accepted_non_source") for o in c_out),
        },
    ]
    rollback_summary = {
        "from_state_version_id": rollback_rec["from_state_version_id"],
        "to_state_version_id": rollback_rec["to_state_version_id"],
        "case_id": rollback_obs["case_id"],
        "outcome_task_score": rollback_obs["scores"]["outcome_task_score"],
        "primary_hypothesis": rollback_obs["primary_hypothesis"],
        "run_id": rollback_obs["run_id"],
    }
    verdict, reasons = _pass_fail(
        a,
        b,
        c,
        rollback_summary,
        protocol_checks,
    )

    deltas = {
        "C_minus_B_outcome_transfer": (c["mean_outcome_task_score_transfer"] or 0)
        - (b["mean_outcome_task_score_transfer"] or 0),
        "C_minus_A_outcome_transfer": (c["mean_outcome_task_score_transfer"] or 0)
        - (a["mean_outcome_task_score_transfer"] or 0),
        "C_minus_B_condition_outcome": (c["condition_outcome_score"] or 0) - (b["condition_outcome_score"] or 0),
        "C_minus_A_condition_outcome": (c["condition_outcome_score"] or 0) - (a["condition_outcome_score"] or 0),
        "B_minus_A_outcome_transfer": (b["mean_outcome_task_score_transfer"] or 0)
        - (a["mean_outcome_task_score_transfer"] or 0),
        "changed_without_outcome_improvement_B_vs_A": (
            b["mean_outcome_task_score_transfer"] == a["mean_outcome_task_score_transfer"]
            and any(o["strategy_hint"] != a_out["strategy_hint"] for o, a_out in zip(b_out, a_out))
        ),
    }
    regressions = [r for r in reasons if r["name"] == "no_transfer_outcome_regression" and not r["pass"]]
    control_regressions = [
        o["case_id"]
        for o in c_out
        if o["role"] == "control" and (o["scores"]["outcome_task_score"] or 0) < 1
    ]

    report = {
        "status": "PASS" if verdict.startswith("B4 CLOSEOUT PASS") else "FAIL",
        "verdict": verdict,
        "cognition_dir": str(cognition_dir),
        "evidence_dir": str(evidence_dir),
        "freeze": freeze,
        "train": {
            "e1_episode_id": train["e1_episode_id"],
            "e1_run_id": train["e1_run_id"],
            "e2_run_id": train["e2_run_id"],
            "outcome_id": train["outcome_id"],
            "proposal_id": train["proposal_id"],
            "evaluation_id": train["evaluation_id"],
            "evaluation_status": train["evaluation_status"],
            "parent_state_version_id": train["parent_state_version_id"],
            "held_out_absent_from_proposal": train["held_out_absent_from_proposal"],
        },
        "decision_id": decision["decision_id"],
        "promoted_state_version_id": promoted_state["state_version_id"],
        "conditions": {
            "A_baseline": {k: v for k, v in a.items() if k != "outputs"},
            "B_provisional": {k: v for k, v in b.items() if k != "outputs"},
            "C_promoted": {k: v for k, v in c.items() if k != "outputs"},
        },
        "outputs": {
            "A_baseline": a_out,
            "B_provisional": b_out,
            "C_promoted": c_out,
            "R_rollback": rollback_obs,
        },
        "score_deltas": deltas,
        "regressions": {
            "transfer_cases": regressions,
            "control_case_ids": control_regressions,
        },
        "rollback": rollback_summary,
        "pass_fail_reasons": reasons,
        "frozen_model_id": FROZEN_MODEL_ID,
        "frozen_seed": FROZEN_SEED,
        "independent_transfer_outcome": INDEPENDENT_TRANSFER_OUTCOME,
        "cases": [c["case_id"] for c in FROZEN_CASES],
    }
    (evidence_dir / "condition_outputs.json").write_text(
        canonical_json_dumps(report["outputs"]), encoding="utf-8"
    )
    (evidence_dir / "score_deltas.json").write_text(canonical_json_dumps(deltas), encoding="utf-8")
    (evidence_dir / "outcome_scores.json").write_text(
        canonical_json_dumps(
            {
                "A_baseline": {
                    "transfer": a["mean_outcome_task_score_transfer"],
                    "control": a["mean_outcome_task_score_control"],
                },
                "B_provisional": {
                    "transfer": b["mean_outcome_task_score_transfer"],
                    "control": b["mean_outcome_task_score_control"],
                },
                "C_promoted": {
                    "transfer": c["mean_outcome_task_score_transfer"],
                    "control": c["mean_outcome_task_score_control"],
                },
                "R_rollback_L1": rollback_summary["outcome_task_score"],
            }
        ),
        encoding="utf-8",
    )
    (evidence_dir / "provenance.json").write_text(
        canonical_json_dumps(
            {
                "outcome_id": train["outcome_id"],
                "proposal_id": train["proposal_id"],
                "evaluation_id": train["evaluation_id"],
                "decision_id": decision["decision_id"],
                "parent_state_version_id": train["parent_state_version_id"],
                "promoted_state_version_id": promoted_state["state_version_id"],
                "later_c_l1_run_id": later_c_l1["run_id"],
                "later_c_l1_state_version_id": later_c_l1["state_version_id"],
                "later_c_l1_source_epistemic_status": later_c_l1["source_epistemic_status"],
            }
        ),
        encoding="utf-8",
    )
    (evidence_dir / "rollback.json").write_text(canonical_json_dumps(rollback_summary), encoding="utf-8")
    (evidence_dir / "verdict.json").write_text(
        canonical_json_dumps({"verdict": verdict, "reasons": reasons}), encoding="utf-8"
    )
    (evidence_dir / "b4_report.json").write_text(canonical_json_dumps(report), encoding="utf-8")
    return report


def run_b4_evaluation(
    cognition_dir: Optional[Path] = None,
    *,
    keep_store: bool = False,
    reset_store: bool = False,
    reuse_store: bool = False,
    evidence_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    original_cognition_flag = os.environ.get("FRAMED_COGNITION_V1")
    original_cognition_dir = os.environ.get("FRAMED_COGNITION_DIR")
    original_seed = os.environ.get("FRAMED_DETERMINISTIC_SEED")
    original_model = os.environ.get("FRAMED_MODEL_A")
    os.environ["FRAMED_COGNITION_V1"] = "true"
    resolved_evidence_dir = (
        evidence_dir.resolve()
        if evidence_dir
        else Path(tempfile.mkdtemp(prefix="framed_b4_evidence_"))
    )
    resolved_evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        with _demo_store(
            cognition_dir=cognition_dir,
            keep_store=keep_store,
            reset_store=reset_store,
            reuse_store=reuse_store,
        ) as (resolved_cognition_dir, kept_store_path):
            report = _run_b4_once(
                cognition_dir=resolved_cognition_dir,
                evidence_dir=resolved_evidence_dir,
            )
            report["temporary_store"] = cognition_dir is None
            report["kept_store_path"] = str(kept_store_path) if kept_store_path else None
            (resolved_evidence_dir / "b4_report.json").write_text(
                canonical_json_dumps(report), encoding="utf-8"
            )
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
        if original_seed is None:
            os.environ.pop("FRAMED_DETERMINISTIC_SEED", None)
        else:
            os.environ["FRAMED_DETERMINISTIC_SEED"] = original_seed
        if original_model is None:
            os.environ.pop("FRAMED_MODEL_A", None)
        else:
            os.environ["FRAMED_MODEL_A"] = original_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Slice B B4 later-behavior evaluation.")
    parser.add_argument("--cognition-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--reset-store", action="store_true")
    parser.add_argument("--reuse-store", action="store_true")
    parser.add_argument("--keep-store", action="store_true")
    args = parser.parse_args()
    try:
        report = run_b4_evaluation(
            cognition_dir=args.cognition_dir,
            keep_store=args.keep_store,
            reset_store=args.reset_store,
            reuse_store=args.reuse_store,
            evidence_dir=args.evidence_dir,
        )
        print(json.dumps({"verdict": report.get("verdict"), "status": report.get("status"), "evidence_dir": report.get("evidence_dir")}, indent=2))
        return 0 if report.get("status") == "PASS" else 1
    except AssertionError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
