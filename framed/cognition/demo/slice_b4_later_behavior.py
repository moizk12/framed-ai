"""B4 held-out later-behavior evaluation.

Proves or fails the claim that promotion produces measurably better later
behavior on untouched later cases. Observable case inputs are separated from
hidden evaluator-only ground truth. Proposal generation never sees held-out cases.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    release_ledger_store,
    reset_ledger,
)
from framed.cognition.learning.authority import accept_proposal
from framed.cognition.learning.evaluation import evaluate_proposal
from framed.cognition.learning.outcomes import record_outcome
from framed.cognition.learning.proposals import generate_proposal
from framed.cognition.learning.rollback import rollback_promoted_state

COMPOSITION_FAILURE_MARKER = "COMPOSITION_FAILURE_MODE"
ACTION_REQUEST_COMPOSITION = "request_composition_evidence"
ACTION_FINALIZE_STANDARD = "finalize_standard_critique"
FROZEN_MODEL_ID = "b4-frozen-recognizer"
FROZEN_SEED = "b4-later-behavior-v1"

# Observable inputs only. Recognizer/controller may read these fields.
FROZEN_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "case_id": "L1",
        "role": "transfer",
        "scene_type": "interior_scene",
        "image_bytes": "b4-later-interior-heldout-1",
        "naive_hypothesis": "Interior scene with visible clutter.",
        "naive_confidence": 0.58,
        "composition_risk_signal": "elevated",
    },
    {
        "case_id": "L2",
        "role": "transfer",
        "scene_type": "interior_scene",
        "image_bytes": "b4-later-interior-heldout-2",
        "naive_hypothesis": "Another interior with dense objects.",
        "naive_confidence": 0.57,
        "composition_risk_signal": "elevated",
    },
    {
        "case_id": "L3",
        "role": "near_miss",
        "scene_type": "interior_scene",
        "image_bytes": "b4-later-interior-heldout-near-miss",
        "naive_hypothesis": "Tidy interior with balanced framing.",
        "naive_confidence": 0.59,
        "composition_risk_signal": "low",
    },
    {
        "case_id": "N1",
        "role": "control",
        "scene_type": "people_scene",
        "image_bytes": "b4-control-street-heldout-1",
        "naive_hypothesis": "Layered street composition with pedestrian flow.",
        "naive_confidence": 0.60,
        "composition_risk_signal": "elevated",
    },
    {
        "case_id": "N2",
        "role": "control",
        "scene_type": "screenshot_ui",
        "image_bytes": "b4-control-screenshot-heldout-1",
        "naive_hypothesis": "Software interface screenshot.",
        "naive_confidence": 0.62,
        "composition_risk_signal": "low",
    },
)

# Evaluator-only oracle. Never passed to recognizer, proposal generation, or runtime controller.
FROZEN_ORACLE: Dict[str, Dict[str, str]] = {
    "L1": {"expected_critique_action": ACTION_REQUEST_COMPOSITION},
    "L2": {"expected_critique_action": ACTION_REQUEST_COMPOSITION},
    "L3": {"expected_critique_action": ACTION_FINALIZE_STANDARD},
    "N1": {"expected_critique_action": ACTION_FINALIZE_STANDARD},
    "N2": {"expected_critique_action": ACTION_FINALIZE_STANDARD},
}

FROZEN_METRICS: Dict[str, Any] = {
    "schema": "b4_later_behavior_metrics_v3",
    "defined_before_results": True,
    "later_task": (
        "Choose critique_action from observable scene features and retrieved belief. "
        "Elevated composition risk plus accepted same-scene composition-failure belief "
        f"should yield {ACTION_REQUEST_COMPOSITION!r}; low-risk interiors and non-interior "
        f"scenes should remain {ACTION_FINALIZE_STANDARD!r}."
    ),
    "observable_vs_hidden": (
        "FROZEN_CASES are observable inputs only. FROZEN_ORACLE is evaluator-only ground "
        "truth and is never read by apply_frozen_recognizer or proposal generation."
    ),
    "frozen_recognizer_policy": (
        "Identical frozen recognizer for all conditions. Default action is finalize_standard_critique. "
        "A retrieved same-scene prior containing the failure marker may license "
        "request_composition_evidence only when epistemic_status is accepted and the observable "
        "composition_risk_signal is elevated. Provisional priors may be noted in hypothesis text "
        "but must not change critique_action. Confidence never increases."
    ),
    "outcome_task_score": (
        "1 iff emitted critique_action equals the hidden oracle expected_critique_action for case_id. "
        "Scorer inspects only critique_action and FROZEN_ORACLE."
    ),
    "outcome_scorer_forbidden_inputs": [
        "COMPOSITION_FAILURE_MODE",
        "retrieval occurrence",
        "epistemic_status",
        "promotion state",
        "strategy_hint",
        "internal provenance fields",
        "observable case fields beyond emitted task output",
    ],
    "condition_outcome_score": "mean(outcome_task_score for transfer) + mean(outcome_task_score for near_miss+control)",
    "closeout_pass_criteria": [
        "At least two transfer cases, one near_miss, and two controls frozen before promoted later runs.",
        "mean(C.outcome_task_score transfer) > mean(B.outcome_task_score transfer)",
        "mean(C.outcome_task_score transfer) > mean(A.outcome_task_score transfer)",
        "mean(A.outcome_task_score near_miss+control) == mean(B) == mean(C) == 1.0",
        "mean(B.outcome_task_score transfer) == mean(A.outcome_task_score transfer)",
        "Near-miss L3 stays correct under promotion (no blind interior-wide action).",
        "No transfer case has C.outcome_task_score < B.outcome_task_score",
        "Rollback later transfer outcome_task_score matches provisional, not promoted.",
        "Held-out case ids and oracle actions absent from proposal payload.",
        "Metrics and oracle hashes at scoring equal hashes frozen before later runs.",
        "Provenance closes outcome → proposal → evaluation → decision → promoted state → later C run.",
    ],
}


def observable_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    """Fields visible to the recognizer/controller under evaluation."""
    return {
        "case_id": case["case_id"],
        "role": case["role"],
        "scene_type": case["scene_type"],
        "image_bytes": case["image_bytes"],
        "naive_hypothesis": case["naive_hypothesis"],
        "naive_confidence": case["naive_confidence"],
        "composition_risk_signal": case["composition_risk_signal"],
    }


def freeze_evaluation_protocol(evidence_dir: Path) -> Dict[str, str]:
    """Persist observable cases, hidden oracle, and metrics before promoted later runs."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = {
        "schema": "b4_frozen_observable_cases_v1",
        "cases": [observable_case(c) for c in FROZEN_CASES],
        "frozen_model_id": FROZEN_MODEL_ID,
        "frozen_seed": FROZEN_SEED,
    }
    oracle_payload = {
        "schema": "b4_frozen_oracle_v1",
        "oracle": dict(FROZEN_ORACLE),
    }
    metrics_payload = dict(FROZEN_METRICS)
    cases_hash = artefact_hash(cases_payload)
    oracle_hash = artefact_hash(oracle_payload)
    metrics_hash = artefact_hash(metrics_payload)
    (evidence_dir / "frozen_cases.json").write_text(canonical_json_dumps(cases_payload), encoding="utf-8")
    (evidence_dir / "frozen_oracle.json").write_text(canonical_json_dumps(oracle_payload), encoding="utf-8")
    (evidence_dir / "frozen_metrics.json").write_text(canonical_json_dumps(metrics_payload), encoding="utf-8")
    freeze_record = {
        "schema": "b4_freeze_record_v1",
        "cases_hash": cases_hash,
        "oracle_hash": oracle_hash,
        "metrics_hash": metrics_hash,
        "case_ids": [c["case_id"] for c in FROZEN_CASES],
        "frozen_before_promoted_later_runs": True,
    }
    (evidence_dir / "freeze_record.json").write_text(canonical_json_dumps(freeze_record), encoding="utf-8")
    return {"cases_hash": cases_hash, "oracle_hash": oracle_hash, "metrics_hash": metrics_hash}


def _synthetic_intelligence_with_action(
    hypothesis: str,
    confidence: float,
    critique_action: str,
) -> Dict[str, Any]:
    intel = _synthetic_intelligence(hypothesis, confidence)
    intel["recognition"]["critique_action"] = critique_action
    return intel


def apply_frozen_recognizer(observable: Mapping[str, Any], session: Any) -> Dict[str, Any]:
    """Frozen identical model. Reads observable inputs and cognition context only."""
    hypothesis = str(observable["naive_hypothesis"])
    confidence = float(observable["naive_confidence"])
    critique_action = ACTION_FINALIZE_STANDARD
    risk = str(observable.get("composition_risk_signal") or "low")
    refs = list(session.deliberation_context.memory_references)
    for ref in refs:
        summary = ref.hypothesis_summary or ""
        if ref.scene_signature != observable["scene_type"]:
            continue
        if COMPOSITION_FAILURE_MARKER not in summary:
            continue
        if ref.epistemic_status == "accepted" and risk == "elevated":
            critique_action = ACTION_REQUEST_COMPOSITION
            confidence = min(confidence, 0.50)
        elif ref.epistemic_status == "provisional":
            hypothesis = f"{hypothesis} Provisional prior noted; not adopted as belief."
            confidence = min(confidence, 0.50)
        break
    return _synthetic_intelligence_with_action(hypothesis, confidence, critique_action)


def _observe(case: Dict[str, Any], session: Any, source_episode_id: str) -> Dict[str, Any]:
    refs = list(session.deliberation_context.memory_references)
    source_ref = next((r for r in refs if r.source_episode_id == source_episode_id), None)
    return {
        "case_id": case["case_id"],
        "role": case["role"],
        "scene_type": case["scene_type"],
        "composition_risk_signal": case["composition_risk_signal"],
        "run_id": session.run_id,
        "episode_id": session.episode_id,
        "state_version_id": session.state_version_id,
        "strategy_hint": session.deliberation_context.strategy_hint or "standard",
        "requested_evidence": list(session.deliberation_context.requested_evidence),
        "prior_hypothesis": session.deliberation_context.prior_hypothesis,
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


def _attach_task_output(observed: Dict[str, Any], intelligence: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    recognition = intelligence.get("recognition") or {}
    observed["primary_hypothesis"] = str(recognition.get("what_i_see") or "")
    observed["critique_action"] = str(recognition.get("critique_action") or "")
    observed["confidence"] = float(recognition.get("confidence") or 0.0)
    prov = result.get("cognition_provenance") or {}
    observed["frozen_deliberation_input_hash"] = prov.get("frozen_deliberation_input_hash")
    observed["deliberation_snapshot_hash"] = prov.get("deliberation_snapshot_hash")
    observed["memory_reference_ids"] = list(prov.get("memory_reference_ids") or [])
    return observed


def score_outcome_task(case_id: str, critique_action: str, oracle: Optional[Mapping[str, Mapping[str, str]]] = None) -> Dict[str, Any]:
    """Task-level scorer. Uses only emitted critique_action and hidden oracle."""
    oracle = oracle or FROZEN_ORACLE
    expected = str((oracle.get(case_id) or {}).get("expected_critique_action") or "")
    emitted = str(critique_action or "")
    return {
        "case_id": case_id,
        "outcome_task_score": 1 if expected and emitted == expected else 0,
        "expected_critique_action": expected,
        "emitted_critique_action": emitted,
    }


def score_observation(case: Dict[str, Any], observed: Dict[str, Any]) -> Dict[str, Any]:
    outcome = score_outcome_task(case["case_id"], observed.get("critique_action", ""))
    primary_hyp = str(observed.get("primary_hypothesis") or "")
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
        "critique_action": observed.get("critique_action"),
    }
    return {
        **outcome,
        "role": case["role"],
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
        intelligence = apply_frozen_recognizer(observable_case(case), session)
        out = finalize_cognition_run(session, result, intelligence)
        observed = _observe(case, session, source_episode_id)
        observed = _attach_task_output(observed, intelligence, out)
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
    guard_outcomes = [o["scores"]["outcome_task_score"] for o in outputs if o["role"] in ("near_miss", "control")]
    mean_outcome_t = _mean(transfer_outcomes)
    mean_outcome_g = _mean(guard_outcomes)
    condition_outcome_score = (
        None if mean_outcome_t is None or mean_outcome_g is None else mean_outcome_t + mean_outcome_g
    )
    mechanism_transfer = [o["scores"]["transfer_task_score"] for o in outputs if o["role"] == "transfer"]
    mechanism_control = [o["scores"]["control_task_score"] for o in outputs if o["role"] in ("near_miss", "control")]
    return {
        "mean_outcome_task_score_transfer": mean_outcome_t,
        "mean_outcome_task_score_guard": mean_outcome_g,
        "condition_outcome_score": condition_outcome_score,
        "mean_transfer_task_score": _mean(mechanism_transfer),
        "mean_control_task_score": _mean(mechanism_control),
        "n_transfer": len(transfer_outcomes),
        "n_guard": len(guard_outcomes),
        "outputs": outputs,
    }


def _pass_fail(
    a: Dict[str, Any],
    b: Dict[str, Any],
    c: Dict[str, Any],
    rollback: Dict[str, Any],
    checks: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    reasons: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, **extra: Any) -> None:
        reasons.append({"name": name, "pass": passed, **extra})

    add(
        "frozen_case_counts",
        a["n_transfer"] >= 2 and a["n_guard"] >= 3,
        n_transfer=a["n_transfer"],
        n_guard=a["n_guard"],
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
        "guard_outcomes_unchanged",
        a["mean_outcome_task_score_guard"] == 1.0
        and b["mean_outcome_task_score_guard"] == 1.0
        and c["mean_outcome_task_score_guard"] == 1.0,
        A=a["mean_outcome_task_score_guard"],
        B=b["mean_outcome_task_score_guard"],
        C=c["mean_outcome_task_score_guard"],
    )
    l3_c = next((o for o in c["outputs"] if o["case_id"] == "L3"), None)
    add(
        "near_miss_not_blindly_promoted",
        l3_c is not None and l3_c.get("critique_action") == ACTION_FINALIZE_STANDARD,
        L3_action=None if l3_c is None else l3_c.get("critique_action"),
    )
    add(
        "provisional_outcome_does_not_already_solve_task",
        a["mean_outcome_task_score_transfer"] == b["mean_outcome_task_score_transfer"],
        A=a["mean_outcome_task_score_transfer"],
        B=b["mean_outcome_task_score_transfer"],
    )
    transfer_regressions = []
    for _a_out, b_out, c_out in zip(a["outputs"], b["outputs"], c["outputs"]):
        if _a_out["role"] != "transfer":
            continue
        b_score = b_out["scores"]["outcome_task_score"] or 0
        c_score = c_out["scores"]["outcome_task_score"] or 0
        if c_score < b_score:
            transfer_regressions.append(_a_out["case_id"])
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
        "B4 CLOSEOUT PASS — LEAKAGE-FREE INDEPENDENT OUTCOME IMPROVEMENT PROVEN"
        if all(r["pass"] for r in reasons)
        else "B4 CLOSEOUT FAIL — INDEPENDENT OUTCOME IMPROVEMENT NOT PROVEN"
    )
    return verdict, reasons


def _oracle_leak_targets() -> List[str]:
    targets = list(FROZEN_ORACLE.keys())
    for entry in FROZEN_ORACLE.values():
        targets.append(entry["expected_critique_action"])
    return targets


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
        proposal_blob = canonical_json_dumps(proposal)
        leaked_ids = [cid for cid in FROZEN_ORACLE if cid in proposal_blob]
        leaked_targets = [t for t in _oracle_leak_targets() if t in proposal_blob]
        if leaked_ids or leaked_targets:
            raise AssertionError(f"Proposal must not mention held-out targets: ids={leaked_ids} targets={leaked_targets}")
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
            "held_out_absent_from_proposal": leaked_ids == [] and leaked_targets == [],
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
            "schema": "b4_frozen_observable_cases_v1",
            "cases": [observable_case(c) for c in FROZEN_CASES],
            "frozen_model_id": FROZEN_MODEL_ID,
            "frozen_seed": FROZEN_SEED,
        }
    )
    oracle_now = artefact_hash({"schema": "b4_frozen_oracle_v1", "oracle": dict(FROZEN_ORACLE)})
    protocol_checks = [
        {"name": "held_out_absent_from_proposal", "pass": train["held_out_absent_from_proposal"]},
        {"name": "metrics_hash_unchanged", "pass": metrics_now == freeze["metrics_hash"], "frozen": freeze["metrics_hash"], "now": metrics_now},
        {"name": "cases_hash_unchanged", "pass": cases_now == freeze["cases_hash"], "frozen": freeze["cases_hash"], "now": cases_now},
        {"name": "oracle_hash_unchanged", "pass": oracle_now == freeze["oracle_hash"], "frozen": freeze["oracle_hash"], "now": oracle_now},
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
        "critique_action": rollback_obs["critique_action"],
        "run_id": rollback_obs["run_id"],
    }
    verdict, reasons = _pass_fail(a, b, c, rollback_summary, protocol_checks)

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
    guard_regressions = [
        o["case_id"]
        for o in c_out
        if o["role"] in ("near_miss", "control") and (o["scores"]["outcome_task_score"] or 0) < 1
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
            "guard_case_ids": guard_regressions,
        },
        "rollback": rollback_summary,
        "pass_fail_reasons": reasons,
        "frozen_model_id": FROZEN_MODEL_ID,
        "frozen_seed": FROZEN_SEED,
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
                    "guard": a["mean_outcome_task_score_guard"],
                },
                "B_provisional": {
                    "transfer": b["mean_outcome_task_score_transfer"],
                    "guard": b["mean_outcome_task_score_guard"],
                },
                "C_promoted": {
                    "transfer": c["mean_outcome_task_score_transfer"],
                    "guard": c["mean_outcome_task_score_guard"],
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
