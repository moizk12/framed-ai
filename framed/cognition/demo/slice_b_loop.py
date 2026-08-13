"""Slice B E1→E2→outcome→proposal→eval→accept→E3→rollback demonstration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from framed.cognition.contracts.memory import RetrievalQuery
from framed.cognition.contracts.runs import RunMode, RunPurpose
from framed.cognition.demo.slice_a_e1_e2 import (
    _demo_store,
    _run_episode,
    _synthetic_intelligence,
    _synthetic_result,
    _write_temp_image,
)
from framed.cognition.identity import get_identity
from framed.cognition.integration.pipeline_hook import asset_id_from_path
from framed.cognition.ledger.artefact_store import canonical_json_dumps
from framed.cognition.ledger.sqlite_store import clear_ledger, get_ledger, reset_ledger
from framed.cognition.learning.authority import accept_proposal
from framed.cognition.learning.evaluation import evaluate_proposal
from framed.cognition.learning.outcomes import record_outcome
from framed.cognition.learning.proposals import generate_proposal
from framed.cognition.learning.rollback import rollback_promoted_state
from framed.cognition.retrieval.service import retrieve_memories


def _query(workspace_id: str, actor_id: str, asset_id: str) -> RetrievalQuery:
    return RetrievalQuery(
        workspace_id=workspace_id,
        actor_id=actor_id,
        asset_id=asset_id,
        goal_type="critique",
        goal_instance_id=None,
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
    )


def _run_slice_b_demo_once(*, cognition_dir: Path, evidence_dir: Path) -> Dict[str, Any]:
    os.environ["FRAMED_COGNITION_DIR"] = str(cognition_dir)
    ledger = reset_ledger(cognition_dir / "cognition_ledger.sqlite3")
    ident = get_identity()
    workspace_id = ident["workspace_id"]
    actor_id = ident["actor_id"]
    ledger.ensure_initial_states(workspace_id)
    ledger.activate_state(workspace_id, "state_memory_enabled")

    e1_path = _write_temp_image(b"slice_b_e1_cluttered_room")
    e2_path = _write_temp_image(b"slice_b_e2_related_interior")
    e3_path = _write_temp_image(b"slice_b_e3_later_interior")
    try:
        shared_result = _synthetic_result("interior_scene")
        e1_out = _run_episode(
            image_path=e1_path,
            result=shared_result,
            run_mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.LIVE,
            state_label="state_memory_enabled",
            intelligence=_synthetic_intelligence(
                "Cluttered interior with weak composition — prior failure mode noted.",
                0.52,
            ),
        )
        e1_episode_id = e1_out["session"].episode_id
        e1_run_id = e1_out["session"].run_id

        e2_base = _run_episode(
            image_path=e2_path,
            result=shared_result,
            run_mode=RunMode.BASELINE,
            run_purpose=RunPurpose.BASELINE,
            state_label="state_baseline",
            intelligence=_synthetic_intelligence("Interior scene with visible clutter.", 0.58),
        )
        e2_mem = _run_episode(
            image_path=e2_path,
            result=shared_result,
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
        assert e2_mem["session"].memory_reference_ids, "E2 must retrieve E1"
        assert e2_mem["session"].deliberation_context.strategy_hint == "consider_prior_provisional_experience"
        e2_refs = [
            r
            for r in ledger.export_replay_bundle([e2_mem["session"].run_id])["memory_references"]
            if r["source_episode_id"] == e1_episode_id
        ]
        assert e2_refs, "E2 must retrieve E1 as a memory reference"
        assert e2_refs[0]["epistemic_status"] == "provisional"
        pre_promotion_state = ledger.get_active_state(workspace_id)
        assert e1_episode_id not in (pre_promotion_state["snapshot"].get("promoted_episode_ids") or [])

        outcome = record_outcome(
            workspace_id=workspace_id,
            source_episode_id=e1_episode_id,
            source_run_id=e1_run_id,
            kind="testdaemon_eval",
            verdict="useful",
            created_by="testdaemon",
            note="E1 prior failure mode was useful for E2 deliberation.",
            ledger=ledger,
        )
        proposal = generate_proposal(outcome_id=outcome["outcome_id"], ledger=ledger)
        assert proposal["created_by"] == "proposal_generator"
        assert ledger.get_active_state(workspace_id)["state_version_id"] == pre_promotion_state["state_version_id"]

        evaluation = evaluate_proposal(
            proposal_id=proposal["proposal_id"],
            ledger=ledger,
            evidence_dir=evidence_dir,
        )
        assert evaluation["status"] == "pass", evaluation

        decision = accept_proposal(
            proposal_id=proposal["proposal_id"],
            authority_kind="testdaemon",
            actor_id="slice-b-testdaemon",
            ledger=ledger,
        )
        promoted_state = ledger.get_active_state(workspace_id)
        assert promoted_state["state_version_id"] == decision["resulting_state_version_id"]
        assert e1_episode_id in promoted_state["snapshot"]["promoted_episode_ids"]

        e3_out = _run_episode(
            image_path=e3_path,
            result=shared_result,
            run_mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.LIVE,
            state_label=None,
            intelligence=_synthetic_intelligence(
                "Later interior — consume promoted belief from E1.",
                0.50,
            ),
        )
        assert e3_out["session"].state_version_id == promoted_state["state_version_id"]
        assert e3_out["session"].deliberation_context.strategy_hint == "consider_promoted_belief"
        e3_refs = [
            r
            for r in ledger.export_replay_bundle([e3_out["session"].run_id])["memory_references"]
            if r["source_episode_id"] == e1_episode_id
        ]
        assert e3_refs, "E3 must retrieve promoted E1"
        assert e3_refs[0]["epistemic_status"] == "accepted"
        assert e3_refs[0]["memory_role"] == "promoted_belief"
        assert e3_refs[0]["trust_level"] == "medium"

        db_path = ledger.db_path
        artefact_root = ledger.artefacts.root
        clear_ledger()
        restarted = reset_ledger(db_path, artefact_root=artefact_root)
        restarted_state = restarted.get_active_state(workspace_id)
        assert restarted_state["state_version_id"] == promoted_state["state_version_id"]
        assert e1_episode_id in restarted_state["snapshot"]["promoted_episode_ids"]

        rollback = rollback_promoted_state(
            workspace_id=workspace_id,
            authority_kind="testdaemon",
            actor_id="slice-b-testdaemon",
            ledger=restarted,
        )
        restored = restarted.get_active_state(workspace_id)
        assert restored["state_version_id"] == pre_promotion_state["state_version_id"]
        assert restored["state_version_id"] == rollback["to_state_version_id"]
        later_q = _query(workspace_id, actor_id, asset_id_from_path(e3_path))
        rolled = retrieve_memories(later_q, ledger=restarted, state_snapshot=restored["snapshot"])
        rolled_e1 = next((r for r in rolled.references if r.source_episode_id == e1_episode_id), None)
        assert rolled_e1 is not None
        assert rolled_e1.epistemic_status == "provisional"

        report = {
            "status": "PASS",
            "cognition_dir": str(cognition_dir),
            "evidence_dir": str(evidence_dir),
            "e1_episode_id": e1_episode_id,
            "e1_run_id": e1_run_id,
            "e2_run_id": e2_mem["session"].run_id,
            "e3_run_id": e3_out["session"].run_id,
            "e3_state_version_id": e3_out["session"].state_version_id,
            "outcome_id": outcome["outcome_id"],
            "proposal_id": proposal["proposal_id"],
            "evaluation_id": evaluation["evaluation_id"],
            "decision_id": decision["decision_id"],
            "promoted_state_version_id": promoted_state["state_version_id"],
            "parent_state_version_id": pre_promotion_state["state_version_id"],
            "restored_state_version_id": restored["state_version_id"],
            "promoted_episode_ids": promoted_state["snapshot"]["promoted_episode_ids"],
            "e2_epistemic_status": e2_refs[0]["epistemic_status"],
            "e3_epistemic_status": e3_refs[0]["epistemic_status"],
            "e3_strategy_hint": e3_out["session"].deliberation_context.strategy_hint,
            "rollback_epistemic_status": rolled_e1.epistemic_status,
            "learned": {
                "kind": "promote_episode_belief",
                "source_episode_id": e1_episode_id,
                "belief_policy_version": promoted_state["snapshot"].get("belief_policy_version"),
            },
        }
        (evidence_dir / "slice_b_demo_report.json").write_text(
            canonical_json_dumps(report), encoding="utf-8"
        )
        return report
    finally:
        for p in (e1_path, e2_path, e3_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def run_slice_b_demo(
    cognition_dir: Optional[Path] = None,
    *,
    keep_store: bool = False,
    reset_store: bool = False,
    reuse_store: bool = False,
    evidence_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    original_cognition_flag = os.environ.get("FRAMED_COGNITION_V1")
    original_cognition_dir = os.environ.get("FRAMED_COGNITION_DIR")
    os.environ["FRAMED_COGNITION_V1"] = "true"
    resolved_evidence_dir = (
        evidence_dir.resolve()
        if evidence_dir
        else Path(tempfile.mkdtemp(prefix="framed_slice_b_demo_evidence_"))
    )
    resolved_evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        with _demo_store(
            cognition_dir=cognition_dir,
            keep_store=keep_store,
            reset_store=reset_store,
            reuse_store=reuse_store,
        ) as (resolved_cognition_dir, kept_store_path):
            report = _run_slice_b_demo_once(
                cognition_dir=resolved_cognition_dir,
                evidence_dir=resolved_evidence_dir,
            )
            report["temporary_store"] = cognition_dir is None
            report["kept_store_path"] = str(kept_store_path) if kept_store_path else None
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Slice B controlled-learning demo.")
    parser.add_argument("--cognition-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--reset-store", action="store_true")
    parser.add_argument("--reuse-store", action="store_true")
    parser.add_argument("--keep-store", action="store_true")
    args = parser.parse_args()
    try:
        report = run_slice_b_demo(
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
