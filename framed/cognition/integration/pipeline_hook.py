"""Pipeline cognition hook and legacy write gate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from framed.cognition.config import cognition_enabled
from framed.cognition.constants import PERCEPTION_SNAPSHOT_SCHEMA
from framed.cognition.context.builder import (
    DeliberationContext,
    build_deliberation_context,
    compare_deliberation_snapshots,
    compute_deliberation_delta,
    snapshot_to_legacy_dict,
)
from framed.cognition.context.formatting import build_cognition_context
from framed.cognition.contracts.memory import RetrievalQuery
from framed.cognition.contracts.runs import (
    CognitiveRun,
    RunMode,
    RunPurpose,
    SameAssetPolicy,
    is_retrieval_eligible,
    purpose_from_mode,
    validate_mode_purpose,
)
from framed.cognition.contracts.snapshot import DeliberationSnapshot, snapshot_from_intelligence
from framed.cognition.identity import get_identity
from framed.cognition.ledger.artefact_store import ArtefactStore, artefact_hash
from framed.cognition.ledger.sqlite_store import get_ledger
from framed.cognition.retrieval.service import retrieve_memories


@dataclass
class CognitionSession:
    episode_id: str
    run_id: str
    actor_id: str
    workspace_id: str
    asset_id: str
    state_version_id: str
    run_mode: RunMode
    run_purpose: RunPurpose
    retrieval_enabled: bool
    baseline_run_id: Optional[str] = None
    comparison_group_id: Optional[str] = None
    deliberation_context: DeliberationContext = field(default_factory=DeliberationContext)
    perception_artefact_hash: Optional[str] = None
    memory_reference_ids: List[str] = field(default_factory=list)
    context_fingerprint: Optional[str] = None
    cognition_context: Optional[Dict[str, Any]] = None
    baseline_snapshot: Optional[DeliberationSnapshot] = None
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    confidence_provenance: Dict[str, Any] = field(default_factory=dict)


def legacy_writes_allowed() -> bool:
    return not cognition_enabled()


def build_perception_snapshot_v1(result: Dict[str, Any]) -> Dict[str, Any]:
    ve = result.get("visual_evidence") or {}
    sg = ve.get("scene_gate") or {}
    anchors = result.get("semantic_anchors") if isinstance(result.get("semantic_anchors"), dict) else {}
    return {
        "schema": PERCEPTION_SNAPSHOT_SCHEMA,
        "scene_type": sg.get("scene_type"),
        "signals": sg.get("signals"),
        "category": anchors.get("scene_type"),
    }


def perception_artefact_from_result(result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    payload = build_perception_snapshot_v1(result)
    return artefact_hash(payload), payload


def build_provenance_manifest(
    *,
    state_version_id: str,
    state_snapshot_hash: Optional[str],
    prompt_provenance: Dict[str, Any],
    model_provenance: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": "provenance_manifest_v1",
        "code_commit": os.getenv("FRAMED_CODE_COMMIT", "local"),
        "python_version": platform.python_version(),
        "schema_version": 3,
        "model_provenance": model_provenance,
        "prompt_provenance": prompt_provenance,
        "retrieval_policy_version": "v1",
        "state_version_id": state_version_id,
        "state_snapshot_hash": state_snapshot_hash,
        "feature_flags": {
            "FRAMED_COGNITION_V1": cognition_enabled(),
        },
        "deterministic_seed": os.getenv("FRAMED_DETERMINISTIC_SEED"),
    }


def asset_id_from_path(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _signatures_from_result(result: Dict[str, Any]) -> tuple[str, str]:
    ve = result.get("visual_evidence") or {}
    sg = ve.get("scene_gate") or {}
    scene = str(sg.get("scene_type") or "unknown")
    category = scene
    if scene in ("interior_scene", "object_dense"):
        category = "cluttered_room_weak_composition"
    elif scene == "screenshot_ui":
        category = "screenshot_or_ui_image"
    elif scene == "people_scene":
        category = "layered_street_composition"
    elif scene == "abstract_art":
        category = "fine_art_reproduction"
    return scene, category


def _same_asset_policy_from_snapshot(snap: Dict[str, Any]) -> SameAssetPolicy:
    raw = str(snap.get("same_asset_policy") or "exclude")
    try:
        return SameAssetPolicy(raw)
    except ValueError:
        if raw == "allow_retake":
            return SameAssetPolicy.ALLOW_RELATED_REVISION
        return SameAssetPolicy.EXCLUDE


def _context_fingerprint(
    *,
    perception_hash: str,
    state_version_id: str,
    retrieval_enabled: bool,
    memory_ref_ids: List[str],
    run_mode: RunMode,
    run_purpose: RunPurpose,
) -> str:
    return artefact_hash(
        {
            "perception": perception_hash,
            "state_version_id": state_version_id,
            "retrieval_enabled": retrieval_enabled,
            "memory_ref_ids": sorted(memory_ref_ids),
            "run_mode": run_mode.value,
            "run_purpose": run_purpose.value,
        }
    )


def begin_cognition_run(
    *,
    result: Dict[str, Any],
    image_path: str,
    asset_filename: Optional[str],
    goal_type: str = "critique",
    goal_instance_id: Optional[str] = None,
    run_mode: RunMode = RunMode.MEMORY_ENABLED,
    run_purpose: Optional[RunPurpose] = None,
    state_label: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    comparison_group_id: Optional[str] = None,
    exclude_episode_ids: Optional[Tuple[str, ...]] = None,
    exclude_run_ids: Optional[Tuple[str, ...]] = None,
    same_asset_policy: Optional[SameAssetPolicy] = None,
) -> Optional[CognitionSession]:
    if not cognition_enabled():
        return None
    ledger = get_ledger()
    ident = get_identity()
    actor_id = ident["actor_id"]
    workspace_id = ident["workspace_id"]
    ledger.ensure_demo_states(workspace_id)
    if state_label:
        state_version_id = ledger.activate_state(workspace_id, state_label)
    else:
        state = ledger.get_active_state(workspace_id)
        state_version_id = state["state_version_id"]
    state = ledger.get_active_state(workspace_id)
    snap = state["snapshot"]
    purpose = run_purpose or purpose_from_mode(run_mode)
    if run_mode == RunMode.BASELINE and run_purpose is None:
        purpose = RunPurpose.BASELINE
    elif run_mode == RunMode.CONTROL and run_purpose is None:
        purpose = RunPurpose.CONTROL
    elif run_mode == RunMode.REPLAY and run_purpose is None:
        purpose = RunPurpose.REPLAY
    validate_mode_purpose(run_mode, purpose)
    retrieval_enabled = (
        bool(snap.get("retrieval_enabled", True))
        and purpose not in (RunPurpose.BASELINE, RunPurpose.CONTROL, RunPurpose.REPLAY)
    )
    asset_id = asset_id_from_path(image_path)
    episode_id = ledger.open_episode(
        workspace_id=workspace_id,
        actor_id=actor_id,
        asset_id=asset_id,
        goal_type=goal_type,
        goal_instance_id=goal_instance_id,
        state_version_id=state_version_id,
        asset_filename=asset_filename,
    )
    run_id = str(uuid.uuid4())
    perception_hash, perception_payload = perception_artefact_from_result(result)
    ledger.put_artefact("perception_snapshot", "v1", perception_payload)
    scene_sig, cat_sig = _signatures_from_result(result)
    refs: List[Any] = []
    rejected: List[Dict[str, Any]] = []
    retrieval_candidates: List[Dict[str, Any]] = []
    policy = same_asset_policy or _same_asset_policy_from_snapshot(snap)
    ex_episodes = tuple(exclude_episode_ids or ())
    ex_runs = tuple(exclude_run_ids or ())
    if baseline_run_id:
        baseline_run = ledger.get_run(baseline_run_id)
        if baseline_run:
            ex_episodes = ex_episodes + (baseline_run["episode_id"],)
            ex_runs = ex_runs + (baseline_run_id,)
    if retrieval_enabled:
        q = RetrievalQuery(
            workspace_id=workspace_id,
            actor_id=actor_id,
            asset_id=asset_id,
            goal_type=goal_type,
            goal_instance_id=goal_instance_id,
            scene_signature=scene_sig,
            category_signature=cat_sig,
            exclude_episode_ids=ex_episodes + (episode_id,),
            exclude_run_ids=ex_runs,
            comparison_group_id=comparison_group_id,
            same_asset_policy=policy,
        )
        retrieval = retrieve_memories(q, ledger=ledger, state_snapshot=snap)
        refs = retrieval.references
        retrieval_candidates = retrieval.candidates
        rejected = list(retrieval.rejected)
        if baseline_run_id:
            br = ledger.get_run(baseline_run_id)
            if br:
                rejected.append(
                    {
                        "episode_id": br["episode_id"],
                        "source_run_id": baseline_run_id,
                        "run_purpose": br.get("run_purpose"),
                        "asset_id": asset_id,
                        "rejection_reason": "excluded_by_experiment",
                        "excluded_by_experiment": True,
                        "ineligible_run_purpose": br.get("run_purpose"),
                        "same_asset": True,
                    }
                )
    ctx_fp = _context_fingerprint(
        perception_hash=perception_hash,
        state_version_id=state_version_id,
        retrieval_enabled=retrieval_enabled,
        memory_ref_ids=[r.memory_ref_id for r in refs],
        run_mode=run_mode,
        run_purpose=purpose,
    )
    run = CognitiveRun(
        run_id=run_id,
        episode_id=episode_id,
        mode=run_mode,
        run_purpose=purpose,
        state_version_id=state_version_id,
        context_fingerprint=ctx_fp,
        retrieval_enabled=retrieval_enabled,
        model_provenance={"model": os.getenv("FRAMED_MODEL_A", "default"), "seed": os.getenv("FRAMED_DETERMINISTIC_SEED")},
        prompt_provenance={"policy": "slice_a_v1"},
        started_at=ArtefactStore.utc_now(),
        baseline_run_id=baseline_run_id,
        comparison_group_id=comparison_group_id,
        retrieval_eligible=is_retrieval_eligible(purpose),
    )
    provenance_manifest = build_provenance_manifest(
        state_version_id=state_version_id,
        state_snapshot_hash=state.get("snapshot") and artefact_hash(state["snapshot"]),
        prompt_provenance=run.prompt_provenance,
        model_provenance=run.model_provenance,
    )
    ledger.create_run(run, provenance_manifest=provenance_manifest)
    ledger.append_event(
        episode_id=episode_id,
        run_id=run_id,
        event_type="experience_opened",
        payload={
            "goal_type": goal_type,
            "asset_id": asset_id,
            "mode": run_mode.value,
            "run_purpose": purpose.value,
            "baseline_run_id": baseline_run_id,
            "comparison_group_id": comparison_group_id,
        },
    )
    ledger.append_event(
        episode_id=episode_id,
        run_id=run_id,
        event_type="perception_completed",
        payload={"perception_artefact_hash": perception_hash},
        artefact_hash=perception_hash,
    )
    if retrieval_enabled:
        ledger.append_event(
            episode_id=episode_id,
            run_id=run_id,
            event_type="retrieval_performed",
            payload={
                "candidates": retrieval_candidates,
                "selected": [r.memory_ref_id for r in refs],
                "rejected": rejected,
            },
        )
        for ref in refs:
            ledger.store_memory_reference(run_id, ref, episode_id)
    baseline_hypothesis = "I see a scene worth naming with care."
    baseline_confidence = 0.55
    ctx = build_deliberation_context(refs, baseline_hypothesis, baseline_confidence)
    cognition_context = build_cognition_context(refs)
    baseline_snapshot: Optional[DeliberationSnapshot] = None
    if purpose == RunPurpose.BASELINE:
        pass
    elif baseline_run_id or purpose == RunPurpose.MEMORY_ENABLED:
        legacy = ledger.find_compatible_baseline_snapshot(
            workspace_id,
            asset_id,
            perception_hash,
            baseline_run_id=baseline_run_id,
        )
        if legacy:
            baseline_snapshot = DeliberationSnapshot(
                primary_hypothesis=str(legacy.get("primary_hypothesis", "")),
                confidence=float(legacy.get("confidence", 0.5)),
                strategy=str(legacy.get("strategy", "standard")),
                requested_evidence=list(legacy.get("requested_evidence") or []),
                perception_artefact_hash=perception_hash,
                scene_signature=scene_sig,
                category_signature=cat_sig,
                run_id=baseline_run_id or legacy.get("run_id"),
            )
    return CognitionSession(
        episode_id=episode_id,
        run_id=run_id,
        actor_id=actor_id,
        workspace_id=workspace_id,
        asset_id=asset_id,
        state_version_id=state_version_id,
        run_mode=run_mode,
        run_purpose=purpose,
        retrieval_enabled=retrieval_enabled,
        baseline_run_id=baseline_run_id,
        comparison_group_id=comparison_group_id,
        deliberation_context=ctx,
        perception_artefact_hash=perception_hash,
        memory_reference_ids=[r.memory_ref_id for r in refs],
        context_fingerprint=ctx_fp,
        cognition_context=cognition_context,
        baseline_snapshot=baseline_snapshot,
        rejected_candidates=rejected,
    )


def fail_cognition_run(
    session: CognitionSession,
    *,
    error_code: str,
    safe_message: str,
    stage: str,
    internal_exception_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Mark cognition run failed without indexing partial success as retrievable memory."""
    ledger = get_ledger()
    ledger.append_event(
        episode_id=session.episode_id,
        run_id=session.run_id,
        event_type="run_failed",
        payload={
            "error_code": error_code,
            "safe_message": safe_message,
            "stage": stage,
            "internal_exception_type": internal_exception_type,
            "run_purpose": session.run_purpose.value,
        },
    )
    ledger.fail_episode(
        session.episode_id,
        failure_code=error_code,
        failure_message=safe_message,
    )
    ledger.complete_run(session.run_id, failure_code=error_code, failure_stage=stage)
    return {
        "status": "failed",
        "episode_id": session.episode_id,
        "run_id": session.run_id,
        "error_code": error_code,
        "stage": stage,
    }


def finalize_cognition_run(
    session: CognitionSession,
    result: Dict[str, Any],
    intelligence_output: Dict[str, Any],
    baseline_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ledger = get_ledger()
    scene_sig, cat_sig = _signatures_from_result(result)
    strategy = session.deliberation_context.strategy_hint or "standard"
    snap_obj = snapshot_from_intelligence(
        intelligence_output,
        run_id=session.run_id,
        state_version_id=session.state_version_id,
        perception_artefact_hash=session.perception_artefact_hash or "",
        context_fingerprint=session.context_fingerprint or "",
        memory_reference_ids=session.memory_reference_ids,
        scene_signature=scene_sig,
        category_signature=cat_sig,
        strategy=strategy,
        requested_evidence=session.deliberation_context.requested_evidence,
    )
    raw_confidence = float(snap_obj.confidence)
    baseline_for_compare: Optional[DeliberationSnapshot] = session.baseline_snapshot
    if baseline_snapshot and not baseline_for_compare:
        baseline_for_compare = DeliberationSnapshot(
            primary_hypothesis=str(baseline_snapshot.get("primary_hypothesis", "")),
            confidence=float(baseline_snapshot.get("confidence", 0.5)),
            strategy=str(baseline_snapshot.get("strategy", "standard")),
            requested_evidence=list(baseline_snapshot.get("requested_evidence") or []),
            perception_artefact_hash=session.perception_artefact_hash or "",
            scene_signature=scene_sig,
            category_signature=cat_sig,
            run_id=session.baseline_run_id,
        )
    if not baseline_for_compare and session.memory_reference_ids:
        legacy = ledger.find_compatible_baseline_snapshot(
            session.workspace_id,
            session.asset_id,
            session.perception_artefact_hash or "",
            baseline_run_id=session.baseline_run_id,
        )
        if legacy:
            baseline_for_compare = DeliberationSnapshot(
                primary_hypothesis=str(legacy.get("primary_hypothesis", "")),
                confidence=float(legacy.get("confidence", 0.5)),
                strategy=str(legacy.get("strategy", "standard")),
                requested_evidence=list(legacy.get("requested_evidence") or []),
                perception_artefact_hash=session.perception_artefact_hash or "",
                scene_signature=scene_sig,
                category_signature=cat_sig,
                run_id=session.baseline_run_id,
            )

    baseline_confidence = float(baseline_for_compare.confidence) if baseline_for_compare else raw_confidence
    clamp_applied = False
    final_confidence = raw_confidence
    comparison_status = "no_compatible_baseline"
    if session.memory_reference_ids and baseline_for_compare:
        comparison_status = "compatible_baseline"
        if raw_confidence > baseline_confidence:
            final_confidence = baseline_confidence
            clamp_applied = True
    elif session.memory_reference_ids:
        comparison_status = "missing_baseline"

    snap_obj = DeliberationSnapshot(
        **{
            **snap_obj.to_dict(),
            "confidence": final_confidence,
        }
    )
    session.confidence_provenance = {
        "raw_confidence": raw_confidence,
        "baseline_confidence": baseline_confidence if baseline_for_compare else None,
        "final_confidence": final_confidence,
        "clamp_applied": clamp_applied,
        "comparison_status": comparison_status,
    }

    snap = snap_obj.to_dict()
    snap["perception_artefact_hash"] = session.perception_artefact_hash
    snap["confidence_provenance"] = session.confidence_provenance
    snap_hash = ledger.put_artefact("deliberation_snapshot", "v1", snap)

    deltas = []
    delta_payload = None
    baseline_link_payload = None
    if session.run_purpose == RunPurpose.BASELINE:
        baseline_link_payload = snap
    if baseline_for_compare:
        delta_objs = compare_deliberation_snapshots(baseline_for_compare, snap_obj, session.memory_reference_ids)
        deltas = [d.__dict__ for d in delta_objs if d.field_changed != "_compatibility"]
        if deltas:
            delta_payload = {
                "deltas": deltas,
                "baseline_run_id": baseline_for_compare.run_id or session.baseline_run_id,
            }

    fp = artefact_hash({"episode": session.episode_id, "run": session.run_id, "snap": snap_hash})
    event_id = ledger.finalize_run_atomic(
        episode_id=session.episode_id,
        run_id=session.run_id,
        run_purpose=session.run_purpose,
        scene_signature=scene_sig,
        category_signature=cat_sig,
        goal_type="critique",
        goal_instance_id=None,
        final_fingerprint=fp,
        perception_artefact_hash=session.perception_artefact_hash,
        deliberation_snapshot=snap,
        deliberation_snapshot_hash=snap_hash,
        experience_closed_payload={"status": "closed", "run_purpose": session.run_purpose.value},
        delta_payload=delta_payload,
        baseline_link_payload=baseline_link_payload,
    )
    result.setdefault("cognition_provenance", {})
    result["cognition_provenance"].update(
        {
            "episode_id": session.episode_id,
            "run_id": session.run_id,
            "state_version_id": session.state_version_id,
            "context_fingerprint": session.context_fingerprint,
            "memory_reference_ids": session.memory_reference_ids,
            "deliberation_event_id": event_id,
            "deltas": deltas,
            "cognition_context_used": bool(session.cognition_context),
            "run_purpose": session.run_purpose.value,
            "baseline_run_id": session.baseline_run_id,
            "rejected_candidates": session.rejected_candidates,
            "confidence_provenance": session.confidence_provenance,
        }
    )
    intelligence_output.setdefault("_cognition_provenance", {})
    intelligence_output["_cognition_provenance"]["memory_reference_ids"] = session.memory_reference_ids
    intelligence_output["_cognition_provenance"]["confidence_provenance"] = session.confidence_provenance
    return result
