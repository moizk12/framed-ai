"""Live cognitive loop repair tests (Slice A blockers)."""

from __future__ import annotations

import inspect
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from framed.analysis.analysis_cache import strip_cognition_from_result
from framed.analysis.intelligence_formatting import is_likely_art_reproduction, is_likely_digital_display
from framed.cognition.config import cognition_enabled
from framed.cognition.context.formatting import build_cognition_context, format_cognition_context_for_prompt
from framed.cognition.contracts.memory import MemoryReference, ScoreComponents
from framed.cognition.contracts.runs import RunMode
from framed.cognition.integration.pipeline_hook import begin_cognition_run, legacy_writes_allowed
from framed.cognition.ledger.sqlite_store import reset_ledger


@pytest.fixture
def cognition_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(tmp_path))
    ledger = reset_ledger(tmp_path / "test.sqlite3")
    yield ledger
    reset_ledger()


def _ref() -> MemoryReference:
    return MemoryReference(
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
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
        hypothesis_summary="Prior clutter failure",
        confidence_at_source=0.5,
        scores=ScoreComponents(1, 1, 1, 0, 0.1, 0.85),
        match_reason="test",
    )


def test_cognition_context_reaches_recognition_prompt():
    from framed.analysis.intelligence_layers import reason_about_recognition

    sig = inspect.signature(reason_about_recognition)
    assert "cognition_context" in sig.parameters
    ctx = build_cognition_context([_ref()])
    formatted = format_cognition_context_for_prompt(ctx)
    assert "PRIOR EXPERIENCE" in formatted
    assert "m1" in formatted
    assert "Temporal memory incomplete" not in formatted


def test_empty_cognition_context_omits_section():
    assert format_cognition_context_for_prompt(None) == ""
    assert format_cognition_context_for_prompt({"retrieved_experiences": []}) == ""


def test_legacy_temporal_not_used_when_cognition_on(monkeypatch):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    assert legacy_writes_allowed() is False


def test_cache_strip_removes_cognition_identity():
    result = {
        "perception": {"technical": {"available": True}},
        "intelligence": {"recognition": {}},
        "cognition_provenance": {"run_id": "old"},
        "pattern_signature": "sig",
    }
    stripped = strip_cognition_from_result(result)
    assert "intelligence" not in stripped
    assert "cognition_provenance" not in stripped
    assert "pattern_signature" not in stripped
    assert "perception" in stripped


def test_state_activation_durable_after_ensure(cognition_env):
    ledger = cognition_env
    ws = str(uuid.uuid4())
    ledger.ensure_demo_states(ws)
    ledger.activate_state(ws, "state_baseline")
    active = ledger.get_active_state(ws)
    assert active["label"] == "state_baseline"
    ledger.ensure_demo_states(ws)
    active2 = ledger.get_active_state(ws)
    assert active2["label"] == "state_baseline"


def test_fresco_not_digital_display():
    ve = {
        "scene_gate": {
            "scene_type": "unknown",
            "signals": {"clip_caption": "Michelangelo fresco detail in the Sistine Chapel"},
        }
    }
    assert is_likely_art_reproduction(ve) is True
    assert is_likely_digital_display(ve) is False


def test_screenshot_still_digital():
    ve = {
        "scene_gate": {
            "scene_type": "screenshot_ui",
            "signals": {"clip_caption": "code editor on laptop screen"},
        },
        "material_condition": {"edge_degradation": 0.9, "color_uniformity": 0.95},
        "organic_growth": {"green_coverage": 0.0},
    }
    assert is_likely_digital_display(ve) is True


def test_static_feedback_js_served():
    from framed import create_app

    app = create_app({"TESTING": True})
    client = app.test_client()
    resp = client.get("/static/feedback.js")
    assert resp.status_code == 200


def test_analyze_route_exists():
    from framed import create_app

    app = create_app({"TESTING": True})
    client = app.test_client()
    resp = client.post("/analyze")
    assert resp.status_code in (400, 415)


def test_same_image_two_runs_different_ids(cognition_env, monkeypatch, tmp_path):
    import os

    from framed.cognition.integration.pipeline_hook import finalize_cognition_run

    img = tmp_path / "same.jpg"
    img.write_bytes(b"same-image-bytes")
    result = {
        "visual_evidence": {
            "scene_gate": {"scene_type": "interior_scene", "signals": {}},
        }
    }
    intel = {"recognition": {"what_i_see": "room", "confidence": 0.5}}
    s1 = begin_cognition_run(result=result, image_path=str(img), asset_filename="same.jpg", run_mode=RunMode.MEMORY_ENABLED)
    assert s1
    finalize_cognition_run(s1, result, intel)
    s2 = begin_cognition_run(result=result, image_path=str(img), asset_filename="same.jpg", run_mode=RunMode.MEMORY_ENABLED)
    assert s2
    assert s1.run_id != s2.run_id
    assert s1.episode_id != s2.episode_id


def test_framed_intelligence_accepts_cognition_context():
    from framed.analysis.intelligence_core import framed_intelligence

    sig = inspect.signature(framed_intelligence)
    assert "cognition_context" in sig.parameters
