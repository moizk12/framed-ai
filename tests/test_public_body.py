"""Public body contracts: fake provider clients, real orchestration boundaries."""
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from framed import create_app
from framed.public_api import (
    PublicAnalysisUnavailable,
    build_public_analysis_dto,
)
from framed.public_limits import AnalysisLimiter
from framed.public_runtime import runtime_defaults, validate_runtime
from framed.public_store import MemoryPublicRepository


@pytest.mark.parametrize("responses,fallback", [(True, False), (True, True), (False, False)])
def test_cloud_image_parity_and_text_only_expression(tmp_path, monkeypatch, responses, fallback):
    from framed.analysis.providers.openai_provider import OpenAIProvider
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)
    def respond(**kwargs):
        calls.append(kwargs)
        if fallback:
            raise RuntimeError("unsupported endpoint")
        return SimpleNamespace(output_text="ok", usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=chat)))
    if responses:
        client.responses = SimpleNamespace(create=respond)
    provider = OpenAIProvider({"model_name": "fake"}, "reasoning")
    monkeypatch.setattr(provider, "_get_client", lambda: client)
    image = tmp_path / "photo.png"
    image.write_bytes(b"unique-image-bytes")
    provider.call("recognize", image_path=str(image))
    encoded = base64.b64encode(image.read_bytes()).decode()
    assert calls and all(encoded in json.dumps(call) for call in calls)
    calls.clear()
    provider.call("later reasoning")
    assert all("image" not in json.dumps(call) for call in calls)
    calls.clear()
    provider.role = "expression"
    provider.call("express")
    assert calls[0]["messages"] == [{"role": "user", "content": "express"}]


def test_public_core_never_invokes_longitudinal_layers(monkeypatch):
    from framed.analysis import intelligence_core as core
    from framed.analysis import public_reasoning
    forbidden = Mock(side_effect=AssertionError("research mechanism invoked"))
    for name in ("reason_about_layers_2_7", "reason_about_evolution", "reason_about_trajectory", "reason_about_past_errors", "get_governor_bias", "reason_about_thinking", "reason_about_feeling", "reason_about_mentorship"):
        monkeypatch.setattr(core, name, forbidden)
    recognition = Mock(return_value={"what_i_see": "A window", "confidence": .8, "evidence": ["light"]})
    monkeypatch.setattr(core, "reason_about_recognition", recognition)
    monkeypatch.setattr(public_reasoning, "reason_standalone", lambda r: {"observations": ["Window light"], "questions": [], "why_i_believe_this": "Visible edges"})
    result = core.framed_intelligence({}, {}, public_safe=True, temporal_memory={"private": 1}, user_history={"private": 1}, cognition_context={"private": 1}, image_path="photo.png")
    assert result["recognition"]["what_i_see"] == "A window"
    assert not {"temporal", "continuity", "self_critique"} & result.keys()
    assert recognition.call_args.kwargs["cognition_context"] is None
    forbidden.assert_not_called()


def test_expression_public_contract_excludes_history(monkeypatch):
    from framed.analysis import expression_layer as expression
    captured = Mock(return_value={"content": "Window light separates the shapes. Consider a closer crop.", "error": None})
    monkeypatch.setattr(expression, "call_model_b", captured)
    result = expression.generate_poetic_critique({"recognition": {"what_i_see": "Window", "confidence": .8}, "temporal": {"private": "HISTORY_SENTINEL"}, "continuity": {"user_pattern": "HISTORY_SENTINEL"}}, public_safe=True)
    assert result
    args = captured.call_args.kwargs
    assert "standalone public analysis" in args["system_prompt"]
    assert "HISTORY_SENTINEL" not in args["prompt"]
    assert "used to struggle" not in args["prompt"]
    assert "image_path" not in args


@pytest.mark.parametrize("failure", [RuntimeError("429 insufficient_quota account SECRET"), ValueError("provider SECRET payload"), TimeoutError("SECRET timed out"), TypeError("malformed SECRET response")])
def test_provider_failures_are_sanitized_and_do_not_persist(tmp_path, failure):
    repository = MemoryPublicRepository()
    def fail(*args):
        raise failure
    app = create_app({"TESTING": True, "PUBLIC_REPOSITORY": repository, "PUBLIC_ANALYSIS_RUNNER": fail, "PUBLIC_UPLOAD_TEMP_DIR": str(tmp_path)})
    store = app.extensions["framed_public_store"]
    store.record_analysis = Mock()
    content = io.BytesIO()
    Image.new("RGB", (10, 10)).save(content, format="PNG")
    content.seek(0)
    response = app.test_client().post("/api/v1/analyses", data={"image": (content, "photo.png")})
    assert response.status_code in (500, 503)
    body = response.get_json()
    assert set(body) == {"request_id", "error", "message"}
    assert "SECRET" not in json.dumps(body)
    assert not body.get("critique")
    store.record_analysis.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("content", ["not json", "[]", '{"what_i_see": 42}', '{"what_i_see":"", "confidence":.8}', '{"what_i_see":"window", "confidence": 2, "evidence":[]}'])
def test_public_recognition_rejects_malformed_response(monkeypatch, content):
    from framed.analysis import intelligence_layers as layers
    monkeypatch.setattr(layers, "call_model_a", lambda **kw: {"content": content})
    result = layers.reason_about_recognition({}, public_safe=True)
    assert result.get("error") and not result["what_i_see"]


@pytest.mark.parametrize("critique", ["", "Error code: 429 SECRET", '[PLACEHOLDER] mock', 'Traceback (most recent call last): SECRET', '{"error": "SECRET"}', None])
def test_dto_rejects_empty_or_runtime_critique(critique):
    with pytest.raises(PublicAnalysisUnavailable):
        build_public_analysis_dto({"critique": critique}, request_id="r", analysis_id="a", duration_ms=1)


def test_rate_limit_allows_then_rejects_without_affecting_probes():
    app = create_app({"TESTING": True, "PUBLIC_RATE_LIMIT": 2})
    client = app.test_client()
    assert client.post("/api/v1/analyses").status_code == 400
    assert client.post("/api/v1/analyses").status_code == 400
    response = client.post("/api/v1/analyses", headers={"X-Forwarded-For": "different-client"})
    assert response.status_code == 429
    assert response.get_json()["error"] == {"code": "rate_limited"}
    assert int(response.headers["Retry-After"]) > 0
    for path in ("/health", "/version", "/", "/static/css/public-beta.css"):
        assert client.get(path).status_code == 200
    now = [0]
    limiter = AnalysisLimiter(1, 10, lambda: now[0])
    assert limiter.admit() == 0
    assert limiter.admit() == 10
    now[0] = 10
    assert limiter.admit() == 0


def test_measurements_and_grounding_are_allowlisted(monkeypatch):
    monkeypatch.setenv("ENABLE_DENSE_GROUNDING_PROBE", "true")
    internal = {"critique": "A study of light.", "perception": {"technical": {"brightness": 100.12, "contrast": 40.75, "sharpness": 123.21}}, "visual_evidence": {"organic_growth": {"green_coverage": .253}, "grounding": [{"source": "heuristic", "x": 0, "y": 0, "w": 1, "h": 1, "confidence": .99}]}}
    dto = build_public_analysis_dto(internal, request_id="r", analysis_id="a", duration_ms=1)
    assert len(dto["evidence"]["measured_signals"]) == 4
    assert dto["evidence"]["measured_signals"][0]["value"] == "100 / 255"
    assert dto["evidence"]["grounding"] == {"state": "empty", "boxes": []}
    internal["visual_evidence"]["grounding"][0]["source"] = "detector"
    assert build_public_analysis_dto(internal, request_id="r", analysis_id="a", duration_ms=1)["evidence"]["grounding"]["state"] == "available"
    internal["perception"]["technical"] = {"brightness": float("nan"), "contrast": None}
    internal["visual_evidence"] = {}
    assert not build_public_analysis_dto(internal, request_id="r", analysis_id="a", duration_ms=1)["evidence"]["measured_signals"]


def test_timeout_config_rejects_incoherent_budget():
    config = runtime_defaults()
    validate_runtime(config)
    config["PUBLIC_WORKER_TIMEOUT_SECONDS"] = config["PUBLIC_ANALYSIS_TIMEOUT_SECONDS"]
    with pytest.raises(RuntimeError, match="Worker timeout"):
        validate_runtime(config)


def test_empty_public_expression_never_returns_fallback(monkeypatch):
    from framed.analysis import expression_layer
    from framed.analysis.critique_finalization import CritiqueRuntimeError
    monkeypatch.setattr(expression_layer, "call_model_b", lambda **kw: {"content": "", "error": None})
    with pytest.raises(CritiqueRuntimeError):
        expression_layer.generate_poetic_critique({"recognition": {"what_i_see": "Window"}}, public_safe=True)


def test_public_finalization_does_not_retrieve_rules(monkeypatch):
    from framed.analysis import critique_finalization as finalization
    forbidden = Mock(side_effect=AssertionError("research rules retrieved"))
    monkeypatch.setattr(finalization, "_active_correction_rules", forbidden)
    monkeypatch.setattr(finalization, "_reflect", lambda *a: {"requires_regeneration": False})
    result = finalization.finalize_critique_with_reflection("Window light defines the frame.", {"recognition": {"what_i_see": "Window"}}, public_safe=True)
    assert result["critique"]
    forbidden.assert_not_called()


def test_public_pipeline_does_not_touch_research_or_caches(monkeypatch, tmp_path):
    from framed.analysis import (
        intelligence_core,
        pipeline,
        public_reasoning,
        temporal_memory,
    )
    from framed.cognition.integration import pipeline_hook
    monkeypatch.setenv("FRAMED_ENABLE_INTELLIGENCE_CORE", "true")
    forbidden = Mock(side_effect=AssertionError("research or cache invoked"))
    for name in ("get_cached_analysis", "save_cached_analysis", "update_echo_memory"):
        monkeypatch.setattr(pipeline, name, forbidden)
    for name in ("begin_cognition_run", "finalize_cognition_run", "legacy_writes_allowed"):
        monkeypatch.setattr(pipeline_hook, name, forbidden)
    for name in ("format_temporal_memory_for_intelligence", "track_user_trajectory", "store_interpretation"):
        monkeypatch.setattr(temporal_memory, name, forbidden)
    for name in ("get_clip_description", "predict_nima_score", "analyze_color", "analyze_color_harmony", "detect_objects_and_framing", "analyze_lines_and_symmetry", "analyze_lighting_direction", "analyze_tonal_range", "analyze_subject_emotion", "analyze_background_clutter", "extract_visual_features"):
        monkeypatch.setattr(pipeline, name, lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "get_clip_inventory", lambda *a: [])
    monkeypatch.setattr(pipeline, "get_nima_model", lambda: None)
    monkeypatch.setattr(intelligence_core, "reason_about_recognition", lambda *a, **kw: {"what_i_see": "A blue frame", "evidence": [], "confidence": .7})
    monkeypatch.setattr(public_reasoning, "reason_standalone", lambda r: {"observations": [], "questions": [], "why_i_believe_this": "Blue pixels"})
    path = tmp_path / "photo.png"
    Image.new("RGB", (40, 40), (50, 100, 200)).save(path)
    result = pipeline.run_full_analysis(str(path), public_safe=True)
    assert result["intelligence"]["recognition"]["what_i_see"] == "A blue frame"
    forbidden.assert_not_called()


def test_model_warmup_reuses_initialized_instances(monkeypatch):
    from framed.analysis import models
    yolo, clip, processor = object(), object(), object()
    monkeypatch.setattr(models, "_yolo_model", yolo)
    monkeypatch.setattr(models, "_clip_model", clip)
    monkeypatch.setattr(models, "_clip_processor", processor)
    models.warm_public_models()
    models.warm_public_models()
    assert models.get_yolo_model() is yolo
    assert models.get_clip_model()[:2] == (clip, processor)
    assert models.public_models_ready()
