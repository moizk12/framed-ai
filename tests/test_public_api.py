"""Track A v1 public API contract and boundary tests."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from framed import create_app
from framed.public_api import PublicAnalysisUnavailable


def _image_bytes(image_format: str = "PNG") -> io.BytesIO:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), (80, 120, 160)).save(output, format=image_format)
    output.seek(0)
    return output


def _internal_result() -> dict:
    return {
        "critique": "The diagonal light gives the frame direction; protect that tension with a tighter crop.",
        "perception": {"semantics": {"available": True, "caption": "A blue geometric study."}},
        "visual_evidence": {
            "scene_gate": {"scene_type": "abstract study"},
            "grounding": [{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "label": "diagonal light", "internal": "drop"}],
            "theme_claim_license": {"tier": "cautious", "reasons": ["limited scene evidence"]},
        },
        "intelligence": {
            "recognition": {"what_i_see": "A blue geometric study with diagonal light.", "confidence": 0.72},
            "meta_cognition": {"private": "must never leak"},
        },
        "cognition": {"run_id": "research-run"},
        "echo_memory": ["private"],
        "pattern_signature": "private-signature",
    }


@pytest.fixture()
def app(tmp_path):
    created_paths = []

    def runner(path, filename):
        created_paths.append(Path(path))
        assert Path(path).exists()
        assert filename == "photo.png"
        return _internal_result(), 1250

    app = create_app(
        {
            "TESTING": True,
            "PUBLIC_ANALYSIS_RUNNER": runner,
            "PUBLIC_UPLOAD_TEMP_DIR": str(tmp_path),
        }
    )
    app.config["CREATED_PUBLIC_PATHS"] = created_paths
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def _create_analysis(client):
    return client.post(
        "/api/v1/analyses",
        data={"image": (_image_bytes(), "photo.png"), "mentor_mode": "balanced"},
        content_type="multipart/form-data",
    )


def test_create_analysis_matches_v1_allowlisted_contract_and_removes_upload(app, client):
    response = _create_analysis(client)
    assert response.status_code == 201
    payload = response.get_json()
    assert set(payload) == {"request_id", "analysis_id", "status", "critique", "evidence", "limitations", "meta"}
    assert payload["request_id"].startswith("req-")
    assert payload["analysis_id"].startswith("ana-")
    assert payload["status"] == "complete"
    assert payload["meta"] == {"duration_ms": 1250, "cached": False, "contract_version": "1"}
    assert payload["evidence"]["recognition"]["text"].startswith("A blue geometric")
    assert payload["evidence"]["scene"] == {"type": "abstract study"}
    assert payload["evidence"]["grounding"] == {"state": "disabled", "boxes": []}
    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    assert keys(payload).isdisjoint({"cognition", "echo_memory", "run_id", "pattern_signature", "meta_cognition", "private"})
    assert "must never leak" not in json.dumps(payload).lower()
    assert all(not path.exists() for path in app.config["CREATED_PUBLIC_PATHS"])


def test_feedback_is_attached_to_an_existing_analysis(client, app):
    analysis_id = _create_analysis(client).get_json()["analysis_id"]
    response = client.post(
        "/api/v1/feedback",
        json={"analysis_id": analysis_id, "useful": True, "comment": "Specific and useful."},
    )
    assert response.status_code == 201
    feedback_payload = response.get_json()
    assert feedback_payload["status"] == "recorded"
    assert feedback_payload["analysis_id"] == analysis_id
    assert feedback_payload["request_id"].startswith("req-")
    assert feedback_payload["meta"] == {"contract_version": "1"}
    stored = app.extensions["framed_public_store"].feedback_for(analysis_id)
    assert len(stored) == 1
    assert stored[0]["useful"] is True
    assert stored[0]["comment"] == "Specific and useful."


@pytest.mark.parametrize(
    ("data", "expected_status", "expected_code"),
    [
        ({}, 400, "missing_image"),
        ({"image": (io.BytesIO(b"text"), "notes.txt")}, 415, "unsupported_media_type"),
        ({"image": (io.BytesIO(b"not an image"), "photo.png")}, 400, "invalid_image"),
        ({"image": (_image_bytes(), "photo.jpg")}, 415, "unsupported_media_type"),
    ],
)
def test_upload_failures_use_safe_error_envelopes(client, data, expected_status, expected_code):
    response = client.post("/api/v1/analyses", data=data, content_type="multipart/form-data")
    assert response.status_code == expected_status
    payload = response.get_json()
    assert payload["request_id"].startswith("req-")
    assert payload["error"] == {"code": expected_code}
    assert "internal" not in payload["message"].lower()


def test_oversized_upload_returns_413_json():
    app = create_app({"TESTING": True, "MAX_CONTENT_LENGTH": 128})
    response = app.test_client().post(
        "/api/v1/analyses",
        data={"image": (io.BytesIO(b"x" * 1024), "photo.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert response.get_json()["error"] == {"code": "payload_too_large"}


def test_analysis_core_failure_is_503_not_internal_detail(app):
    app.config["PUBLIC_ANALYSIS_RUNNER"] = lambda *_: (_ for _ in ()).throw(PublicAnalysisUnavailable("provider secret"))
    response = _create_analysis(app.test_client())
    assert response.status_code == 503
    payload = response.get_json()
    assert payload["error"] == {"code": "analysis_unavailable"}
    assert "secret" not in json.dumps(payload)


def test_unexpected_analysis_failure_is_safe_500(app):
    app.config["PUBLIC_ANALYSIS_RUNNER"] = lambda *_: (_ for _ in ()).throw(ValueError("private stack detail"))
    response = _create_analysis(app.test_client())
    assert response.status_code == 500
    payload = response.get_json()
    assert payload["error"] == {"code": "internal_error"}
    assert "private stack detail" not in json.dumps(payload)


def test_feedback_validation_and_unknown_analysis(client):
    unsupported = client.post("/api/v1/feedback", data="{}", content_type="text/plain")
    assert unsupported.status_code == 415
    invalid = client.post("/api/v1/feedback", json={"analysis_id": "ana-missing", "useful": "yes"})
    assert invalid.status_code == 400
    missing_id = "ana-" + "0" * 32
    missing = client.post("/api/v1/feedback", json={"analysis_id": missing_id, "useful": True})
    assert missing.status_code == 404
    assert missing.get_json()["error"] == {"code": "analysis_not_found"}


def test_public_safe_pipeline_disables_echo_and_cache_path():
    source = (Path(__file__).parents[1] / "framed" / "analysis" / "pipeline.py").read_text(encoding="utf-8")
    assert "disable_cache=public_safe" in source
    assert "if not critical_errors and not public_safe:" in source
    assert "if file_hash and not public_safe:" in source
    assert "if not public_safe:" in source
