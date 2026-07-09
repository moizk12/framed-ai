"""IC_0022: dense grounding probe unit tests (flag off = no boxes; mock on = schema boxes)."""

from unittest.mock import patch

import config
from framed.analysis.perception import run_dense_grounding_probe
from framed.analysis.visual_evidence import GroundingBox, _attach_grounding_probe, grounding_boxes_to_dicts


def test_flag_off_yields_empty_grounding():
    ve = {"organic_growth": {"green_coverage": 0.2, "confidence": 0.8}}
    with patch.object(config, "ENABLE_DENSE_GROUNDING_PROBE", False):
        _attach_grounding_probe(ve, "/tmp/fake.jpg")
    assert ve["grounding"] == []


def test_mock_backend_returns_schema_boxes():
    with patch.object(config, "GROUNDING_PROBE_BACKEND", "mock"):
        boxes = run_dense_grounding_probe("/tmp/fake.jpg")
    assert len(boxes) >= 1
    for box in boxes:
        assert set(box.keys()) == {"label", "x", "y", "w", "h", "confidence", "source"}
        assert 0.0 <= box["x"] <= 1.0
        assert 0.0 <= box["w"] <= 1.0


def test_flag_on_mock_attaches_boxes():
    ve = {"organic_growth": {"green_coverage": 0.01, "confidence": 0.5}}
    with patch.object(config, "ENABLE_DENSE_GROUNDING_PROBE", True):
        with patch.object(config, "GROUNDING_PROBE_BACKEND", "mock"):
            _attach_grounding_probe(ve, "/tmp/x.jpg")
    assert len(ve["grounding"]) >= 1
    assert ve["grounding"][0]["source"] == "mock"


def test_grounding_box_roundtrip():
    box = GroundingBox("text", 0.1, 0.2, 0.3, 0.4, 0.9, "mock")
    d = grounding_boxes_to_dicts([box])[0]
    assert d["label"] == "text"
    assert d["source"] == "mock"
