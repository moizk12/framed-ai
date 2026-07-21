"""IC_0026a: evidence inspector UI payload tests."""

import os

from framed.routes import build_evidence_inspector, clean_result_for_ui


def _sample_result():
    return {
        "metadata": {"photo_id": "p1"},
        "perception": {"semantics": {"available": True, "caption": "test", "tags": []}},
        "visual_evidence": {
            "scene_gate": {"scene_type": "interior_scene"},
            "organic_growth": {"green_coverage": 0.04, "evidence": ["green_coverage=0.04"]},
            "theme_claim_license": {
                "tier": "forbidden",
                "organic_growth": "forbidden",
                "reclamation": "forbidden",
                "weathered_stone": "forbidden",
                "reasons": ["green_capped_forbidden:0.04"],
            },
            "grounding": [],
        },
        "intelligence": {
            "recognition": {"what_i_see": "A room scene", "confidence": 0.7},
        },
        "critique": "The composition feels cluttered.",
    }


def test_build_evidence_inspector_schema():
    inspector = build_evidence_inspector(_sample_result())
    for key in (
        "recognition",
        "scene",
        "evidence",
        "grounding",
        "theme_licenses",
        "claim_traces",
        "critique",
        "warnings",
        "provenance",
    ):
        assert key in inspector


def test_critique_not_mutated():
    result = _sample_result()
    before = result["critique"]
    inspector = build_evidence_inspector(result)
    assert result["critique"] == before
    assert inspector["critique"]["text"] == before


def test_grounding_disabled_not_empty_semantics(monkeypatch):
    monkeypatch.delenv("ENABLE_DENSE_GROUNDING_PROBE", raising=False)
    os.environ["ENABLE_DENSE_GROUNDING_PROBE"] = "false"
    inspector = build_evidence_inspector(_sample_result())
    assert inspector["grounding"]["state"] == "disabled"
    assert inspector["grounding"]["render_boxes"] == []


def test_clean_result_for_ui_includes_inspector():
    ui = clean_result_for_ui(_sample_result())
    assert "evidence_inspector" in ui
    assert ui["evidence_inspector"]["scene"]["scene_type"] == "interior_scene"
