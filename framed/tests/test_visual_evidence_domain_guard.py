"""IC_0015-A: visual evidence domain guard unit tests."""

from framed.analysis.visual_evidence import (
    MIN_GREEN_FOR_RECLAMATION,
    apply_domain_guard,
    detect_organic_integration,
    validate_visual_evidence,
)


def _ve(green=0.002, relationship="reclamation", condition="degraded", edge=0.75, rough=0.006):
    return {
        "organic_growth": {
            "green_coverage": green,
            "salience": "minimal",
            "confidence": 0.5,
        },
        "material_condition": {
            "condition": condition,
            "edge_degradation": edge,
            "surface_roughness": rough,
            "confidence": 0.8,
        },
        "organic_integration": {
            "relationship": relationship,
            "integration_level": "high",
            "confidence": 0.85,
            "evidence": ["relationship=reclamation"],
        },
        "overall_confidence": 0.7,
    }


def test_apply_domain_guard_suppresses_reclamation_at_low_green():
    ve = apply_domain_guard(_ve(green=0.002, relationship="reclamation"))
    assert ve["organic_integration"]["relationship"] == "none"
    assert ve["organic_growth"].get("applicable") is False
    assert "suppressed_reason" in ve["organic_integration"]


def test_apply_domain_guard_neutralizes_edge_only_degraded():
    ve = apply_domain_guard(_ve(green=0.001, condition="degraded", edge=0.8, rough=0.005))
    assert ve["material_condition"]["condition"] in ("not_applicable", "neutral")


def test_apply_domain_guard_allows_reclamation_above_threshold():
    ve = apply_domain_guard(_ve(green=0.08, relationship="reclamation"))
    assert ve["organic_integration"]["relationship"] == "reclamation"
    assert ve["organic_growth"].get("applicable") is True


def test_validate_no_reclamation_contradiction_after_guard():
    ve = apply_domain_guard(_ve(green=0.002, relationship="reclamation"))
    val = validate_visual_evidence(ve)
    assert val["is_valid"] is True
    assert not any("Reclamation relationship" in i for i in val.get("issues", []))


def test_detect_organic_integration_suppressed_when_green_sparse(tmp_path):
    # 10x10 white image — negligible green
    import cv2
    import numpy as np

    p = tmp_path / "white.png"
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(p), img)
    result = detect_organic_integration(str(p))
    assert result["relationship"] == "none"
    assert result.get("green_coverage", 1.0) < MIN_GREEN_FOR_RECLAMATION
