"""IC_0015-B: downstream prompt suppression unit tests."""

from framed.analysis.intelligence_formatting import (
    domain_guard_prompt_block,
    format_visual_evidence,
    organic_evidence_suppressed,
    sanitize_primary_when_suppressed,
    ui_screen_scene_hint,
)


def _suppressed_ve(**overrides):
    base = {
        "domain_guard_applied": True,
        "organic_growth": {
            "green_coverage": 0.002,
            "salience": "minimal",
            "applicable": False,
            "suppressed_reason": "green_coverage_below_threshold",
        },
        "material_condition": {
            "condition": "neutral",
            "edge_degradation": 0.72,
            "color_uniformity": 0.97,
        },
        "organic_integration": {
            "relationship": "none",
            "integration_level": "none",
        },
        "scene_gate": {
            "scene_type": "interior_scene",
            "is_surface_study": False,
            "signals": {
                "places_scene_category": "artificial",
                "yolo_objects": ["tv"],
                "clip_caption": "a photo with intentional motion blur",
            },
        },
        "overall_confidence": 0.53,
    }
    base.update(overrides)
    return base


def _normal_ve():
    return {
        "domain_guard_applied": False,
        "organic_growth": {
            "green_coverage": 0.12,
            "salience": "moderate",
            "applicable": True,
            "green_locations": "scattered",
            "confidence": 0.7,
        },
        "material_condition": {
            "condition": "weathered",
            "surface_roughness": 0.4,
            "edge_degradation": 0.5,
            "confidence": 0.8,
        },
        "organic_integration": {
            "relationship": "reclamation",
            "integration_level": "high",
            "overlap_ratio": 0.3,
            "confidence": 0.85,
        },
        "scene_gate": {
            "scene_type": "surface_study",
            "is_surface_study": True,
        },
        "overall_confidence": 0.75,
    }


def test_organic_evidence_suppressed_true_when_guard_active():
    assert organic_evidence_suppressed(_suppressed_ve()) is True


def test_organic_evidence_suppressed_false_without_guard():
    assert organic_evidence_suppressed(_normal_ve()) is False


def test_format_visual_evidence_omits_organic_when_suppressed():
    text = format_visual_evidence(_suppressed_ve())
    assert "NOT APPLICABLE" in text
    assert "domain guard active" in text
    assert "Organic Growth: coverage=" not in text
    assert "Organic Integration: relationship=" not in text
    assert "reclamation" not in text.lower() or "do not infer ivy/reclamation" in text


def test_format_visual_evidence_unchanged_when_not_suppressed():
    text = format_visual_evidence(_normal_ve())
    assert "Organic Growth: coverage=" in text
    assert "Organic Integration: relationship=reclamation" in text
    assert "NOT APPLICABLE" not in text


def test_ui_hint_appended_for_artificial_scene():
    assert ui_screen_scene_hint(_suppressed_ve()) is True
    text = format_visual_evidence(_suppressed_ve())
    assert "screen/UI/code" in text or "SCENE ROUTING" in text


def test_domain_guard_prompt_block_forbids_organic_narrative():
    block = domain_guard_prompt_block(_suppressed_ve())
    assert "CONSTRAINT (domain guard)" in block
    assert "organic growth" in block.lower()
    assert "screen" in block.lower() or "UI" in block


def test_sanitize_primary_strips_organic_growth():
    primary = (
        "I see an interior scene that appears to be a study with "
        "minimal organic growth and no significant surface weathering."
    )
    cleaned = sanitize_primary_when_suppressed(primary, _suppressed_ve())
    assert "organic growth" not in cleaned.lower()
    assert "surface weathering" not in cleaned.lower()
    assert "screen" in cleaned.lower() or "display" in cleaned.lower() or "interior" in cleaned.lower()


def test_sanitize_primary_unchanged_when_not_suppressed():
    primary = "I see weathered stone with ivy reclaiming the structure."
    assert sanitize_primary_when_suppressed(primary, _normal_ve()) == primary
