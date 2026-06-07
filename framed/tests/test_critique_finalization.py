"""Unit tests for shared critique finalization."""

from unittest.mock import patch

from framed.analysis.critique_finalization import (
    check_vocab_guard,
    finalize_critique_with_reflection,
)

_REASONER = {
    "recognition": {"what_i_see": "weathered stone surface", "confidence": 0.7, "evidence": ["stone"]},
}


def test_no_regen_when_reflection_passes():
    with patch("framed.analysis.critique_finalization._reflect") as mock_reflect:
        mock_reflect.return_value = {"requires_regeneration": False, "quality_score": 0.9}
        out = finalize_critique_with_reflection("plain critique text", _REASONER)
    assert out["critique"] == "plain critique text"
    assert out["regen_count"] == 0
    assert out["downgraded_to_tentative"] is False


def test_regen_once_then_pass():
    calls = {"n": 0}

    def regen():
        calls["n"] += 1
        return "revised critique"

    reflections = [
        {"requires_regeneration": True, "quality_score": 0.5},
        {"requires_regeneration": False, "quality_score": 0.9},
    ]

    with patch("framed.analysis.critique_finalization._reflect", side_effect=reflections):
        out = finalize_critique_with_reflection(
            "bad critique",
            _REASONER,
            regenerate_fn=regen,
        )
    assert calls["n"] == 1
    assert out["regen_count"] == 1
    assert out["critique"] == "revised critique"
    assert out["downgraded_to_tentative"] is False


def test_downgrade_after_max_regens():
    with patch("framed.analysis.critique_finalization._reflect") as mock_reflect:
        mock_reflect.return_value = {"requires_regeneration": True, "quality_score": 0.4}
        out = finalize_critique_with_reflection(
            "still bad",
            _REASONER,
            regenerate_fn=lambda: "still bad again",
        )
    assert out["regen_count"] == 1
    assert out["downgraded_to_tentative"] is True
    assert out["critique"].startswith("One plausible reading is:")
    assert out["reflection_report"]["requires_regeneration"] is False


def test_skips_when_no_reasoner():
    out = finalize_critique_with_reflection("text", {})
    assert out["reflection_report"] is None
    assert out["regen_count"] == 0


def test_vocab_guard_detects_banned_terms():
    rules = ["Do not over-poeticize this."]
    assert check_vocab_guard("the soil whisper of winter", rules) is True
    assert check_vocab_guard("a tapestry of time", rules) is True
    assert check_vocab_guard("a silent symphony of nature", rules) is True
    assert check_vocab_guard("plain factual description", rules) is False


def test_vocab_guard_downgrades_when_rules_active():
    rules = ["Do not over-poeticize this."]
    with patch("framed.analysis.critique_finalization._reflect") as mock_reflect:
        mock_reflect.return_value = {"requires_regeneration": False, "quality_score": 0.9}
        with patch("framed.analysis.critique_finalization._active_correction_rules", return_value=rules):
            out = finalize_critique_with_reflection("earthy tones whisper across stone", _REASONER)
    assert out["vocab_guard_triggered"] is True
    assert out["downgraded_to_tentative"] is True
    assert out["critique"].startswith("One plausible reading is:")


def test_vocab_guard_inactive_without_rules():
    with patch("framed.analysis.critique_finalization._reflect") as mock_reflect:
        mock_reflect.return_value = {"requires_regeneration": False, "quality_score": 0.9}
        with patch("framed.analysis.critique_finalization._active_correction_rules", return_value=[]):
            out = finalize_critique_with_reflection("earthy tones whisper across stone", _REASONER)
    assert out["vocab_guard_triggered"] is False
    assert "whisper" in out["critique"]
