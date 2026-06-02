# REF:D3 JSON parse + evidence strings for Model A prompts (split from intelligence_core.py)
import json
import re
from typing import Any, Dict, Optional


def _safe_parse_layer_json(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from Responses API output; handles plain text, markdown blocks, and empty responses."""
    if not content or not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def format_visual_evidence(visual_evidence: Dict[str, Any]) -> str:
    """Format extract_visual_features output for Model A prompts."""
    if not visual_evidence:
        return "No visual evidence available."

    lines = []

    scene_gate = visual_evidence.get("scene_gate", {}) if isinstance(visual_evidence, dict) else {}
    if isinstance(scene_gate, dict) and scene_gate:
        scene_type = scene_gate.get("scene_type", "unknown")
        is_surface_study = bool(scene_gate.get("is_surface_study", False))
        lines.append(f"- Scene Gate: scene_type={scene_type}, is_surface_study={is_surface_study}")
        if not is_surface_study:
            lines.append(
                "- IMPORTANT: This appears to be a scene depiction. Treat material_condition / organic_integration as background-only metrics; "
                "do NOT center recognition on 'weathered stone / reclamation'."
            )

    organic_growth = visual_evidence.get("organic_growth", {})
    if organic_growth:
        green_coverage = organic_growth.get("green_coverage", 0.0)
        salience = organic_growth.get("salience", "minimal")
        green_locations = organic_growth.get("green_locations", "none")
        confidence = organic_growth.get("confidence", 0.0)
        lines.append(f"- Organic Growth: coverage={green_coverage:.3f}, salience={salience}, locations={green_locations} (confidence: {confidence:.2f})")

    is_surface_study = bool(scene_gate.get("is_surface_study", True)) if isinstance(scene_gate, dict) else True
    if is_surface_study:
        material_condition = visual_evidence.get("material_condition", {})
        if material_condition:
            condition = material_condition.get("condition", "unknown")
            surface_roughness = material_condition.get("surface_roughness", 0.0)
            edge_degradation = material_condition.get("edge_degradation", 0.0)
            confidence = material_condition.get("confidence", 0.0)
            lines.append(f"- Material Condition: {condition}, roughness={surface_roughness:.3f}, edge_degradation={edge_degradation:.3f} (confidence: {confidence:.2f})")

        organic_integration = visual_evidence.get("organic_integration", {})
        if organic_integration:
            relationship = organic_integration.get("relationship", "none")
            integration_level = organic_integration.get("integration_level", "none")
            overlap_ratio = organic_integration.get("overlap_ratio", 0.0)
            confidence = organic_integration.get("confidence", 0.0)
            lines.append(f"- Organic Integration: relationship={relationship}, level={integration_level}, overlap={overlap_ratio:.3f} (confidence: {confidence:.2f})")

    overall_confidence = visual_evidence.get("overall_confidence", 0.0)
    if overall_confidence > 0:
        lines.append(f"- Overall Visual Confidence: {overall_confidence:.2f}")

    return "\n".join(lines) if lines else "Visual evidence incomplete."


def format_semantic_signals(analysis_result: Dict[str, Any]) -> str:
    """Format CLIP/YOLO/technical summary for Model A prompts."""
    if not analysis_result:
        return "No semantic signals available."

    lines = []

    semantics = analysis_result.get("perception", {}).get("semantics", {})
    if semantics.get("available"):
        caption = semantics.get("caption", "")
        tags = semantics.get("tags", [])
        if caption:
            lines.append(f"- CLIP Caption: \"{caption}\"")
        if tags:
            lines.append(f"- CLIP Tags: {', '.join(tags[:10])}")

    composition = analysis_result.get("perception", {}).get("composition", {})
    if composition.get("available"):
        subject_framing = composition.get("subject_framing", {})
        if subject_framing:
            position = subject_framing.get("position", "")
            size = subject_framing.get("size", "")
            if position or size:
                lines.append(f"- Subject: {position}, {size}")

    technical = analysis_result.get("perception", {}).get("technical", {})
    if technical.get("available"):
        brightness = technical.get("brightness")
        contrast = technical.get("contrast")
        sharpness = technical.get("sharpness")
        if brightness is not None:
            lines.append(f"- Technical: brightness={brightness:.1f}, contrast={contrast:.1f}, sharpness={sharpness:.1f}")

    color = analysis_result.get("perception", {}).get("color", {})
    if color.get("available"):
        mood = color.get("mood", "")
        if mood:
            lines.append(f"- Color Mood: {mood}")

    return "\n".join(lines) if lines else "Semantic signals incomplete."


def format_temporal_memory(temporal_memory: Optional[Dict[str, Any]]) -> str:
    if not temporal_memory:
        return "No temporal memory available (first time seeing this pattern)."

    lines = []

    patterns = temporal_memory.get("patterns", [])
    if patterns:
        lines.append("PAST INTERPRETATIONS:")
        for i, pattern in enumerate(patterns[:5], 1):
            interpretations = pattern.get("interpretations", [])
            if interpretations:
                latest = interpretations[-1]
                date = latest.get("date", "unknown")
                interpretation_summary = latest.get("interpretation", {}).get("what_i_see", "N/A")
                confidence = latest.get("confidence", 0.0)
                lines.append(f"  {i}. {date}: \"{interpretation_summary[:100]}...\" (confidence: {confidence:.2f})")

    user_trajectory = temporal_memory.get("user_trajectory", {})
    if user_trajectory:
        themes = user_trajectory.get("themes", [])
        evolution = user_trajectory.get("evolution", [])
        if themes:
            lines.append(f"USER THEMES: {', '.join(themes[:5])}")
        if evolution:
            lines.append("USER EVOLUTION:")
            for entry in evolution[-3:]:
                date = entry.get("date", "unknown")
                state = entry.get("state", "")
                lines.append(f"  - {date}: {state}")

    return "\n".join(lines) if lines else "Temporal memory incomplete."


def format_user_history(user_history: Optional[Dict[str, Any]]) -> str:
    if not user_history:
        return "No user history available (new user or insufficient data)."

    lines = []

    themes = user_history.get("themes", [])
    if themes:
        lines.append(f"RECURRING THEMES: {', '.join(themes[:5])}")

    patterns = user_history.get("patterns", [])
    if patterns:
        lines.append(f"PATTERNS: {', '.join(patterns[:5])}")

    evolution = user_history.get("evolution", [])
    if evolution:
        lines.append("EVOLUTION:")
        for entry in evolution[-3:]:
            date = entry.get("date", "unknown")
            state = entry.get("state", "")
            lines.append(f"  - {date}: {state}")

    return "\n".join(lines) if lines else "User history incomplete."
