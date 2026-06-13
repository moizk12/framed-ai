# REF:D3 JSON parse + evidence strings for Model A prompts (split from intelligence_core.py)
import json
import re
from typing import Any, Dict, List, Optional

_UI_YOLO_OBJECTS = frozenset({"tv", "laptop", "mouse", "keyboard", "monitor", "cell phone"})

_BANNED_WHEN_SUPPRESSED = re.compile(
    r"\b(organic growth|reclamation|ivy|weathered stone|surface weathering|weathering)\b",
    re.I,
)


def organic_evidence_suppressed(visual_evidence: Optional[Dict[str, Any]]) -> bool:
    """True when IC_0015-A domain guard marked organic evidence not applicable."""
    if not visual_evidence or not visual_evidence.get("domain_guard_applied"):
        return False
    og = visual_evidence.get("organic_growth") or {}
    oi = visual_evidence.get("organic_integration") or {}
    return og.get("applicable") is False and oi.get("relationship") in (None, "none")


_UI_CAPTION_HINT = re.compile(
    r"\b(screen|monitor|ui|code|editor|laptop|display|interface|text|keyboard|program)\b",
    re.I,
)


def is_screenshot_ui_scene(visual_evidence: Optional[Dict[str, Any]]) -> bool:
    """True when scene_gate or signals indicate screenshot/UI/code content (IC_0017)."""
    if not visual_evidence:
        return False
    scene_gate = visual_evidence.get("scene_gate") or {}
    if str(scene_gate.get("scene_type", "")).lower() == "screenshot_ui":
        return True
    return ui_screen_scene_hint(visual_evidence)


def ui_screen_scene_hint(visual_evidence: Optional[Dict[str, Any]]) -> bool:
    """Route toward screen/UI/code interpretation when signals suggest digital content."""
    if not visual_evidence:
        return False
    scene_gate = visual_evidence.get("scene_gate") or {}
    signals = scene_gate.get("signals") or {}
    yolo_objects = {str(o).lower() for o in (signals.get("yolo_objects") or [])}
    if yolo_objects & _UI_YOLO_OBJECTS:
        return True
    caption = str(signals.get("clip_caption", "") or "")
    if _UI_CAPTION_HINT.search(caption):
        return True
    places_cat = str(signals.get("places_scene_category", "")).lower()
    green_cov = float((visual_evidence.get("organic_growth") or {}).get("green_coverage", 1.0))
    scene_type = str(scene_gate.get("scene_type", "")).lower()
    if places_cat == "artificial" and green_cov < 0.05 and _UI_CAPTION_HINT.search(caption):
        return True
    if scene_type == "interior_scene" and yolo_objects & {"tv", "laptop", "keyboard", "mouse"}:
        return True
    return False


def screenshot_critique_prompt_block(visual_evidence: Optional[Dict[str, Any]]) -> str:
    """IC_0017: screenshot/UI critique routing — always when scene is screen-like."""
    if not is_screenshot_ui_scene(visual_evidence):
        return ""
    return "\n".join(
        [
            "SCREEN/UI CRITIQUE ROUTING (IC_0017):",
            "- Primary: screen, UI, code editor, webpage screenshot, or photo-of-screen.",
            "- Critique MUST discuss: layout, text readability, hierarchy, contrast, glare, crop, text density, screen/photo quality.",
            "- FORBIDDEN: fine-art mood, street/room photography framing, organic growth, weathered stone, reclamation, poetic symbolism.",
            "- Never describe screenshot content as weathered stone, organic growth, or interior room photography.",
            "- Use terms: screen, UI, interface, layout, readability, text, contrast, hierarchy, crop, display, navigation.",
        ]
    )


def routing_prompt_blocks(
    visual_evidence: Optional[Dict[str, Any]],
    perception_composition: Optional[Dict[str, Any]] = None,
    perception_technical: Optional[Dict[str, Any]] = None,
) -> str:
    """Combined domain-guard + screenshot + composition + technical routing prompt constraints."""
    if perception_composition is None and visual_evidence:
        perception_composition = visual_evidence.get("perception_composition")
    if perception_technical is None and visual_evidence:
        perception_technical = visual_evidence.get("perception_technical")
    parts = [
        domain_guard_prompt_block(visual_evidence),
        screenshot_critique_prompt_block(visual_evidence),
        composition_critique_prompt_block(visual_evidence, perception_composition),
        technical_critique_prompt_block(visual_evidence, perception_technical),
    ]
    return "\n".join(p for p in parts if p)


def domain_guard_prompt_block(visual_evidence: Optional[Dict[str, Any]]) -> str:
    """Prompt constraints when organic evidence is suppressed downstream of domain guard."""
    if not organic_evidence_suppressed(visual_evidence):
        return ""
    lines: List[str] = [
        "CONSTRAINT (domain guard):",
        "- Do NOT interpret this image as weathered stone, ivy reclamation, or organic growth narrative.",
        "- Do NOT mention: organic growth, reclamation, ivy, weathered stone, surface weathering unless visibly supported.",
        "- Base primary recognition on scene_gate.scene_type and visible objects only.",
    ]
    if ui_screen_scene_hint(visual_evidence):
        lines.extend(
            [
                "- Prefer interpretation: screen, UI, code editor, digital display, or photo-of-screen.",
                "- Discuss layout, text readability, contrast, glare, crop — not nature or material weathering.",
            ]
        )
    return "\n".join(lines)


def sanitize_primary_when_suppressed(primary: str, visual_evidence: Optional[Dict[str, Any]]) -> str:
    """Safety net: strip banned organic/weathering terms from Layer 1 primary when guard active."""
    if not primary or not organic_evidence_suppressed(visual_evidence):
        return primary
    if not _BANNED_WHEN_SUPPRESSED.search(primary):
        return primary
    cleaned = _BANNED_WHEN_SUPPRESSED.sub("", primary)
    cleaned = re.sub(r"\b(with|and|no)\s+(significant\s+)?\s*", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;")
    if len(cleaned) < 25:
        if ui_screen_scene_hint(visual_evidence):
            return (
                "I see a screen or digital display showing UI or code content — "
                "layout, text readability, contrast, and crop are the primary subjects."
            )
        scene_gate = visual_evidence.get("scene_gate") or {}
        scene_type = str(scene_gate.get("scene_type", "a scene")).replace("_", " ")
        return f"I see {scene_type} based on visible objects and scene context."
    return cleaned


_FINE_ART_ON_SCREEN = re.compile(
    r"\b(street\s+scene|interior\s+room|fine\s+art|gallery|weathered\s+stone|organic\s+growth|"
    r"reclamation|ivy|poetic|ethereal|souls?)\b",
    re.I,
)


def sanitize_primary_screenshot(primary: str, visual_evidence: Optional[Dict[str, Any]]) -> str:
    """Ensure Layer 1 primary uses screen/UI language when screenshot scene is detected."""
    if not primary or not is_screenshot_ui_scene(visual_evidence):
        return primary
    if _FINE_ART_ON_SCREEN.search(primary) or not _UI_CAPTION_HINT.search(primary):
        return (
            "I see a screen or digital display showing UI, code, or webpage content — "
            "layout, text readability, contrast, hierarchy, and crop are the primary subjects."
        )
    return primary


_COMPOSITION_DEPTH_TERMS = re.compile(
    r"\b(foreground|midground|background|focal point|visual hierarchy|depth|layering|"
    r"framing|balance|leading lines?|negative space|horizon)\b",
    re.I,
)


def is_composition_depth_scene(visual_evidence: Optional[Dict[str, Any]]) -> bool:
    """True when critique should name concrete visual structure (IC_0018)."""
    if not visual_evidence or is_screenshot_ui_scene(visual_evidence):
        return False
    scene_gate = visual_evidence.get("scene_gate") or {}
    scene_type = str(scene_gate.get("scene_type", "")).lower()
    return scene_type != "screenshot_ui"


def format_perception_composition_lines(perception_composition: Optional[Dict[str, Any]]) -> List[str]:
    """Deterministic composition cues for Model A prompts."""
    if not perception_composition or not perception_composition.get("available"):
        return []
    lines: List[str] = []
    subject_framing = perception_composition.get("subject_framing") or {}
    if subject_framing:
        position = subject_framing.get("position", "")
        size = subject_framing.get("size", "")
        if position or size:
            lines.append(f"- Subject framing: position={position}, size={size}")
    for key in ("symmetry", "line_pattern", "line_style"):
        val = perception_composition.get(key)
        if val:
            lines.append(f"- Composition signal: {key}={val}")
    return lines


def composition_critique_prompt_block(
    visual_evidence: Optional[Dict[str, Any]],
    perception_composition: Optional[Dict[str, Any]] = None,
) -> str:
    """IC_0018: require foreground/midground/background and focal hierarchy in critique."""
    if not is_composition_depth_scene(visual_evidence):
        return ""
    comp_lines = format_perception_composition_lines(perception_composition)
    comp_section = "\n".join(comp_lines) if comp_lines else "- (use visible objects and scene_gate to infer depth layers)"
    return "\n".join(
        [
            "COMPOSITION DEPTH ROUTING (IC_0018):",
            "- Critique MUST name concrete visual structure: foreground, midground, background, focal point or hierarchy, depth/layering.",
            "- Mention visual path or how the eye travels through the frame when layers are visible.",
            "- FORBIDDEN: generic praise alone (nice composition, beautiful, stunning) without structural vocabulary.",
            "- Use perception composition signals when present:",
            comp_section,
        ]
    )


def count_composition_terms(text: str) -> int:
    """Count IC_0018 composition vocabulary terms (for finalize guard)."""
    if not text:
        return 0
    return len(_COMPOSITION_DEPTH_TERMS.findall(text))


_TECHNICAL_CRITIQUE_TERMS = re.compile(
    r"\b(blur|motion blur|noise|grain|flat light|underexpos|overexpos|focus|sharpness|"
    r"compression|shutter|aperture|white balance|retake|crop)\b",
    re.I,
)


def has_technical_weakness(perception_technical: Optional[Dict[str, Any]]) -> bool:
    """True when pipeline technical stats suggest flawed capture."""
    if not perception_technical:
        return False
    sharpness = perception_technical.get("sharpness")
    contrast = perception_technical.get("contrast")
    brightness = perception_technical.get("brightness")
    if sharpness is not None and sharpness < 100:
        return True
    if contrast is not None and contrast < 35:
        return True
    if brightness is not None and (brightness < 45 or brightness > 210):
        return True
    return False


def is_technical_practicality_scene(
    visual_evidence: Optional[Dict[str, Any]],
    perception_technical: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when critique should name actionable capture/technical advice (IC_0019)."""
    if not visual_evidence or is_screenshot_ui_scene(visual_evidence):
        return False
    if perception_technical is None:
        perception_technical = visual_evidence.get("perception_technical") or {}
    scene_gate = visual_evidence.get("scene_gate") or {}
    scene_type = str(scene_gate.get("scene_type", "")).lower()
    if scene_type in ("object_dense", "interior_scene"):
        return True
    return has_technical_weakness(perception_technical)


def format_perception_technical_lines(perception_technical: Optional[Dict[str, Any]]) -> List[str]:
    """Deterministic technical cues for Model A prompts."""
    if not perception_technical:
        return []
    lines: List[str] = []
    brightness = perception_technical.get("brightness")
    contrast = perception_technical.get("contrast")
    sharpness = perception_technical.get("sharpness")
    if brightness is not None:
        lines.append(f"- Technical stats: brightness={brightness:.1f}, contrast={contrast:.1f}, sharpness={sharpness:.1f}")
    if has_technical_weakness(perception_technical):
        if sharpness is not None and sharpness < 100:
            lines.append("- Weakness signal: low sharpness (possible blur or motion)")
        if contrast is not None and contrast < 35:
            lines.append("- Weakness signal: low contrast (flat light)")
        if brightness is not None and brightness < 45:
            lines.append("- Weakness signal: underexposure")
        if brightness is not None and brightness > 210:
            lines.append("- Weakness signal: overexposure")
    return lines


def technical_critique_prompt_block(
    visual_evidence: Optional[Dict[str, Any]],
    perception_technical: Optional[Dict[str, Any]] = None,
) -> str:
    """IC_0019: require focus/sharpness/exposure/crop/retake advice on weak or cluttered photos."""
    if not is_technical_practicality_scene(visual_evidence, perception_technical):
        return ""
    tech_lines = format_perception_technical_lines(perception_technical)
    tech_section = "\n".join(tech_lines) if tech_lines else "- (infer capture flaws from visible blur, flat light, crop, or clutter)"
    return "\n".join(
        [
            "TECHNICAL PRACTICALITY ROUTING (IC_0019):",
            "- Critique MUST include actionable technical advice: focus, sharpness, blur, exposure, flat light, noise, crop, or retake.",
            "- Name at least one concrete capture flaw or improvement when the image is weak, cluttered, or phone-snapshot quality.",
            "- FORBIDDEN: mood-only or aesthetic prose without any technical vocabulary on flawed photos.",
            "- Use perception technical signals when present:",
            tech_section,
        ]
    )


def count_technical_terms(text: str) -> int:
    """Count IC_0019 technical vocabulary terms (for finalize guard)."""
    if not text:
        return 0
    return len(_TECHNICAL_CRITIQUE_TERMS.findall(text))


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
    suppressed = organic_evidence_suppressed(visual_evidence)

    scene_gate = visual_evidence.get("scene_gate", {}) if isinstance(visual_evidence, dict) else {}
    if isinstance(scene_gate, dict) and scene_gate:
        scene_type = scene_gate.get("scene_type", "unknown")
        is_surface_study = bool(scene_gate.get("is_surface_study", False))
        lines.append(f"- Scene Gate: scene_type={scene_type}, is_surface_study={is_surface_study}")
        signals = scene_gate.get("signals") or {}
        yolo_objects = signals.get("yolo_objects") or []
        if yolo_objects:
            lines.append(f"- Visible objects: {', '.join(str(o) for o in yolo_objects[:8])}")
        clip_caption = signals.get("clip_caption")
        if clip_caption:
            lines.append(f"- CLIP scene caption: \"{clip_caption}\"")
        scene_type = str(scene_gate.get("scene_type", "unknown"))
        if scene_type == "screenshot_ui" or is_screenshot_ui_scene(visual_evidence):
            lines.append(
                "- SCENE ROUTING (IC_0017): screenshot/UI/code — interpret layout, readability, glare, contrast, crop; NOT street/room/fine-art."
            )
        elif scene_type == "object_dense":
            lines.append(
                "- SCENE ROUTING: object-dense/workshop/clutter wall — describe tools, shelves, wall objects; NOT a street scene."
            )
        elif scene_type == "interior_scene" and not ui_screen_scene_hint(visual_evidence):
            lines.append(
                "- SCENE ROUTING: interior room — describe walls, shelves, windows, decay, clutter; NOT a digital display unless monitor/UI visible."
            )
        if suppressed:
            if ui_screen_scene_hint(visual_evidence):
                lines.append(
                    "- SCENE ROUTING: screen/UI/code-like content — interpret layout, readability, glare, crop, contrast."
                )
            else:
                lines.append(
                    "- IMPORTANT: Scene depiction — do NOT center recognition on weathered stone, reclamation, or organic growth."
                )
        elif not is_surface_study:
            lines.append(
                "- IMPORTANT: This appears to be a scene depiction. Treat material_condition / organic_integration as background-only metrics; "
                "do NOT center recognition on 'weathered stone / reclamation'."
            )

    if suppressed:
        material_condition = visual_evidence.get("material_condition", {}) or {}
        condition = material_condition.get("condition", "neutral")
        lines.extend(
            [
                "Visual evidence (domain guard active):",
                "- Organic growth: NOT APPLICABLE (insufficient visible green; do not infer ivy/reclamation)",
                f"- Material surface: {condition} / not a weathering study",
                "- Scene focus: use scene_gate and visible objects, not surface-reclamation narrative",
            ]
        )
    else:
        organic_growth = visual_evidence.get("organic_growth", {})
        if organic_growth:
            green_coverage = organic_growth.get("green_coverage", 0.0)
            salience = organic_growth.get("salience", "minimal")
            green_locations = organic_growth.get("green_locations", "none")
            confidence = organic_growth.get("confidence", 0.0)
            lines.append(
                f"- Organic Growth: coverage={green_coverage:.3f}, salience={salience}, "
                f"locations={green_locations} (confidence: {confidence:.2f})"
            )

        is_surface_study = bool(scene_gate.get("is_surface_study", True)) if isinstance(scene_gate, dict) else True
        if is_surface_study:
            material_condition = visual_evidence.get("material_condition", {})
            if material_condition:
                condition = material_condition.get("condition", "unknown")
                surface_roughness = material_condition.get("surface_roughness", 0.0)
                edge_degradation = material_condition.get("edge_degradation", 0.0)
                confidence = material_condition.get("confidence", 0.0)
                lines.append(
                    f"- Material Condition: {condition}, roughness={surface_roughness:.3f}, "
                    f"edge_degradation={edge_degradation:.3f} (confidence: {confidence:.2f})"
                )

            organic_integration = visual_evidence.get("organic_integration", {})
            if organic_integration:
                relationship = organic_integration.get("relationship", "none")
                integration_level = organic_integration.get("integration_level", "none")
                overlap_ratio = organic_integration.get("overlap_ratio", 0.0)
                confidence = organic_integration.get("confidence", 0.0)
                lines.append(
                    f"- Organic Integration: relationship={relationship}, level={integration_level}, "
                    f"overlap={overlap_ratio:.3f} (confidence: {confidence:.2f})"
                )

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
