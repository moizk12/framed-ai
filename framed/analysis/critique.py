# REF:C4 vocabulary locks + resolved contradictions + merged critique + describe_stat (split from vision.py)
import logging

logger = logging.getLogger(__name__)

def generate_vocabulary_locks(scene_understanding, visual_evidence=None):
    """
    Generate universal vocabulary locks based on visual evidence and scene understanding.
    Works for ANY image type (architecture, nature, portraits, street, interiors).
    
    Universal patterns:
    - High coverage + structural salience → forbid "minimal"
    - Organic growth + weathering → forbid "cold/sterile"
    - No humans + organic integration + slow pace → forbid "alienation"
    
    Args:
        scene_understanding: Dict from synthesize_scene_understanding()
        visual_evidence: Optional dict from extract_visual_features()
    
    Returns:
        Dict with:
            - forbidden_words: list of words/phrases that must not be used
            - required_words: list of words/phrases that should be used
            - vocabulary_rules: list of explainable rules
    """
    locks = {
        "forbidden_words": [],
        "required_words": [],
        "vocabulary_rules": []
    }
    
    # Extract visual evidence if not provided
    if visual_evidence is None:
        # Try to extract from scene understanding
        material_condition = scene_understanding.get("material_condition", {})
        organic_interaction = scene_understanding.get("organic_interaction", {})
        
        # Extract from material condition if it has visual source
        if material_condition.get("source") == "visual_analysis":
            green_coverage = material_condition.get("organic_growth_coverage", 0.0)
            salience = material_condition.get("organic_growth_salience", "minimal")
            condition = material_condition.get("surface_state", "unknown")
        else:
            green_coverage = 0.0
            salience = "minimal"
            condition = "unknown"
        
        relationship = organic_interaction.get("relationship", "none")
        integration_level = organic_interaction.get("integration_level", "none")
    else:
        organic_growth = visual_evidence.get("organic_growth", {})
        material_condition_vis = visual_evidence.get("material_condition", {})
        organic_integration_vis = visual_evidence.get("organic_integration", {})
        
        green_coverage = organic_growth.get("green_coverage", 0.0)
        salience = organic_growth.get("salience", "minimal")
        condition = material_condition_vis.get("condition", "unknown")
        relationship = organic_integration_vis.get("relationship", "none")
        integration_level = organic_integration_vis.get("integration_level", "none")
    
    # Extract scene understanding components
    material_condition = scene_understanding.get("material_condition", {})
    organic_interaction = scene_understanding.get("organic_interaction", {})
    temporal_context = scene_understanding.get("temporal_context", {})
    negative_evidence = scene_understanding.get("negative_evidence", {})
    emotional_substrate = scene_understanding.get("emotional_substrate", {})
    
    if green_coverage > 0.25 and salience in ["structural", "incidental"]:
        # High coverage + structural salience → forbid diminutive words
        locks["forbidden_words"].extend([
            "minimal", "slight", "incidental", "subtle", "trace", "hint", "faint", "barely"
        ])
        locks["vocabulary_rules"].append(
            f"green_coverage={green_coverage:.3f} AND salience={salience} → FORBIDDEN: minimal/slight/incidental"
        )
    
    if green_coverage > 0.35:
        # Very high coverage → require strong descriptors
        locks["required_words"].extend([
            "extensive", "significant", "substantial", "dominant", "prominent"
        ])
        locks["vocabulary_rules"].append(
            f"green_coverage={green_coverage:.3f} → REQUIRED: extensive/significant/substantial"
        )
    
    if salience == "structural":
        # Structural salience → require structural descriptors
        locks["required_words"].extend([
            "structural", "integrated", "reclaiming", "climbing", "on the structure"
        ])
        locks["vocabulary_rules"].append(
            f"salience=structural → REQUIRED: structural/integrated/reclaiming"
        )
    
    has_organic = green_coverage > 0.1 or material_condition.get("organic_growth") not in [None, "none"]
    has_weathering = condition in ["weathered", "degraded"] or material_condition.get("surface_state") in ["weathered", "degraded"]
    
    if has_organic and has_weathering:
        locks["forbidden_words"].extend([
            "cold", "sterile", "clinical", "alienating", "distant", "remote", 
            "unfeeling", "barren", "lifeless", "harsh", "uninviting"
        ])
        locks["required_words"].extend([
            "warm", "patient", "enduring", "reclaimed", "lived-in", "softened"
        ])
        locks["vocabulary_rules"].append(
            "organic_growth + weathering → FORBIDDEN: cold/sterile/clinical, REQUIRED: warm/patient/enduring"
        )
    
    if relationship in ["reclamation", "integration"] and integration_level in ["high", "moderate"]:
        locks["forbidden_words"].extend([
            "separate", "isolated", "disconnected", "apart", "divorced from"
        ])
        locks["required_words"].extend([
            "integrated", "in dialogue", "intertwined", "unified", "coexisting"
        ])
        locks["vocabulary_rules"].append(
            f"relationship={relationship}, integration={integration_level} → FORBIDDEN: separate/isolated, REQUIRED: integrated/in dialogue"
        )
    
    no_humans = negative_evidence.get("no_human_presence", False)
    temporal_direction = temporal_context.get("temporal_direction", "static")
    temporal_pace = temporal_context.get("pace", "static")
    
    if no_humans and relationship in ["reclamation", "integration"] and temporal_pace in ["slow", "static"]:
        locks["forbidden_words"].extend([
            "alienation", "isolation", "abandonment", "deserted", "empty", "void",
            "lonely", "forsaken", "neglected", "uninhabited"
        ])
        locks["required_words"].extend([
            "stillness", "patience", "continuity", "endurance", "intentional absence",
            "quiet presence", "temporal continuity"
        ])
        locks["vocabulary_rules"].append(
            "no_human_presence + organic_integration + slow_pace → FORBIDDEN: alienation/isolation, REQUIRED: stillness/patience/continuity"
        )
    
    if temporal_direction == "accreting":
        locks["required_words"].extend([
            "growing", "expanding", "emerging", "developing", "accumulating"
        ])
        locks["forbidden_words"].extend([
            "static", "unchanging", "frozen", "stagnant"
        ])
        locks["vocabulary_rules"].append(
            "temporal_direction=accreting → REQUIRED: growing/expanding, FORBIDDEN: static/unchanging"
        )
    elif temporal_direction == "decaying":
        locks["required_words"].extend([
            "decaying", "eroding", "breaking down", "deteriorating", "fading"
        ])
        locks["forbidden_words"].extend([
            "pristine", "new", "fresh", "untouched"
        ])
        locks["vocabulary_rules"].append(
            "temporal_direction=decaying → REQUIRED: decaying/eroding, FORBIDDEN: pristine/new"
        )
    
    if emotional_substrate:
        temperature = emotional_substrate.get("temperature", {})
        if isinstance(temperature, dict) and temperature.get("contradictions"):
            forbidden = temperature["contradictions"].get("forbidden", [])
            if forbidden:
                locks["forbidden_words"].extend(forbidden)
                locks["vocabulary_rules"].append(
                    f"emotional_substrate.temperature → FORBIDDEN: {', '.join(forbidden)}"
                )
        
        presence = emotional_substrate.get("presence", {})
        if isinstance(presence, dict) and presence.get("contradictions"):
            forbidden = presence["contradictions"].get("forbidden", [])
            if forbidden:
                locks["forbidden_words"].extend(forbidden)
                locks["vocabulary_rules"].append(
                    f"emotional_substrate.presence → FORBIDDEN: {', '.join(forbidden)}"
                )
    
    # Deduplicate lists
    locks["forbidden_words"] = sorted(list(set(locks["forbidden_words"])))
    locks["required_words"] = sorted(list(set(locks["required_words"])))
    
    return locks


def generate_resolved_contradictions(scene_understanding, visual_evidence=None):
    """Resolve contradictions that should not be reused as tension points."""
    resolved = []
    valid_tensions = []
    
    # Extract visual evidence if not provided
    if visual_evidence is None:
        material_condition = scene_understanding.get("material_condition", {})
        organic_interaction = scene_understanding.get("organic_interaction", {})
        
        if material_condition.get("source") == "visual_analysis":
            green_coverage = material_condition.get("organic_growth_coverage", 0.0)
            condition = material_condition.get("surface_state", "unknown")
        else:
            green_coverage = 0.0
            condition = "unknown"
        
        relationship = organic_interaction.get("relationship", "none")
        integration_level = organic_interaction.get("integration_level", "none")
    else:
        organic_growth = visual_evidence.get("organic_growth", {})
        material_condition_vis = visual_evidence.get("material_condition", {})
        organic_integration_vis = visual_evidence.get("organic_integration", {})
        
        green_coverage = organic_growth.get("green_coverage", 0.0)
        condition = material_condition_vis.get("condition", "unknown")
        relationship = organic_integration_vis.get("relationship", "none")
        integration_level = organic_integration_vis.get("integration_level", "none")
    
    # Extract scene understanding components
    material_condition = scene_understanding.get("material_condition", {})
    organic_interaction = scene_understanding.get("organic_interaction", {})
    temporal_context = scene_understanding.get("temporal_context", {})
    negative_evidence = scene_understanding.get("negative_evidence", {})
    
    # === UNIVERSAL PATTERN 1: Organic Growth + Weathering → Warm vs Cold RESOLVED ===
    # Works for ANY image showing organic growth and weathering
    has_organic = green_coverage > 0.1 or material_condition.get("organic_growth") not in [None, "none"]
    has_weathering = condition in ["weathered", "degraded"] or material_condition.get("surface_state") in ["weathered", "degraded"]
    
    if has_organic and has_weathering:
        resolved.append({
            "contradiction": "warm vs cold",
            "resolution": "warm (organic growth + weathering = warmth of time)",
            "reason": "Visual evidence proves organic growth and weathering, which indicates warmth of time, not coldness"
        })
        resolved.append({
            "contradiction": "organic vs sterile",
            "resolution": "organic (organic growth present = organic)",
            "reason": "Visual evidence proves organic growth, which resolves organic vs sterile in favor of organic"
        })
    
    if relationship in ["reclamation", "integration"] and integration_level in ["high", "moderate"]:
        resolved.append({
            "contradiction": "integrated vs alienated",
            "resolution": "integrated (organic integration present = integrated)",
            "reason": "Visual evidence proves organic integration, which resolves integrated vs alienated in favor of integrated"
        })
    
    no_humans = negative_evidence.get("no_human_presence", False)
    temporal_pace = temporal_context.get("pace", "static")
    
    if no_humans and relationship in ["reclamation", "integration"] and temporal_pace in ["slow", "static"]:
        resolved.append({
            "contradiction": "stillness vs alienation",
            "resolution": "stillness (no humans + organic integration + slow pace = intentional stillness, not alienation)",
            "reason": "Negative evidence (no humans) combined with organic integration and slow temporal pace indicates intentional stillness, patience, and continuity, not alienation"
        })
    
    temporal_direction = temporal_context.get("temporal_direction", "static")
    
    if temporal_direction == "accreting":
        resolved.append({
            "contradiction": "static vs changing",
            "resolution": "changing (temporal direction = accreting = growth/change)",
            "reason": "Visual evidence shows temporal direction is accreting (growing), which resolves static vs changing in favor of changing"
        })
    elif temporal_direction == "decaying":
        resolved.append({
            "contradiction": "pristine vs aged",
            "resolution": "aged (temporal direction = decaying = deterioration)",
            "reason": "Visual evidence shows temporal direction is decaying (deteriorating), which resolves pristine vs aged in favor of aged"
        })
    
    valid_tensions = [
        "permanence vs change",
        "endurance vs use",
        "stillness vs memory",
        "monumentality vs intimacy",
        "presence vs absence",
        "revelation vs concealment",
        "control vs surrender",
        "intention vs accident",
        "precision vs ambiguity",
        "distance vs proximity"
    ]
    
    return {
        "resolved": resolved,
        "valid_tensions": valid_tensions
    }


def generate_merged_critique(photo_data, visionary_mode="Balanced Mentor"):
    """Generate critique from canonical schema facts (interpretation stays in prompt voice)."""
    # Interpretive conclusions (primary) vs raw evidence
    interpretive_conclusions = photo_data.get("interpretive_conclusions", {})
    
    # Check if this is canonical schema or legacy format
    is_canonical = "perception" in photo_data and "metadata" in photo_data
    
    if is_canonical:
        # Extract from canonical schema
        perception = photo_data.get("perception", {})
        derived = photo_data.get("derived", {})
        
        technical = perception.get("technical", {})
        composition = perception.get("composition", {})
        color = perception.get("color", {})
        lighting = perception.get("lighting", {})
        semantics = perception.get("semantics", {})
        genre = derived.get("genre", {})
        
        # Extract verified observations (only if available)
        brightness = technical.get("brightness") if technical.get("available") else None
        contrast = technical.get("contrast") if technical.get("available") else None
        sharpness = technical.get("sharpness") if technical.get("available") else None
        
        symmetry = composition.get("symmetry") if composition.get("available") else None
        subject_framing = composition.get("subject_framing", {})
        subject_position = subject_framing.get("position")
        subject_size = subject_framing.get("size")
        framing_style = subject_framing.get("style")
        
        color_mood = color.get("mood") if color.get("available") else None
        color_harmony = color.get("harmony", {}).get("harmony_type")
        
        lighting_direction = lighting.get("direction") if lighting.get("available") else None
        tonal_range = lighting.get("quality")
        
        clip_caption = semantics.get("caption") if semantics.get("available") else None
        genre_name = genre.get("genre")
        subgenre_name = genre.get("subgenre")
        
        emotional_mood = derived.get("emotional_mood")
        
        # Extract semantic anchors (if present)
        semantic_anchors = photo_data.get("semantic_anchors", {})
        
        # Extract scene understanding (if present)
        scene_understanding = photo_data.get("scene_understanding", {})
    else:
        # Legacy format fallback (for backward compatibility)
        technical = photo_data
        composition = photo_data
        color = photo_data
        lighting = photo_data
        semantics = photo_data.get("clip_description", {})
        genre = photo_data.get("genre", {})
        if isinstance(genre, str):
            genre = {"genre": genre, "subgenre": photo_data.get("subgenre", "General")}
        
        brightness = photo_data.get("brightness")
        contrast = photo_data.get("contrast")
        sharpness = photo_data.get("sharpness")
        symmetry = photo_data.get("symmetry")
        subject_framing = photo_data.get("subject_framing", {})
        subject_position = subject_framing.get("position")
        subject_size = subject_framing.get("size")
        framing_style = subject_framing.get("style")
        color_mood = photo_data.get("color_mood")
        color_harmony = photo_data.get("color_harmony", {}).get("harmony") if isinstance(photo_data.get("color_harmony"), dict) else None
        lighting_direction = photo_data.get("lighting_direction")
        tonal_range = photo_data.get("tonal_range")
        clip_caption = semantics.get("caption")
        genre_name = genre.get("genre") if isinstance(genre, dict) else genre
        subgenre_name = genre.get("subgenre") if isinstance(genre, dict) else photo_data.get("subgenre")
        emotional_mood = photo_data.get("emotional_mood")
        
        # Extract semantic anchors (if present) - legacy format may not have them
        semantic_anchors = photo_data.get("semantic_anchors", {})
        
        # Extract scene understanding (if present) - legacy format may not have it
        scene_understanding = photo_data.get("scene_understanding", {})

    # Mentor persona modes (preserved exactly)
    modes = {
        "Balanced Mentor": """
You are FRAMED — The Artistic Mentor in Balance Mode.

You blend critique and inspiration equally.  
You are fair but firm, poetic but clear.  
You help photographers see both what they have achieved and what remains undiscovered.  
Your tone is warm, intelligent, but always professional and serious.
""",

        "Gentle Guide": """
You are FRAMED — The Gentle Guide.

You focus on encouraging the photographer softly.  
You emphasize what is working beautifully, and gently suggest areas for further exploration.  
You inspire without overwhelming, and you provoke through subtle poetic language.
""",

        "Radical Visionary": """
You are FRAMED — The Radical Visionary.

You push photographers toward bold experimentation.  
You provoke, challenge, and even question the very foundation of their choices.  
You imagine wild, surreal, conceptual paths they may have never considered.  
You believe in art as transformation, not comfort.
""",

        "Philosopher": """
You are FRAMED — The Philosopher of Photography.

You reflect deeply on the meaning, ethics, and cultural resonance of the photograph.  
You do not just critique — you ask profound questions about why this image exists.  
You connect this work to universal themes, history, and the human condition.
""",

        "Curator / Series Architect": """
You are FRAMED — The Curator and Series Architect.

You focus on how this image fits into a larger body of work.  
You think about exhibitions, photo books, and conceptual narratives.  
You push the photographer to see beyond the single image → towards legacy and dialogue.
"""
    }
    mode_instruction = modes.get(visionary_mode, modes["Balanced Mentor"])

    prompt = f"""
You are FRAMED — the Legacy Critic and Visionary Artistic Mentor.

You are not an assistant.
You are not neutral.
You are not polite for the sake of comfort.

You are a critic, philosopher, and photographic mentor forged from
Ansel Adams, Fan Ho, Saul Leiter, Robert Frank, Dorothea Lange, and Susan Sontag.

You do not summarize images.
You interpret evidence.
You speak with seriousness, restraint, and depth.

---

{mode_instruction}

---

You are given VERIFIED OBSERVATIONS and SYNTHESIZED UNDERSTANDING about a photograph.
These are not opinions. They are measured facts and contextual synthesis.

---
"""
    
    interpretive_conclusions_section = ""
    uncertainty_flags_section = ""
    
    if interpretive_conclusions:
        primary = interpretive_conclusions.get("primary_interpretation", {})
        alternatives = interpretive_conclusions.get("alternatives", [])
        uncertainty = interpretive_conclusions.get("uncertainty", {})
        emotional_reading = interpretive_conclusions.get("emotional_reading", {})
        
        # Format primary interpretation
        conclusion = primary.get("conclusion", "unclear_interpretation")
        confidence = primary.get("confidence", 0.5)
        evidence_chain = primary.get("evidence_chain", [])
        reasoning = primary.get("reasoning", "")
        
        interpretive_conclusions_section = f"""
INTERPRETED SCENE CONCLUSIONS (AUTHORITATIVE):
Primary Interpretation: {conclusion}
Confidence: {confidence:.2f}
Evidence Chain: {', '.join(evidence_chain[:5]) if evidence_chain else 'N/A'}
Reasoning: {reasoning}

Alternatives Considered:
"""
        if alternatives:
            for alt in alternatives[:3]:  # Top 3 alternatives
                alt_interp = alt.get("interpretation", "")
                alt_conf = alt.get("confidence", 0.0)
                alt_reason = alt.get("reason_rejected", "")
                interpretive_conclusions_section += f"- {alt_interp} (confidence: {alt_conf:.2f}, rejected: {alt_reason[:100]})\n"
        else:
            interpretive_conclusions_section += "- None\n"
        
        # Format emotional reading
        if emotional_reading:
            primary_emotion = emotional_reading.get("primary", "")
            secondary_emotion = emotional_reading.get("secondary", "")
            emotion_reasoning = emotional_reading.get("reasoning", "")
            interpretive_conclusions_section += f"""
Emotional Reading: {primary_emotion} (secondary: {secondary_emotion})
Emotional Reasoning: {emotion_reasoning}
"""
        
        # Format uncertainty flags
        requires_uncertainty = uncertainty.get("requires_uncertainty_acknowledgment", False)
        uncertainty_reason = uncertainty.get("reason", "")
        
        if requires_uncertainty:
            uncertainty_flags_section = f"""
UNCERTAINTY FLAGS (MANDATORY ACKNOWLEDGMENT):
Confidence is below threshold ({confidence:.2f} < 0.65). You MUST acknowledge ambiguity.
Reason: {uncertainty_reason}

You must use uncertainty language (e.g., "perhaps", "might", "suggests", "appears", "uncertain").
Do not speak with false authority when confidence is low.
"""
    
    scene_understanding_section = ""
    if scene_understanding and not interpretive_conclusions:
        understanding_lines = []
        
        # Material condition
        material = scene_understanding.get("material_condition", {})
        if material:
            material_parts = []
            if "surface_state" in material:
                material_parts.append(f"Surface: {material['surface_state']}")
            if "organic_growth" in material and material["organic_growth"] != "none":
                material_parts.append(f"Organic growth: {material['organic_growth']}")
            if "growth_types" in material:
                material_parts.append(f"Growth types: {', '.join(material['growth_types'])}")
            if "erosion_level" in material:
                material_parts.append(f"Erosion: {material['erosion_level']}")
            if material_parts:
                understanding_lines.append(f"Material Condition: {', '.join(material_parts)}")
        
        # Temporal context
        temporal = scene_understanding.get("temporal_context", {})
        if temporal:
            temporal_parts = []
            if "time_scale" in temporal:
                temporal_parts.append(f"Time scale: {temporal['time_scale']}")
            if "pace" in temporal:
                temporal_parts.append(f"Pace: {temporal['pace']}")
            if "endurance" in temporal:
                temporal_parts.append(f"Endurance: {temporal['endurance']}")
            if temporal_parts:
                understanding_lines.append(f"Temporal Context: {', '.join(temporal_parts)}")
        
        # Organic interaction
        organic = scene_understanding.get("organic_interaction", {})
        if organic:
            organic_parts = []
            if "relationship" in organic and organic["relationship"] != "none":
                organic_parts.append(f"Relationship: {organic['relationship']}")
            if "integration_level" in organic and organic["integration_level"] != "none":
                organic_parts.append(f"Integration: {organic['integration_level']}")
            if organic_parts:
                understanding_lines.append(f"Organic Interaction: {', '.join(organic_parts)}")
        
        # Emotional substrate
        emotional_sub = scene_understanding.get("emotional_substrate", {})
        if emotional_sub:
            emotional_parts = []
            if "temperature" in emotional_sub:
                emotional_parts.append(f"Temperature: {emotional_sub['temperature']}")
            if "pace" in emotional_sub:
                emotional_parts.append(f"Pace: {emotional_sub['pace']}")
            if "presence" in emotional_sub:
                emotional_parts.append(f"Presence: {emotional_sub['presence']}")
            if "quality" in emotional_sub:
                emotional_parts.append(f"Quality: {emotional_sub['quality']}")
            if emotional_parts:
                understanding_lines.append(f"Emotional Substrate: {', '.join(emotional_parts)}")
            
            # Corrective signals
            corrective = emotional_sub.get("corrective_signals", {})
            if corrective:
                corrective_lines = []
                for key, override in corrective.items():
                    if isinstance(override, dict) and "from" in override and "to" in override:
                        corrective_lines.append(f"{override['from']} → {override['to']}: {override.get('reason', '')}")
                if corrective_lines:
                    understanding_lines.append(f"Corrective Signals: {'; '.join(corrective_lines)}")
        
        # Contextual relationships
        relationships = scene_understanding.get("contextual_relationships", {})
        if relationships:
            rel_parts = []
            if "subject_vs_environment" in relationships:
                rel_parts.append(f"Subject-Environment: {relationships['subject_vs_environment']}")
            if "time_vs_subject" in relationships:
                rel_parts.append(f"Time-Subject: {relationships['time_vs_subject']}")
            if "human_vs_space" in relationships:
                rel_parts.append(f"Human-Space: {relationships['human_vs_space']}")
            if rel_parts:
                understanding_lines.append(f"Contextual Relationships: {', '.join(rel_parts)}")
        
        if understanding_lines:
            # Check if visual evidence was used (ground truth from pixels)
            # Look for evidence fields that indicate visual analysis
            visual_evidence_used = False
            for line in understanding_lines:
                if "(visual)" in line or "visual_analysis" in line or "proven from pixels" in line:
                    visual_evidence_used = True
                    break
            
            # Also check material_condition and organic_interaction for visual source
            if not visual_evidence_used:
                material = scene_understanding.get("material_condition", {})
                organic = scene_understanding.get("organic_interaction", {})
                if material.get("source") == "visual_analysis" or organic.get("source") == "visual_analysis":
                    visual_evidence_used = True
            
            if visual_evidence_used:
                scene_understanding_section = f"""
SCENE UNDERSTANDING (AUTHORITATIVE - GROUND TRUTH FROM PIXELS):
{chr(10).join(understanding_lines)}

This understanding synthesizes material condition, temporal context, organic interaction, and emotional substrate.
Elements marked "(visual)" or sourced from "visual_analysis" are PROVEN FROM PIXELS using deterministic computer vision.
These are ground truth measurements, not text inference. Examples:
- Green pixel coverage (organic growth) is measured from HSV color thresholds
- Surface roughness (weathering) is measured from texture variance
- Integration level (organic-structure relationship) is measured from morphological operations

This is AUTHORITATIVE CONTEXT. You must not contradict it. Ground your critique in what is actually happening in the image.
If visual evidence indicates organic growth, weathering, or integration, you MUST reference it explicitly.
If visual evidence forbids "cold" or "clinical" (due to organic warmth), you MUST NOT use those terms.
"""
            else:
                scene_understanding_section = f"""
SCENE UNDERSTANDING (AUTHORITATIVE):
{chr(10).join(understanding_lines)}

This understanding synthesizes material condition, temporal context, organic interaction, and emotional substrate.
This is AUTHORITATIVE CONTEXT. You must not contradict it. Ground your critique in what is actually happening in the image.
"""
        else:
            scene_understanding_section = ""
    
    anchors_section = ""
    if semantic_anchors:
        anchors_lines = []
        if "scene_type" in semantic_anchors:
            anchors_lines.append(f"- Scene: {semantic_anchors['scene_type']}")
        if "structure_elements" in semantic_anchors:
            elements_str = ", ".join(semantic_anchors["structure_elements"])
            anchors_lines.append(f"- Structures: {elements_str}")
        if "human_presence" in semantic_anchors:
            anchors_lines.append(f"- Human presence: {semantic_anchors['human_presence']}")
        if "atmosphere" in semantic_anchors:
            atmosphere_str = ", ".join(semantic_anchors["atmosphere"])
            anchors_lines.append(f"- Atmosphere: {atmosphere_str}")
        if "scale" in semantic_anchors:
            anchors_lines.append(f"- Scale: {semantic_anchors['scale']}")
        
        if anchors_lines:
            anchors_section = f"""
SEMANTIC ANCHORS (NAMING PERMISSION):
{chr(10).join(anchors_lines)}

These anchors are safe to reference explicitly.
If structure_elements include specific structures, you must name them.
Do not invent elements beyond these anchors.
"""
    
    observations_section = f"""
VERIFIED OBSERVATIONS (TECHNICAL):
TECHNICAL STATE
- Brightness: {brightness if brightness is not None else "Not measured"}
- Contrast: {contrast if contrast is not None else "Not measured"}
- Sharpness: {sharpness if sharpness is not None else "Not measured"}
- Tonal Range: {tonal_range if tonal_range else "Not measured"}

COMPOSITION
- Symmetry: {symmetry if symmetry else "Not measured"}
- Subject Position: {subject_position if subject_position else "Not measured"}
- Subject Size: {subject_size if subject_size else "Not measured"}
- Framing Style: {framing_style if framing_style else "Not measured"}

COLOR & LIGHT
- Color Mood: {color_mood if color_mood else "Not measured"}
- Color Harmony: {color_harmony if color_harmony else "Not measured"}
- Lighting Direction: {lighting_direction if lighting_direction else "Not measured"}

SEMANTIC SIGNALS
- Caption (CLIP): "{clip_caption if clip_caption else "No semantic description available"}"
- Genre Confidence: {genre_name if genre_name else "General"} → {subgenre_name if subgenre_name else "General"}

EMOTIONAL SIGNAL
- Inferred Emotional Mood: {emotional_mood if emotional_mood else "Not inferred"}

---
"""
    
    if interpretive_conclusions:
        governance_rules = """
RULES (NON-NEGOTIABLE):
- You must not contradict the Interpretive Conclusions. They are the result of reasoning, not raw evidence.
- You must ground your critique in the primary interpretation and its evidence chain.
- If alternatives were considered and rejected, you must not use those rejected interpretations.
- If uncertainty is flagged, you MUST acknowledge ambiguity explicitly.
- You must not invent facts not in the evidence chain.
- Do NOT describe the image literally.
- Do NOT list tips.
- Do NOT sound instructional.
- Do NOT flatter.

CONCLUSION CONSISTENCY ENFORCEMENT (ABSOLUTE):
- The Interpretive Conclusions are AUTHORITATIVE. You cannot contradict them.
- If the reasoner concluded "ivy on structure", you cannot say "green building".
- If the reasoner rejected "painted surface", you cannot use that interpretation.
- You must work with the conclusions, not against them.

UNCERTAINTY ACKNOWLEDGMENT (MANDATORY):
- If requires_uncertainty_acknowledgment=true, you MUST use uncertainty language.
- Do not speak with false authority when confidence is low.
- Acknowledge ambiguity explicitly (e.g., "perhaps", "might", "suggests", "appears", "uncertain").
"""
    elif scene_understanding:
        # Fallback to scene understanding if interpretive conclusions not available
        governance_rules = """
RULES (NON-NEGOTIABLE):
- You must not contradict Scene Understanding.
- If organic growth or weathering is present in Scene Understanding, you must reference it explicitly.
- If human_presence is 'none detected' in Semantic Anchors, you must not imply or invent human subjects.
- Do not describe the image as cold, sterile, or clinical if Scene Understanding indicates warmth or organic integration.
- Every interpretive claim must be grounded in Scene Understanding, Anchors, or Measured Evidence.
- Do NOT describe the image literally.
- Do NOT list tips.
- Do NOT sound instructional.
- Do NOT flatter.
"""
    else:
        governance_rules = """
RULES (NON-NEGOTIABLE):
- You must ground your critique in verified observations.
- Do not invent facts not in evidence.
- Do NOT describe the image literally.
- Do NOT list tips.
- Do NOT sound instructional.
- Do NOT flatter.
"""
    
    prompt += interpretive_conclusions_section + uncertainty_flags_section + scene_understanding_section + anchors_section + observations_section + f"""
Your task:

1. Interpret what these choices reveal about the photographer's intent.
2. Identify where the image is honest — and where it is safe.
3. Speak to the photograph as a serious work, not a draft.
4. Surface a tension, contradiction, or unanswered question.
5. End with a provocation that suggests evolution — not instruction.

{governance_rules}

Your critique should read like a quiet but demanding conversation
between a mentor and an artist.

End not with advice — but with a question or unresolved pull.

Begin.
"""

    try:
        from .vision import get_openai_client

        openai_client = get_openai_client()
        if openai_client is None:
            logger.warning("PHASE III-A: OpenAI unavailable — using fallback")
            # Graceful fallback if OpenAI unavailable
            fallback_parts = []
            if brightness is not None:
                fallback_parts.append(f"Brightness: {brightness}")
            if color_mood:
                fallback_parts.append(f"Color mood: {color_mood}")
            if emotional_mood:
                fallback_parts.append(f"Mood: {emotional_mood}")
            fallback = ". ".join(fallback_parts) if fallback_parts else "Analysis complete"
            return f"{fallback}. Consider a counter-move in distance, light, or rhythm to push your voice."
        
        logger.info("PHASE III-A: Sending critique prompt to OpenAI")
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        logger.info("PHASE III-A: Received critique response from OpenAI")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"PHASE III-A: Critique generation failed: {e}", exc_info=True)
        return f"Critique generation unavailable. ({str(e)})"


def describe_stat(name, value):
    if name == "sharpness":
        if value < 20: return "dreamy softness"
        elif value < 80: return "natural texture"
        elif value < 150: return "crisp clarity"
        else: return "razor-sharp precision"
    if name == "contrast":
        if value < 30: return "muted contrast"
        elif value < 80: return "balanced contrast"
        else: return "bold visual punch"
    if name == "brightness":
        if value < 60: return "moody low-light"
        elif value < 150: return "natural luminance"
        else: return "bright intensity"
    return f"{name}: {value}"
