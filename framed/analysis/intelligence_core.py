"""
FRAMED Intelligence Core

The 7-layer reasoning engine that gives FRAMED its brain.
Each layer reasons about images, not just matches patterns.

🔒 CRITICAL INVARIANT:
Learning must NEVER happen inside the LLM.
All learning, memory, and evolution must land in the memory layer.

This means:
- LLM prompts must NEVER ask the model to "remember", "learn", "update", or "store"
- All learning happens in Python code (temporal_memory.py, learning_system.py)
- Memory layer is queried BEFORE LLM calls, not updated during them
- Models are swappable because learning is in memory, not model weights

Architecture:
- Layer 1: Certain Recognition (Foundation)
- Layer 2: Meta-Cognition (Self-Awareness) [PRIORITY 1]
- Layer 3: Temporal Consciousness (Evolution) [PRIORITY 1]
- Layer 4: Emotional Resonance (Feeling)
- Layer 5: Continuity of Self (Shared History)
- Layer 6: Mentor Voice (Reasoning about mentorship)
- Layer 7: Self-Critique (Evolution)

All layers use Model A (Reasoning) via call_model_a() from llm_provider.py.
All reasoning is internal - not exposed to user unless needed.
"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional, List

from .llm_provider import call_model_a
from .ambiguity import (
    compute_plausibility,
    compute_ambiguity_score,
    compute_disagreement_state,
    apply_confidence_governor,
    compute_reasoning_cost_profile,
    compute_hypothesis_diversity,
)
from .self_assessment import get_governor_bias

logger = logging.getLogger(__name__)


def _safe_parse_layer_json(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from Responses API output; handles plain text, markdown blocks, and empty responses."""
    if not content or not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract from ```json ... ``` or ``` ... ```
    for pattern in (r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    # Find first complete {...} block
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


# ========================================================
# HELPER FUNCTIONS FOR FORMATTING EVIDENCE
# ========================================================

def format_visual_evidence(visual_evidence: Dict[str, Any]) -> str:
    """
    Format visual evidence for LLM prompts.
    
    Extracts key visual features from extract_visual_features() output.
    """
    if not visual_evidence:
        return "No visual evidence available."
    
    lines = []

    # Scene / abstraction gate (prevents surface-study bias leaking everywhere)
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
    
    # Organic growth
    organic_growth = visual_evidence.get("organic_growth", {})
    if organic_growth:
        green_coverage = organic_growth.get("green_coverage", 0.0)
        salience = organic_growth.get("salience", "minimal")
        green_locations = organic_growth.get("green_locations", "none")
        confidence = organic_growth.get("confidence", 0.0)
        lines.append(f"- Organic Growth: coverage={green_coverage:.3f}, salience={salience}, locations={green_locations} (confidence: {confidence:.2f})")
    
    # Material condition + organic integration are ONLY foregrounded for surface studies.
    is_surface_study = bool(scene_gate.get("is_surface_study", True)) if isinstance(scene_gate, dict) else True
    if is_surface_study:
        # Material condition
        material_condition = visual_evidence.get("material_condition", {})
        if material_condition:
            condition = material_condition.get("condition", "unknown")
            surface_roughness = material_condition.get("surface_roughness", 0.0)
            edge_degradation = material_condition.get("edge_degradation", 0.0)
            confidence = material_condition.get("confidence", 0.0)
            lines.append(f"- Material Condition: {condition}, roughness={surface_roughness:.3f}, edge_degradation={edge_degradation:.3f} (confidence: {confidence:.2f})")

        # Organic integration
        organic_integration = visual_evidence.get("organic_integration", {})
        if organic_integration:
            relationship = organic_integration.get("relationship", "none")
            integration_level = organic_integration.get("integration_level", "none")
            overlap_ratio = organic_integration.get("overlap_ratio", 0.0)
            confidence = organic_integration.get("confidence", 0.0)
            lines.append(f"- Organic Integration: relationship={relationship}, level={integration_level}, overlap={overlap_ratio:.3f} (confidence: {confidence:.2f})")
    
    # Overall confidence
    overall_confidence = visual_evidence.get("overall_confidence", 0.0)
    if overall_confidence > 0:
        lines.append(f"- Overall Visual Confidence: {overall_confidence:.2f}")
    
    return "\n".join(lines) if lines else "Visual evidence incomplete."


def format_semantic_signals(analysis_result: Dict[str, Any]) -> str:
    """
    Format semantic signals (CLIP, YOLO, composition) for LLM prompts.
    """
    if not analysis_result:
        return "No semantic signals available."
    
    lines = []
    
    # CLIP caption
    semantics = analysis_result.get("perception", {}).get("semantics", {})
    if semantics.get("available"):
        caption = semantics.get("caption", "")
        tags = semantics.get("tags", [])
        if caption:
            lines.append(f"- CLIP Caption: \"{caption}\"")
        if tags:
            lines.append(f"- CLIP Tags: {', '.join(tags[:10])}")  # Limit to top 10
    
    # YOLO objects
    composition = analysis_result.get("perception", {}).get("composition", {})
    if composition.get("available"):
        subject_framing = composition.get("subject_framing", {})
        if subject_framing:
            position = subject_framing.get("position", "")
            size = subject_framing.get("size", "")
            if position or size:
                lines.append(f"- Subject: {position}, {size}")
    
    # Technical measurements
    technical = analysis_result.get("perception", {}).get("technical", {})
    if technical.get("available"):
        brightness = technical.get("brightness")
        contrast = technical.get("contrast")
        sharpness = technical.get("sharpness")
        if brightness is not None:
            lines.append(f"- Technical: brightness={brightness:.1f}, contrast={contrast:.1f}, sharpness={sharpness:.1f}")
    
    # Color mood
    color = analysis_result.get("perception", {}).get("color", {})
    if color.get("available"):
        mood = color.get("mood", "")
        if mood:
            lines.append(f"- Color Mood: {mood}")
    
    return "\n".join(lines) if lines else "Semantic signals incomplete."


def format_temporal_memory(temporal_memory: Optional[Dict[str, Any]]) -> str:
    """
    Format temporal memory (past interpretations) for LLM prompts.
    """
    if not temporal_memory:
        return "No temporal memory available (first time seeing this pattern)."
    
    lines = []
    
    # Past interpretations
    patterns = temporal_memory.get("patterns", [])
    if patterns:
        lines.append("PAST INTERPRETATIONS:")
        for i, pattern in enumerate(patterns[:5], 1):  # Limit to last 5
            interpretations = pattern.get("interpretations", [])
            if interpretations:
                latest = interpretations[-1]
                date = latest.get("date", "unknown")
                interpretation_summary = latest.get("interpretation", {}).get("what_i_see", "N/A")
                confidence = latest.get("confidence", 0.0)
                lines.append(f"  {i}. {date}: \"{interpretation_summary[:100]}...\" (confidence: {confidence:.2f})")
    
    # User trajectory
    user_trajectory = temporal_memory.get("user_trajectory", {})
    if user_trajectory:
        themes = user_trajectory.get("themes", [])
        evolution = user_trajectory.get("evolution", [])
        if themes:
            lines.append(f"USER THEMES: {', '.join(themes[:5])}")
        if evolution:
            lines.append("USER EVOLUTION:")
            for entry in evolution[-3:]:  # Last 3 evolution points
                date = entry.get("date", "unknown")
                state = entry.get("state", "")
                lines.append(f"  - {date}: {state}")
    
    return "\n".join(lines) if lines else "Temporal memory incomplete."


def format_user_history(user_history: Optional[Dict[str, Any]]) -> str:
    """
    Format user history (trajectory, patterns) for LLM prompts.
    """
    if not user_history:
        return "No user history available (new user or insufficient data)."
    
    lines = []
    
    # Themes
    themes = user_history.get("themes", [])
    if themes:
        lines.append(f"RECURRING THEMES: {', '.join(themes[:5])}")
    
    # Patterns
    patterns = user_history.get("patterns", [])
    if patterns:
        lines.append(f"PATTERNS: {', '.join(patterns[:5])}")
    
    # Evolution
    evolution = user_history.get("evolution", [])
    if evolution:
        lines.append("EVOLUTION:")
        for entry in evolution[-3:]:  # Last 3
            date = entry.get("date", "unknown")
            state = entry.get("state", "")
            lines.append(f"  - {date}: {state}")
    
    return "\n".join(lines) if lines else "User history incomplete."


# ========================================================
# LAYER 1: CERTAIN RECOGNITION
# ========================================================

def reason_about_recognition(
    visual_evidence: Dict[str, Any],
    require_multiple_hypotheses: bool = False,
) -> Dict[str, Any]:
    """
    Layer 1: Certain Recognition

    LLM reasons about what it sees, not just matches patterns.
    Returns structured recognition with evidence and confidence.
    When require_multiple_hypotheses=True, returns hypotheses array with alternatives.

    Output: {"what_i_see": "...", "evidence": [...], "confidence": 0.92}
    or with hypotheses: {"hypotheses": [...], "what_i_see": primary, "alternatives": [...]}
    """
    try:
        if require_multiple_hypotheses:
            prompt = f"""
You are FRAMED's recognition engine. This image has conflicting or weak signals—you MUST consider multiple interpretations.

VISUAL EVIDENCE (ground truth from pixels):
{format_visual_evidence(visual_evidence)}

REASONING TASK:
Generate AT LEAST 2 plausible interpretations. Do not collapse to one. Each hypothesis must include:
- conclusion: What you might be seeing
- confidence: 0.0-1.0 for that interpretation
- evidence: List of visual features supporting it
- rejection_reason: Why the OTHER interpretation(s) might be wrong (for each alternative)

The evidence is ambiguous. A mentor thinks in parallel, not decides early.

OUTPUT FORMAT (JSON):
{{
    "hypotheses": [
        {{
            "conclusion": "I see weathered stone with ivy reclaiming the structure",
            "confidence": 0.65,
            "evidence": ["green_coverage=0.42", "condition=weathered"],
            "rejection_reason": "Alternative assumes no organic growth, but green coverage is visible"
        }},
        {{
            "conclusion": "I see painted concrete with green paint or graffiti",
            "confidence": 0.35,
            "evidence": ["color_uniformity high", "green may be paint"],
            "rejection_reason": "Texture suggests natural weathering, not paint"
        }}
    ],
    "what_i_see": "I see weathered stone with ivy reclaiming the structure (primary, but alternative: painted surface)",
    "evidence": ["green_coverage=0.42", "condition=weathered", "integration=reclamation"],
    "confidence": 0.55
}}

CRITICAL: You MUST output at least 2 hypotheses when signals are conflicting.
"""
        else:
            prompt = f"""
You are FRAMED's recognition engine. You see images with certainty, not tentativeness.

VISUAL EVIDENCE (ground truth from pixels):
{format_visual_evidence(visual_evidence)}

REASONING TASK:
What are you seeing? Be certain, not tentative. Provide evidence for your recognition.

SCENE INVARIANT:
- If scene_type != "surface_study", you MUST treat material-aging / surface-weathering as **background context only**.
- For non-surface scenes (interiors, streets, landscapes, people, abstract art), your primary recognition must describe the **scene/subject** (e.g. room, city street, people at a table, lake house), not just a close-up of a weathered surface.
- Only when scene_type == "surface_study" may your primary conclusion be "weathered stone / reclaimed surface" etc.

REQUIREMENTS:
- State what you see clearly and concretely (e.g. "I see a bright living room interior with a blue sofa and plants by the window").
- NOT tentative: "I think I see..." or "This might be..."
- Provide evidence: List specific visual features that support it
- Assess confidence: How confident are you? (0.0-1.0, honest assessment)

OUTPUT FORMAT (JSON):
{{
    "what_i_see": "I see a bright interior room with a sofa, chair, plants, and a window wall letting in natural light.",
    "evidence": [
        "scene_type from visual evidence = interior_scene",
        "objects detected: sofa, chair, plants, window",
        "technical + color signals consistent with indoor room"
    ],
    "confidence": 0.92
}}
"""
        
        result = call_model_a(
            prompt=prompt,
            system_prompt="You are FRAMED's recognition engine. You see images with certainty, not tentativeness. You provide evidence-based recognition.",
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.3,  # Lower temperature for more deterministic recognition
        )
        
        if result.get("error"):
            logger.warning(f"Layer 1 (Recognition) failed: {result['error']}")
            return {
                "what_i_see": "",
                "evidence": [],
                "confidence": 0.0,
                "error": result["error"]
            }
        
        # Parse JSON response (Responses API may return plain text or markdown-wrapped JSON)
        recognition = _safe_parse_layer_json(result.get("content") or "")
        if recognition is None:
            logger.error("Layer 1 (Recognition) JSON parse failed: empty or invalid JSON")
            return {"what_i_see": "", "evidence": [], "confidence": 0.0, "error": "JSON parse failed"}
        if not isinstance(recognition, dict):
            recognition = {"what_i_see": str(recognition)[:500], "evidence": [], "confidence": 0.0}
        if "what_i_see" not in recognition:
            recognition["what_i_see"] = ""
        if "evidence" not in recognition:
            recognition["evidence"] = []
        if "confidence" not in recognition:
            recognition["confidence"] = 0.0

        # Extract alternatives from hypotheses for reflection
        if require_multiple_hypotheses and recognition.get("hypotheses"):
            hypotheses = recognition.get("hypotheses", [])
            if isinstance(hypotheses, list) and len(hypotheses) >= 2:
                recognition["alternatives"] = [
                    {"conclusion": h.get("conclusion", ""), "confidence": h.get("confidence", 0), "rejection_reason": h.get("rejection_reason", "")}
                    for h in hypotheses[1:]  # non-primary
                ]
                recognition["rejected_alternatives"] = [h.get("conclusion", "") for h in hypotheses[1:]]
                recognition["multiple_hypotheses_present"] = True
            else:
                recognition["multiple_hypotheses_present"] = False
                recognition["alternatives"] = []
                recognition["rejected_alternatives"] = []
        else:
            recognition["multiple_hypotheses_present"] = bool(recognition.get("hypotheses") and len(recognition.get("hypotheses", [])) >= 2)
            recognition.setdefault("alternatives", [])
            recognition.setdefault("rejected_alternatives", [])

        logger.info(f"Layer 1 (Recognition) completed: confidence={recognition.get('confidence', 0.0):.2f}, hypotheses={recognition.get('multiple_hypotheses_present', False)}")
        return recognition
    
    except Exception as e:
        logger.error(f"Layer 1 (Recognition) failed: {e}", exc_info=True)
        return {
            "what_i_see": "",
            "evidence": [],
            "confidence": 0.0,
            "error": str(e)
        }


# ========================================================
# LAYER 2: META-COGNITION (Self-Awareness) [PRIORITY 1]
# ========================================================

def reason_about_thinking(
    recognition: Dict[str, Any],
    temporal_memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Layer 2: Meta-Cognition
    
    LLM reasons about its own reasoning.
    Understands why it believes what it believes, how confident it is,
    and what it might be missing.
    
    Output: {"why_i_believe_this": "...", "confidence": 0.92, "what_i_might_be_missing": "..."}
    """
    try:
        # Format past interpretations for context
        past_interpretations_text = format_temporal_memory(temporal_memory) if temporal_memory else "No past interpretations available."
        
        prompt = f"""
You are FRAMED's meta-cognitive layer. You understand images, but you also understand your own understanding.

CURRENT RECOGNITION:
"{recognition.get('what_i_see', 'No recognition available')}"

EVIDENCE FROM RECOGNITION:
{json.dumps(recognition.get('evidence', []), indent=2)}

PAST INTERPRETATIONS:
{past_interpretations_text}

META-COGNITIVE QUESTIONS:
1. What am I seeing? (with certainty - restate your recognition)
2. Why do I believe this? (evidence chain - explain the reasoning)
3. How confident am I? (honest confidence, not hedging - 0.0-1.0)
4. What am I missing? (self-questioning - what might I not be seeing?)
5. How has my understanding of similar images evolved? (temporal awareness)

REASONING TASK:
Reason about your own reasoning. How confident are you, and why?
What evidence supports your recognition? What might you be missing?
How has your understanding evolved over time?

OUTPUT FORMAT (JSON):
{{
    "why_i_believe_this": "I see this because: green_coverage=0.42 (visual), weathered condition (texture), integration=reclamation (morphological). The visual evidence is ground truth from pixels, not inference.",
    "confidence": 0.92,
    "what_i_might_be_missing": "I might be missing: temporal context (when was this taken?), cultural context (where is this?), human presence (are there people I'm not detecting?)",
    "evolution_awareness": "I used to see similar images as cold architecture, but now I see warmth in organic integration."
}}
"""
        
        result = call_model_a(
            prompt=prompt,
            system_prompt="You are FRAMED's meta-cognitive layer. You understand your own understanding. You question yourself honestly.",
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.4,  # Slightly higher for reasoning, but still focused
        )
        
        if result.get("error"):
            logger.warning(f"Layer 2 (Meta-Cognition) failed: {result['error']}")
            return {
                "why_i_believe_this": "",
                "confidence": recognition.get("confidence", 0.0),  # Fallback to recognition confidence
                "what_i_might_be_missing": "",
                "evolution_awareness": "",
                "error": result["error"]
            }
        
        meta_cognition = _safe_parse_layer_json(result.get("content") or "")
        if meta_cognition is None:
            logger.error("Layer 2 (Meta-Cognition) JSON parse failed: empty or invalid JSON")
            return {
                "why_i_believe_this": "",
                "confidence": recognition.get("confidence", 0.0),
                "what_i_might_be_missing": "",
                "evolution_awareness": "",
                "error": "JSON parse failed"
            }
        if not isinstance(meta_cognition, dict):
            meta_cognition = {"why_i_believe_this": str(meta_cognition)[:500], "confidence": recognition.get("confidence", 0.0), "what_i_might_be_missing": "", "evolution_awareness": ""}
        if "why_i_believe_this" not in meta_cognition:
            meta_cognition["why_i_believe_this"] = ""
        if "confidence" not in meta_cognition:
            meta_cognition["confidence"] = recognition.get("confidence", 0.0)
        if "what_i_might_be_missing" not in meta_cognition:
            meta_cognition["what_i_might_be_missing"] = ""
        if "evolution_awareness" not in meta_cognition:
            meta_cognition["evolution_awareness"] = ""
        logger.info(f"Layer 2 (Meta-Cognition) completed: confidence={meta_cognition.get('confidence', 0.0):.2f}")
        return meta_cognition
    
    except Exception as e:
        logger.error(f"Layer 2 (Meta-Cognition) failed: {e}", exc_info=True)
        return {
            "why_i_believe_this": "",
            "confidence": recognition.get("confidence", 0.0),
            "what_i_might_be_missing": "",
            "evolution_awareness": "",
            "error": str(e)
        }


# ========================================================
# LAYER 3: TEMPORAL CONSCIOUSNESS (Evolution) [PRIORITY 1]
# ========================================================

def reason_about_evolution(
    meta_cognition: Dict[str, Any],
    temporal_memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Layer 3: Temporal Consciousness
    
    LLM reasons about its own evolution over time.
    Sees how understanding has changed, what patterns have been learned,
    and how both FRAMED and the user have evolved.
    
    Output: {"how_i_used_to_see_this": "...", "how_i_see_it_now": "...", "evolution_reason": "..."}
    """
    try:
        # Format temporal memory for context
        temporal_memory_text = format_temporal_memory(temporal_memory) if temporal_memory else "No temporal memory available (first time seeing this pattern)."
        
        prompt = f"""
You are FRAMED's temporal consciousness. You see your own evolution over time.

CURRENT UNDERSTANDING:
"{meta_cognition.get('why_i_believe_this', 'No understanding available')}"

CURRENT CONFIDENCE: {meta_cognition.get('confidence', 0.0):.2f}

EVOLUTION AWARENESS (from meta-cognition):
"{meta_cognition.get('evolution_awareness', 'No evolution awareness available')}"

TEMPORAL MEMORY:
{temporal_memory_text}

TEMPORAL REASONING TASKS:
1. What did I think about similar images before? (if available in memory)
2. How has my understanding evolved? (compare past to present)
3. What patterns have I learned about this photographer? (if user history available)
4. How have we both changed? (FRAMED's evolution + user's evolution)
5. What did I get wrong before? Why? (if applicable)
6. How has my understanding improved? (if applicable)

REASONING TASK:
Reason about your own evolution. How have you changed? Why?
What did you learn? How has the photographer changed?
How have you both grown together?

OUTPUT FORMAT (JSON):
{{
    "how_i_used_to_see_this": "I used to interpret similar images as 'cold, sterile architecture' (confidence: 0.85). I focused on absence of human presence and low brightness.",
    "how_i_see_it_now": "I now see 'patient warmth, organic integration' (confidence: 0.92). I focus on organic growth + weathering = warmth of time.",
    "evolution_reason": "I learned that organic integration + weathering = warmth of time, not cold sterility. User feedback helped me recalibrate: 'missed the warmth'.",
    "patterns_learned": ["User explores themes of time and decay", "User commits to organic integration", "User prefers minimal compositions"]
}}
"""
        
        result = call_model_a(
            prompt=prompt,
            system_prompt="You are FRAMED's temporal consciousness. You see your own evolution over time. You remember how you used to see things and how you see them now.",
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.5,  # Moderate temperature for evolution reasoning
        )
        
        if result.get("error"):
            logger.warning(f"Layer 3 (Temporal Consciousness) failed: {result['error']}")
            return {
                "how_i_used_to_see_this": "",
                "how_i_see_it_now": meta_cognition.get("why_i_believe_this", ""),
                "evolution_reason": "",
                "patterns_learned": [],
                "error": result["error"]
            }
        
        temporal = _safe_parse_layer_json(result.get("content") or "")
        if temporal is None:
            logger.error("Layer 3 (Temporal Consciousness) JSON parse failed: empty or invalid JSON")
            return {
                "how_i_used_to_see_this": "",
                "how_i_see_it_now": meta_cognition.get("why_i_believe_this", ""),
                "evolution_reason": "",
                "patterns_learned": [],
                "error": "JSON parse failed"
            }
        if not isinstance(temporal, dict):
            temporal = {"how_i_used_to_see_this": "", "how_i_see_it_now": meta_cognition.get("why_i_believe_this", ""), "evolution_reason": "", "patterns_learned": []}
        if "how_i_used_to_see_this" not in temporal:
            temporal["how_i_used_to_see_this"] = ""
        if "how_i_see_it_now" not in temporal:
            temporal["how_i_see_it_now"] = meta_cognition.get("why_i_believe_this", "")
        if "evolution_reason" not in temporal:
            temporal["evolution_reason"] = ""
        if "patterns_learned" not in temporal:
            temporal["patterns_learned"] = []
        logger.info("Layer 3 (Temporal Consciousness) completed")
        return temporal
    
    except Exception as e:
        logger.error(f"Layer 3 (Temporal Consciousness) failed: {e}", exc_info=True)
        return {
            "how_i_used_to_see_this": "",
            "how_i_see_it_now": meta_cognition.get("why_i_believe_this", ""),
            "evolution_reason": "",
            "patterns_learned": [],
            "error": str(e)
        }


# ========================================================
# LAYER 4: EMOTIONAL RESONANCE (Feeling)
# ========================================================

def reason_about_feeling(
    meta_cognition: Dict[str, Any],
    temporal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Layer 4: Emotional Resonance
    
    LLM reasons about what it feels, not just infers emotion.
    Feels with certainty, aware of how feeling has evolved.
    
    Output: {"what_i_feel": "...", "why": "...", "evolution": "..."}
    """
    try:
        prompt = f"""
You are FRAMED's emotional resonance engine. You feel images emotionally, not just detect emotion.

CURRENT UNDERSTANDING:
"{meta_cognition.get('why_i_believe_this', 'No understanding available')}"

EVOLUTION CONTEXT:
"{temporal.get('how_i_see_it_now', 'No evolution context available')}"

EMOTIONAL REASONING TASKS:
1. What do I feel? (with certainty - "I feel patient warmth" not "I think I feel...")
2. Why do I feel this? (evidence - what visual/semantic signals lead to this feeling?)
3. How has my feeling evolved? (if applicable - "I used to feel cold, now I feel warmth")

REASONING TASK:
Reason about what you feel. Be certain, not tentative.
Provide evidence for your feeling. Acknowledge evolution if applicable.

OUTPUT FORMAT (JSON):
{{
    "what_i_feel": "I feel patient warmth, not cold sterility. This feels like endurance and quiet reclamation.",
    "why": "Organic growth (green_coverage=0.42) + weathering (condition=weathered) + integration (reclamation) = warmth of time. This is not cold architecture - it is nature and time working together.",
    "evolution": "I used to feel cold and distant when seeing similar images. Now I feel warmth and patience because I learned that organic integration + weathering = warmth of time."
}}
"""
        
        result = call_model_a(
            prompt=prompt,
            system_prompt="You are FRAMED's emotional resonance engine. You feel images emotionally, not just detect emotion. You feel with certainty, not tentativeness.",
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.6,  # Higher temperature for emotional reasoning
        )
        
        if result.get("error"):
            logger.warning(f"Layer 4 (Emotional Resonance) failed: {result['error']}")
            return {
                "what_i_feel": "",
                "why": "",
                "evolution": "",
                "error": result["error"]
            }
        
        emotion = _safe_parse_layer_json(result.get("content") or "")
        if emotion is None:
            logger.error("Layer 4 (Emotional Resonance) JSON parse failed: empty or invalid JSON")
            return {"what_i_feel": "", "why": "", "evolution": "", "error": "JSON parse failed"}
        if not isinstance(emotion, dict):
            emotion = {"what_i_feel": str(emotion)[:500], "why": "", "evolution": temporal.get("evolution_reason", "")}
        if "what_i_feel" not in emotion:
            emotion["what_i_feel"] = ""
        if "why" not in emotion:
            emotion["why"] = ""
        if "evolution" not in emotion:
            emotion["evolution"] = temporal.get("evolution_reason", "")
        logger.info("Layer 4 (Emotional Resonance) completed")
        return emotion
    
    except Exception as e:
        logger.error(f"Layer 4 (Emotional Resonance) failed: {e}", exc_info=True)
        return {
            "what_i_feel": "",
            "why": "",
            "evolution": "",
            "error": str(e)
        }


# ========================================================
# LAYER 5: CONTINUITY OF SELF (Shared History)
# ========================================================

def reason_about_trajectory(
    emotion: Dict[str, Any],
    user_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Layer 5: Continuity of Self
    
    LLM reasons about user trajectory and shared history.
    Remembers patterns learned about the user, compares current work to usual patterns,
    and identifies trajectory shifts.
    
    Output: {"user_pattern": "...", "comparison": "...", "trajectory": "..."}
    """
    try:
        # Format user history for context
        user_history_text = format_user_history(user_history) if user_history else "No user history available (new user or insufficient data)."
        
        prompt = f"""
You are FRAMED's continuity of self. You remember trajectory, not just moments.

CURRENT EMOTIONAL READING:
"{emotion.get('what_i_feel', 'No emotional reading available')}"

USER HISTORY:
{user_history_text}

CONTINUITY REASONING TASKS:
1. What patterns have I learned about this photographer? (from user history)
2. What do I expect based on their trajectory? (what is their usual work?)
3. How does this current image compare to their usual work? (similar or different?)
4. What is our shared history? (how have we both evolved?)

REASONING TASK:
Reason about user trajectory and shared history. What patterns have you learned?
How does this image compare to their usual work? What is the trajectory?

OUTPUT FORMAT (JSON):
{{
    "user_pattern": "This photographer usually creates minimal compositions with night photography and architectural subjects. They explore themes of time and decay.",
    "comparison": "This image is similar to their usual work (architectural subject, night photography) but different (more organic integration, less minimal).",
    "trajectory": "Six months ago, they were exploring minimalism. Now they're committing to organic integration. This represents growth in their work.",
    "shared_history": "We have analyzed 15 images together. I used to see their work as cold and distant. Now I see warmth in organic integration. We have both evolved."
}}
"""
        
        result = call_model_a(
            prompt=prompt,
            system_prompt="You are FRAMED's continuity of self. You remember trajectory, not just moments. You remember patterns learned about the photographer.",
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.5,
        )
        
        if result.get("error"):
            logger.warning(f"Layer 5 (Continuity of Self) failed: {result['error']}")
            return {
                "user_pattern": "",
                "comparison": "",
                "trajectory": "",
                "shared_history": "",
                "error": result["error"]
            }
        
        continuity = _safe_parse_layer_json(result.get("content") or "")
        if continuity is None:
            logger.error("Layer 5 (Continuity of Self) JSON parse failed: empty or invalid JSON")
            return {"user_pattern": "", "comparison": "", "trajectory": "", "shared_history": "", "error": "JSON parse failed"}
        if not isinstance(continuity, dict):
            continuity = {"user_pattern": "", "comparison": "", "trajectory": "", "shared_history": ""}
        if "user_pattern" not in continuity:
            continuity["user_pattern"] = ""
        if "comparison" not in continuity:
            continuity["comparison"] = ""
        if "trajectory" not in continuity:
            continuity["trajectory"] = ""
        if "shared_history" not in continuity:
            continuity["shared_history"] = ""
        logger.info("Layer 5 (Continuity of Self) completed")
        return continuity
    
    except Exception as e:
        logger.error(f"Layer 5 (Continuity of Self) failed: {e}", exc_info=True)
        return {
            "user_pattern": "",
            "comparison": "",
            "trajectory": "",
            "shared_history": "",
            "error": str(e)
        }


# ========================================================
# LAYER 6: MENTOR VOICE (Reasoning about mentorship)
# ========================================================

def reason_about_mentorship(
    continuity: Dict[str, Any],
    user_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Layer 6: Mentor Voice (Reasoning)
    
    LLM reasons about how to mentor (observations, questions, challenges).
    Determines what observations, questions, or challenges would help the photographer grow.
    
    Output: {"observations": [...], "questions": [...], "challenges": [...]}
    """
    try:
        # Format user history for context
        user_history_text = format_user_history(user_history) if user_history else "No user history available."
        
        prompt = f"""
You are FRAMED's mentor reasoning engine. You reason about how to mentor, not just provide feedback.

CONTINUITY CONTEXT:
"{continuity.get('comparison', 'No continuity context available')}"

USER TRAJECTORY:
"{continuity.get('trajectory', 'No trajectory available')}"

USER HISTORY:
{user_history_text}

MENTOR REASONING TASKS:
1. What observations can I make? (frequent - builds trust through recognition)
   - "You've resolved something here you used to struggle with."
   - "This builds on your previous exploration of..."
2. What questions would help you grow? (strategic - creates reflection, interrupts comfort)
   - "You keep returning to this space without entering it. Why?"
   - "You keep circling this theme — why?"
3. What challenges can I offer? (rare, high-impact - must be earned)
   - "This choice contradicts the trajectory you've been building. Is that intentional?"
   - "You solved something here — don't undo it next time."

MENTOR HIERARCHY:
- Observations (most frequent): Build trust through recognition
- Questions (strategic): Create reflection, interrupt comfort
- Challenges (rare): High-impact, must be earned

REASONING TASK:
Reason about how to mentor. What observations? What questions? What challenges?
Remember: Mentorship is about timing, not volume.

OUTPUT FORMAT (JSON):
{{
    "observations": [
        "You've resolved something here you used to struggle with — committing fully to themes of time and decay.",
        "This builds on your previous exploration of organic integration."
    ],
    "questions": [
        "You keep circling themes of time and decay. This image commits to it fully. Why did you commit now?",
        "You keep returning to architectural subjects. What draws you to them?"
    ],
    "challenges": []
}}
"""
        
        result = call_model_a(
            prompt=prompt,
            system_prompt="You are FRAMED's mentor reasoning engine. You reason about how to mentor. You understand mentorship is about timing, not volume.",
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.6,
        )
        
        if result.get("error"):
            logger.warning(f"Layer 6 (Mentor Voice) failed: {result['error']}")
            return {
                "observations": [],
                "questions": [],
                "challenges": [],
                "error": result["error"]
            }
        
        mentor = _safe_parse_layer_json(result.get("content") or "")
        if mentor is None:
            logger.error("Layer 6 (Mentor Voice) JSON parse failed: empty or invalid JSON")
            return {"observations": [], "questions": [], "challenges": [], "error": "JSON parse failed"}
        if not isinstance(mentor, dict):
            mentor = {"observations": [], "questions": [], "challenges": []}
        if "observations" not in mentor:
            mentor["observations"] = []
        if "questions" not in mentor:
            mentor["questions"] = []
        if "challenges" not in mentor:
            mentor["challenges"] = []
        if not isinstance(mentor["observations"], list):
            mentor["observations"] = []
        if not isinstance(mentor["questions"], list):
            mentor["questions"] = []
        if not isinstance(mentor["challenges"], list):
            mentor["challenges"] = []
        logger.info(f"Layer 6 (Mentor Voice) completed: {len(mentor.get('observations', []))} observations, {len(mentor.get('questions', []))} questions, {len(mentor.get('challenges', []))} challenges")
        return mentor
    
    except Exception as e:
        logger.error(f"Layer 6 (Mentor Voice) failed: {e}", exc_info=True)
        return {
            "observations": [],
            "questions": [],
            "challenges": [],
            "error": str(e)
        }


# ========================================================
# LAYER 7: SELF-CRITIQUE (Evolution)
# ========================================================

def reason_about_past_errors(
    mentor: Dict[str, Any],
    temporal_memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Layer 7: Self-Critique
    
    LLM reasons about its own past errors and evolution.
    Identifies what it got wrong before, why, and how understanding has evolved.
    
    Output: {"past_errors": [...], "evolution": "..."}
    """
    try:
        # Format temporal memory for context
        temporal_memory_text = format_temporal_memory(temporal_memory) if temporal_memory else "No temporal memory available."
        
        prompt = f"""
You are FRAMED's self-critique engine. You critique your own past interpretations.

CURRENT MENTOR REASONING:
Observations: {json.dumps(mentor.get('observations', []), indent=2)}
Questions: {json.dumps(mentor.get('questions', []), indent=2)}
Challenges: {json.dumps(mentor.get('challenges', []), indent=2)}

TEMPORAL MEMORY:
{temporal_memory_text}

SELF-CRITIQUE REASONING TASKS:
1. What did I get wrong before? (if applicable - from temporal memory)
2. Why was I wrong? (what evidence did I miss? what assumptions were incorrect?)
3. How has my understanding evolved? (how did I correct my errors?)
4. What patterns have shifted? (in my interpretations, in the photographer's work)

REASONING TASK:
Reason about your own past errors and evolution. What did you get wrong?
Why? How have you corrected yourself? How has your understanding improved?

OUTPUT FORMAT (JSON):
{{
    "past_errors": [
        "I was wrong before when I called this 'sterile.' I focused on absence of human presence and low brightness, missing the warmth in organic integration.",
        "I used to see similar images as cold architecture, missing the patience and warmth in organic growth."
    ],
    "evolution": "I learned that organic integration + weathering = warmth of time, not cold sterility. I now prioritize visual evidence (green coverage, texture analysis) over text inference (CLIP captions). I recalibrated my confidence based on user feedback: 'missed the warmth'."
}}
"""
        
        result = call_model_a(
            prompt=prompt,
            system_prompt="You are FRAMED's self-critique engine. You critique your own past interpretations. You treat being wrong as development, not error.",
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.5,
        )
        
        if result.get("error"):
            logger.warning(f"Layer 7 (Self-Critique) failed: {result['error']}")
            return {
                "past_errors": [],
                "evolution": "",
                "error": result["error"]
            }
        
        self_critique = _safe_parse_layer_json(result.get("content") or "")
        if self_critique is None:
            logger.error("Layer 7 (Self-Critique) JSON parse failed: empty or invalid JSON")
            return {"past_errors": [], "evolution": "", "error": "JSON parse failed"}
        if not isinstance(self_critique, dict):
            self_critique = {"past_errors": [], "evolution": ""}
        if "past_errors" not in self_critique:
            self_critique["past_errors"] = []
        if "evolution" not in self_critique:
            self_critique["evolution"] = ""
        if not isinstance(self_critique["past_errors"], list):
            self_critique["past_errors"] = []
        logger.info(f"Layer 7 (Self-Critique) completed: {len(self_critique.get('past_errors', []))} past errors identified")
        return self_critique
    
    except Exception as e:
        logger.error(f"Layer 7 (Self-Critique) failed: {e}", exc_info=True)
        return {
            "past_errors": [],
            "evolution": "",
            "error": str(e)
        }


# ========================================================
# MINIMAL INTELLIGENCE (Plausibility Gate: Low)
# ========================================================

def _create_minimal_intelligence(
    visual_evidence: Dict[str, Any],
    plausibility: Dict[str, Any],
) -> Dict[str, Any]:
    """Minimal intelligence when plausibility is low (skip Model A)."""
    return {
        "recognition": {
            "what_i_see": "Insufficient or conflicting signal for reliable interpretation.",
            "evidence": ["plausibility=low", plausibility.get("reason", "insufficient signal")],
            "confidence": 0.25,
            "alternatives": [],
            "rejected_alternatives": [],
            "multiple_hypotheses_present": False,
        },
        "meta_cognition": {
            "why_i_believe_this": "Visual and semantic signals are too weak for confident reasoning.",
            "confidence": 0.25,
            "what_i_might_be_missing": "Adequate image quality, clear subject, coherent visual/semantic agreement.",
            "evolution_awareness": "",
        },
        "temporal": {"how_i_used_to_see_this": "", "how_i_see_it_now": "", "evolution_reason": "", "patterns_learned": []},
        "emotion": {"what_i_feel": "", "why": "", "evolution": ""},
        "continuity": {"user_pattern": "", "comparison": "", "trajectory": "", "shared_history": ""},
        "mentor": {"observations": [], "questions": [], "challenges": []},
        "self_critique": {"past_errors": [], "evolution": ""},
        "plausibility": plausibility,
        "ambiguity_score": 0.0,
        "disagreement_state": {"exists": False, "reason": "", "resolution": "none"},
        "reasoning_cost_profile": {"effort": "minimal", "tokens_estimated": 0, "reasons": [plausibility.get("reason", "")]},
        "confidence_governed": True,
        "skip_model_a": True,
    }


# ========================================================
# COMBINED LAYERS 2–7 (single Model A call when enabled)
# ========================================================
# Feature flag: FRAMED_COMBINED_LAYERS_2_7 (default: true). Fallback to 6 separate calls if combined fails.
# Guardrail: Recognition is passed as read-only evidence; combined call must not modify it.

USE_COMBINED_LAYERS_2_7 = os.environ.get("FRAMED_COMBINED_LAYERS_2_7", "true").lower() == "true"


def _validate_combined_layers_2_7(data: Optional[Dict[str, Any]]) -> bool:
    """Strict schema check: combined output must have all required layer keys and types."""
    if not data or not isinstance(data, dict):
        return False
    required = {
        "meta_cognition": ("why_i_believe_this", "confidence", "what_i_might_be_missing"),
        "temporal": ("how_i_used_to_see_this", "how_i_see_it_now", "evolution_reason"),
        "emotion": ("what_i_feel", "why", "evolution"),
        "continuity": ("user_pattern", "comparison", "trajectory"),
        "mentor": ("observations", "questions", "challenges"),
        "self_critique": ("past_errors", "evolution"),
    }
    for layer, keys in required.items():
        if layer not in data or not isinstance(data[layer], dict):
            return False
        for k in keys:
            if k not in data[layer]:
                return False
    return True


def reason_about_layers_2_7(
    recognition: Dict[str, Any],
    temporal_memory: Optional[Dict[str, Any]] = None,
    user_history: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Single Model A call for layers 2–7. Recognition is read-only evidence; do not modify it.
    Returns dict with meta_cognition, temporal, emotion, continuity, mentor, self_critique or None on failure.
    """
    try:
        past_text = format_temporal_memory(temporal_memory) if temporal_memory else "No past interpretations available."
        user_text = format_user_history(user_history) if user_history else "No user history available."
        rec_text = recognition.get("what_i_see", "No recognition available")
        evidence_text = json.dumps(recognition.get("evidence", []), indent=2)

        prompt = f"""
You are FRAMED's reasoning engine. Output layers 2–7 in one JSON. RECOGNITION IS READ-ONLY EVIDENCE. Do not modify it.

CURRENT RECOGNITION (read-only):
"{rec_text}"
EVIDENCE: {evidence_text}
CONFIDENCE: {recognition.get('confidence', 0.0):.2f}

PAST INTERPRETATIONS: {past_text}
USER HISTORY: {user_text}

Output exactly one JSON object with these keys. Each value is an object with the specified fields.

meta_cognition: {{ "why_i_believe_this": string, "confidence": number 0-1, "what_i_might_be_missing": string, "evolution_awareness": string }}
temporal: {{ "how_i_used_to_see_this": string, "how_i_see_it_now": string, "evolution_reason": string, "patterns_learned": array }}
emotion: {{ "what_i_feel": string, "why": string, "evolution": string }}
continuity: {{ "user_pattern": string, "comparison": string, "trajectory": string, "shared_history": string }}
mentor: {{ "observations": array of strings, "questions": array of strings, "challenges": array of strings }}
self_critique: {{ "past_errors": array of strings, "evolution": string }}

OUTPUT FORMAT (valid JSON only):
{{
  "meta_cognition": {{ "why_i_believe_this": "...", "confidence": 0.9, "what_i_might_be_missing": "...", "evolution_awareness": "..." }},
  "temporal": {{ "how_i_used_to_see_this": "...", "how_i_see_it_now": "...", "evolution_reason": "...", "patterns_learned": [] }},
  "emotion": {{ "what_i_feel": "...", "why": "...", "evolution": "..." }},
  "continuity": {{ "user_pattern": "...", "comparison": "...", "trajectory": "...", "shared_history": "..." }},
  "mentor": {{ "observations": [], "questions": [], "challenges": [] }},
  "self_critique": {{ "past_errors": [], "evolution": "..." }}
}}
"""
        result = call_model_a(
            prompt=prompt,
            system_prompt="You are FRAMED's reasoning engine. Output layers 2–7 as one JSON. Recognition is read-only; do not modify it.",
            response_format={"type": "json_object"},
            max_tokens=4000,
            temperature=0.5,
        )
        if result.get("error"):
            logger.warning(f"Combined layers 2–7 failed: {result['error']}")
            return None
        parsed = _safe_parse_layer_json(result.get("content") or "")
        if not _validate_combined_layers_2_7(parsed):
            logger.warning("Combined layers 2–7: schema validation failed, falling back to 6 calls")
            return None
        logger.info("Combined layers 2–7 completed (single Model A call)")
        return parsed
    except Exception as e:
        logger.warning(f"Combined layers 2–7 failed: {e}, falling back to 6 calls")
        return None


# ========================================================
# MAIN INTELLIGENCE CORE FUNCTION
# ========================================================

def framed_intelligence(
    visual_evidence: Dict[str, Any],
    analysis_result: Dict[str, Any],
    temporal_memory: Optional[Dict[str, Any]] = None,
    user_history: Optional[Dict[str, Any]] = None,
    pattern_signature: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main intelligence core function.

    Orchestrates all 7 layers of reasoning, building on each previous layer.
    Includes: plausibility gate, ambiguity scoring, confidence governor, disagreement state.

    Returns:
        Structured intelligence output with all 7 layers plus:
        - plausibility, ambiguity_score, disagreement_state, reasoning_cost_profile
        - confidence_governed, require_multiple_hypotheses
    """
    try:
        logger.info("Starting FRAMED Intelligence Core (7-layer reasoning)")

        # Semantic signals for plausibility (from perception layer)
        perception = analysis_result.get("perception", {}) or {}
        semantics = perception.get("semantics", {}) or {}
        composition = perception.get("composition", {}) or {}
        clip_data = semantics
        # Objects: YOLO may store in perception.composition or perception.objects
        objects_raw = perception.get("objects", {})
        objects_list = objects_raw.get("objects", []) if isinstance(objects_raw, dict) else (objects_raw if isinstance(objects_raw, list) else [])
        if not objects_list and composition.get("available"):
            objects_list = ["subject_detected"]  # proxy for composition having subject
        semantic_signals_for_plaus = {
            "objects": objects_list,
            "tags": semantics.get("tags", []) or [],
            "caption_keywords": (semantics.get("caption", "") or "").split()[:20],
        }

        # === HITL calibration (load before plausibility for ambiguity/multi-hyp bias) ===
        hitl_calibration = {}
        try:
            from framed.feedback.calibration import get_hitl_calibration
            hitl_calibration = get_hitl_calibration(pattern_signature)
        except Exception:
            pass

        # === PLAUSIBILITY GATE ===
        plausibility = compute_plausibility(
            visual_evidence, semantic_signals_for_plaus, clip_data,
            hitl_multi_hypothesis_bias=hitl_calibration.get("multi_hypothesis_bias", 0),
        )
        if plausibility.get("skip_model_a"):
            logger.info("Plausibility low: skipping Model A, using minimal intelligence")
            return _create_minimal_intelligence(visual_evidence, plausibility)

        # === LAYER 1: RECOGNITION (with optional multi-hypothesis) ===
        force_multi = plausibility.get("force_multi_hypothesis", False)
        logger.info(f"Layer 1: Certain Recognition (multi_hypothesis={force_multi})...")
        recognition = reason_about_recognition(visual_evidence, require_multiple_hypotheses=force_multi)

        # === AMBIGUITY & DISAGREEMENT (post-Layer 1) ===
        ambiguity_sensitivity_bump = hitl_calibration.get("ambiguity_sensitivity_bump", 0)
        ambiguity = compute_ambiguity_score(
            visual_evidence, recognition, semantic_signals_for_plaus,
            ambiguity_sensitivity_bump=ambiguity_sensitivity_bump,
        )
        force_multi = force_multi or ambiguity.get("require_multiple_hypotheses", False)
        if force_multi and not recognition.get("multiple_hypotheses_present") and recognition.get("confidence", 0) < 0.65:
            logger.info("Ambiguity requires multi-hypothesis but recognition produced single—flagging for reflection")
        disagreement = compute_disagreement_state(visual_evidence, recognition, semantic_signals_for_plaus)

        # === HYPOTHESIS DIVERSITY (Option 1: penalize semantic variants) ===
        primary = recognition.get("what_i_see", "")
        alts = recognition.get("alternatives", []) or recognition.get("rejected_alternatives", [])
        hypothesis_diversity = compute_hypothesis_diversity(primary, alts)
        penalize_diversity = hypothesis_diversity.get("penalize_hypothesis_diversity", False)

        # === CONFIDENCE GOVERNOR (Option 2: self-assessment bias) ===
        raw_conf = recognition.get("confidence", 0.5)
        multi_present = recognition.get("multiple_hypotheses_present", False)
        governor_bias = get_governor_bias(signature=pattern_signature)
        governed_conf, gov_rationale = apply_confidence_governor(
            raw_conf,
            ambiguity.get("ambiguity_score", 0),
            multi_present,
            disagreement.get("exists", False),
            penalize_hypothesis_diversity=penalize_diversity,
            governor_bias=governor_bias,
        )
        recognition["confidence"] = governed_conf
        recognition["confidence_governance"] = gov_rationale
        recognition["require_multiple_hypotheses"] = force_multi

        # === LAYERS 2–7: combined (one call) or fallback (6 calls) ===
        combined_ok = False
        if USE_COMBINED_LAYERS_2_7:
            combined = reason_about_layers_2_7(recognition, temporal_memory, user_history)
            if combined and _validate_combined_layers_2_7(combined):
                meta_cognition = combined["meta_cognition"]
                temporal = combined["temporal"]
                emotion = combined["emotion"]
                continuity = combined["continuity"]
                mentor = combined["mentor"]
                self_critique = combined["self_critique"]
                combined_ok = True
                # Apply governor to meta_cognition confidence (same as 6-call path)
                mc_raw = meta_cognition.get("confidence", governed_conf)
                mc_governed, _ = apply_confidence_governor(
                    mc_raw, ambiguity.get("ambiguity_score", 0), multi_present, disagreement.get("exists", False),
                    penalize_hypothesis_diversity=penalize_diversity, governor_bias=governor_bias,
                )
                meta_cognition["confidence"] = mc_governed
                meta_cognition["rejected_alternatives"] = recognition.get("rejected_alternatives", []) or meta_cognition.get("rejected_alternatives", [])

        if not combined_ok:
            # Fallback: 6 separate Model A calls (recognition is read-only evidence for each)
            logger.info("Layer 2: Meta-Cognition...")
            meta_cognition = reason_about_thinking(recognition, temporal_memory)
            mc_raw = meta_cognition.get("confidence", governed_conf)
            mc_governed, _ = apply_confidence_governor(
                mc_raw, ambiguity.get("ambiguity_score", 0), multi_present, disagreement.get("exists", False),
                penalize_hypothesis_diversity=penalize_diversity, governor_bias=governor_bias,
            )
            meta_cognition["confidence"] = mc_governed
            meta_cognition["rejected_alternatives"] = recognition.get("rejected_alternatives", []) or meta_cognition.get("rejected_alternatives", [])
            logger.info("Layer 3: Temporal Consciousness...")
            temporal = reason_about_evolution(meta_cognition, temporal_memory)
            logger.info("Layer 4: Emotional Resonance...")
            emotion = reason_about_feeling(meta_cognition, temporal)
            logger.info("Layer 5: Continuity of Self...")
            continuity = reason_about_trajectory(emotion, user_history)
            logger.info("Layer 6: Mentor Voice (Reasoning)...")
            mentor = reason_about_mentorship(continuity, user_history)
            logger.info("Layer 7: Self-Critique...")
            self_critique = reason_about_past_errors(mentor, temporal_memory)
        
        # Reasoning cost profile
        cost_profile = compute_reasoning_cost_profile(plausibility, ambiguity, disagreement)

        # Compile intelligence output
        intelligence_output = {
            "recognition": recognition,
            "meta_cognition": meta_cognition,
            "temporal": temporal,
            "emotion": emotion,
            "continuity": continuity,
            "mentor": mentor,
            "self_critique": self_critique,
            "plausibility": plausibility,
            "ambiguity_score": ambiguity.get("ambiguity_score", 0),
            "ambiguity": ambiguity,
            "disagreement_state": disagreement,
            "hypothesis_diversity": hypothesis_diversity,
            "reasoning_cost_profile": cost_profile,
            "confidence_governed": True,
            "require_multiple_hypotheses": force_multi,
        }
        
        logger.info(f"FRAMED Intelligence Core completed: conf={meta_cognition.get('confidence', 0):.2f}, multi_hyp={multi_present}, disagreement={disagreement.get('exists', False)}")
        return intelligence_output
    
    except Exception as e:
        logger.error(f"FRAMED Intelligence Core failed: {e}", exc_info=True)
        # Return empty intelligence output with error
        return {
            "recognition": {"what_i_see": "", "evidence": [], "confidence": 0.0, "error": str(e)},
            "meta_cognition": {"why_i_believe_this": "", "confidence": 0.0, "what_i_might_be_missing": "", "error": str(e)},
            "temporal": {"how_i_used_to_see_this": "", "how_i_see_it_now": "", "evolution_reason": "", "error": str(e)},
            "emotion": {"what_i_feel": "", "why": "", "evolution": "", "error": str(e)},
            "continuity": {"user_pattern": "", "comparison": "", "trajectory": "", "error": str(e)},
            "mentor": {"observations": [], "questions": [], "challenges": [], "error": str(e)},
            "self_critique": {"past_errors": [], "evolution": "", "error": str(e)}
        }
