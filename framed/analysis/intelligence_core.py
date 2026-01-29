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
from typing import Dict, Any, Optional, List

from .llm_provider import call_model_a

logger = logging.getLogger(__name__)


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
    
    # Organic growth
    organic_growth = visual_evidence.get("organic_growth", {})
    if organic_growth:
        green_coverage = organic_growth.get("green_coverage", 0.0)
        salience = organic_growth.get("salience", "minimal")
        green_locations = organic_growth.get("green_locations", "none")
        confidence = organic_growth.get("confidence", 0.0)
        lines.append(f"- Organic Growth: coverage={green_coverage:.3f}, salience={salience}, locations={green_locations} (confidence: {confidence:.2f})")
    
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

def reason_about_recognition(visual_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Layer 1: Certain Recognition
    
    LLM reasons about what it sees, not just matches patterns.
    Returns structured recognition with evidence and confidence.
    
    Output: {"what_i_see": "...", "evidence": [...], "confidence": 0.92}
    """
    try:
        prompt = f"""
You are FRAMED's recognition engine. You see images with certainty, not tentativeness.

VISUAL EVIDENCE (ground truth from pixels):
{format_visual_evidence(visual_evidence)}

REASONING TASK:
What are you seeing? Be certain, not tentative. Provide evidence for your recognition.

REQUIREMENTS:
- State what you see clearly: "I see weathered stone with ivy reclaiming the structure"
- NOT tentative: "I think I see..." or "This might be..."
- Provide evidence: List specific visual features that support your recognition
- Assess confidence: How confident are you? (0.0-1.0, honest assessment)

OUTPUT FORMAT (JSON):
{{
    "what_i_see": "I see weathered stone with ivy reclaiming the structure. This is nature integrating with architecture over time.",
    "evidence": [
        "green_coverage=0.42 (visual analysis)",
        "condition=weathered (texture analysis)",
        "integration=reclamation (morphological analysis)"
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
        
        # Parse JSON response
        try:
            recognition = json.loads(result["content"])
            # Validate structure
            if not isinstance(recognition, dict):
                raise ValueError("Recognition output is not a dictionary")
            if "what_i_see" not in recognition:
                recognition["what_i_see"] = ""
            if "evidence" not in recognition:
                recognition["evidence"] = []
            if "confidence" not in recognition:
                recognition["confidence"] = 0.0
            
            logger.info(f"Layer 1 (Recognition) completed: confidence={recognition.get('confidence', 0.0):.2f}")
            return recognition
        
        except json.JSONDecodeError as e:
            logger.error(f"Layer 1 (Recognition) JSON parse failed: {e}")
            return {
                "what_i_see": "",
                "evidence": [],
                "confidence": 0.0,
                "error": f"JSON parse error: {str(e)}"
            }
    
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
        
        try:
            meta_cognition = json.loads(result["content"])
            # Validate structure
            if not isinstance(meta_cognition, dict):
                raise ValueError("Meta-cognition output is not a dictionary")
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
        
        except json.JSONDecodeError as e:
            logger.error(f"Layer 2 (Meta-Cognition) JSON parse failed: {e}")
            return {
                "why_i_believe_this": "",
                "confidence": recognition.get("confidence", 0.0),
                "what_i_might_be_missing": "",
                "evolution_awareness": "",
                "error": f"JSON parse error: {str(e)}"
            }
    
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
        
        try:
            temporal = json.loads(result["content"])
            # Validate structure
            if not isinstance(temporal, dict):
                raise ValueError("Temporal output is not a dictionary")
            if "how_i_used_to_see_this" not in temporal:
                temporal["how_i_used_to_see_this"] = ""
            if "how_i_see_it_now" not in temporal:
                temporal["how_i_see_it_now"] = meta_cognition.get("why_i_believe_this", "")
            if "evolution_reason" not in temporal:
                temporal["evolution_reason"] = ""
            if "patterns_learned" not in temporal:
                temporal["patterns_learned"] = []
            
            logger.info(f"Layer 3 (Temporal Consciousness) completed")
            return temporal
        
        except json.JSONDecodeError as e:
            logger.error(f"Layer 3 (Temporal Consciousness) JSON parse failed: {e}")
            return {
                "how_i_used_to_see_this": "",
                "how_i_see_it_now": meta_cognition.get("why_i_believe_this", ""),
                "evolution_reason": "",
                "patterns_learned": [],
                "error": f"JSON parse error: {str(e)}"
            }
    
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
        
        try:
            emotion = json.loads(result["content"])
            # Validate structure
            if not isinstance(emotion, dict):
                raise ValueError("Emotion output is not a dictionary")
            if "what_i_feel" not in emotion:
                emotion["what_i_feel"] = ""
            if "why" not in emotion:
                emotion["why"] = ""
            if "evolution" not in emotion:
                emotion["evolution"] = temporal.get("evolution_reason", "")
            
            logger.info(f"Layer 4 (Emotional Resonance) completed")
            return emotion
        
        except json.JSONDecodeError as e:
            logger.error(f"Layer 4 (Emotional Resonance) JSON parse failed: {e}")
            return {
                "what_i_feel": "",
                "why": "",
                "evolution": "",
                "error": f"JSON parse error: {str(e)}"
            }
    
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
        
        try:
            continuity = json.loads(result["content"])
            # Validate structure
            if not isinstance(continuity, dict):
                raise ValueError("Continuity output is not a dictionary")
            if "user_pattern" not in continuity:
                continuity["user_pattern"] = ""
            if "comparison" not in continuity:
                continuity["comparison"] = ""
            if "trajectory" not in continuity:
                continuity["trajectory"] = ""
            if "shared_history" not in continuity:
                continuity["shared_history"] = ""
            
            logger.info(f"Layer 5 (Continuity of Self) completed")
            return continuity
        
        except json.JSONDecodeError as e:
            logger.error(f"Layer 5 (Continuity of Self) JSON parse failed: {e}")
            return {
                "user_pattern": "",
                "comparison": "",
                "trajectory": "",
                "shared_history": "",
                "error": f"JSON parse error: {str(e)}"
            }
    
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
        
        try:
            mentor = json.loads(result["content"])
            # Validate structure
            if not isinstance(mentor, dict):
                raise ValueError("Mentor output is not a dictionary")
            if "observations" not in mentor:
                mentor["observations"] = []
            if "questions" not in mentor:
                mentor["questions"] = []
            if "challenges" not in mentor:
                mentor["challenges"] = []
            
            # Ensure lists are actually lists
            if not isinstance(mentor["observations"], list):
                mentor["observations"] = []
            if not isinstance(mentor["questions"], list):
                mentor["questions"] = []
            if not isinstance(mentor["challenges"], list):
                mentor["challenges"] = []
            
            logger.info(f"Layer 6 (Mentor Voice) completed: {len(mentor.get('observations', []))} observations, {len(mentor.get('questions', []))} questions, {len(mentor.get('challenges', []))} challenges")
            return mentor
        
        except json.JSONDecodeError as e:
            logger.error(f"Layer 6 (Mentor Voice) JSON parse failed: {e}")
            return {
                "observations": [],
                "questions": [],
                "challenges": [],
                "error": f"JSON parse error: {str(e)}"
            }
    
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
        
        try:
            self_critique = json.loads(result["content"])
            # Validate structure
            if not isinstance(self_critique, dict):
                raise ValueError("Self-critique output is not a dictionary")
            if "past_errors" not in self_critique:
                self_critique["past_errors"] = []
            if "evolution" not in self_critique:
                self_critique["evolution"] = ""
            
            # Ensure past_errors is a list
            if not isinstance(self_critique["past_errors"], list):
                self_critique["past_errors"] = []
            
            logger.info(f"Layer 7 (Self-Critique) completed: {len(self_critique.get('past_errors', []))} past errors identified")
            return self_critique
        
        except json.JSONDecodeError as e:
            logger.error(f"Layer 7 (Self-Critique) JSON parse failed: {e}")
            return {
                "past_errors": [],
                "evolution": "",
                "error": f"JSON parse error: {str(e)}"
            }
    
    except Exception as e:
        logger.error(f"Layer 7 (Self-Critique) failed: {e}", exc_info=True)
        return {
            "past_errors": [],
            "evolution": "",
            "error": str(e)
        }


# ========================================================
# MAIN INTELLIGENCE CORE FUNCTION
# ========================================================

def framed_intelligence(
    visual_evidence: Dict[str, Any],
    analysis_result: Dict[str, Any],
    temporal_memory: Optional[Dict[str, Any]] = None,
    user_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main intelligence core function.
    
    Orchestrates all 7 layers of reasoning, building on each previous layer.
    This is where FRAMED's brain lives.
    
    Args:
        visual_evidence: Visual analysis results from extract_visual_features() (YOLO, CLIP, OpenCV)
        analysis_result: Full analysis result (canonical schema) for semantic signals
        temporal_memory: Past interpretations and evolution (from temporal_memory.py)
        user_history: User trajectory and patterns (from temporal_memory.py)
    
    Returns:
        Structured intelligence output with all 7 layers:
        {
            "recognition": {...},
            "meta_cognition": {...},
            "temporal": {...},
            "emotion": {...},
            "continuity": {...},
            "mentor": {...},
            "self_critique": {...}
        }
    """
    try:
        logger.info("Starting FRAMED Intelligence Core (7-layer reasoning)")
        
        # Layer 1: Certain Recognition
        logger.info("Layer 1: Certain Recognition...")
        recognition = reason_about_recognition(visual_evidence)
        
        # Layer 2: Meta-Cognition
        logger.info("Layer 2: Meta-Cognition...")
        meta_cognition = reason_about_thinking(recognition, temporal_memory)
        
        # Layer 3: Temporal Consciousness
        logger.info("Layer 3: Temporal Consciousness...")
        temporal = reason_about_evolution(meta_cognition, temporal_memory)
        
        # Layer 4: Emotional Resonance
        logger.info("Layer 4: Emotional Resonance...")
        emotion = reason_about_feeling(meta_cognition, temporal)
        
        # Layer 5: Continuity of Self
        logger.info("Layer 5: Continuity of Self...")
        continuity = reason_about_trajectory(emotion, user_history)
        
        # Layer 6: Mentor Voice (Reasoning)
        logger.info("Layer 6: Mentor Voice (Reasoning)...")
        mentor = reason_about_mentorship(continuity, user_history)
        
        # Layer 7: Self-Critique
        logger.info("Layer 7: Self-Critique...")
        self_critique = reason_about_past_errors(mentor, temporal_memory)
        
        # Compile intelligence output
        intelligence_output = {
            "recognition": recognition,
            "meta_cognition": meta_cognition,
            "temporal": temporal,
            "emotion": emotion,
            "continuity": continuity,
            "mentor": mentor,
            "self_critique": self_critique
        }
        
        logger.info("FRAMED Intelligence Core completed successfully")
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
