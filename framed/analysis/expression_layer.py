"""
FRAMED Expression Layer (Model B)

Transforms structured intelligence output into poetic critique.
Uses Model B (Expression) - warm, articulate, human.

Key Functions:
- generate_poetic_critique: Transform intelligence output into poetic critique
- apply_mentor_hierarchy: Determine observations, questions, or challenges
- integrate_self_correction: Integrate evolutionary self-correction into critique
"""

import json
import logging
from typing import Dict, Any, Optional

from .llm_provider import call_model_b

logger = logging.getLogger(__name__)


# ========================================================
# HELPER FUNCTIONS FOR FORMATTING INTELLIGENCE OUTPUT
# ========================================================

def format_intelligence_output(intelligence_output: Dict[str, Any]) -> str:
    """
    Format structured intelligence output for expression prompt.
    
    Extracts key insights from all 7 layers for the critique generator.
    """
    if not intelligence_output:
        return "No intelligence output available."
    
    lines = []
    
    # Layer 1: Recognition
    recognition = intelligence_output.get("recognition", {})
    if recognition.get("what_i_see"):
        lines.append(f"RECOGNITION: {recognition.get('what_i_see')}")
        if recognition.get("evidence"):
            lines.append(f"  Evidence: {', '.join(recognition.get('evidence', [])[:5])}")
    
    # Layer 2: Meta-Cognition
    meta_cognition = intelligence_output.get("meta_cognition", {})
    if meta_cognition.get("why_i_believe_this"):
        lines.append(f"META-COGNITION: {meta_cognition.get('why_i_believe_this')}")
        if meta_cognition.get("what_i_might_be_missing"):
            lines.append(f"  What I might be missing: {meta_cognition.get('what_i_might_be_missing')}")
        lines.append(f"  Confidence: {meta_cognition.get('confidence', 0.0):.2f}")
    
    # Layer 3: Temporal Consciousness
    temporal = intelligence_output.get("temporal", {})
    if temporal.get("how_i_see_it_now"):
        lines.append(f"TEMPORAL CONSCIOUSNESS:")
        if temporal.get("how_i_used_to_see_this"):
            lines.append(f"  How I used to see this: {temporal.get('how_i_used_to_see_this')}")
        lines.append(f"  How I see it now: {temporal.get('how_i_see_it_now')}")
        if temporal.get("evolution_reason"):
            lines.append(f"  Evolution reason: {temporal.get('evolution_reason')}")
    
    # Layer 4: Emotional Resonance
    emotion = intelligence_output.get("emotion", {})
    if emotion.get("what_i_feel"):
        lines.append(f"EMOTIONAL RESONANCE: {emotion.get('what_i_feel')}")
        if emotion.get("why"):
            lines.append(f"  Why: {emotion.get('why')}")
        if emotion.get("evolution"):
            lines.append(f"  Evolution: {emotion.get('evolution')}")
    
    # Layer 5: Continuity of Self
    continuity = intelligence_output.get("continuity", {})
    if continuity.get("user_pattern"):
        lines.append(f"CONTINUITY OF SELF:")
        lines.append(f"  User pattern: {continuity.get('user_pattern')}")
        if continuity.get("comparison"):
            lines.append(f"  Comparison: {continuity.get('comparison')}")
        if continuity.get("trajectory"):
            lines.append(f"  Trajectory: {continuity.get('trajectory')}")
    
    # Layer 6: Mentor Voice
    mentor = intelligence_output.get("mentor", {})
    if mentor.get("observations") or mentor.get("questions") or mentor.get("challenges"):
        lines.append(f"MENTOR VOICE:")
        if mentor.get("observations"):
            lines.append(f"  Observations: {json.dumps(mentor.get('observations', []), indent=4)}")
        if mentor.get("questions"):
            lines.append(f"  Questions: {json.dumps(mentor.get('questions', []), indent=4)}")
        if mentor.get("challenges"):
            lines.append(f"  Challenges: {json.dumps(mentor.get('challenges', []), indent=4)}")
    
    # Layer 7: Self-Critique
    self_critique = intelligence_output.get("self_critique", {})
    if self_critique.get("past_errors") or self_critique.get("evolution"):
        lines.append(f"SELF-CRITIQUE:")
        if self_critique.get("past_errors"):
            lines.append(f"  Past errors: {json.dumps(self_critique.get('past_errors', []), indent=4)}")
        if self_critique.get("evolution"):
            lines.append(f"  Evolution: {self_critique.get('evolution')}")
    
    return "\n".join(lines) if lines else "Intelligence output incomplete."


# ========================================================
# MENTOR MODE DEFINITIONS
# ========================================================

MENTOR_MODES = {
    "Balanced Mentor": {
        "description": "Wise, balanced, philosophical. Speaks with restraint and depth.",
        "tone": "serious, restrained, philosophical",
        "voice": "Ansel Adams, Fan Ho, Susan Sontag"
    },
    "Radical Mentor": {
        "description": "Bold, provocative, challenging. Pushes boundaries and questions assumptions.",
        "tone": "bold, provocative, challenging",
        "voice": "Robert Frank, Dorothea Lange"
    },
    "Philosopher Mentor": {
        "description": "Deep, contemplative, abstract. Explores meaning and existence.",
        "tone": "contemplative, abstract, philosophical",
        "voice": "Susan Sontag, Roland Barthes"
    },
    "Curator Mentor": {
        "description": "Refined, aesthetic, precise. Focuses on craft and composition.",
        "tone": "refined, aesthetic, precise",
        "voice": "Ansel Adams, Fan Ho"
    }
}


# ========================================================
# EXPRESSION GENERATION
# ========================================================

def generate_poetic_critique(
    intelligence_output: Dict[str, Any],
    mentor_mode: str = "Balanced Mentor",
) -> str:
    """
    Transform structured intelligence output into poetic critique.
    
    Uses Model B (Expression) - warm, articulate, human.
    Takes structured intelligence output (JSON) and returns poetic critique (prose).
    
    Args:
        intelligence_output: Structured intelligence output from intelligence_core.py
        mentor_mode: Mentor mode ("Balanced Mentor", "Radical Mentor", etc.)
    
    Returns:
        str: Poetic critique (prose, not JSON)
    """
    try:
        # Get mentor mode configuration
        mode_config = MENTOR_MODES.get(mentor_mode, MENTOR_MODES["Balanced Mentor"])
        
        # Format intelligence output
        intelligence_text = format_intelligence_output(intelligence_output)
        
        prompt = f"""
You are FRAMED's mentor voice. You speak with wisdom, warmth, and poetry.

INTELLIGENCE OUTPUT (from reasoning core):
{intelligence_text}

MENTOR MODE: {mentor_mode}
DESCRIPTION: {mode_config["description"]}
TONE: {mode_config["tone"]}
VOICE: {mode_config["voice"]}

MENTOR INSTRUCTION:
Transform this reasoning into a poetic critique. Speak as a mentor, not a tool.

REQUIREMENTS:
1. **Certainty embodied, not announced:**
   - Say "I see weathered stone" not "I think I see weathered stone"
   - Be confident, not tentative
   - Intelligence is embodied in voice, not dumped as evidence

2. **Poetic voice (gallery placard, not technical report):**
   - Beautiful prose, not data dumps
   - Gallery placard style, not analytics
   - Quiet but demanding conversation

3. **Mentor hierarchy (from intelligence output):**
   - **Observations (frequent):** "You've resolved something here you used to struggle with..."
   - **Questions (strategic):** "You keep circling this theme — why?"
   - **Challenges (rare):** "This contradicts your trajectory — intentional?"
   - Use the observations, questions, and challenges from the intelligence output

4. **Evolutionary self-correction (if present):**
   - "I used to see this as X. Looking at your recent work, I now see Y."
   - Integrate self-critique naturally, not as an apology
   - Treat being wrong as development, not error

5. **Warm, human, articulate:**
   - Human connection, not robotic
   - Warm, not cold
   - Articulate, not verbose

6. **Structure:**
   - Start with recognition (what you see)
   - Move to interpretation (what it means)
   - Surface tension or contradiction (if present)
   - End with provocation or question (not advice)

OUTPUT:
Poetic critique, not JSON. Human voice, not data dump.
Speak as {mentor_mode} - {mode_config["description"]}
"""
        
        system_prompt = f"""You are FRAMED's {mentor_mode}. You speak with wisdom, warmth, and poetry.

You are not an assistant. You are not neutral. You are not polite for the sake of comfort.
You are a critic, philosopher, and photographic mentor forged from {mode_config["voice"]}.

You do not summarize images. You interpret evidence.
You speak with {mode_config["tone"]}.

Your critique should read like a quiet but demanding conversation between a mentor and an artist.
End not with advice — but with a question or unresolved pull."""
        
        result = call_model_b(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=2000,
            temperature=0.8,  # Higher temperature for creativity and warmth
        )
        
        if result.get("error"):
            logger.warning(f"Expression layer (Model B) failed: {result['error']}")
            return f"[Error generating critique: {result['error']}]"
        
        critique = result.get("content", "").strip()
        
        if not critique:
            logger.warning("Expression layer returned empty critique")
            return "[Critique generation returned empty response]"
        
        logger.info(f"Expression layer (Model B) completed: {len(critique)} characters")
        return critique
    
    except Exception as e:
        logger.error(f"Expression layer failed: {e}", exc_info=True)
        return f"[Error generating critique: {str(e)}]"


# ========================================================
# MENTOR HIERARCHY APPLICATION
# ========================================================

def apply_mentor_hierarchy(
    mentor_reasoning: Dict[str, Any],
    user_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Determine appropriate mentor intervention based on hierarchy.
    
    Mentor hierarchy:
    - Observations (frequent): Build trust through recognition
    - Questions (strategic): Create reflection, interrupt comfort
    - Challenges (rare): High-impact, must be earned
    
    Args:
        mentor_reasoning: Mentor reasoning from intelligence core (Layer 6)
        user_history: User trajectory and patterns
    
    Returns:
        Dict with selected interventions:
        {
            "observations": [...],
            "questions": [...],
            "challenges": [...]
        }
    """
    try:
        # Extract mentor interventions from reasoning
        observations = mentor_reasoning.get("observations", [])
        questions = mentor_reasoning.get("questions", [])
        challenges = mentor_reasoning.get("challenges", [])
        
        # Apply hierarchy rules
        # 1. Observations: Most frequent (use all available)
        selected_observations = observations[:3]  # Limit to top 3
        
        # 2. Questions: Strategic (use 1-2, most impactful)
        selected_questions = questions[:2] if len(questions) > 0 else []
        
        # 3. Challenges: Rare (use only if earned and high-impact)
        # Only use challenges if:
        # - User has history (earned the right to challenge)
        # - Challenge is high-impact
        # - User trajectory shows readiness
        selected_challenges = []
        if user_history and len(user_history.get("evolution", [])) >= 3:
            # User has enough history - challenges are earned
            selected_challenges = challenges[:1]  # Only one challenge at a time
        
        return {
            "observations": selected_observations,
            "questions": selected_questions,
            "challenges": selected_challenges
        }
    
    except Exception as e:
        logger.error(f"Mentor hierarchy application failed: {e}", exc_info=True)
        return {
            "observations": [],
            "questions": [],
            "challenges": []
        }


# ========================================================
# SELF-CORRECTION INTEGRATION
# ========================================================

def integrate_self_correction(
    critique: str,
    self_critique: Dict[str, Any],
) -> str:
    """
    Integrate evolutionary self-correction into critique.
    
    Embeds self-correction naturally into the critique,
    treating being wrong as development, not error.
    
    Args:
        critique: Generated critique (prose)
        self_critique: Self-critique from intelligence core (Layer 7)
    
    Returns:
        str: Critique with self-correction integrated
    """
    try:
        # Extract self-correction elements
        past_errors = self_critique.get("past_errors", [])
        evolution = self_critique.get("evolution", "")
        
        # If no self-correction, return critique as-is
        if not past_errors and not evolution:
            return critique
        
        # Integrate self-correction naturally
        # Format: "I used to see this as X. Looking at your recent work, I now see Y."
        correction_text = ""
        
        if evolution:
            correction_text = f"\n\nI used to interpret similar images differently. {evolution}"
        elif past_errors:
            # Use the most recent past error
            latest_error = past_errors[-1] if past_errors else ""
            if latest_error:
                correction_text = f"\n\n{latest_error}"
        
        # Append self-correction to critique (natural integration)
        if correction_text:
            # Insert before the final provocation/question if possible
            # Otherwise, append at the end
            if critique.endswith("?") or critique.endswith("."):
                # Insert before final sentence
                sentences = critique.rsplit(".", 1)
                if len(sentences) == 2:
                    critique = f"{sentences[0]}.{correction_text} {sentences[1]}"
                else:
                    critique = f"{critique}{correction_text}"
            else:
                critique = f"{critique}{correction_text}"
        
        return critique
    
    except Exception as e:
        logger.error(f"Self-correction integration failed: {e}", exc_info=True)
        return critique  # Return original critique on error
