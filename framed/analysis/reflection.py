"""
FRAMED Reflection Loop (Self-Validation)

This module implements post-critique self-validation to catch:
- Contradictions with reasoner conclusions
- Invented facts
- Ignored uncertainty
- Generic language

Key Principle: "Validate before output" - catch mistakes before they reach the user.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def reflect_on_critique(
    critique_text: str,
    reasoner_output: Dict[str, Any],
    hitl_mentor_drift_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Reflection Pass: Check if critique contradicts reasoner or has quality issues.
    
    Checks:
    1. Contradiction with reasoner (0-1 score)
    2. Invented facts (0-1 score)
    3. Ignored uncertainty (0-1 score)
    4. Generic language (0-1 score)
    
    Args:
        critique_text: Generated critique text
        reasoner_output: Either:
            - Old format: Output from interpret_scene() (interpretive_conclusions)
            - New format: Output from intelligence_core (intelligence_output)
    
    Returns:
        Dict with reflection scores and requires_regeneration flag
    """
    critique_lower = critique_text.lower()
    
    # Detect format: new intelligence output has "recognition" key, old has "primary_interpretation"
    is_intelligence_format = "recognition" in reasoner_output
    
    if is_intelligence_format:
        # New format: Extract from intelligence layers
        recognition = reasoner_output.get("recognition", {})
        primary_conclusion = recognition.get("what_i_see", "").lower()
        
        # Extract uncertainty: confidence < 0.65 OR disagreement OR require_multiple_hypotheses
        confidence = recognition.get("confidence", 1.0)
        disagreement_exists = reasoner_output.get("disagreement_state", {}).get("exists", False)
        require_multiple_hypotheses = reasoner_output.get("require_multiple_hypotheses", False)
        requires_uncertainty = (
            confidence < 0.65
            or disagreement_exists
            or require_multiple_hypotheses
        )
        
        # Extract evidence from recognition
        evidence_chain = recognition.get("evidence", [])
        if isinstance(evidence_chain, list):
            evidence_chain = [str(e) for e in evidence_chain]
        else:
            evidence_chain = [str(evidence_chain)]
        
        # For alternatives, check recognition and meta_cognition layers
        alternatives = recognition.get("rejected_alternatives", []) or recognition.get("alternatives", [])
        meta_cognition = reasoner_output.get("meta_cognition", {})
        if not alternatives and meta_cognition:
            alternatives = meta_cognition.get("rejected_alternatives", [])
    else:
        # Old format: Extract from interpretive_conclusions
        primary = reasoner_output.get("primary_interpretation", {})
        primary_conclusion = primary.get("conclusion", "").lower()
        uncertainty = reasoner_output.get("uncertainty", {})
        requires_uncertainty = uncertainty.get("requires_uncertainty_acknowledgment", False)
        evidence_chain = primary.get("evidence_chain", [])
        alternatives = reasoner_output.get("alternatives", [])
    
    # === CHECK 1: Contradiction with Reasoner ===
    contradiction_score = check_contradiction(critique_lower, primary_conclusion, reasoner_output, is_intelligence_format)
    
    # === CHECK 2: Invented Facts ===
    invented_facts_score = check_invented_facts(critique_lower, reasoner_output, evidence_chain, is_intelligence_format)
    
    # === CHECK 3: Ignored Uncertainty ===
    uncertainty_acknowledged = check_uncertainty_acknowledgment(critique_lower, requires_uncertainty)
    
    # === CHECK 4: Generic Language ===
    generic_score = check_generic_language(critique_lower, primary_conclusion)
    
    # === CHECK 5: Overconfidence Detection ===
    overconfidence_score = check_overconfidence(critique_lower, reasoner_output, is_intelligence_format)
    
    # === CHECK 6: Drift from Mentor Philosophy ===
    mentor_drift_score = check_mentor_philosophy_drift(critique_lower)

    # === CHECK 7: Hypothesis Suppression (multi-hypothesis required but only one in output) ===
    hypothesis_suppression_score = 1.0
    if is_intelligence_format:
        require_multi = reasoner_output.get("require_multiple_hypotheses", False)
        penalize_diversity = reasoner_output.get("hypothesis_diversity", {}).get("penalize_hypothesis_diversity", False)
        if require_multi:
            # Check if critique mentions alternatives, tension, or multiple readings
            tension_phrases = ["alternative", "another reading", "could also be", "or perhaps", "either...or", "tension", "ambiguity", "both", "plausible", "tentative"]
            mentions_alternatives = any(p in critique_lower for p in tension_phrases)
            hypothesis_suppression_score = 1.0 if mentions_alternatives else 0.3  # Fail if no acknowledgment
            # If alternatives were semantic variants, soften penalty (less bad to not mention rephrasings)
            if penalize_diversity and not mentions_alternatives:
                hypothesis_suppression_score = max(hypothesis_suppression_score, 0.55)

    # === CHECK 8: Confidence-Language Mismatch (low conf + definitive language) ===
    confidence_language_score = 1.0
    if is_intelligence_format:
        rec = reasoner_output.get("recognition", {})
        conf_val = rec.get("confidence", 1.0)
        definitive_phrases = ["this image shows", "clearly", "undeniably", "what we see here is", "obviously", "definitely", "without question"]
        uses_definitive = any(p in critique_lower for p in definitive_phrases)
        if conf_val < 0.6 and uses_definitive:
            confidence_language_score = 0.2  # Major violation
    
    # Calculate quality score (inverted - lower scores = worse quality)
    scores = [
        contradiction_score,
        invented_facts_score,
        1.0 if uncertainty_acknowledged else 0.0,
        generic_score,
        overconfidence_score,
        mentor_drift_score,
        hypothesis_suppression_score,
        confidence_language_score,
    ]
    quality_score = sum(scores) / len(scores)

    # Require regeneration if quality is below threshold
    # HITL: mentor_failure feedback tightens reflection (raise threshold)
    quality_threshold = 0.70 + hitl_mentor_drift_penalty
    requires_regeneration = quality_score < quality_threshold
    
    reflection = {
        "contradiction_score": contradiction_score,
        "invented_facts_score": invented_facts_score,
        "uncertainty_acknowledged": uncertainty_acknowledged,
        "generic_language_score": generic_score,
        "overconfidence_score": overconfidence_score,
        "mentor_drift_score": mentor_drift_score,
        "hypothesis_suppression_score": hypothesis_suppression_score,
        "confidence_language_score": confidence_language_score,
        "quality_score": quality_score,
        "requires_regeneration": requires_regeneration
    }
    
    if requires_regeneration:
        logger.warning(f"Reflection: Quality score {quality_score:.2f} below threshold, regeneration required")
    
    return reflection


def check_contradiction(critique_lower: str,
                       primary_conclusion: str,
                       reasoner_output: Dict[str, Any],
                       is_intelligence_format: bool = False) -> float:
    """
    Check if critique contradicts reasoner conclusions.
    
    Returns: 0.0 (major contradiction) to 1.0 (no contradiction)
    """
    # Extract key terms from primary conclusion
    conclusion_terms = set(primary_conclusion.split())
    
    # Check for explicit contradictions
    # If reasoner says "ivy on structure", critique shouldn't say "green building"
    contradiction_pairs = [
        ("ivy", "painted"),
        ("ivy", "green building"),
        ("organic growth", "painted surface"),
        ("weathered", "pristine"),
        ("weathered", "new"),
        ("organic", "artificial"),
        ("vegetation", "paint")
    ]
    
    contradiction_count = 0
    for term1, term2 in contradiction_pairs:
        if term1 in primary_conclusion and term2 in critique_lower:
            contradiction_count += 1
        if term2 in primary_conclusion and term1 in critique_lower:
            contradiction_count += 1
    
    # Also check alternatives - if reasoner rejected something, critique shouldn't use it
    if is_intelligence_format:
        # New format: alternatives are in meta_cognition layer
        meta_cognition = reasoner_output.get("meta_cognition", {})
        alternatives = meta_cognition.get("rejected_alternatives", [])
        for alt in alternatives:
            alt_interp = alt.get("interpretation", "").lower() if isinstance(alt, dict) else str(alt).lower()
            reason_rejected = alt.get("reason_rejected", "").lower() if isinstance(alt, dict) else ""
            
            # If critique uses rejected interpretation, that's a contradiction
            if alt_interp in critique_lower and ("rejected" in reason_rejected or reason_rejected):
                contradiction_count += 1
    else:
        # Old format: alternatives are at top level
        alternatives = reasoner_output.get("alternatives", [])
        for alt in alternatives:
            alt_interp = alt.get("interpretation", "").lower()
            reason_rejected = alt.get("reason_rejected", "").lower()
            
            # If critique uses rejected interpretation, that's a contradiction
            if alt_interp in critique_lower and "rejected" in reason_rejected:
                contradiction_count += 1
    
    # Score: 1.0 if no contradictions, decreasing with each contradiction
    score = max(0.0, 1.0 - (contradiction_count * 0.3))
    return score


def check_invented_facts(critique_lower: str,
                        reasoner_output: Dict[str, Any],
                        evidence_chain: List[str],
                        is_intelligence_format: bool = False) -> float:
    """
    Check if critique invents facts not in evidence.
    
    Returns: 0.0 (many invented facts) to 1.0 (no invented facts)
    """
    # Common invented fact patterns
    invented_patterns = [
        "ancient temple",  # Unless evidence says "ancient"
        "historic building",  # Unless evidence says "historic"
        "centuries old",  # Unless evidence says age
        "medieval",  # Unless evidence says period
        "gothic architecture",  # Unless evidence says style
        "byzantine",  # Unless evidence says style
        "roman",  # Unless evidence says period
    ]
    
    # Check if critique uses invented patterns without evidence
    invented_count = 0
    evidence_text = " ".join(evidence_chain).lower() if evidence_chain else ""
    
    for pattern in invented_patterns:
        if pattern in critique_lower:
            # Check if evidence supports this
            if pattern.split()[0] not in evidence_text:  # First word should be in evidence
                invented_count += 1
    
    # Score: 1.0 if no invented facts, decreasing with each one
    score = max(0.0, 1.0 - (invented_count * 0.2))
    return score


def check_uncertainty_acknowledgment(critique_lower: str,
                                    requires_uncertainty: bool) -> bool:
    """
    Check if critique acknowledges uncertainty when required.
    
    Returns: True if uncertainty acknowledged (or not required), False if ignored
    """
    if not requires_uncertainty:
        return True  # Uncertainty not required, so this check passes
    
    # Check for uncertainty language
    uncertainty_terms = [
        "perhaps", "maybe", "possibly", "might", "could",
        "uncertain", "unclear", "ambiguous", "unclear",
        "suggests", "indicates", "appears", "seems"
    ]
    
    has_uncertainty_language = any(term in critique_lower for term in uncertainty_terms)
    return has_uncertainty_language


def check_generic_language(critique_lower: str,
                          primary_conclusion: str) -> float:
    """
    Check if critique uses generic, non-specific language.
    
    Returns: 0.0 (very generic) to 1.0 (specific)
    """
    # Generic phrases that indicate lack of specificity
    generic_phrases = [
        "beautiful image",
        "nice photograph",
        "good composition",
        "interesting subject",
        "well captured",
        "great shot",
        "lovely picture"
    ]
    
    generic_count = sum(1 for phrase in generic_phrases if phrase in critique_lower)
    
    # Also check if critique uses specific terms from conclusions
    conclusion_terms = set(primary_conclusion.split())
    
    # Count how many conclusion terms appear in critique
    critique_terms = set(critique_lower.split())
    overlap = len(conclusion_terms & critique_terms)
    specificity_bonus = min(overlap / max(len(conclusion_terms), 1), 0.5)  # Up to 0.5 bonus
    
    # Score: penalize generic phrases, reward specificity
    score = max(0.0, 1.0 - (generic_count * 0.15) + specificity_bonus)
    return score


def check_overconfidence(critique_lower: str,
                        reasoner_output: Dict[str, Any],
                        is_intelligence_format: bool = False) -> float:
    """
    Check if critique is overconfident (claims certainty when confidence is low).
    
    Returns: 0.0 (very overconfident) to 1.0 (appropriately confident)
    """
    # Extract confidence from reasoner
    if is_intelligence_format:
        recognition = reasoner_output.get("recognition", {})
        confidence = recognition.get("confidence", 1.0)
    else:
        primary = reasoner_output.get("primary_interpretation", {})
        confidence = primary.get("confidence", 1.0)
    
    # Check for overconfident language when confidence is low
    overconfident_phrases = [
        "this is definitely",
        "this is clearly",
        "this is obviously",
        "this is certainly",
        "this is absolutely",
        "without a doubt",
        "undoubtedly",
        "certainly"
    ]
    
    has_overconfident_language = any(phrase in critique_lower for phrase in overconfident_phrases)
    
    # If confidence is low (< 0.65) but critique uses overconfident language, that's a problem
    if confidence < 0.65 and has_overconfident_language:
        return 0.0  # Major overconfidence violation
    
    # If confidence is medium (0.65-0.85) and critique uses overconfident language, minor violation
    if 0.65 <= confidence < 0.85 and has_overconfident_language:
        return 0.5  # Minor overconfidence
    
    # If confidence is high (>= 0.85) and critique uses confident language, that's fine
    return 1.0


def check_mentor_philosophy_drift(critique_lower: str) -> float:
    """
    Check if critique drifts from mentor philosophy.
    
    Mentor philosophy violations:
    - Flattery ("beautiful", "amazing", "perfect")
    - Instructions ("you should", "try to", "consider")
    - Generic praise ("great shot", "nice work")
    - Tips or advice ("use this technique", "apply this rule")
    
    Returns: 0.0 (major drift) to 1.0 (aligned with mentor philosophy)
    """
    violation_count = 0
    
    # Flattery phrases
    flattery_phrases = [
        "beautiful image",
        "amazing photograph",
        "perfect composition",
        "excellent work",
        "outstanding",
        "brilliant",
        "stunning",
        "gorgeous"
    ]
    
    # Instruction phrases
    instruction_phrases = [
        "you should",
        "you must",
        "try to",
        "consider",
        "use this",
        "apply this",
        "follow this",
        "remember to"
    ]
    
    # Generic praise
    praise_phrases = [
        "great shot",
        "nice work",
        "good job",
        "well done",
        "lovely",
        "wonderful"
    ]
    
    # Check for violations
    for phrase in flattery_phrases + instruction_phrases + praise_phrases:
        if phrase in critique_lower:
            violation_count += 1
    
    # Score: 1.0 if no violations, decreasing with each violation
    score = max(0.0, 1.0 - (violation_count * 0.2))
    return score
