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


def reflect_on_critique(critique_text: str,
                       interpretive_conclusions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reflection Pass: Check if critique contradicts reasoner or has quality issues.
    
    Checks:
    1. Contradiction with reasoner (0-1 score)
    2. Invented facts (0-1 score)
    3. Ignored uncertainty (0-1 score)
    4. Generic language (0-1 score)
    
    Args:
        critique_text: Generated critique text
        interpretive_conclusions: Output from interpret_scene()
    
    Returns:
        Dict with reflection scores and requires_regeneration flag
    """
    critique_lower = critique_text.lower()
    
    # Extract reasoner conclusions
    primary = interpretive_conclusions.get("primary_interpretation", {})
    primary_conclusion = primary.get("conclusion", "").lower()
    uncertainty = interpretive_conclusions.get("uncertainty", {})
    requires_uncertainty = uncertainty.get("requires_uncertainty_acknowledgment", False)
    
    # === CHECK 1: Contradiction with Reasoner ===
    contradiction_score = check_contradiction(critique_lower, primary_conclusion, interpretive_conclusions)
    
    # === CHECK 2: Invented Facts ===
    invented_facts_score = check_invented_facts(critique_lower, interpretive_conclusions)
    
    # === CHECK 3: Ignored Uncertainty ===
    uncertainty_acknowledged = check_uncertainty_acknowledgment(critique_lower, requires_uncertainty)
    
    # === CHECK 4: Generic Language ===
    generic_score = check_generic_language(critique_lower, interpretive_conclusions)
    
    # Calculate quality score (inverted - lower scores = worse quality)
    scores = [
        contradiction_score,  # Lower = more contradiction
        invented_facts_score,  # Lower = more invented facts
        1.0 if uncertainty_acknowledged else 0.0,  # 0 if uncertainty ignored
        generic_score  # Lower = more generic
    ]
    quality_score = sum(scores) / len(scores)
    
    # Require regeneration if quality is below threshold
    requires_regeneration = quality_score < 0.70
    
    reflection = {
        "contradiction_score": contradiction_score,
        "invented_facts_score": invented_facts_score,
        "uncertainty_acknowledged": uncertainty_acknowledged,
        "generic_language_score": generic_score,
        "quality_score": quality_score,
        "requires_regeneration": requires_regeneration
    }
    
    if requires_regeneration:
        logger.warning(f"Reflection: Quality score {quality_score:.2f} below threshold, regeneration required")
    
    return reflection


def check_contradiction(critique_lower: str,
                       primary_conclusion: str,
                       conclusions: Dict[str, Any]) -> float:
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
    alternatives = conclusions.get("alternatives", [])
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
                        conclusions: Dict[str, Any]) -> float:
    """
    Check if critique invents facts not in evidence.
    
    Returns: 0.0 (many invented facts) to 1.0 (no invented facts)
    """
    # Extract evidence chain from primary interpretation
    primary = conclusions.get("primary_interpretation", {})
    evidence_chain = primary.get("evidence_chain", [])
    
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
    for pattern in invented_patterns:
        if pattern in critique_lower:
            # Check if evidence supports this
            evidence_text = " ".join(evidence_chain).lower()
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
                          conclusions: Dict[str, Any]) -> float:
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
    primary = conclusions.get("primary_interpretation", {})
    conclusion_text = primary.get("conclusion", "").lower()
    conclusion_terms = set(conclusion_text.split())
    
    # Count how many conclusion terms appear in critique
    critique_terms = set(critique_lower.split())
    overlap = len(conclusion_terms & critique_terms)
    specificity_bonus = min(overlap / max(len(conclusion_terms), 1), 0.5)  # Up to 0.5 bonus
    
    # Score: penalize generic phrases, reward specificity
    score = max(0.0, 1.0 - (generic_count * 0.15) + specificity_bonus)
    return score
