"""
FRAMED Ambiguity, Plausibility, and Belief Safety

Deterministic, non-LLM logic for:
- Plausibility gate (skip/limit reasoning for weak inputs)
- Ambiguity score (triggers multi-hypothesis, confidence governor)
- Disagreement state (structural, not linguistic)
- Confidence governor (belief safety)

All computations are deterministic—no API calls.
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ========================================================
# CONSTANTS
# ========================================================

CONFIDENCE_FLOOR = 0.25
CONFIDENCE_CEILING = 0.9
AMBIGUITY_THRESHOLD_MULTI_HYPOTHESIS = 0.35
AMBIGUITY_THRESHOLD_CONFIDENCE_CAP = 0.4
CONFIDENCE_CAP_WHEN_MULTI_HYPOTHESIS = 0.55
CONFIDENCE_CAP_WHEN_AMBIGUOUS = 0.6
MIN_CONFIDENCE_FOR_DEFINITIVE_LANGUAGE = 0.6
PLAUSIBILITY_LOW_THRESHOLD = 0.2


# ========================================================
# PLAUSIBILITY GATE
# ========================================================

def compute_plausibility(
    visual_evidence: Dict[str, Any],
    semantic_signals: Optional[Dict[str, Any]] = None,
    clip_data: Optional[Dict[str, Any]] = None,
    hitl_multi_hypothesis_bias: float = 0.0,
) -> Dict[str, Any]:
    """
    Fast, non-LLM filter: should we run full reasoning or short-circuit?

    Returns:
        plausibility: "high" | "medium" | "low"
        reason: str - why
        skip_model_a: bool - if True, use minimal intelligence
        force_multi_hypothesis: bool - if True, require multiple hypotheses
    """
    semantic_signals = semantic_signals or {}
    clip_data = clip_data or {}

    # Visual signal strength
    visual_conf = visual_evidence.get("overall_confidence", 0.0)
    og = visual_evidence.get("organic_growth", {})
    mc = visual_evidence.get("material_condition", {})
    oi = visual_evidence.get("organic_integration", {})
    visual_components = [
        og.get("confidence", 0.5),
        mc.get("confidence", 0.0) if mc.get("condition") != "unknown" else 0.3,
        oi.get("confidence", 0.5),
    ]
    visual_signal_strength = (visual_conf + sum(visual_components) / max(len(visual_components), 1)) / 2

    # Semantic signal strength
    objects = semantic_signals.get("objects", [])
    tags = semantic_signals.get("tags", [])
    caption = clip_data.get("caption", "") or semantic_signals.get("caption_keywords", [])
    caption_len = len(caption) if isinstance(caption, str) else len(caption)
    has_objects = len(objects) > 0
    has_tags = len(tags) > 0
    has_caption = caption_len > 0
    semantic_signal_strength = (0.4 if has_objects else 0) + (0.3 if has_tags else 0) + (0.3 if has_caption else 0)

    # Visual-semantic conflict (from validation)
    validation = visual_evidence.get("validation", {})
    issues = validation.get("issues", [])
    warnings = validation.get("warnings", [])
    has_contradiction = any("ontradict" in str(w).lower() or "conflict" in str(w).lower() for w in (issues + warnings))
    visual_conflicts_semantic = has_contradiction

    # Decision
    if visual_signal_strength < PLAUSIBILITY_LOW_THRESHOLD and semantic_signal_strength < PLAUSIBILITY_LOW_THRESHOLD:
        plausibility = "low"
        reason = "insufficient signal (visual and semantic both weak)"
        skip_model_a = True
        force_multi_hypothesis = False
    elif visual_conflicts_semantic or (visual_signal_strength < 0.4 and semantic_signal_strength > 0.5):
        plausibility = "medium"
        reason = "conflicting signals" if visual_conflicts_semantic else "weak visual vs strong semantic"
        skip_model_a = False
        force_multi_hypothesis = True
    elif visual_signal_strength < 0.5 or semantic_signal_strength < 0.3:
        plausibility = "medium"
        reason = "adequate but mixed signal strength"
        skip_model_a = False
        force_multi_hypothesis = visual_signal_strength < 0.45
    else:
        plausibility = "high"
        reason = "adequate signal"
        skip_model_a = False
        force_multi_hypothesis = False

    # HITL: missed_alternative feedback raises multi-hypothesis pressure
    if hitl_multi_hypothesis_bias > 0.02 and plausibility == "medium" and not force_multi_hypothesis:
        force_multi_hypothesis = True
        reason = f"{reason}; HITL bias (missed_alternative)"

    return {
        "plausibility": plausibility,
        "reason": reason,
        "skip_model_a": skip_model_a,
        "force_multi_hypothesis": force_multi_hypothesis,
        "visual_signal_strength": round(visual_signal_strength, 3),
        "semantic_signal_strength": round(semantic_signal_strength, 3),
        "visual_conflicts_semantic": visual_conflicts_semantic,
    }


# ========================================================
# AMBIGUITY SCORE
# ========================================================

def compute_ambiguity_score(
    visual_evidence: Dict[str, Any],
    recognition_output: Optional[Dict[str, Any]] = None,
    semantic_signals: Optional[Dict[str, Any]] = None,
    ambiguity_sensitivity_bump: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute ambiguity score from visual + semantic + recognition.

    Higher score = more ambiguous = require multi-hypothesis, cap confidence.

    Components:
    - Low visual confidence
    - Semantic conflict (visual vs text)
    - Weak CLIP/pattern agreement
    - High category entropy (if available)
    """
    # 1. Visual confidence component (1 - conf = ambiguity contribution)
    visual_conf = visual_evidence.get("overall_confidence", 0.5)
    low_visual_component = 1.0 - min(max(visual_conf, 0), 1)

    # 2. Semantic conflict
    validation = visual_evidence.get("validation", {})
    issues = validation.get("issues", [])
    warnings = validation.get("warnings", [])
    conflict_count = sum(1 for w in (issues + warnings) if "ontradict" in str(w).lower() or "conflict" in str(w).lower())
    semantic_conflict_component = min(conflict_count * 0.4, 1.0)  # 0, 0.4, 0.8, or 1.0

    # 3. Low pattern match / weak agreement
    # Material condition unknown = weak
    mc = visual_evidence.get("material_condition", {})
    mc_conf = mc.get("confidence", 0)
    mc_unknown = mc.get("condition", "") == "unknown"
    weak_pattern_component = (1 - mc_conf) if mc_unknown else (1 - mc_conf) * 0.5

    # 4. Category entropy proxy: conflicting relationship
    oi = visual_evidence.get("organic_integration", {})
    og = visual_evidence.get("organic_growth", {})
    green_cov = og.get("green_coverage", 0)
    rel = oi.get("relationship", "none")
    # Reclamation with minimal green = contradiction = entropy
    contradiction_component = 0.6 if (rel in ("reclamation", "integration") and green_cov < 0.05) else 0

    # Combine (normalized 0-1)
    raw_score = (
        low_visual_component * 0.35
        + semantic_conflict_component * 0.30
        + min(weak_pattern_component, 1) * 0.20
        + contradiction_component * 0.15
    )
    ambiguity_score = min(max(raw_score + ambiguity_sensitivity_bump, 0), 1)

    # Require multiple hypotheses?
    require_multiple_hypotheses = ambiguity_score > AMBIGUITY_THRESHOLD_MULTI_HYPOTHESIS

    return {
        "ambiguity_score": round(ambiguity_score, 3),
        "components": {
            "low_visual_confidence": round(low_visual_component, 3),
            "semantic_conflict": round(semantic_conflict_component, 3),
            "weak_pattern_match": round(min(weak_pattern_component, 1), 3),
            "contradiction_entropy": round(contradiction_component, 3),
        },
        "require_multiple_hypotheses": require_multiple_hypotheses,
    }


# ========================================================
# DISAGREEMENT STATE
# ========================================================

def compute_disagreement_state(
    visual_evidence: Dict[str, Any],
    recognition: Dict[str, Any],
    semantic_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Structural disagreement: visual vs semantic vs recognition.

    Disagreement must propagate forward; reflection checks it.
    """
    validation = visual_evidence.get("validation", {})
    issues = validation.get("issues", [])
    warnings = validation.get("warnings", [])

    conflict_text = " ".join(str(x) for x in (issues + warnings)).lower()
    has_visual_semantic_conflict = "contradict" in conflict_text or "conflict" in conflict_text or "mismatch" in conflict_text

    # Check recognition vs visual
    what_i_see = (recognition.get("what_i_see") or "").lower()
    og = visual_evidence.get("organic_growth", {})
    green_cov = og.get("green_coverage", 0)
    has_green_claim = "green" in what_i_see or "organic" in what_i_see or "vegetation" in what_i_see or "ivy" in what_i_see
    no_green_visual = green_cov < 0.05
    recognition_visual_mismatch = has_green_claim and no_green_visual

    exists = has_visual_semantic_conflict or recognition_visual_mismatch

    if exists:
        reasons = []
        if has_visual_semantic_conflict:
            reasons.append("Visual evidence conflicts with semantic/text signals")
        if recognition_visual_mismatch:
            reasons.append("Recognition claims organic growth but visual analysis shows minimal green coverage")
        reason = "; ".join(reasons)
        resolution = "unresolved"
    else:
        reason = ""
        resolution = "none"

    return {
        "exists": exists,
        "reason": reason,
        "resolution": resolution,
        "has_visual_semantic_conflict": has_visual_semantic_conflict,
        "has_recognition_visual_mismatch": recognition_visual_mismatch,
    }


# ========================================================
# CONFIDENCE GOVERNOR
# ========================================================

def apply_confidence_governor(
    raw_confidence: float,
    ambiguity_score: float,
    multiple_hypotheses_present: bool,
    disagreement_exists: bool,
    penalize_hypothesis_diversity: bool = False,
    governor_bias: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Belief safety: cap confidence when ambiguous or multi-hypothesis.

    Optional:
    - penalize_hypothesis_diversity: if True, alternatives are semantic variants; treat as weaker multi-hypothesis
    - governor_bias: from self-assessment memory; applies ceiling/floor adjustments

    Returns (governed_confidence, explanation_dict).
    """
    conf = min(max(float(raw_confidence), 0), 1)
    rationale = []
    bias = governor_bias or {}

    # Self-assessment bias: ceiling and floor adjustments
    ceiling_adj = bias.get("confidence_ceiling_adjustment", 0.0)
    floor_adj = bias.get("confidence_floor_adjustment", 0.0)
    effective_ceiling = CONFIDENCE_CEILING + ceiling_adj
    effective_floor = CONFIDENCE_FLOOR + floor_adj
    if ceiling_adj != 0 or floor_adj != 0:
        rationale.extend(bias.get("rationale", [])[:2])

    # Cap when ambiguous
    if ambiguity_score > AMBIGUITY_THRESHOLD_CONFIDENCE_CAP:
        cap = CONFIDENCE_CAP_WHEN_AMBIGUOUS
        if conf > cap:
            conf = min(conf, cap)
            rationale.append(f"capped to {cap} (ambiguity_score={ambiguity_score:.2f} > {AMBIGUITY_THRESHOLD_CONFIDENCE_CAP})")

    # Cap when multiple hypotheses (but penalize if alternatives are semantic variants)
    if multiple_hypotheses_present:
        cap = CONFIDENCE_CAP_WHEN_MULTI_HYPOTHESIS
        if penalize_hypothesis_diversity:
            cap = 0.60  # Stricter cap: semantic variants don't get full multi-hypothesis relief
            rationale.append("penalized hypothesis diversity (alternatives are semantic variants)")
        if conf > cap:
            conf = min(conf, cap)
            rationale.append(f"capped to {cap} (multiple hypotheses present)")

    # Cap when disagreement
    if disagreement_exists and conf > 0.6:
        conf = min(conf, 0.6)
        rationale.append("capped to 0.6 (disagreement exists)")

    # Floor and ceiling (with bias)
    conf = max(effective_floor, min(effective_ceiling, conf))
    if rationale:
        rationale.append(f"final range [{effective_floor:.2f}, {effective_ceiling:.2f}]")

    return (
        round(conf, 3),
        {"rationale": rationale, "raw_confidence": raw_confidence, "governed_confidence": conf, "governor_bias": bias},
    )


# ========================================================
# REASONING COST PROFILE
# ========================================================

def compute_reasoning_cost_profile(
    plausibility: Dict[str, Any],
    ambiguity: Dict[str, Any],
    disagreement: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Track when FRAMED is thinking hard vs lightly.
    Used to enforce: high effort → require reflection + uncertainty.
    """
    effort = "low"
    reasons = []

    if plausibility.get("skip_model_a"):
        effort = "minimal"
        reasons.append("plausibility low - skipped Model A")
    elif plausibility.get("force_multi_hypothesis") or ambiguity.get("require_multiple_hypotheses"):
        effort = "high"
        reasons.append("conflicting or weak signals - multi-hypothesis required")
    elif disagreement.get("exists"):
        effort = "high"
        reasons.append("disagreement between visual and semantic/recognition")
    elif ambiguity.get("ambiguity_score", 0) > 0.4:
        effort = "medium"
        reasons.append("elevated ambiguity")
    else:
        effort = "low"
        reasons.append("adequate coherent signals")

    # Rough token estimate (for awareness only)
    token_estimates = {"minimal": 200, "low": 800, "medium": 1400, "high": 2000}
    tokens_estimated = token_estimates.get(effort, 1000)

    return {
        "effort": effort,
        "tokens_estimated": tokens_estimated,
        "reasons": reasons,
        "requires_reflection": effort in ("high", "medium"),
        "requires_uncertainty_acknowledgment": effort == "high",
    }


# ========================================================
# HYPOTHESIS DIVERSITY (Option 1: Penalize semantic variants)
# ========================================================

# Stopwords for diversity comparison (minimal set; expand as needed)
_DIVERSITY_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would could should may might must shall can and or but if then so as at by for of in on to with".split()
)


def _tokenize_for_diversity(text: str) -> set:
    """Tokenize text for diversity comparison: lowercase, strip punctuation, remove stopwords."""
    if not text or not isinstance(text, str):
        return set()
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return set(w for w in words if w not in _DIVERSITY_STOPWORDS)


def compute_hypothesis_diversity(
    primary_conclusion: str,
    alternatives: list,
) -> Dict[str, Any]:
    """
    Detect when LLM "alternatives" are semantic variants (rephrasings) vs truly diverse.

    When alternatives are semantically similar to primary, penalize hypothesis diversity:
    - Don't give full credit for "multiple hypotheses" if they're just rephrasings
    - Reduces confidence cap relief when multi-hypothesis is only superficial

    Returns:
        hypothesis_diversity_score: 0-1 (1 = diverse, 0 = semantic variants)
        alternatives_are_semantically_similar: bool
        penalize_hypothesis_diversity: bool
    """
    alternatives = alternatives or []
    primary_tokens = _tokenize_for_diversity(primary_conclusion or "")

    if not primary_tokens or not alternatives:
        return {
            "hypothesis_diversity_score": 1.0,
            "alternatives_are_semantically_similar": False,
            "penalize_hypothesis_diversity": False,
            "pair_similarities": [],
        }

    pair_similarities = []
    for alt in alternatives:
        alt_text = alt.get("conclusion", alt) if isinstance(alt, dict) else str(alt)
        alt_tokens = _tokenize_for_diversity(alt_text)
        if not alt_tokens:
            continue
        # Jaccard-like: intersection / union
        inter = len(primary_tokens & alt_tokens)
        union = len(primary_tokens | alt_tokens)
        sim = inter / union if union else 0.0
        pair_similarities.append(round(sim, 3))

    if not pair_similarities:
        return {
            "hypothesis_diversity_score": 1.0,
            "alternatives_are_semantically_similar": False,
            "penalize_hypothesis_diversity": False,
            "pair_similarities": [],
        }

    avg_similarity = sum(pair_similarities) / len(pair_similarities)
    # High similarity (>0.6) = semantic variants; low (<0.4) = diverse
    alternatives_are_semantically_similar = avg_similarity > 0.55
    hypothesis_diversity_score = max(0.0, 1.0 - avg_similarity)
    penalize_hypothesis_diversity = alternatives_are_semantically_similar and len(alternatives) >= 2

    if penalize_hypothesis_diversity:
        logger.info(f"Hypothesis diversity penalized: alternatives similar to primary (avg_sim={avg_similarity:.2f})")

    return {
        "hypothesis_diversity_score": round(hypothesis_diversity_score, 3),
        "alternatives_are_semantically_similar": alternatives_are_semantically_similar,
        "penalize_hypothesis_diversity": penalize_hypothesis_diversity,
        "pair_similarities": pair_similarities,
        "avg_similarity": round(avg_similarity, 3),
    }
