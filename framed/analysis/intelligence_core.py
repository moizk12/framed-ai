"""Seven-layer Model A pipeline (call_model_a). Invariant: no learning inside the LLM—memory lives in Python (temporal_memory, learning_system)."""

import json
import logging
import os
from typing import Dict, Any, Optional, List

from .llm_provider import call_model_a
from .intelligence_formatting import (
    _safe_parse_layer_json,
    format_visual_evidence,
    format_temporal_memory,
    format_user_history,
    infer_category_lexicon_key,
)
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

# REF:D3 — see intelligence_formatting.py


from .intelligence_layers import (
    USE_COMBINED_LAYERS_2_7,
    _create_minimal_intelligence,
    _validate_combined_layers_2_7,
    reason_about_layers_2_7,
    reason_about_recognition,
    reason_about_thinking,
    reason_about_evolution,
    reason_about_feeling,
    reason_about_trajectory,
    reason_about_mentorship,
    reason_about_past_errors,
)

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
        technical = perception.get("technical", {}) or {}
        category_key = infer_category_lexicon_key(visual_evidence) if visual_evidence else None
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
        ve_l1 = dict(visual_evidence)
        if composition.get("available"):
            ve_l1["perception_composition"] = composition
        if technical.get("available") or technical.get("brightness") is not None:
            ve_l1["perception_technical"] = technical
        if category_key:
            ve_l1["inferred_category_key"] = category_key
        recognition = reason_about_recognition(ve_l1, require_multiple_hypotheses=force_multi)

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
