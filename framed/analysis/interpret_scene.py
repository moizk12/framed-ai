"""
FRAMED Interpretive Reasoner (The Brain)

This module implements the interpretive reasoning layer that:
- Receives visual evidence, semantic signals, and technical stats
- Generates plausible interpretations via plausibility gate
- Uses LLM to reason about what is happening (multi-hypothesis)
- Outputs structured conclusions with confidence and uncertainty

Key Principle: "Reason first, then speak" - separate interpretation from expression.

This is the silent brain. It never writes prose. It never sounds artistic.
It only answers: What is happening? What else could it be? How confident?
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# Lazy-load OpenAI client
_openai_client = None

def get_openai_client():
    """Lazy-load OpenAI client (same pattern as vision.py)"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                from config import OPENAI_API_KEY as config_key
                api_key = config_key
            except ImportError:
                pass
        
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
            logger.info("Interpretive reasoner: OpenAI client initialized")
        else:
            logger.warning("Interpretive reasoner: OpenAI API key not found")
    
    return _openai_client


def generate_plausible_interpretations(visual_evidence: Dict[str, Any], 
                                      semantic_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Plausibility Gate: Cheap filter to limit interpretation space before expensive LLM call.
    
    Generates plausible interpretations based on visual evidence and semantic signals.
    LLM will only choose from these, not invent new categories.
    
    Args:
        visual_evidence: Dict from extract_visual_features()
        semantic_signals: Dict with clip_inventory, clip_caption, yolo_objects
    
    Returns:
        List of plausible interpretations with confidence hints
    """
    plausible = []
    
    # Extract visual evidence components
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition = visual_evidence.get("material_condition", {})
    organic_integration = visual_evidence.get("organic_integration", {})
    
    green_coverage = organic_growth.get("green_coverage", 0.0)
    distribution = organic_growth.get("distribution", "none")
    salience = organic_growth.get("salience", "minimal")
    condition = material_condition.get("condition", "unknown")
    surface_roughness = material_condition.get("surface_roughness", 0.0)
    texture_variance = material_condition.get("texture_variance", 0.0)
    relationship = organic_integration.get("relationship", "none")
    
    # Extract semantic signals
    clip_inventory = semantic_signals.get("clip_inventory", [])
    clip_caption = semantic_signals.get("clip_caption", "")
    yolo_objects = semantic_signals.get("yolo_objects", [])
    
    # Convert clip_inventory to list of strings if needed
    if isinstance(clip_inventory, list):
        inventory_text = " ".join([str(item) for item in clip_inventory]).lower()
    else:
        inventory_text = str(clip_inventory).lower()
    
    caption_text = (clip_caption or "").lower()
    all_text = f"{inventory_text} {caption_text}".lower()
    
    # === ORGANIC GROWTH INTERPRETATIONS ===
    if green_coverage > 0.25:
        if distribution == "vertical_surfaces":
            # Check if CLIP or caption mentions vegetation
            vegetation_terms = ["ivy", "moss", "vegetation", "vine", "climbing", "plant", "greenery"]
            has_vegetation_signal = any(term in all_text for term in vegetation_terms)
            
            if has_vegetation_signal:
                plausible.append({
                    "interpretation": "ivy_on_structure",
                    "confidence_hint": 0.75,
                    "evidence": f"green_coverage={green_coverage:.2f}, vertical_surfaces, CLIP/caption mentions vegetation"
                })
            elif texture_variance > 0.5 and surface_roughness > 0.4:
                # High texture variance + roughness suggests organic, not paint
                plausible.append({
                    "interpretation": "organic_growth_on_structure",
                    "confidence_hint": 0.65,
                    "evidence": f"green_coverage={green_coverage:.2f}, vertical_surfaces, high texture variance"
                })
        elif distribution == "foreground":
            plausible.append({
                "interpretation": "foreground_vegetation",
                "confidence_hint": 0.70,
                "evidence": f"green_coverage={green_coverage:.2f}, foreground distribution"
            })
        elif distribution == "background":
            plausible.append({
                "interpretation": "background_vegetation",
                "confidence_hint": 0.65,
                "evidence": f"green_coverage={green_coverage:.2f}, background distribution"
            })
    
    # === PAINTED SURFACE INTERPRETATIONS ===
    # Check for color uniformity (paint is uniform, organic growth is not)
    color_uniformity = material_condition.get("color_uniformity", 0.0)
    
    if green_coverage > 0.2 and color_uniformity > 0.8 and texture_variance < 0.2:
        # High coverage + high uniformity + low texture variance = likely paint
        plausible.append({
            "interpretation": "painted_surface",
            "confidence_hint": 0.65,
            "evidence": f"green_coverage={green_coverage:.2f}, high color uniformity, low texture variance"
        })
    
    # === WEATHERED/AGED SURFACE INTERPRETATIONS ===
    if condition in ["weathered", "degraded"] and surface_roughness > 0.5:
        plausible.append({
            "interpretation": "weathered_surface",
            "confidence_hint": 0.80,
            "evidence": f"condition={condition}, surface_roughness={surface_roughness:.2f}"
        })
    
    # === ORGANIC INTEGRATION INTERPRETATIONS ===
    if relationship in ["reclamation", "integration"] and green_coverage > 0.1:
        plausible.append({
            "interpretation": "organic_integration",
            "confidence_hint": 0.75,
            "evidence": f"relationship={relationship}, green_coverage={green_coverage:.2f}"
        })
    
    # === PRISTINE SURFACE INTERPRETATIONS ===
    if condition == "pristine" and green_coverage < 0.1 and color_uniformity > 0.7:
        plausible.append({
            "interpretation": "pristine_surface",
            "confidence_hint": 0.70,
            "evidence": f"condition=pristine, minimal organic growth"
        })
    
    # If no plausible interpretations found, add a generic one
    if not plausible:
        plausible.append({
            "interpretation": "unclear_interpretation",
            "confidence_hint": 0.50,
            "evidence": "Insufficient evidence for specific interpretation"
        })
    
    return plausible


def interpret_scene(visual_evidence: Dict[str, Any],
                   semantic_signals: Dict[str, Any],
                   technical_stats: Dict[str, Any],
                   interpretive_memory_patterns: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Interpretive Reasoner: Silent brain that interprets evidence and produces probabilistic conclusions.
    
    This is NOT the critique voice. This is pure reasoning.
    Answers only 5 questions:
    1. What is most likely happening?
    2. What else could be happening?
    3. Why did you reject alternatives?
    4. How confident are you (0-1)?
    5. What emotional reading follows (one sentence max)?
    
    Args:
        visual_evidence: Dict from extract_visual_features()
        semantic_signals: Dict with clip_inventory, clip_caption, yolo_objects
        technical_stats: Dict with brightness, contrast, sharpness, color_mood
        interpretive_memory_patterns: Optional list of historical patterns (for learning)
    
    Returns:
        Dict with structured interpretive conclusions:
        {
            "primary_interpretation": {...},
            "alternatives": [...],
            "uncertainty": {...},
            "emotional_reading": {...}
        }
    """
    # Step 1: Generate plausible interpretations (plausibility gate)
    plausible = generate_plausible_interpretations(visual_evidence, semantic_signals)
    
    # Step 2: Format evidence for LLM
    evidence_text = format_evidence_for_reasoner(visual_evidence, semantic_signals, technical_stats)
    
    # Step 3: Format memory patterns if available
    memory_context = ""
    if interpretive_memory_patterns:
        memory_context = format_memory_patterns(interpretive_memory_patterns)
    
    # Step 4: Construct reasoning prompt
    reasoning_prompt = f"""You are FRAMED's internal reasoning engine. You do not write critiques. You interpret evidence and produce probabilistic conclusions.

EVIDENCE:
{evidence_text}

PLAUSIBLE INTERPRETATIONS (choose from these only):
{format_plausible_interpretations(plausible)}

HISTORICAL PATTERNS (if available):
{memory_context}

Answer these 5 questions ONLY:
1. What is most likely happening in this image? (Choose from plausible interpretations or reject all)
2. What else could be happening? (List alternatives from plausible interpretations)
3. Why did you reject alternatives? (Explain why other interpretations are less likely)
4. How confident are you (0-1)? (Be honest about uncertainty)
5. What emotional reading follows from this interpretation? (One sentence max, no prose)

Output STRICT JSON only:
{{
    "primary_interpretation": {{
        "conclusion": "string",
        "confidence": 0.0-1.0,
        "evidence_chain": ["evidence1", "evidence2", ...],
        "reasoning": "brief explanation"
    }},
    "alternatives": [
        {{
            "interpretation": "string",
            "confidence": 0.0-1.0,
            "reason_rejected": "why this is less likely"
        }}
    ],
    "uncertainty": {{
        "present": true/false,
        "confidence_threshold": 0.65,
        "requires_uncertainty_acknowledgment": true/false,
        "reason": "why uncertain if present"
    }},
    "emotional_reading": {{
        "primary": "string",
        "secondary": "string",
        "confidence": 0.0-1.0,
        "reasoning": "one sentence max"
    }}
}}

Do not write prose. Do not philosophize. Only reason."""
    
    # Step 5: Call LLM
    try:
        client = get_openai_client()
        if client is None:
            logger.warning("Interpretive reasoner: OpenAI unavailable, using fallback")
            return generate_fallback_conclusions(visual_evidence, semantic_signals, plausible)
        
        logger.info("Interpretive reasoner: Sending reasoning prompt to OpenAI")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a scientific reasoning engine. Output only structured JSON. No prose."},
                {"role": "user", "content": reasoning_prompt}
            ],
            temperature=0.3,  # Low temperature for consistent reasoning
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        conclusions = json.loads(result_text)
        
        logger.info(f"Interpretive reasoner: Received conclusions (confidence={conclusions.get('primary_interpretation', {}).get('confidence', 0.0):.2f})")
        
        # Step 6: Validate and normalize output
        return validate_and_normalize_conclusions(conclusions, plausible)
        
    except Exception as e:
        logger.error(f"Interpretive reasoner failed: {e}", exc_info=True)
        return generate_fallback_conclusions(visual_evidence, semantic_signals, plausible)


def format_evidence_for_reasoner(visual_evidence: Dict[str, Any],
                                semantic_signals: Dict[str, Any],
                                technical_stats: Dict[str, Any]) -> str:
    """Format evidence for reasoning prompt"""
    lines = []
    
    # Visual evidence (highest reliability)
    lines.append("VISUAL EVIDENCE (ground truth from pixels):")
    organic = visual_evidence.get("organic_growth", {})
    material = visual_evidence.get("material_condition", {})
    integration = visual_evidence.get("organic_integration", {})
    
    if organic.get("green_coverage", 0) > 0:
        lines.append(f"- Green coverage: {organic['green_coverage']:.3f} (confidence: {organic.get('confidence', 0.0):.2f})")
        lines.append(f"- Distribution: {organic.get('distribution', 'unknown')}")
        lines.append(f"- Salience: {organic.get('salience', 'unknown')}")
    
    if material.get("condition") != "unknown":
        lines.append(f"- Material condition: {material.get('condition')} (confidence: {material.get('confidence', 0.0):.2f})")
        lines.append(f"- Surface roughness: {material.get('surface_roughness', 0.0):.3f}")
        lines.append(f"- Texture variance: {material.get('texture_variance', 0.0):.3f}")
    
    if integration.get("relationship") != "none":
        lines.append(f"- Organic integration: {integration.get('relationship')} (confidence: {integration.get('confidence', 0.0):.2f})")
        lines.append(f"- Overlap ratio: {integration.get('overlap_ratio', 0.0):.3f}")
    
    # Semantic signals (medium reliability)
    lines.append("\nSEMANTIC SIGNALS (inference from models):")
    clip_inventory = semantic_signals.get("clip_inventory", [])
    if clip_inventory:
        lines.append(f"- CLIP inventory: {', '.join([str(item) for item in clip_inventory[:10]])}")
    
    clip_caption = semantic_signals.get("clip_caption", "")
    if clip_caption:
        lines.append(f"- CLIP caption: {clip_caption}")
    
    yolo_objects = semantic_signals.get("yolo_objects", [])
    if yolo_objects:
        # Handle both dict format and string format
        obj_names = []
        for obj in yolo_objects[:5]:
            if isinstance(obj, dict):
                obj_names.append(str(obj.get("name", obj)))
            else:
                obj_names.append(str(obj))
        lines.append(f"- YOLO objects: {', '.join(obj_names)}")
    
    # Technical stats (lower reliability)
    lines.append("\nTECHNICAL STATS (measurements):")
    if technical_stats.get("brightness") is not None:
        lines.append(f"- Brightness: {technical_stats['brightness']:.1f}")
    if technical_stats.get("contrast") is not None:
        lines.append(f"- Contrast: {technical_stats['contrast']:.1f}")
    if technical_stats.get("color_mood"):
        lines.append(f"- Color mood: {technical_stats['color_mood']}")
    
    return "\n".join(lines)


def format_plausible_interpretations(plausible: List[Dict[str, Any]]) -> str:
    """Format plausible interpretations for prompt"""
    lines = []
    for i, interp in enumerate(plausible, 1):
        lines.append(f"{i}. {interp['interpretation']} (confidence hint: {interp['confidence_hint']:.2f})")
        lines.append(f"   Evidence: {interp.get('evidence', 'N/A')}")
    return "\n".join(lines)


def format_memory_patterns(patterns: List[Dict[str, Any]]) -> str:
    """Format historical patterns for prompt"""
    if not patterns:
        return "No historical patterns available."
    
    lines = ["Historical patterns (for reference):"]
    for pattern in patterns[:5]:  # Limit to top 5
        sig = pattern.get("pattern_signature", {})
        lines.append(f"- Pattern: {sig}")
        lines.append(f"  Chosen: {pattern.get('chosen_interpretation')} (confidence: {pattern.get('confidence', 0.0):.2f})")
        if pattern.get("user_feedback"):
            lines.append(f"  Feedback: {pattern['user_feedback']}")
    
    return "\n".join(lines)


def validate_and_normalize_conclusions(conclusions: Dict[str, Any], 
                                      plausible: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and normalize LLM output"""
    # Ensure required fields exist
    if "primary_interpretation" not in conclusions:
        conclusions["primary_interpretation"] = {
            "conclusion": "unclear_interpretation",
            "confidence": 0.50,
            "evidence_chain": [],
            "reasoning": "No primary interpretation provided"
        }
    
    # Ensure confidence is in valid range
    primary = conclusions["primary_interpretation"]
    primary["confidence"] = max(0.0, min(1.0, primary.get("confidence", 0.5)))
    
    # Ensure uncertainty flags are set
    if "uncertainty" not in conclusions:
        conclusions["uncertainty"] = {
            "present": primary["confidence"] < 0.65,
            "confidence_threshold": 0.65,
            "requires_uncertainty_acknowledgment": primary["confidence"] < 0.65,
            "reason": "Low confidence" if primary["confidence"] < 0.65 else "Confident"
        }
    else:
        uncertainty = conclusions["uncertainty"]
        uncertainty["requires_uncertainty_acknowledgment"] = uncertainty.get("present", False) or primary["confidence"] < 0.65
    
    # Ensure alternatives exist
    if "alternatives" not in conclusions:
        conclusions["alternatives"] = []
    
    # Ensure emotional reading exists
    if "emotional_reading" not in conclusions:
        conclusions["emotional_reading"] = {
            "primary": "neutral",
            "secondary": "neutral",
            "confidence": 0.50,
            "reasoning": "No emotional reading provided"
        }
    
    return conclusions


def generate_fallback_conclusions(visual_evidence: Dict[str, Any],
                                 semantic_signals: Dict[str, Any],
                                 plausible: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate fallback conclusions when LLM is unavailable"""
    # Use highest confidence plausible interpretation
    if plausible:
        best = max(plausible, key=lambda x: x.get("confidence_hint", 0.0))
        conclusion = best["interpretation"]
        confidence = best["confidence_hint"]
    else:
        conclusion = "unclear_interpretation"
        confidence = 0.50
    
    return {
        "primary_interpretation": {
            "conclusion": conclusion,
            "confidence": confidence,
            "evidence_chain": ["fallback_mode"],
            "reasoning": "LLM unavailable, using plausibility gate fallback"
        },
        "alternatives": [
            {
                "interpretation": alt["interpretation"],
                "confidence": alt["confidence_hint"] * 0.8,  # Lower confidence for alternatives
                "reason_rejected": "Not selected as primary"
            }
            for alt in plausible if alt["interpretation"] != conclusion
        ],
        "uncertainty": {
            "present": confidence < 0.65,
            "confidence_threshold": 0.65,
            "requires_uncertainty_acknowledgment": confidence < 0.65,
            "reason": "Fallback mode - lower confidence"
        },
        "emotional_reading": {
            "primary": "neutral",
            "secondary": "neutral",
            "confidence": 0.50,
            "reasoning": "Fallback mode - no emotional analysis"
        }
    }
