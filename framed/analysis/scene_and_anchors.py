# REF:C3 semantic anchors + emotional substrate + scene understanding (split from vision.py)
import logging
import os
from .visual_evidence import extract_visual_features

logger = logging.getLogger(__name__)

def generate_semantic_anchors(clip_inventory, clip_tags, clip_caption, yolo_objects, composition_data):
    """
    Generate semantic anchors from multiple signals using consensus logic.
    
    Semantic Anchors are high-confidence, low-risk labels derived from multiple signals.
    They are not facts — they are permissions. Only keys that pass thresholds are included.
    
    Args:
        clip_inventory: List of strings from get_clip_inventory()
        clip_tags: List of strings from get_clip_description() tags
        clip_caption: String from get_clip_description() caption
        yolo_objects: List of strings from YOLO object detection
        composition_data: Dict with symmetry, subject_size, etc.
    
    Returns:
        Dict with sparse semantic anchors (only keys that meet thresholds)
        Missing key = not permitted, present key = safe to reference
    """
    anchors = {}
    
    # Normalize inputs to lowercase for matching
    # Handle both dict format (new) and string format (legacy) for clip_inventory
    inventory_lower = []
    for item in (clip_inventory or []):
        if isinstance(item, dict):
            # New format: extract "item" field
            inventory_lower.append(str(item.get("item", "")).lower())
        else:
            # Legacy format: string
            inventory_lower.append(str(item).lower())
    
    tags_lower = [str(tag).lower() for tag in (clip_tags or [])]
    caption_lower = str(clip_caption or "").lower()
    # Handle both dict format and string format for yolo_objects
    yolo_lower = []
    for obj in (yolo_objects or []):
        if isinstance(obj, dict):
            # Dict format: extract "name" field or convert to string
            yolo_lower.append(str(obj.get("name", obj)).lower())
        else:
            # String format
            yolo_lower.append(str(obj).lower())
    
    all_text = " ".join(inventory_lower + tags_lower + [caption_lower] + yolo_lower).lower()
    
    # Keyword sets for matching
    ARCHITECTURE_TERMS = ["mosque", "cathedral", "temple", "church", "religious architecture", 
                          "building", "structure", "tower", "dome", "minaret", "spire",
                          "architectural facade", "interior space"]
    STRUCTURE_TERMS = ["dome", "minaret", "tower", "spire", "arch", "column", "wall", 
                      "building", "structure", "facade"]
    TIME_TERMS = ["night", "daytime", "dawn", "dusk", "sunset", "sunrise"]
    ATMOSPHERE_TERMS = ["fog", "mist", "haze", "rain", "snow", "atmospheric", "smoke"]
    LIGHTING_TERMS = ["artificial lighting", "neon lights", "street lights", "illuminated"]
    HUMAN_TERMS = ["person", "people", "crowd", "human figure", "silhouette"]
    NO_HUMAN_TERMS = ["no people", "empty", "unoccupied"]
    SCALE_TERMS = ["monumental", "large structure", "tall building", "vast"]
    INTIMATE_TERMS = ["intimate space", "small building", "close-up"]
    
    # === SCENE_TYPE ===
    # Requires: 2+ signals (CLIP inventory + CLIP tags, or CLIP + composition)
    architecture_signals = sum(1 for term in ARCHITECTURE_TERMS if term in all_text)
    time_signals = sum(1 for term in TIME_TERMS if term in all_text)
    
    if architecture_signals >= 1 and time_signals >= 1:
        # Build scene type string (canonicalized)
        arch_type = None
        for term in ["mosque", "cathedral", "temple", "church"]:
            if term in all_text:
                arch_type = term
                break
        if not arch_type:
            arch_type = "religious architecture" if any(t in all_text for t in ["religious", "mosque", "cathedral", "temple", "church"]) else "architecture"
        
        time_type = None
        for term in TIME_TERMS:
            if term in all_text:
                time_type = term
                break
        
        if arch_type and time_type:
            anchors["scene_type"] = f"{arch_type} at {time_type}"
        elif arch_type:
            anchors["scene_type"] = arch_type
    elif architecture_signals >= 2:  # Architecture mentioned multiple times
        for term in ["mosque", "cathedral", "temple", "church"]:
            if term in all_text:
                anchors["scene_type"] = term
                break
        if "scene_type" not in anchors:
            anchors["scene_type"] = "architecture"
    
    # === STRUCTURE_ELEMENTS ===
    # Requires: 1+ signal (CLIP tags or CLIP inventory)
    structure_found = []
    for term in STRUCTURE_TERMS:
        if term in all_text:
            structure_found.append(term)
    
    if structure_found:
        # Canonicalize: remove duplicates, sort
        structure_found = sorted(list(set(structure_found)))
        anchors["structure_elements"] = structure_found[:5]  # Limit to top 5
    
    # === HUMAN_PRESENCE ===
    # Requires: 1 signal (YOLO or CLIP)
    human_detected = any(term in all_text for term in HUMAN_TERMS)
    no_human_detected = any(term in all_text for term in NO_HUMAN_TERMS)
    
    if human_detected:
        anchors["human_presence"] = "present"
    elif no_human_detected or (not human_detected and len(yolo_lower) > 0 and "person" not in " ".join(yolo_lower)):
        anchors["human_presence"] = "none detected"
    
    # === ATMOSPHERE ===
    # Requires: 1+ signal (CLIP tags or technical analysis)
    atmosphere_found = []
    for term in ATMOSPHERE_TERMS:
        if term in all_text:
            atmosphere_found.append(term)
    
    lighting_found = []
    for term in LIGHTING_TERMS:
        if term in all_text:
            lighting_found.append(term)
    
    if atmosphere_found or lighting_found:
        combined = atmosphere_found + lighting_found
        anchors["atmosphere"] = sorted(list(set(combined)))[:4]  # Limit to top 4
    
    # === SCALE ===
    # Requires: 1 signal (composition analysis or YOLO "tower"/"building")
    scale_signals = sum(1 for term in SCALE_TERMS if term in all_text)
    intimate_signals = sum(1 for term in INTIMATE_TERMS if term in all_text)
    
    # Also check composition data
    subject_size = composition_data.get("subject_size", "") if composition_data else ""
    subject_size_lower = subject_size.lower() if subject_size else ""
    
    if scale_signals >= 1 or "large" in subject_size_lower or "extreme" in subject_size_lower:
        anchors["scale"] = "monumental"
    elif intimate_signals >= 1 or "small" in subject_size_lower or "tiny" in subject_size_lower:
        anchors["scale"] = "intimate"
    elif any(term in yolo_lower for term in ["tower", "building", "structure"]) and len(yolo_lower) > 0:
        # YOLO detected large structures
        anchors["scale"] = "monumental"
    
    return anchors


def synthesize_emotional_substrate_constrained(visual_evidence, technical_data, clip_data, scene_context):
    """
    Synthesize emotional substrate with full explainability and constraints.
    Every emotional output must be explainable by upstream signals.
    
    Universal: works for any image type.
    
    Args:
        visual_evidence: Dict from extract_visual_features() - ground truth from pixels
        technical_data: Dict with brightness, contrast, sharpness, color_mood
        clip_data: Dict with caption, tags, inventory
        scene_context: Dict with temporal, organic interaction, etc.
    
    Returns:
        Dict with emotional substrate, each field containing:
            - value: str - the emotional value
            - evidence: list - explainable evidence
            - confidence: float - confidence level
            - contradictions: dict - forbidden states and reasons
    """
    emotional_substrate = {}
    
    # Extract visual evidence
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition = visual_evidence.get("material_condition", {})
    organic_integration = visual_evidence.get("organic_integration", {})
    
    green_coverage = organic_growth.get("green_coverage", 0.0)
    condition = material_condition.get("condition", "unknown")
    surface_roughness = material_condition.get("surface_roughness", 0.0)
    relationship = organic_integration.get("relationship", "none")
    integration_level = organic_integration.get("integration_level", "none")
    
    # Extract technical data
    color_mood = technical_data.get("color_mood")
    brightness = technical_data.get("brightness")
    sharpness = technical_data.get("sharpness")
    
    # Extract scene context
    temporal_pace = scene_context.get("temporal_context", {}).get("pace", "static")
    organic_rel = scene_context.get("organic_interaction", {}).get("relationship", "none")
    
    # === TEMPERATURE SYNTHESIS (with evidence) ===
    # Visual evidence has highest priority (ground truth)
    if green_coverage > 0.35 and condition in ["weathered", "degraded"]:
        # Organic growth + weathering = warmth of time (proven from pixels)
        emotional_substrate["temperature"] = {
            "value": "warm_patience",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                f"condition={condition} (visual)",
                "organic_growth + weathering = warmth of time"
            ],
            "confidence": 0.92,  # High - visual evidence
            "source": "visual_analysis",
            "contradictions": {
                "forbidden": ["cold", "clinical", "sterile"],
                "reason": "Organic growth and weathering indicate warmth of time, not coldness"
            }
        }
    elif green_coverage > 0.2 and relationship in ["reclamation", "integration"]:
        # Organic integration = warmth (proven from pixels)
        emotional_substrate["temperature"] = {
            "value": "warm",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                f"relationship={relationship} (visual)",
                "organic integration suggests warmth"
            ],
            "confidence": 0.85,
            "source": "visual_analysis",
            "contradictions": {
                "forbidden": ["cold", "clinical"],
                "reason": "Organic integration contradicts coldness"
            }
        }
    elif condition == "pristine" and green_coverage < 0.1:
        # Pristine + no organic = cold (proven from pixels)
        emotional_substrate["temperature"] = {
            "value": "cold",
            "evidence": [
                f"condition={condition} (visual)",
                f"green_coverage={green_coverage:.3f} (visual)",
                "pristine + no organic = cold"
            ],
            "confidence": 0.80,
            "source": "visual_analysis"
        }
    elif color_mood == "warm" and green_coverage < 0.1:
        # Color says warm but no organic evidence
        emotional_substrate["temperature"] = {
            "value": "warm",
            "evidence": [
                f"color_mood={color_mood} (technical)",
                f"green_coverage={green_coverage:.3f} (visual)"
            ],
            "confidence": 0.70,  # Lower - color alone
            "source": "technical_analysis"
        }
    elif color_mood == "cool" and green_coverage < 0.1:
        emotional_substrate["temperature"] = {
            "value": "cold",
            "evidence": [
                f"color_mood={color_mood} (technical)",
                f"green_coverage={green_coverage:.3f} (visual)"
            ],
            "confidence": 0.70,
            "source": "technical_analysis"
        }
    else:
        emotional_substrate["temperature"] = {
            "value": "neutral",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                f"condition={condition} (visual)",
                "mixed or ambiguous signals"
            ],
            "confidence": 0.60,
            "source": "multi_modal"
        }
    
    # === PRESENCE SYNTHESIS (with evidence) ===
    if integration_level == "high" or relationship == "reclamation":
        # High integration = grounded presence (proven from pixels)
        emotional_substrate["presence"] = {
            "value": "grounded",
            "evidence": [
                f"integration_level={integration_level} (visual)",
                f"relationship={relationship} (visual)",
                "nature integration suggests grounded, lived-in presence"
            ],
            "confidence": 0.88,
            "source": "visual_analysis",
            "contradictions": {
                "forbidden": ["distant", "alienating"],
                "reason": "Nature integration contradicts distance"
            }
        }
    elif green_coverage > 0.2:
        # Organic present = grounded (proven from pixels)
        emotional_substrate["presence"] = {
            "value": "grounded",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                "organic elements suggest grounded presence"
            ],
            "confidence": 0.75,
            "source": "visual_analysis"
        }
    elif condition == "pristine" and green_coverage < 0.05:
        # Pristine + no organic = distant (proven from pixels)
        emotional_substrate["presence"] = {
            "value": "distant",
            "evidence": [
                f"condition={condition} (visual)",
                f"green_coverage={green_coverage:.3f} (visual)",
                "pristine + no organic = distant"
            ],
            "confidence": 0.75,
            "source": "visual_analysis"
        }
    else:
        emotional_substrate["presence"] = {
            "value": "grounded",
            "evidence": ["default assumption"],
            "confidence": 0.50,
            "source": "default"
        }
    
    # === PACE SYNTHESIS (with evidence) ===
    if temporal_pace == "slow" and green_coverage > 0.2:
        # Slow pace + organic = contemplative (proven from pixels)
        emotional_substrate["pace"] = {
            "value": "slow_contemplative",
            "evidence": [
                f"temporal_pace={temporal_pace}",
                f"green_coverage={green_coverage:.3f} (visual)",
                "slow pace + organic growth = contemplative"
            ],
            "confidence": 0.85,
            "source": "multi_modal"
        }
    elif temporal_pace == "fast":
        emotional_substrate["pace"] = {
            "value": "fast_energetic",
            "evidence": [f"temporal_pace={temporal_pace}"],
            "confidence": 0.80,
            "source": "temporal_context"
        }
    elif green_coverage > 0.3:
        # Organic growth = slow (proven from pixels - growth takes time)
        emotional_substrate["pace"] = {
            "value": "slow_contemplative",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                "organic growth indicates slow time"
            ],
            "confidence": 0.80,
            "source": "visual_analysis"
        }
    else:
        emotional_substrate["pace"] = {
            "value": "static_eternal",
            "evidence": ["default assumption"],
            "confidence": 0.60,
            "source": "default"
        }
    
    # === QUALITY SYNTHESIS (with evidence) ===
    if green_coverage > 0.3 and condition in ["weathered", "degraded"] and temporal_pace == "slow":
        # Organic + weathered + slow = enduring calm (proven from pixels)
        emotional_substrate["quality"] = {
            "value": "enduring_calm",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                f"condition={condition} (visual)",
                f"temporal_pace={temporal_pace}",
                "organic + weathered + slow = enduring calm"
            ],
            "confidence": 0.90,
            "source": "multi_modal"
        }
    elif green_coverage > 0.2:
        # Organic present = organic quality (proven from pixels)
        emotional_substrate["quality"] = {
            "value": "organic",
            "evidence": [
                f"green_coverage={green_coverage:.3f} (visual)",
                "organic elements suggest organic quality"
            ],
            "confidence": 0.80,
            "source": "visual_analysis"
        }
    elif sharpness and sharpness > 150 and green_coverage < 0.1:
        # High sharpness + no organic = clinical (proven from pixels)
        emotional_substrate["quality"] = {
            "value": "clinical",
            "evidence": [
                f"sharpness={sharpness} (technical)",
                f"green_coverage={green_coverage:.3f} (visual)",
                "high sharpness + no organic = clinical"
            ],
            "confidence": 0.75,
            "source": "multi_modal"
        }
    else:
        emotional_substrate["quality"] = {
            "value": "calm",
            "evidence": ["default assumption"],
            "confidence": 0.60,
            "source": "default"
        }
    
    return emotional_substrate


def synthesize_scene_understanding(analysis_result):
    """
    Synthesize contextual understanding of "what is happening here" from perception signals.
    
    This is a cognitive layer that answers material condition, temporal context, organic interaction,
    emotional substrate, and contextual relationships - universal to any image type.
    
    NOW ENHANCED: Uses visual evidence (ground truth from pixels) as primary source.
    
    Args:
        analysis_result: Canonical schema analysis result (must have perception layer)
    
    Returns:
        Dict with sparse scene_understanding (only high-confidence elements)
        Missing key = ambiguous, present key = confident understanding
    """
    # Feature flag check
    scene_understanding_enabled = os.getenv("SCENE_UNDERSTANDING_ENABLE", "true").lower() == "true"
    if not scene_understanding_enabled:
        return {}
    
    understanding = {}
    
    # ========================================================
    # SAFE DEFAULTS FOR ALL SCENE UNDERSTANDING VARIABLES
    # Every variable must have a safe default at declaration
    # This prevents NameError when variables are used before assignment
    # ========================================================
    
    # Visual evidence defaults
    visual_evidence = {}
    organic_growth = {}
    material_condition_vis = {}
    organic_integration_vis = {}
    green_coverage = 0.0
    condition_vis = "unknown"
    surface_roughness = 0.0
    relationship_vis = "none"
    integration_level_vis = "none"
    salience = "minimal"
    
    # Perception signal defaults
    perception = analysis_result.get("perception", {})
    technical = perception.get("technical", {})
    composition = perception.get("composition", {})
    color = perception.get("color", {})
    lighting = perception.get("lighting", {})
    semantics = perception.get("semantics", {})
    emotion = perception.get("emotion", {})
    derived = analysis_result.get("derived", {})
    
    # Text signal defaults
    clip_caption = ""
    clip_tags = []
    clip_inventory = []
    all_text = ""
    
    # Technical measurement defaults
    brightness = None
    contrast = None
    sharpness = None
    color_mood = None
    lighting_direction = None
    
    # Signal count defaults
    organic_signals = 0
    historical_signals = 0
    motion_signals = 0
    stillness_signals = 0
    
    # === EXTRACT VISUAL EVIDENCE (GROUND TRUTH) ===
    # This is the new foundation - visual evidence from pixels
    image_path = analysis_result.get("_image_path")  # Temporarily stored during analysis
    if image_path and os.path.exists(image_path):
        try:
            visual_evidence = extract_visual_features(image_path)
        except Exception as e:
            logger.warning(f"Visual feature extraction failed: {e}")
            visual_evidence = {}
    
    # Extract visual evidence components (with safe defaults)
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition_vis = visual_evidence.get("material_condition", {})
    organic_integration_vis = visual_evidence.get("organic_integration", {})
    
    green_coverage = organic_growth.get("green_coverage", 0.0)
    condition_vis = material_condition_vis.get("condition", "unknown")
    surface_roughness = material_condition_vis.get("surface_roughness", 0.0)
    relationship_vis = organic_integration_vis.get("relationship", "none")
    integration_level_vis = organic_integration_vis.get("integration_level", "none")
    salience = organic_growth.get("salience", "minimal")
    
    # Extract perception signals (for fallback and fusion)
    perception = analysis_result.get("perception", {})
    technical = perception.get("technical", {})
    composition = perception.get("composition", {})
    color = perception.get("color", {})
    lighting = perception.get("lighting", {})
    semantics = perception.get("semantics", {})
    emotion = perception.get("emotion", {})
    derived = analysis_result.get("derived", {})
    
    # Collect all text signals for keyword matching (secondary to visual)
    clip_caption = (semantics.get("caption") or "").lower() if semantics.get("available") else ""
    clip_tags = [tag.lower() for tag in (semantics.get("tags", []) or [])]
    clip_inventory = analysis_result.get("_clip_inventory", [])  # May be stored temporarily
    all_text = " ".join([clip_caption] + clip_tags + [str(item).lower() for item in (clip_inventory or [])]).lower()
    
    # Technical measurements
    brightness = technical.get("brightness") if technical.get("available") else None
    contrast = technical.get("contrast") if technical.get("available") else None
    sharpness = technical.get("sharpness") if technical.get("available") else None
    color_mood = color.get("mood") if color.get("available") else None
    lighting_direction = lighting.get("direction") if lighting.get("available") else None
    
    # Calculate signal counts (with safe defaults)
    organic_growth_terms = [
        "ivy", "moss", "vegetation", "growth", "overgrown", "reclaimed", "patina", "weathering", 
        "eroded", "aged", "weathered stone", "aged surface", "eroded facade", "patinated",
        "ivy covered", "moss covered", "green growth", "climbing plants", "plant growth",
        "nature reclaiming", "organic growth", "vegetation on surface", "greenery"
    ]
    organic_signals = sum(1 for term in organic_growth_terms if term in all_text)
    
    historical_terms = ["historical", "ancient", "old", "vintage", "cathedral", "temple", "monument", "heritage"]
    historical_signals = sum(1 for term in historical_terms if term in all_text)
    
    motion_terms = ["motion", "movement", "dynamic", "action", "busy", "chaotic", "energetic"]
    motion_signals = sum(1 for term in motion_terms if term in all_text)
    
    stillness_terms = ["still", "static", "quiet", "calm", "peaceful", "serene", "enduring", "patient"]
    stillness_signals = sum(1 for term in stillness_terms if term in all_text)
    
    # === MATERIAL CONDITION ===
    # PRIORITY: Visual evidence (ground truth) > Text matching (inference)
    material_condition = {}
    
    # Extract salience from organic growth (needed for both material condition and temporal context)
    salience = organic_growth.get("salience", "minimal")  # Default to "minimal" if not present
    
    # Use visual evidence as primary source (proven from pixels)
    if green_coverage > 0.35:
        # Visual evidence: extensive organic growth (proven)
        material_condition["organic_growth"] = "extensive"
        material_condition["organic_growth_coverage"] = green_coverage  # NEW: separate coverage
        material_condition["organic_growth_salience"] = salience  # NEW: structural | incidental | peripheral
        material_condition["surface_state"] = condition_vis if condition_vis != "unknown" else "weathered"
        material_condition["erosion_level"] = "moderate" if surface_roughness > 0.15 else "light"
        material_condition["evidence"] = [
            f"green_coverage={green_coverage:.3f} (visual)",
            f"salience={salience} (visual)",
            f"surface_roughness={surface_roughness:.3f} (visual)",
            "proven from pixels"
        ]
        material_condition["confidence"] = organic_growth.get("confidence", 0.95)
        material_condition["source"] = "visual_analysis"
        
        # Growth types from visual spatial distribution and salience
        green_locations = organic_growth.get("green_locations", "")
        if salience == "structural":
            material_condition["growth_types"] = ["ivy", "structural vegetation"]  # Ivy on structure
        elif salience == "incidental":
            material_condition["growth_types"] = ["foreground vegetation", "greenery"]
        elif salience == "peripheral":
            material_condition["growth_types"] = ["background vegetation", "landscape"]
        elif green_locations == "vertical_surfaces":
            material_condition["growth_types"] = ["ivy"]  # Likely ivy on walls
        elif green_locations in ["foreground", "distributed"]:
            material_condition["growth_types"] = ["vegetation", "greenery"]
    elif green_coverage > 0.2:
        # Visual evidence: moderate organic growth (proven)
        material_condition["organic_growth"] = "moderate"
        material_condition["organic_growth_coverage"] = green_coverage  # NEW: separate coverage
        material_condition["organic_growth_salience"] = salience  # NEW: structural | incidental | peripheral
        material_condition["surface_state"] = condition_vis if condition_vis != "unknown" else "weathered"
        material_condition["erosion_level"] = "light"
        material_condition["evidence"] = [
            f"green_coverage={green_coverage:.3f} (visual)",
            f"salience={salience} (visual)",
            "proven from pixels"
        ]
        material_condition["confidence"] = organic_growth.get("confidence", 0.85)
        material_condition["source"] = "visual_analysis"
    elif green_coverage > 0.1:
        # Visual evidence: minimal organic growth (proven)
        material_condition["organic_growth"] = "minimal"
        material_condition["organic_growth_coverage"] = green_coverage  # NEW: separate coverage
        material_condition["organic_growth_salience"] = salience  # NEW: structural | incidental | peripheral
        material_condition["surface_state"] = condition_vis if condition_vis != "unknown" else "moderate"
        material_condition["evidence"] = [
            f"green_coverage={green_coverage:.3f} (visual)",
            f"salience={salience} (visual)",
            "proven from pixels"
        ]
        material_condition["confidence"] = organic_growth.get("confidence", 0.70)
        material_condition["source"] = "visual_analysis"
    else:
        # No visual evidence of organic growth - check text as fallback
        organic_growth_terms = [
            "ivy", "moss", "vegetation", "growth", "overgrown", "reclaimed", "patina", "weathering", 
            "eroded", "aged", "weathered stone", "aged surface", "eroded facade", "patinated",
            "ivy covered", "moss covered", "green growth", "climbing plants", "plant growth",
            "nature reclaiming", "organic growth", "vegetation on surface", "greenery"
        ]
        organic_signals = sum(1 for term in organic_growth_terms if term in all_text)
        
        if organic_signals >= 2:
            material_condition["organic_growth"] = "extensive"
            material_condition["surface_state"] = "weathered"
            growth_types = [term for term in ["ivy", "moss", "patina", "weathering"] if term in all_text]
            if growth_types:
                material_condition["growth_types"] = growth_types[:3]
            material_condition["erosion_level"] = "moderate" if organic_signals >= 3 else "light"
            material_condition["evidence"] = [f"text_signals={organic_signals}"]
            material_condition["confidence"] = 0.70  # Lower - text inference
            material_condition["source"] = "clip_inventory"
        elif organic_signals >= 1:
            material_condition["organic_growth"] = "moderate"
            material_condition["surface_state"] = "weathered"
            material_condition["erosion_level"] = "light"
            material_condition["evidence"] = [f"text_signals={organic_signals}"]
            material_condition["confidence"] = 0.65
            material_condition["source"] = "clip_inventory"
        else:
            # Use visual condition if available
            if condition_vis != "unknown":
                material_condition["surface_state"] = condition_vis
                material_condition["evidence"] = [f"condition={condition_vis} (visual)"]
                material_condition["confidence"] = material_condition_vis.get("confidence", 0.80)
                material_condition["source"] = "visual_analysis"
            elif sharpness and sharpness > 100:
                material_condition["surface_state"] = "pristine"
                material_condition["organic_growth"] = "none"
                material_condition["evidence"] = [f"sharpness={sharpness} (technical)"]
                material_condition["confidence"] = 0.70
                material_condition["source"] = "technical_analysis"
            elif sharpness and sharpness < 30:
                material_condition["surface_state"] = "degraded"
                material_condition["evidence"] = [f"sharpness={sharpness} (technical)"]
                material_condition["confidence"] = 0.70
                material_condition["source"] = "technical_analysis"
    
    # Age indicators (combine visual + text)
    age_indicators = []
    if surface_roughness > 0.15:
        age_indicators.append("weathered (visual)")
    if condition_vis in ["weathered", "degraded"]:
        age_indicators.append(f"{condition_vis} (visual)")
    
    # Add text-based age indicators as secondary
    age_terms = [
        "old", "aged", "ancient", "historical", "vintage", "weathered", "patina", "eroded", 
        "time", "endurance", "weathered stone", "aged surface", "eroded facade", "patinated",
        "timeworn", "historic building", "ancient structure", "aged architecture"
    ]
    age_signals_text = [term for term in age_terms if term in all_text]
    if age_signals_text:
        age_indicators.extend(age_signals_text[:3])
    
    if age_indicators:
        material_condition["age_indicators"] = age_indicators[:5]
    
    # Maintenance state (combine visual + text)
    if "abandoned" in all_text or "neglected" in all_text:
        material_condition["maintenance_state"] = "neglected"
    elif "well maintained" in all_text or "pristine" in all_text:
        material_condition["maintenance_state"] = "well_maintained"
    elif green_coverage > 0.1 or organic_signals >= 1:
        material_condition["maintenance_state"] = "in_use"
    elif condition_vis == "pristine" and sharpness and sharpness > 100:
        material_condition["maintenance_state"] = "well_maintained"
    
    if material_condition:
        understanding["material_condition"] = material_condition
    
    # === TEMPORAL CONTEXT ===
    temporal_context = {}
    
    # Extract visual evidence for temporal direction
    # IMPORTANT: Extract salience early to avoid NameError
    condition_vis = material_condition_vis.get("condition", "unknown")
    green_coverage = organic_growth.get("green_coverage", 0.0)
    salience = organic_growth.get("salience", "minimal")  # Default to "minimal" if not present
    surface_roughness = material_condition_vis.get("surface_roughness", 0.0)
    edge_degradation = material_condition_vis.get("edge_degradation", 0.0)
    
    # Time scale inference
    historical_terms = ["historical", "ancient", "old", "vintage", "cathedral", "temple", "monument", "heritage"]
    historical_signals = sum(1 for term in historical_terms if term in all_text)
    
    if historical_signals >= 2:
        temporal_context["time_scale"] = "historical"
    elif "contemporary" in all_text or "modern" in all_text or "new" in all_text:
        temporal_context["time_scale"] = "contemporary"
    elif "moment" in all_text or "decisive" in all_text:
        temporal_context["time_scale"] = "momentary"
    else:
        temporal_context["time_scale"] = "timeless"
    
    # Pace inference
    # Note: motion_signals and stillness_signals already calculated above with safe defaults
    
    if motion_signals >= 2:
        temporal_context["pace"] = "fast"
        temporal_context["moment_type"] = "decisive"
    elif stillness_signals >= 2:
        temporal_context["pace"] = "slow"
        temporal_context["moment_type"] = "still"
    elif organic_signals >= 1:  # Organic growth suggests slow time
        temporal_context["pace"] = "slow"
        temporal_context["moment_type"] = "eternal"
    else:
        temporal_context["pace"] = "static"
        temporal_context["moment_type"] = "in_between"
    
    # Temporal direction (NEW): accreting | decaying | static
    # This distinguishes growth from decay, not just pace
    if salience == "structural" and green_coverage > 0.2:
        # Ivy on structure = accreting (nature growing on structure)
        temporal_context["temporal_direction"] = "accreting"
    elif condition_vis in ["weathered", "degraded"] and surface_roughness > 0.15:
        # Weathered/degraded = decaying (structure breaking down)
        temporal_context["temporal_direction"] = "decaying"
    elif condition_vis == "pristine" and green_coverage < 0.1:
        # Pristine + no organic = static (no change)
        temporal_context["temporal_direction"] = "static"
    elif green_coverage > 0.2:
        # Organic growth present = accreting
        temporal_context["temporal_direction"] = "accreting"
    elif "decay" in all_text or "falling" in all_text or "crumbling" in all_text:
        temporal_context["temporal_direction"] = "decaying"
    elif "growth" in all_text or "new" in all_text or "emerging" in all_text:
        temporal_context["temporal_direction"] = "accreting"
    else:
        temporal_context["temporal_direction"] = "static"
    
    # Endurance
    if organic_signals >= 1 and historical_signals >= 1:
        temporal_context["endurance"] = "enduring"
        temporal_context["change_indicators"] = ["vegetation growth", "weathering"]
    elif historical_signals >= 1:
        temporal_context["endurance"] = "enduring"
    elif "decay" in all_text or "falling" in all_text:
        temporal_context["endurance"] = "decaying"
    elif "growth" in all_text or "new" in all_text:
        temporal_context["endurance"] = "growing"
    else:
        temporal_context["endurance"] = "transient"
    
    if temporal_context:
        understanding["temporal_context"] = temporal_context
    
    # === ORGANIC INTERACTION ===
    # PRIORITY: Visual evidence (ground truth) > Text matching (inference)
    organic_interaction = {}
    
    # Use visual evidence as primary source (proven from pixels)
    if relationship_vis != "none" and integration_level_vis != "none":
        # Visual evidence: proven relationship from pixels
        organic_interaction["relationship"] = relationship_vis
        organic_interaction["integration_level"] = integration_level_vis
        organic_interaction["overlap_ratio"] = organic_integration_vis.get("overlap_ratio", 0.0)
        organic_interaction["evidence"] = [
            f"overlap_ratio={organic_interaction['overlap_ratio']:.3f} (visual)",
            f"relationship={relationship_vis} (visual)",
            "proven from pixels"
        ]
        organic_interaction["confidence"] = organic_integration_vis.get("confidence", 0.90)
        organic_interaction["source"] = "visual_analysis"
        
        # Dominance inference from visual evidence
        if relationship_vis == "reclamation" and green_coverage > 0.4:
            organic_interaction["dominance"] = "nature"  # Nature reclaiming structure
        elif relationship_vis == "reclamation":
            organic_interaction["dominance"] = "balanced"  # Balanced reclamation
        elif relationship_vis == "integration":
            organic_interaction["dominance"] = "balanced"
        else:
            organic_interaction["dominance"] = "structure"
        
        # Specific indicators from visual spatial distribution
        green_locations = organic_growth.get("green_locations", "")
        if green_locations == "vertical_surfaces":
            organic_interaction["specific_indicators"] = ["ivy on structure"]
        elif green_locations in ["foreground", "distributed"]:
            organic_interaction["specific_indicators"] = ["vegetation integrated"]
    elif green_coverage > 0.2:
        # Visual evidence: organic present but unclear relationship
        organic_interaction["relationship"] = "coexistence"
        organic_interaction["integration_level"] = "moderate"
        organic_interaction["evidence"] = [
            f"green_coverage={green_coverage:.3f} (visual)",
            "organic present but relationship unclear"
        ]
        organic_interaction["confidence"] = 0.75
        organic_interaction["source"] = "visual_analysis"
        organic_interaction["dominance"] = "balanced"
    else:
        # Fallback to text matching if no visual evidence
        organic_growth_terms = [
            "ivy", "moss", "vegetation", "growth", "overgrown", "reclaimed", "patina", "weathering"
        ]
        organic_signals = sum(1 for term in organic_growth_terms if term in all_text)
        
        if organic_signals >= 2:
            organic_interaction["relationship"] = "reclamation"
            organic_interaction["integration_level"] = "high"
            organic_interaction["dominance"] = "balanced"
            organic_interaction["specific_indicators"] = [term for term in ["ivy", "moss", "vegetation"] if term in all_text][:3]
            organic_interaction["evidence"] = [f"text_signals={organic_signals}"]
            organic_interaction["confidence"] = 0.70  # Lower - text inference
            organic_interaction["source"] = "clip_inventory"
        elif organic_signals >= 1:
            organic_interaction["relationship"] = "coexistence"
            organic_interaction["integration_level"] = "moderate"
            organic_interaction["dominance"] = "balanced"
            organic_interaction["evidence"] = [f"text_signals={organic_signals}"]
            organic_interaction["confidence"] = 0.65
            organic_interaction["source"] = "clip_inventory"
        elif "nature" in all_text and ("building" in all_text or "structure" in all_text):
            organic_interaction["relationship"] = "harmony"
            organic_interaction["integration_level"] = "moderate"
            organic_interaction["dominance"] = "balanced"
            organic_interaction["evidence"] = ["text_inference"]
            organic_interaction["confidence"] = 0.60
            organic_interaction["source"] = "clip_inventory"
    
    if organic_interaction.get("relationship") != "none":
        understanding["organic_interaction"] = organic_interaction
    
    # === EMOTIONAL SUBSTRATE ===
    # Use constrained emotional synthesis with visual evidence as primary source
    technical_data = {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "color_mood": color_mood
    }
    
    clip_data = {
        "caption": clip_caption,
        "tags": clip_tags,
        "inventory": clip_inventory
    }
    
    scene_context = {
        "temporal_context": understanding.get("temporal_context", {}),
        "organic_interaction": understanding.get("organic_interaction", {})
    }
    
    # Synthesize emotional substrate with full explainability
    emotional_substrate = synthesize_emotional_substrate_constrained(
        visual_evidence, technical_data, clip_data, scene_context
    )
    
    if emotional_substrate:
        understanding["emotional_substrate"] = emotional_substrate
    
    # === NEGATIVE EVIDENCE (NEW) ===
    # Track what is NOT present to prevent incorrect interpretations
    # "no people" ≠ alienation, "no people" = stillness / endurance / pause
    negative_evidence = {}
    
    # Check for human presence (from emotion detection or YOLO)
    human_presence_detected = emotion.get("subject_type") == "human subject" if emotion.get("available") else False
    yolo_objects = analysis_result.get("perception", {}).get("objects", {}).get("objects", [])
    yolo_has_people = any(obj.lower() in ["person", "people", "human", "man", "woman", "child"] for obj in yolo_objects)
    
    if not human_presence_detected and not yolo_has_people:
        negative_evidence["no_human_presence"] = True
        negative_evidence["human_presence_evidence"] = "No humans detected in emotion analysis or object detection"
    
    # Check for motion (from temporal context or text)
    motion_terms = ["motion", "movement", "dynamic", "action", "busy", "chaotic", "energetic"]
    motion_signals_count = sum(1 for term in motion_terms if term in all_text)
    motion_detected = motion_signals_count >= 2
    if not motion_detected:
        negative_evidence["no_motion_detected"] = True
        negative_evidence["motion_evidence"] = "No motion signals detected in temporal analysis"
    
    # Check for artificial surface uniformity (pristine, clean surfaces)
    # This distinguishes "no organic" from "artificial uniformity"
    if condition_vis == "pristine" and green_coverage < 0.05 and surface_roughness < 0.05:
        negative_evidence["no_artificial_surface_uniformity"] = True
        negative_evidence["uniformity_evidence"] = "Pristine condition with minimal organic growth suggests artificial uniformity"
    elif condition_vis != "pristine" or green_coverage > 0.05:
        negative_evidence["no_artificial_surface_uniformity"] = False
    
    if negative_evidence:
        understanding["negative_evidence"] = negative_evidence
    
    # === CONTEXTUAL RELATIONSHIPS ===
    contextual_relationships = {}
    
    # Subject vs environment
    # Use visual evidence if available, otherwise text signals
    organic_growth_terms = [
        "ivy", "moss", "vegetation", "growth", "overgrown", "reclaimed", "patina", "weathering"
    ]
    organic_signals_count = sum(1 for term in organic_growth_terms if term in all_text)
    has_organic_visual = green_coverage > 0.1
    has_organic_text = organic_signals_count >= 1
    
    if has_organic_visual or has_organic_text:
        contextual_relationships["subject_vs_environment"] = "in_dialogue"
    elif "isolated" in all_text:
        contextual_relationships["subject_vs_environment"] = "isolated"
    else:
        contextual_relationships["subject_vs_environment"] = "integrated"
    
    # Time vs subject
    if temporal_context.get("endurance") == "enduring":
        contextual_relationships["time_vs_subject"] = "enduring"
    elif temporal_context.get("endurance") == "decaying":
        contextual_relationships["time_vs_subject"] = "decaying"
    else:
        contextual_relationships["time_vs_subject"] = "fleeting"
    
    # Human vs space
    human_presence = emotion.get("subject_type") == "human subject" if emotion.get("available") else False
    if not human_presence and organic_signals >= 1:
        contextual_relationships["human_vs_space"] = "intentional_stillness"
    elif not human_presence:
        contextual_relationships["human_vs_space"] = "alienation"
    else:
        contextual_relationships["human_vs_space"] = "presence" "active_occupation"
    
    # Organic vs inorganic
    if organic_interaction.get("relationship") == "reclamation":
        contextual_relationships["organic_vs_inorganic"] = "reclamation"
    elif organic_interaction.get("relationship") == "harmony":
        contextual_relationships["organic_vs_inorganic"] = "harmony"
    elif organic_interaction.get("relationship") == "coexistence":
        contextual_relationships["organic_vs_inorganic"] = "coexistence"
    else:
        contextual_relationships["organic_vs_inorganic"] = "none"
    
    if contextual_relationships:
        understanding["contextual_relationships"] = contextual_relationships
    
    return understanding
