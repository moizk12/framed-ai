"""Deterministic visual grounding (organic growth, material, integration, Places365-CLIP fallback)."""
import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# IC_0015-A: domain guard thresholds (reclamation leak fix)
MIN_GREEN_FOR_RECLAMATION = 0.05
MIN_GREEN_FOR_ORGANIC_SALIENCE = 0.03
EDGE_DEGRADED_THRESHOLD = 0.60
LOW_TEXTURE_THRESHOLD = 0.02

def detect_organic_growth(image_path):
    """
    Detect organic growth (vegetation, ivy, moss, plants) using HSV color thresholds.
    Universal: works for any image type (architecture, nature, street, portraits).
    
    Deterministic, provable, explainable.
    Returns ground truth about green pixel coverage and spatial distribution.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with:
            - green_coverage: float (0.0-1.0) - percentage of image that is green
            - green_locations: str - spatial distribution ("vertical_surfaces", "foreground", "background", "distributed")
            - green_clusters: int - number of distinct green regions
            - evidence: list of strings - explainable evidence
            - confidence: float (0.0-1.0) - confidence in detection
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"green_coverage": 0.0, "green_locations": "none", "green_clusters": 0, 
                   "evidence": ["image_load_failed"], "confidence": 0.0}
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        total_pixels = h * w
        
        # Edge case: Check if image is too dark or too bright (affects color detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        is_dark = mean_brightness < 30
        is_bright = mean_brightness > 225
        
        # Adjust HSV thresholds for edge cases
        if is_dark:
            # Dark images: lower saturation threshold, allow darker values
            lower_green = np.array([40, 30, 20])
            upper_green = np.array([80, 255, 255])
        elif is_bright:
            # Bright images: higher saturation threshold to avoid false positives
            lower_green = np.array([40, 60, 60])
            upper_green = np.array([80, 255, 255])
        else:
            # Normal images: standard thresholds
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
        
        # HSV green range (covers ivy, moss, grass, leaves, vegetation)
        # Hue: 40-80 (green range in HSV)
        # Saturation: 50-255 (avoid grey/desaturated) - adjusted for edge cases
        # Value: 50-255 (avoid too dark) - adjusted for edge cases
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate coverage
        green_pixels = np.sum(green_mask > 0)
        green_coverage = green_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Spatial distribution analysis
        # Divide image into regions and check where green is concentrated
        h_third = h // 3
        w_third = w // 3
        
        # Check vertical surfaces (left/right edges) - common for ivy on buildings
        left_region = green_mask[:, :w_third]
        right_region = green_mask[:, -w_third:]
        vertical_coverage = (np.sum(left_region > 0) + np.sum(right_region > 0)) / (2 * h * w_third) if w_third > 0 else 0
        
        # Check foreground (bottom third)
        foreground_region = green_mask[-h_third:, :]
        foreground_coverage = np.sum(foreground_region > 0) / (h_third * w) if h_third > 0 else 0
        
        # Check background (top third)
        background_region = green_mask[:h_third, :]
        background_coverage = np.sum(background_region > 0) / (h_third * w) if h_third > 0 else 0
        
        # Determine spatial distribution
        if vertical_coverage > 0.3:
            green_locations = "vertical_surfaces"  # Likely ivy on walls
        elif foreground_coverage > 0.4:
            green_locations = "foreground"  # Grass, plants in front
        elif background_coverage > 0.4:
            green_locations = "background"  # Trees, landscape behind
        elif green_coverage > 0.1:
            green_locations = "distributed"  # Scattered throughout
        else:
            green_locations = "minimal"
        
        # Count distinct green clusters (connected components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
        # Filter out tiny noise (clusters < 0.1% of image)
        min_cluster_size = total_pixels * 0.001
        green_clusters = sum(1 for stat in stats[1:] if stat[cv2.CC_STAT_AREA] >= min_cluster_size)
        
        # Evidence list (explainable)
        evidence = [
            f"green_coverage={green_coverage:.3f}",
            f"green_clusters={green_clusters}",
            f"spatial_distribution={green_locations}"
        ]
        
        # Confidence: high if significant coverage, lower if minimal
        # Adjust for edge cases (dark/bright images may have lower confidence)
        base_confidence = 0.95 if green_coverage > 0.3 else (
            0.85 if green_coverage > 0.1 else (
            0.70 if green_coverage > 0.05 else 0.50
            )
        )
        
        # Reduce confidence for edge cases
        if is_dark:
            confidence = base_confidence * 0.85  # Dark images: harder to detect colors accurately
        elif is_bright:
            confidence = base_confidence * 0.90  # Bright images: slight reduction
        else:
            confidence = base_confidence
        
        # Determine salience (structural vs incidental vs peripheral)
        # This distinguishes ivy on facade from grass in foreground
        if green_locations == "vertical_surfaces" and green_coverage > 0.2:
            salience = "structural"  # Organic growth on structure (e.g., ivy on walls)
        elif green_locations == "foreground" and green_coverage > 0.3:
            salience = "incidental"  # Foreground vegetation (e.g., grass, plants)
        elif green_locations == "background" and green_coverage > 0.2:
            salience = "peripheral"  # Background vegetation (e.g., trees, landscape)
        elif green_coverage > 0.1:
            salience = "distributed"  # Scattered throughout
        else:
            salience = "minimal"  # Minimal or none
        
        return {
            "green_coverage": float(green_coverage),
            "green_locations": green_locations,
            "green_clusters": int(green_clusters),
            "salience": salience,  # NEW: structural | incidental | peripheral | distributed | minimal
            "evidence": evidence,
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.warning(f"Organic growth detection failed: {e}")
        return {"green_coverage": 0.0, "green_locations": "none", "green_clusters": 0,
               "salience": "minimal", "evidence": [f"error: {str(e)}"], "confidence": 0.0}


def detect_material_condition(image_path):
    """
    Detect material condition (weathered/smooth/pristine/degraded) using texture variance.
    Universal: works for any surface type (stone, concrete, wood, fabric, skin).
    
    Deterministic, provable, explainable.
    Returns ground truth about surface roughness and edge quality.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with:
            - surface_roughness: float - texture variance (high = rough/weathered, low = smooth)
            - edge_degradation: float - edge quality (high = degraded/aged, low = sharp/new)
            - condition: str - "weathered" | "pristine" | "moderate" | "degraded"
            - evidence: list of strings - explainable evidence
            - confidence: float (0.0-1.0) - confidence in detection
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"surface_roughness": 0.0, "edge_degradation": 0.0, "condition": "unknown",
                   "evidence": ["image_load_failed"], "confidence": 0.0}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # === TEXTURE VARIANCE (Roughness Detection) ===
        # Use local variance to detect surface roughness
        # High variance = rough/weathered, low variance = smooth/pristine
        
        # Calculate local variance using a sliding window
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Global texture variance (normalized 0-1)
        # Ensure we get a scalar value, not an array
        mean_variance = float(np.mean(local_variance))
        max_possible_variance = 255.0 ** 2  # Maximum variance for 8-bit image
        normalized_variance = min(mean_variance / max_possible_variance, 1.0)
        surface_roughness = float(normalized_variance)
        
        # === EDGE DEGRADATION (Age Indicators) ===
        # Sharp edges = new/pristine, degraded edges = aged/weathered
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge sharpness (how well-defined are the edges?)
        # Use gradient magnitude as proxy for sharpness
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize edge sharpness (0-1)
        max_gradient = np.max(gradient_magnitude) if np.max(gradient_magnitude) > 0 else 1.0
        mean_gradient = np.mean(gradient_magnitude[edges > 0]) if np.sum(edges > 0) > 0 else 0.0
        edge_sharpness = float(mean_gradient / max_gradient) if max_gradient > 0 else 0.0
        
        # Edge degradation is inverse of sharpness
        edge_degradation = 1.0 - edge_sharpness
        
        # === CONDITION INFERENCE ===
        # Thresholds are tuned for general images (architecture, nature, portraits, etc.)
        if surface_roughness > 0.15 and edge_degradation > 0.4:
            condition = "weathered"  # Rough texture + degraded edges
        elif surface_roughness < 0.05 and edge_degradation < 0.2:
            condition = "pristine"  # Smooth texture + sharp edges
        elif edge_degradation > EDGE_DEGRADED_THRESHOLD and surface_roughness < LOW_TEXTURE_THRESHOLD:
            condition = "neutral"  # IC_0015-A: edge density alone is not weathering
        elif surface_roughness > 0.2 or edge_degradation > 0.6:
            condition = "degraded"  # Very rough or very degraded
        else:
            condition = "moderate"  # In between
        
        # Evidence list (explainable)
        evidence = [
            f"texture_variance={surface_roughness:.3f}",
            f"edge_degradation={edge_degradation:.3f}",
            f"condition={condition}"
        ]
        
        # Edge case: Check if image is too dark or too bright (affects texture analysis)
        # Ensure we get a scalar value, not an array
        mean_brightness = float(np.mean(gray))
        is_dark = mean_brightness < 30
        is_bright = mean_brightness > 225
        
        # Confidence: higher when signals are clear
        base_confidence = 0.90 if ((surface_roughness > 0.15 and edge_degradation > 0.4) or 
                                   (surface_roughness < 0.05 and edge_degradation < 0.2)) else (
            0.80 if (surface_roughness > 0.1 or edge_degradation > 0.3) else 0.65
        )
        
        # Adjust confidence for edge cases
        if is_dark:
            confidence = base_confidence * 0.80  # Dark images: texture analysis less reliable
        elif is_bright:
            confidence = base_confidence * 0.85  # Bright images: slight reduction
        else:
            confidence = base_confidence
        
        # === COLOR UNIFORMITY (Paint vs Organic) ===
        # High uniformity = likely paint/manufactured, low uniformity = likely organic/natural
        # Analyze color variance in the green regions (if any)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_ch, sat_ch, val_ch = cv2.split(hsv)
        
        # Calculate color uniformity (inverse of variance)
        # For green regions specifically (if green coverage > 0.1)
        green_mask_temp = cv2.inRange(hsv, np.array([40, 50, 50], dtype=np.uint8),
                                      np.array([80, 255, 255], dtype=np.uint8))
        green_pixels = np.sum(green_mask_temp > 0)
        total_pixels = int(height * width)
        
        if green_pixels > total_pixels * 0.1:  # If significant green coverage
            # Calculate hue variance in green regions
            green_hue = hue_ch[green_mask_temp > 0]
            if len(green_hue) > 0:
                hue_variance = np.var(green_hue.astype(np.float32))
                max_hue_variance = 180.0 ** 2  # Max variance for hue (0-180)
                normalized_hue_variance = min(hue_variance / max_hue_variance, 1.0)
                color_uniformity = 1.0 - normalized_hue_variance  # Uniformity = inverse of variance
            else:
                color_uniformity = 0.5  # Default if no green pixels
        else:
            # For non-green regions, calculate overall color uniformity
            hue_variance = np.var(hue_ch.astype(np.float32))
            max_hue_variance = 180.0 ** 2
            normalized_hue_variance = min(hue_variance / max_hue_variance, 1.0)
            color_uniformity = 1.0 - normalized_hue_variance
        
        # Also calculate texture variance (already computed above)
        texture_variance = surface_roughness
        
        return {
            "surface_roughness": surface_roughness,
            "edge_degradation": edge_degradation,
            "texture_variance": texture_variance,  # Add explicit texture_variance field
            "color_uniformity": float(color_uniformity),  # NEW: for paint vs organic detection
            "condition": condition,
            "evidence": evidence + [f"color_uniformity={color_uniformity:.3f}"],
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.warning(f"Material condition detection failed: {e}")
        return {"surface_roughness": 0.0, "edge_degradation": 0.0, "condition": "unknown",
               "evidence": [f"error: {str(e)}"], "confidence": 0.0}


def detect_organic_integration(image_path, green_mask=None, structure_edges=None):
    """
    Detect if organic elements are integrated with structure using morphological operations.
    Universal: works for any image type (architecture, nature, street, portraits).
    
    Determines relationship: reclamation, integration, coexistence, or none.
    
    Args:
        image_path: Path to image file
        green_mask: Optional pre-computed green mask (if None, computed here)
        structure_edges: Optional pre-computed structure edges (if None, computed here)
    
    Returns:
        Dict with:
            - overlap_ratio: float (0.0-1.0) - how much green overlaps structure
            - relationship: str - "reclamation" | "integration" | "coexistence" | "none"
            - integration_level: str - "high" | "moderate" | "low" | "none"
            - evidence: list of strings - explainable evidence
            - confidence: float (0.0-1.0) - confidence in detection
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"overlap_ratio": 0.0, "relationship": "none", "integration_level": "none",
                   "evidence": ["image_load_failed"], "confidence": 0.0}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Compute green mask if not provided
        if green_mask is None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Compute structure edges if not provided
        # Structure = strong edges (buildings, objects, defined forms)
        if structure_edges is None:
            # Use Canny with higher thresholds to detect strong structural edges
            structure_edges = cv2.Canny(gray, 100, 200)
            # Dilate edges slightly to capture nearby pixels
            kernel = np.ones((3, 3), np.uint8)
            structure_edges = cv2.dilate(structure_edges, kernel, iterations=1)
        
        # Calculate overlap using morphological operations
        # Overlap = green pixels that are near or on structure edges
        overlap = cv2.bitwise_and(green_mask, structure_edges)
        
        green_pixels = np.sum(green_mask > 0)
        overlap_pixels = np.sum(overlap > 0)
        total_pixels = h * w
        green_coverage = green_pixels / total_pixels if total_pixels > 0 else 0.0

        if green_pixels > 0:
            overlap_ratio = overlap_pixels / green_pixels
        else:
            overlap_ratio = 0.0

        # IC_0015-A: insufficient green — no reclamation/integration inference
        if green_coverage < MIN_GREEN_FOR_RECLAMATION:
            return {
                "overlap_ratio": float(overlap_ratio),
                "proximity_ratio": 0.0,
                "relationship": "none",
                "integration_level": "none",
                "green_coverage": float(green_coverage),
                "evidence": [
                    f"green_coverage={green_coverage:.3f}",
                    "domain_guard:integration_suppressed",
                ],
                "confidence": 0.3,
            }

        # Also check proximity (green near structure, even if not directly overlapping)
        # Dilate structure edges to create a "nearby" zone
        dilated_edges = cv2.dilate(structure_edges, np.ones((15, 15), np.uint8), iterations=1)
        proximity_mask = cv2.bitwise_and(green_mask, dilated_edges)
        proximity_pixels = np.sum(proximity_mask > 0)
        proximity_ratio = proximity_pixels / green_pixels if green_pixels > 0 else 0.0

        # Relationship inference
        if overlap_ratio > 0.6 or proximity_ratio > 0.8:
            relationship = "reclamation"  # Organic on/around structure
            integration_level = "high"
        elif overlap_ratio > 0.3 or proximity_ratio > 0.5:
            relationship = "integration"  # Partial integration
            integration_level = "moderate"
        elif green_pixels > 0 and (overlap_ratio > 0.1 or proximity_ratio > 0.2):
            relationship = "coexistence"  # Some interaction
            integration_level = "low"
        else:
            relationship = "none"  # No interaction
            integration_level = "none"
        
        # Evidence list (explainable)
        evidence = [
            f"overlap_ratio={overlap_ratio:.3f}",
            f"proximity_ratio={proximity_ratio:.3f}",
            f"relationship={relationship}"
        ]
        
        # Confidence: higher when overlap is clear
        if overlap_ratio > 0.6:
            confidence = 0.95  # High - clear overlap
        elif overlap_ratio > 0.3:
            confidence = 0.85  # Medium-high - moderate overlap
        elif green_pixels > 0:
            confidence = 0.70  # Medium - some green but unclear relationship
        else:
            confidence = 0.50  # Low - no green detected
        
        return {
            "overlap_ratio": float(overlap_ratio),
            "proximity_ratio": float(proximity_ratio),
            "relationship": relationship,
            "integration_level": integration_level,
            "evidence": evidence,
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.warning(f"Organic integration detection failed: {e}")
        return {"overlap_ratio": 0.0, "relationship": "none", "integration_level": "none",
               "evidence": [f"error: {str(e)}"], "confidence": 0.0}


def extract_places365_signals(image_path):
    """
    Extract scene and attribute signals using ResNet50-Places365.
    
    Position in pipeline: After Visual Evidence, Before Interpretive Reasoner
    Purpose: Provides scene category probabilities, indoor/outdoor, man-made vs natural,
             and high-level attributes (religious, historical, urban, etc.)
    
    What it provides:
    - scene_category: Top scene category (e.g., "cathedral", "forest", "street")
    - scene_probabilities: Top 5 scene categories with probabilities
    - indoor_outdoor: "indoor" | "outdoor" | "unknown"
    - man_made_natural: "man_made" | "natural" | "mixed" | "unknown"
    - attributes: High-level attributes (religious, historical, urban, etc.)
    
    What it does NOT provide:
    - Emotional meaning
    - Critique
    - Reasoning
    - Decisions
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with Places365 signals
    """
    try:
        import torch
        from framed.analysis.vision import get_clip_model

        clip_model, clip_processor, device = get_clip_model()
        image = Image.open(image_path).convert("RGB")
        
        # Scene category candidates (Places365-like categories)
        scene_candidates = [
            "cathedral", "church", "mosque", "temple", "religious building",
            "forest", "woodland", "nature", "landscape",
            "street", "urban", "city", "road", "sidewalk",
            "indoor", "interior", "room", "hall",
            "outdoor", "exterior", "outdoors",
            "building", "architecture", "structure",
            "man-made", "artificial", "constructed",
            "natural", "wild", "organic"
        ]
        
        inputs = clip_processor(text=scene_candidates, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Get top 5 scene categories
        top_indices = probs.argsort()[-5:][::-1]
        scene_probabilities = [
            {"category": scene_candidates[i], "probability": float(probs[i])}
            for i in top_indices
        ]
        
        top_category = scene_candidates[top_indices[0]]
        top_prob = float(probs[top_indices[0]])
        
        # Infer indoor/outdoor
        indoor_probs = sum(probs[i] for i, cat in enumerate(scene_candidates) if cat in ["indoor", "interior", "room", "hall"])
        outdoor_probs = sum(probs[i] for i, cat in enumerate(scene_candidates) if cat in ["outdoor", "exterior", "outdoors", "forest", "landscape"])
        
        if indoor_probs > outdoor_probs and indoor_probs > 0.3:
            indoor_outdoor = "indoor"
        elif outdoor_probs > indoor_probs and outdoor_probs > 0.3:
            indoor_outdoor = "outdoor"
        else:
            indoor_outdoor = "unknown"
        
        # Infer man-made vs natural
        man_made_probs = sum(probs[i] for i, cat in enumerate(scene_candidates) if cat in ["man-made", "artificial", "constructed", "building", "architecture", "structure", "street", "urban", "city"])
        natural_probs = sum(probs[i] for i, cat in enumerate(scene_candidates) if cat in ["natural", "wild", "organic", "forest", "woodland", "nature"])
        
        if man_made_probs > natural_probs and man_made_probs > 0.3:
            man_made_natural = "man_made"
        elif natural_probs > man_made_probs and natural_probs > 0.3:
            man_made_natural = "natural"
        elif man_made_probs > 0.2 and natural_probs > 0.2:
            man_made_natural = "mixed"
        else:
            man_made_natural = "unknown"
        
        # Extract attributes
        attributes = []
        if top_category in ["cathedral", "church", "mosque", "temple", "religious building"]:
            attributes.append("religious")
        if top_category in ["cathedral", "church", "mosque", "temple", "building", "architecture"]:
            attributes.append("historical")
        if top_category in ["street", "urban", "city", "road"]:
            attributes.append("urban")
        
        return {
            "scene_category": top_category,
            "scene_probabilities": scene_probabilities[:5],
            "indoor_outdoor": indoor_outdoor,
            "man_made_natural": man_made_natural,
            "attributes": attributes,
            "confidence": top_prob,
            "source": "clip_fallback"  # CLIP-based approximation until Places365 weights loaded
        }
    
    except Exception as e:
        logger.warning(f"Places365 signal extraction failed: {e}")
        return {}


def apply_domain_guard(visual_evidence):
    """
    IC_0015-A: Suppress organic-growth / reclamation signals when domain cues are absent.
    Prevents edge-density and sparse-green noise from becoming default interpretation.
    """
    og = dict(visual_evidence.get("organic_growth") or {})
    mc = dict(visual_evidence.get("material_condition") or {})
    oi = dict(visual_evidence.get("organic_integration") or {})

    gc = float(og.get("green_coverage", 0.0) or 0.0)
    rel = oi.get("relationship", "none")

    if gc < MIN_GREEN_FOR_ORGANIC_SALIENCE:
        og["applicable"] = False
        og["salience"] = "minimal"
        og["suppressed_reason"] = "green_coverage_below_threshold"
    else:
        og["applicable"] = True

    if gc < MIN_GREEN_FOR_RECLAMATION and rel in ("reclamation", "integration"):
        oi["relationship"] = "none"
        oi["integration_level"] = "none"
        oi["confidence"] = min(float(oi.get("confidence", 0.5)), 0.3)
        evidence = list(oi.get("evidence") or [])
        if "domain_guard:integration_suppressed" not in evidence:
            evidence.append("domain_guard:integration_suppressed")
        oi["evidence"] = evidence
        oi["suppressed_reason"] = "green_coverage_below_threshold"

    cond = mc.get("condition", "unknown")
    edge_deg = float(mc.get("edge_degradation", 0.0) or 0.0)
    rough = float(mc.get("surface_roughness", 0.0) or 0.0)

    if gc < MIN_GREEN_FOR_ORGANIC_SALIENCE:
        if cond == "degraded" and edge_deg > EDGE_DEGRADED_THRESHOLD and rough < LOW_TEXTURE_THRESHOLD:
            mc["condition"] = "not_applicable"
            mc["confidence"] = min(float(mc.get("confidence", 0.8)), 0.4)
            mc["suppressed_reason"] = "edge_density_not_weathering"
        elif cond in ("degraded", "weathered") and gc < 0.01:
            mc["condition"] = "neutral"
            mc["suppressed_reason"] = "no_organic_context_for_material"

    visual_evidence["organic_growth"] = og
    visual_evidence["material_condition"] = mc
    visual_evidence["organic_integration"] = oi
    visual_evidence["domain_guard_applied"] = True
    return visual_evidence


def validate_visual_evidence(visual_evidence, min_confidence=0.60):
    """
    Validate visual evidence quality and detect potential issues.
    Returns validation results and warnings.
    
    Args:
        visual_evidence: Dict from extract_visual_features()
        min_confidence: Minimum confidence threshold (default 0.60)
    
    Returns:
        Dict with:
            - is_valid: bool - whether evidence meets quality thresholds
            - warnings: list - warnings about evidence quality
            - issues: list - critical issues that should prevent use
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "issues": []
    }
    
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition = visual_evidence.get("material_condition", {})
    organic_integration = visual_evidence.get("organic_integration", {})
    overall_confidence = visual_evidence.get("overall_confidence", 0.0)
    
    # Check overall confidence
    if overall_confidence < min_confidence:
        validation["warnings"].append(f"Overall confidence ({overall_confidence:.2f}) below threshold ({min_confidence})")
        if overall_confidence < 0.40:
            validation["issues"].append("Very low overall confidence - visual evidence may be unreliable")
            validation["is_valid"] = False
    
    # Check individual component confidences
    og_confidence = organic_growth.get("confidence", 0.0)
    mc_confidence = material_condition.get("confidence", 0.0)
    oi_confidence = organic_integration.get("confidence", 0.0)
    
    if og_confidence < 0.50 and organic_growth.get("green_coverage", 0.0) > 0.1:
        validation["warnings"].append("Organic growth detected but confidence is low - may be false positive")
    
    if mc_confidence < 0.50:
        validation["warnings"].append("Material condition confidence is low - texture analysis may be unreliable")
    
    if oi_confidence < 0.50 and organic_integration.get("relationship") != "none":
        validation["warnings"].append("Organic integration detected but confidence is low")
    
    # Check for contradictions within visual evidence
    green_coverage = organic_growth.get("green_coverage", 0.0)
    condition = material_condition.get("condition", "unknown")
    
    if green_coverage > 0.3 and condition == "pristine":
        validation["warnings"].append("Contradiction: High organic growth but pristine condition - may indicate error")
    
    if green_coverage < MIN_GREEN_FOR_RECLAMATION and organic_integration.get("relationship") == "reclamation":
        validation["issues"].append(
            "Contradiction: Reclamation relationship but minimal green coverage (domain guard should have suppressed)"
        )
        validation["is_valid"] = False

    return validation


def detect_contradictions(visual_evidence, text_inference, min_confidence_diff=0.15):
    """
    Detect contradictions between visual evidence (ground truth) and text inference.
    Visual evidence with high confidence should override text inference.
    
    Args:
        visual_evidence: Dict from extract_visual_features()
        text_inference: Dict with text-based inferences (e.g., from CLIP)
        min_confidence_diff: Minimum confidence difference to trigger override (default 0.15)
    
    Returns:
        Dict with:
            - contradictions: list of detected contradictions
            - overrides: list of text inferences that should be overridden by visual evidence
            - recommendations: list of recommendations for handling contradictions
    """
    contradictions = {
        "contradictions": [],
        "overrides": [],
        "recommendations": []
    }
    
    # Extract visual evidence
    organic_growth = visual_evidence.get("organic_growth", {})
    material_condition = visual_evidence.get("material_condition", {})
    organic_integration = visual_evidence.get("organic_integration", {})
    
    green_coverage = organic_growth.get("green_coverage", 0.0)
    og_confidence = organic_growth.get("confidence", 0.0)
    condition_vis = material_condition.get("condition", "unknown")
    mc_confidence = material_condition.get("confidence", 0.0)
    relationship_vis = organic_integration.get("relationship", "none")
    oi_confidence = organic_integration.get("confidence", 0.0)
    
    # Check text inference (if available)
    text_has_organic = text_inference.get("has_organic_growth", False)
    text_condition = text_inference.get("condition", "unknown")
    text_relationship = text_inference.get("organic_relationship", "none")
    
    # Contradiction 1: Organic growth
    if green_coverage > 0.2 and og_confidence > 0.70 and not text_has_organic:
        contradictions["contradictions"].append({
            "type": "organic_growth_mismatch",
            "visual": f"green_coverage={green_coverage:.3f} (confidence={og_confidence:.2f})",
            "text": "no organic growth detected",
            "severity": "high" if og_confidence > 0.85 else "medium"
        })
        contradictions["overrides"].append({
            "field": "has_organic_growth",
            "from": False,
            "to": True,
            "reason": f"Visual evidence proves {green_coverage:.1%} green coverage with {og_confidence:.0%} confidence"
        })
        contradictions["recommendations"].append("Trust visual evidence - text inference missed organic growth")
    
    # Contradiction 2: Material condition
    if condition_vis != "unknown" and mc_confidence > 0.75:
        if text_condition != "unknown" and text_condition != condition_vis:
            # Check if visual confidence is significantly higher
            if mc_confidence > 0.80:
                contradictions["contradictions"].append({
                    "type": "condition_mismatch",
                    "visual": f"{condition_vis} (confidence={mc_confidence:.2f})",
                    "text": text_condition,
                    "severity": "high"
                })
                contradictions["overrides"].append({
                    "field": "condition",
                    "from": text_condition,
                    "to": condition_vis,
                    "reason": f"Visual texture analysis proves {condition_vis} with {mc_confidence:.0%} confidence"
                })
                contradictions["recommendations"].append("Trust visual evidence - texture analysis is ground truth")
    
    # Contradiction 3: Organic integration
    if relationship_vis != "none" and oi_confidence > 0.75:
        if text_relationship != relationship_vis and text_relationship != "none":
            contradictions["contradictions"].append({
                "type": "integration_mismatch",
                "visual": f"{relationship_vis} (confidence={oi_confidence:.2f})",
                "text": text_relationship,
                "severity": "high" if oi_confidence > 0.85 else "medium"
            })
            contradictions["overrides"].append({
                "field": "organic_relationship",
                "from": text_relationship,
                "to": relationship_vis,
                "reason": f"Visual morphological analysis proves {relationship_vis} with {oi_confidence:.0%} confidence"
            })
            contradictions["recommendations"].append("Trust visual evidence - morphological operations are ground truth")
    
    return contradictions


def extract_visual_features(image_path):
    """
    Extract all visual features using deterministic computer vision.
    Universal: works for any image type.
    
    This is the main entry point for visual grounding.
    Returns ground truth that text matching cannot provide.
    
    ENHANCED: Now includes validation and edge case handling.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dict with all visual features:
            - organic_growth: dict from detect_organic_growth()
            - material_condition: dict from detect_material_condition()
            - organic_integration: dict from detect_organic_integration()
            - overall_confidence: float - overall confidence in visual analysis
            - validation: dict from validate_visual_evidence()
    """
    try:
        # Extract all visual features
        organic_growth = detect_organic_growth(image_path)
        material_condition = detect_material_condition(image_path)
        
        # For integration detection, we can reuse the green mask
        img = cv2.imread(image_path)
        if img is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            structure_edges = cv2.Canny(gray, 100, 200)
            kernel = np.ones((3, 3), np.uint8)
            structure_edges = cv2.dilate(structure_edges, kernel, iterations=1)
            
            organic_integration = detect_organic_integration(image_path, green_mask, structure_edges)
        else:
            organic_integration = {"overlap_ratio": 0.0, "relationship": "none", "integration_level": "none",
                                  "evidence": ["image_load_failed"], "confidence": 0.0}
        
        # Overall confidence (weighted average)
        confidences = [
            organic_growth.get("confidence", 0.0),
            material_condition.get("confidence", 0.0),
            organic_integration.get("confidence", 0.0)
        ]
        overall_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        visual_evidence = {
            "organic_growth": organic_growth,
            "material_condition": material_condition,
            "organic_integration": organic_integration,
            "overall_confidence": overall_confidence
        }

        visual_evidence = apply_domain_guard(visual_evidence)

        # Validate visual evidence (post-guard)
        validation = validate_visual_evidence(visual_evidence)
        visual_evidence["validation"] = validation
        
        # Log validation warnings and issues
        if validation["warnings"]:
            logger.info(f"Visual evidence validation warnings: {validation['warnings']}")
        if validation["issues"]:
            logger.warning(f"Visual evidence validation issues: {validation['issues']}")
        
        return visual_evidence
    except Exception as e:
        logger.warning(f"Visual feature extraction failed: {e}")
        return {
            "organic_growth": {"green_coverage": 0.0, "confidence": 0.0},
            "material_condition": {"condition": "unknown", "confidence": 0.0},
            "organic_integration": {"relationship": "none", "confidence": 0.0},
            "overall_confidence": 0.0,
            "validation": {"is_valid": False, "warnings": [], "issues": [f"Extraction failed: {str(e)}"]},
        }