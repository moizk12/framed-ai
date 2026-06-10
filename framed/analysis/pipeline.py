"""Pipeline orchestration: analyze_image + run_full_analysis (canonical schema)."""

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .analysis_cache import compute_file_hash, get_cached_analysis, save_cached_analysis
from .derived_fields import detect_genre, infer_emotion, interpret_visual_features
from .echo_memory import update_echo_memory
from .models import get_nima_model
from .runtime_paths import PERCEPTION_MAX_WORKERS, ensure_directories
from .stage_timing import log_stage_done
from .scene_and_anchors import generate_semantic_anchors, synthesize_scene_understanding
from .schema import create_empty_analysis_result, validate_schema
from .visual_evidence import detect_contradictions, extract_places365_signals, extract_visual_features
from .perception import (
    analyze_background_clutter,
    analyze_color,
    analyze_color_harmony,
    analyze_lighting_direction,
    analyze_lines_and_symmetry,
    analyze_subject_emotion,
    analyze_tonal_range,
    detect_objects_and_framing,
    get_clip_description,
    get_clip_inventory,
    predict_nima_score,
)

logger = logging.getLogger(__name__)


def run_full_analysis(image_path: str, photo_id: str = "", filename: str = "") -> Dict[str, Any]:
    """Run the full pipeline and update ECHO memory on success."""
    ensure_directories()
    try:
        analysis_result = analyze_image(image_path, photo_id=photo_id, filename=filename)
        if not validate_schema(analysis_result):
            analysis_result.setdefault("errors", {})["schema_validation"] = "Result does not conform to canonical schema"
        critical_errors = (analysis_result.get("errors", {}) or {}).get("critical") or (analysis_result.get("errors", {}) or {}).get(
            "image_load"
        )
        if not critical_errors:
            update_echo_memory(analysis_result)
        return analysis_result
    except Exception as e:
        logger.error("Fatal error in run_full_analysis: %s", e, exc_info=True)
        result = create_empty_analysis_result()
        result["errors"]["pipeline"] = str(e)
        return result


def analyze_image(path: str, photo_id: str = "", filename: str = "", disable_cache: bool = False) -> Dict[str, Any]:
    ensure_directories()
    logger.info("Analyzing image: %s", path)
    t_request = time.perf_counter()

    file_hash = compute_file_hash(path)
    if not photo_id:
        photo_id = file_hash[:16] if file_hash else str(uuid.uuid4())

    cached_result = None if disable_cache else get_cached_analysis(file_hash)
    if cached_result:
        cached_result["metadata"]["photo_id"] = photo_id
        cached_result["metadata"]["filename"] = filename
        return cached_result

    result = create_empty_analysis_result()
    from datetime import datetime

    result["metadata"] = {
        "photo_id": photo_id,
        "filename": filename or os.path.basename(path),
        "file_hash": file_hash,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    try:
        img = cv2.imread(path)
        if img is None:
            result["errors"]["image_load"] = "Could not load image"
            return result
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            brightness = round(float(np.mean(gray)), 2)
            contrast = round(float(gray.std()), 2)
            sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
            result["perception"]["technical"]["available"] = True
            result["perception"]["technical"]["brightness"] = brightness
            result["perception"]["technical"]["contrast"] = contrast
            result["perception"]["technical"]["sharpness"] = sharpness
        except Exception as e:
            result["errors"]["technical"] = str(e)

        nima_model = None
        try:
            nima_model = get_nima_model()
        except Exception as e:
            result["errors"]["nima_load"] = str(e)

        clip_default = {"caption": None, "tags": [], "genre_hint": None}
        objects_default = {
            "objects": [],
            "object_narrative": None,
            "subject_position": None,
            "subject_size": None,
            "framing_description": None,
            "spatial_interpretation": None,
        }
        nima_default = {"mean_score": None, "distribution": {}}
        color_default = {"palette": [], "mood": None}
        harmony_default = {"dominant_color": None, "harmony": None}
        lines_default = {"line_pattern": None, "line_style": None, "symmetry": None}
        lighting_default = {"direction": None}
        tonal_default = {"tonal_range": None}
        emotion_default = {"subject_type": None, "emotion": None}
        clutter_default = {"clutter_level": None}

        tasks = [
            ("clip", get_clip_description, (path,), clip_default, "clip"),
            ("clip_inventory", get_clip_inventory, (path,), [], "clip_inventory"),
            ("nima", predict_nima_score, (nima_model, path), nima_default, "nima"),
            ("color", analyze_color, (path,), color_default, "color"),
            ("color_harmony", analyze_color_harmony, (path,), harmony_default, "color_harmony"),
            ("objects", detect_objects_and_framing, (path,), objects_default, "objects"),
            ("lines_symmetry", analyze_lines_and_symmetry, (path,), lines_default, "lines_symmetry"),
            ("lighting", analyze_lighting_direction, (path,), lighting_default, "lighting"),
            ("tonal_range", analyze_tonal_range, (path,), tonal_default, "tonal_range"),
            ("subject_emotion", analyze_subject_emotion, (path,), emotion_default, "emotion"),
            ("clutter", analyze_background_clutter, (path,), clutter_default, "clutter"),
            ("visual_evidence", extract_visual_features, (path,), {}, "visual_evidence"),
        ]

        def _run_one(name, func, args, default, error_key):
            try:
                return (name, func(*args), error_key, None)
            except Exception as e:
                return (name, default, error_key, str(e))

        t_stage = time.perf_counter()
        out: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=PERCEPTION_MAX_WORKERS) as executor:
            futures = {executor.submit(_run_one, *t): t[0] for t in tasks}
            for future in as_completed(futures):
                name, value, error_key, error_msg = future.result()
                if error_msg and error_key:
                    result["errors"][error_key] = error_msg
                out[name] = value
        log_stage_done("perception_parallel", t_request, t_stage)

        clip_data = out.get("clip", clip_default)
        clip_inventory = out.get("clip_inventory", [])
        nima_result = out.get("nima", nima_default)
        color_analysis = out.get("color", color_default)
        color_harmony = out.get("color_harmony", harmony_default)
        object_data = out.get("objects", objects_default)
        lines_symmetry = out.get("lines_symmetry", lines_default)
        lighting_direction = out.get("lighting", lighting_default)
        tonal_range = out.get("tonal_range", tonal_default)
        subject_emotion = out.get("subject_emotion", emotion_default)
        clutter_result = out.get("clutter", clutter_default)
        visual_evidence = out.get("visual_evidence", {})

        try:
            objs = object_data.get("objects", []) if isinstance(object_data, dict) else []
            result["objects"] = [{"name": str(o)} for o in (objs or [])]
        except Exception:
            result["objects"] = []

        if clip_data.get("caption"):
            result["perception"]["semantics"]["available"] = True
            result["perception"]["semantics"]["caption"] = clip_data.get("caption")
            result["perception"]["semantics"]["tags"] = clip_data.get("tags", [])
            result["perception"]["semantics"]["genre_hint"] = clip_data.get("genre_hint")
            result["confidence"]["clip"] = True

        if nima_result.get("mean_score") is not None:
            result["perception"]["aesthetics"]["available"] = True
            result["perception"]["aesthetics"]["mean_score"] = nima_result.get("mean_score")
            result["perception"]["aesthetics"]["distribution"] = nima_result.get("distribution", {})
            result["confidence"]["nima"] = True

        if color_analysis.get("palette"):
            result["perception"]["color"]["available"] = True
            result["perception"]["color"]["palette"] = color_analysis.get("palette", [])
            result["perception"]["color"]["mood"] = color_analysis.get("mood")

        if color_harmony.get("dominant_color"):
            result["perception"]["color"]["harmony"]["dominant_color"] = color_harmony.get("dominant_color")
            result["perception"]["color"]["harmony"]["harmony_type"] = color_harmony.get("harmony")

        if object_data.get("objects"):
            result["perception"]["composition"]["available"] = True
            result["perception"]["composition"]["subject_framing"] = {
                "position": object_data.get("subject_position"),
                "size": object_data.get("subject_size"),
                "style": object_data.get("framing_description"),
                "interpretation": object_data.get("spatial_interpretation"),
            }
            result["confidence"]["yolo"] = True

        if lines_symmetry.get("line_pattern"):
            result["perception"]["composition"]["line_pattern"] = lines_symmetry.get("line_pattern")
            result["perception"]["composition"]["line_style"] = lines_symmetry.get("line_style")
            result["perception"]["composition"]["symmetry"] = lines_symmetry.get("symmetry")

        if lighting_direction.get("direction"):
            result["perception"]["lighting"]["available"] = True
            result["perception"]["lighting"]["direction"] = lighting_direction.get("direction")
        if tonal_range.get("tonal_range"):
            result["perception"]["lighting"]["quality"] = tonal_range.get("tonal_range")

        if subject_emotion.get("subject_type"):
            result["perception"]["emotion"]["available"] = True
            result["perception"]["emotion"]["subject_type"] = subject_emotion.get("subject_type")
            result["perception"]["emotion"]["emotion"] = subject_emotion.get("emotion")
        result["confidence"]["deepface"] = False

        legacy_dict = {
            "brightness": result["perception"]["technical"].get("brightness"),
            "contrast": result["perception"]["technical"].get("contrast"),
            "sharpness": result["perception"]["technical"].get("sharpness"),
            "clip_description": clip_data,
            "color_palette": result["perception"]["color"].get("palette", []),
            "color_mood": result["perception"]["color"].get("mood"),
            "lighting_direction": result["perception"]["lighting"].get("direction"),
            "tonal_range": result["perception"]["lighting"].get("quality"),
            "subject_emotion": subject_emotion,
            "line_pattern": result["perception"]["composition"].get("line_pattern"),
            "line_style": result["perception"]["composition"].get("line_style"),
            "symmetry": result["perception"]["composition"].get("symmetry"),
            "objects": object_data.get("objects", []),
            "subject_framing": result["perception"]["composition"].get("subject_framing", {}),
            "background_clutter": clutter_result,
        }

        def _run_derived(name, func, default, error_key):
            try:
                return (name, func(legacy_dict), error_key, None)
            except Exception as e:
                return (name, default, error_key, str(e))

        derived_tasks = [
            ("visual_interp", interpret_visual_features, {}, "visual_interpretation"),
            ("emotion_result", infer_emotion, {"emotional_mood": None}, "emotion_inference"),
            ("genre_info", detect_genre, {"genre": None, "subgenre": None}, "genre_detection"),
        ]
        t_stage = time.perf_counter()
        derived_out: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=3) as exec_derived:
            futures = {exec_derived.submit(_run_derived, t[0], t[1], t[2], t[3]): t[0] for t in derived_tasks}
            for future in as_completed(futures):
                name, value, error_key, error_msg = future.result()
                if error_msg and error_key:
                    result["errors"][error_key] = error_msg
                derived_out[name] = value
        log_stage_done("derived_fields", t_request, t_stage)

        visual_interp = derived_out.get("visual_interp", {})
        emotion_result = derived_out.get("emotion_result", {})
        genre_info = derived_out.get("genre_info", {})
        if visual_interp:
            result["derived"]["visual_interpretation"] = visual_interp
        if emotion_result.get("emotional_mood"):
            result["derived"]["emotional_mood"] = emotion_result.get("emotional_mood")
        if genre_info.get("genre"):
            result["derived"]["genre"]["genre"] = genre_info.get("genre")
            result["derived"]["genre"]["subgenre"] = genre_info.get("subgenre")

        if visual_evidence:
            if clip_data.get("caption") or clip_data.get("tags"):
                text_inference = {
                    "has_organic_growth": any(
                        term
                        in (clip_data.get("caption", "") + " " + " ".join(clip_data.get("tags", []))).lower()
                        for term in ["ivy", "moss", "vegetation", "green", "growth"]
                    ),
                    "condition": "unknown",
                    "organic_relationship": "none",
                }
                contradictions = detect_contradictions(visual_evidence, text_inference)
                if contradictions.get("overrides"):
                    pass
            result["visual_evidence"] = visual_evidence

        try:
            from .negative_evidence import detect_negative_evidence

            negative_evidence = detect_negative_evidence(result)
            if negative_evidence:
                if not result.get("visual_evidence"):
                    result["visual_evidence"] = {}
                result["visual_evidence"]["negative_evidence"] = negative_evidence
        except Exception as e:
            logger.warning("Negative evidence detection failed (non-fatal): %s", e)

        places365_signals = {}
        try:
            places365_signals = extract_places365_signals(path)
            if places365_signals:
                result.setdefault("perception", {}).setdefault("scene", {})["places365"] = places365_signals
        except Exception as e:
            logger.warning("Places365 signal extraction failed (non-fatal): %s", e)

        try:
            places = (result.get("perception", {}).get("scene", {}) or {}).get("places365", {}) or {}
            scene_category = str(places.get("scene_category", "") or "").lower()
            indoor_outdoor = str(places.get("indoor_outdoor", "") or "").lower()

            yolo_objs = [str(o).lower() for o in (object_data.get("objects", []) or [])]
            caption = str(clip_data.get("caption", "") or "").lower()
            tags = [str(t).lower() for t in (clip_data.get("tags", []) or [])]
            inv_items = []
            if isinstance(clip_inventory, list):
                for item in clip_inventory:
                    if isinstance(item, dict):
                        inv_items.append(str(item.get("item", "")).lower())
                    else:
                        inv_items.append(str(item).lower())
            text_blob = " ".join([caption] + tags + inv_items + yolo_objs + [scene_category]).lower()

            def _count_any(candidates):
                return sum(1 for o in yolo_objs if any(tok in o for tok in candidates))

            num_people = _count_any(["person", "people", "man", "woman", "child", "face"]) or sum(
                1 for t in tags if "person" in t or "people" in t
            )
            num_vehicles = _count_any(["car", "truck", "bus", "taxi", "bicycle", "motorcycle"])
            num_buildings = _count_any(["building", "tower", "skyscraper", "bridge"]) or ("city" in text_blob)
            num_tools = _count_any(
                ["hammer", "wrench", "screwdriver", "pliers", "saw", "drill", "tool", "tools"]
            ) or sum(1 for t in tags if any(tok in t for tok in ("tool", "hammer", "wrench", "screwdriver")))
            has_street_cues = (
                num_vehicles > 0
                or any(
                    k in text_blob
                    for k in [
                        "sidewalk",
                        "crosswalk",
                        "road",
                        "asphalt",
                        "pavement",
                        "traffic",
                        "intersection",
                        "downtown",
                        "urban street",
                    ]
                )
            )
            has_interior_cues = any(
                k in text_blob
                for k in [
                    "living room",
                    "interior",
                    "bedroom",
                    "room",
                    "office",
                    "abandoned",
                    "shelf",
                    "shelves",
                    "window",
                    "wall",
                    "desk",
                    "sofa",
                    "couch",
                    "chair",
                    "table",
                ]
            )
            looks_abstract_terms = any(k in text_blob for k in ["abstract", "nonrepresentational", "non-representational"])
            looks_painting = any(k in text_blob for k in ["painting", "canvas", "acrylic", "oil painting", "artwork"])

            avg_saturation = None
            try:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                avg_saturation = float(hsv_img[:, :, 1].mean() / 255.0)
            except Exception:
                avg_saturation = None

            _ui_yolo = frozenset({"tv", "laptop", "mouse", "keyboard", "monitor", "cell phone"})
            _ui_text_tokens = (
                "screen", "monitor", "ui", "code", "editor", "laptop", "display",
                "interface", "text", "keyboard", "program", "website", "webpage",
                "browser", "terminal", "software", "application", "wikipedia", "github",
                "news", "reddit", "arxiv", "html", "css", "javascript",
            )
            _ve_mc = (result.get("visual_evidence") or {}).get("material_condition") or {}
            _edge_deg = float(_ve_mc.get("edge_degradation", 0.0) or 0.0)

            has_ui_yolo = bool(set(yolo_objs) & _ui_yolo)
            has_ui_text = any(tok in text_blob for tok in _ui_text_tokens)
            has_ui_signal = has_ui_yolo or (
                has_ui_text and scene_category in ("artificial", "indoor", "man-made", "")
            )
            looks_like_screen_capture = (
                not has_street_cues
                and num_vehicles == 0
                and _edge_deg > 0.52
                and scene_category in ("artificial", "indoor", "man-made", "")
            )

            scene_type = "unknown"
            if looks_painting or (looks_abstract_terms and ("painting" in text_blob or "canvas" in text_blob)):
                scene_type = "abstract_art"
            elif avg_saturation is not None and avg_saturation > 0.50 and num_people == 0 and num_vehicles == 0 and not num_buildings:
                scene_type = "abstract_art"
            elif (has_ui_signal or looks_like_screen_capture) and not has_street_cues and num_vehicles == 0:
                scene_type = "screenshot_ui"
            elif num_people > 0 and not looks_like_screen_capture:
                scene_type = "people_scene"
            elif (
                (num_tools >= 2 or any(k in text_blob for k in ["tool", "tools", "workshop", "garage", "pegboard"]))
                and not has_street_cues
                and num_vehicles == 0
            ):
                scene_type = "object_dense"
            elif indoor_outdoor == "indoor" or has_interior_cues:
                scene_type = "interior_scene"
            elif has_street_cues and (indoor_outdoor != "indoor"):
                scene_type = "street_scene"
            elif indoor_outdoor == "outdoor" and any(
                k in text_blob for k in ["mountain", "mountains", "lake", "river", "forest", "field", "landscape", "tree", "trees", "house", "cabin"]
            ):
                scene_type = "landscape_scene"

            og = (visual_evidence or {}).get("organic_growth", {}) or {}
            mc = (visual_evidence or {}).get("material_condition", {}) or {}
            green_cov = float(og.get("green_coverage", 0.0) or 0.0)
            edge_deg = float(mc.get("edge_degradation", 0.0) or 0.0)
            rough = float(mc.get("surface_roughness", 0.0) or 0.0)

            scene_is_depiction = scene_type in {
                "abstract_art",
                "people_scene",
                "interior_scene",
                "street_scene",
                "landscape_scene",
                "object_dense",
                "screenshot_ui",
            }
            is_surface_study = (not scene_is_depiction) and (edge_deg > 0.6 or rough > 0.12) and num_people == 0 and num_vehicles == 0 and not num_buildings
            if is_surface_study:
                scene_type = "surface_study"

            result.setdefault("visual_evidence", {}).setdefault("scene_gate", {})["scene_type"] = scene_type
            result["visual_evidence"]["scene_gate"]["is_surface_study"] = bool(is_surface_study)
            result["visual_evidence"]["scene_gate"]["signals"] = {
                "places_scene_category": scene_category,
                "places_indoor_outdoor": indoor_outdoor,
                "yolo_objects": yolo_objs[:20],
                "clip_caption": caption[:200],
                "green_coverage": round(green_cov, 3),
                "edge_degradation": round(edge_deg, 3),
                "surface_roughness": round(rough, 3),
            }
        except Exception as e:
            logger.warning("Scene gate failed (non-fatal): %s", e)

        USE_INTELLIGENCE_CORE = os.getenv("FRAMED_USE_INTELLIGENCE_CORE", "true").lower() == "true"
        if not USE_INTELLIGENCE_CORE:
            try:
                from .interpret_scene import interpret_scene
                from .interpretive_memory import create_pattern_signature, query_memory_patterns, store_interpretation

                if isinstance(clip_inventory, list) and len(clip_inventory) > 0 and isinstance(clip_inventory[0], dict):
                    clip_inventory_items = [item["item"] for item in clip_inventory]
                else:
                    clip_inventory_items = clip_inventory if isinstance(clip_inventory, list) else []

                semantic_signals = {
                    "clip_inventory": clip_inventory_items,
                    "clip_inventory_full": clip_inventory if isinstance(clip_inventory, list) and clip_inventory and isinstance(clip_inventory[0], dict) else [],
                    "clip_caption": clip_data.get("caption", ""),
                    "yolo_objects": object_data.get("objects", []),
                }
                technical_stats = {
                    "brightness": result["perception"]["technical"].get("brightness"),
                    "contrast": result["perception"]["technical"].get("contrast"),
                    "sharpness": result["perception"]["technical"].get("sharpness"),
                    "color_mood": result["perception"]["color"].get("mood"),
                }
                pattern_signature = create_pattern_signature(visual_evidence, semantic_signals)
                memory_patterns = query_memory_patterns(pattern_signature, limit=5)
                interpretive_conclusions = interpret_scene(
                    visual_evidence=visual_evidence,
                    semantic_signals=semantic_signals,
                    technical_stats=technical_stats,
                    interpretive_memory_patterns=memory_patterns,
                )
                result["interpretive_conclusions"] = interpretive_conclusions
                primary = interpretive_conclusions.get("primary_interpretation", {})
                chosen_interp = primary.get("conclusion", "unclear_interpretation")
                confidence = primary.get("confidence", 0.5)
                store_interpretation(pattern_signature, chosen_interp, confidence)
            except Exception:
                result["interpretive_conclusions"] = {}

        result["_clip_inventory"] = clip_inventory if isinstance(clip_inventory, list) else []
        result["_image_path"] = path
        if not USE_INTELLIGENCE_CORE:
            try:
                scene_understanding = synthesize_scene_understanding(result)
                if scene_understanding:
                    result["scene_understanding"] = scene_understanding
            except Exception:
                pass
        result.pop("_clip_inventory", None)
        result.pop("_image_path", None)
        result.pop("_visual_evidence", None)

        try:
            composition_for_anchors = {
                "symmetry": result["perception"]["composition"].get("symmetry"),
                "subject_size": result["perception"]["composition"].get("subject_framing", {}).get("size"),
            }
            semantic_anchors = generate_semantic_anchors(
                clip_inventory=clip_inventory if isinstance(clip_inventory, list) else [],
                clip_tags=clip_data.get("tags", []),
                clip_caption=clip_data.get("caption"),
                yolo_objects=object_data.get("objects", []),
                composition_data=composition_for_anchors,
            )
            if semantic_anchors:
                result["semantic_anchors"] = semantic_anchors
        except Exception:
            pass

        ENABLE_INTELLIGENCE_CORE = os.getenv("FRAMED_ENABLE_INTELLIGENCE_CORE", "true").lower() == "true"
        if ENABLE_INTELLIGENCE_CORE:
            try:
                t_stage = time.perf_counter()
                from .intelligence_core import framed_intelligence
                from .temporal_memory import (
                    create_pattern_signature as create_temporal_signature,
                    format_temporal_memory_for_intelligence,
                    store_interpretation as store_temporal_interpretation,
                    track_user_trajectory,
                )

                semantic_signals_for_intelligence = {
                    "objects": object_data.get("objects", []),
                    "tags": clip_data.get("tags", []),
                    "caption_keywords": clip_data.get("caption", "").split()[:20] if clip_data.get("caption") else [],
                }
                temporal_signature = create_temporal_signature(visual_evidence, semantic_signals_for_intelligence)
                temporal_memory_data = format_temporal_memory_for_intelligence(temporal_signature, user_id=photo_id)
                user_history = temporal_memory_data.get("user_trajectory", {})

                intelligence_output = framed_intelligence(
                    visual_evidence=visual_evidence,
                    analysis_result=result,
                    temporal_memory=temporal_memory_data,
                    user_history=user_history,
                    pattern_signature=temporal_signature,
                )
                log_stage_done("intelligence_core", t_request, t_stage)

                result["intelligence"] = intelligence_output
                result["pattern_signature"] = temporal_signature

                confidence = intelligence_output.get("meta_cognition", {}).get("confidence", 0.85)
                store_temporal_interpretation(signature=temporal_signature, interpretation=intelligence_output, confidence=confidence)
                track_user_trajectory(analysis_result=result, intelligence_output=intelligence_output, user_id=photo_id)

                try:
                    from .learning_system import learn_implicitly

                    learn_implicitly(analysis_result=result, intelligence_output=intelligence_output, user_history=user_history)
                except Exception:
                    pass
            except Exception:
                result["intelligence"] = {}
        else:
            result["intelligence"] = {}

        if file_hash:
            t_stage = time.perf_counter()
            save_cached_analysis(file_hash, result)
            log_stage_done("analysis_cache_save", t_request, t_stage)

        return result

    except Exception as e:
        logger.error("Critical error in analyze_image: %s", e, exc_info=True)
        result["errors"]["critical"] = str(e)
        return result

