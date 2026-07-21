from flask import Blueprint, render_template, request, jsonify, current_app
import copy
import os
import uuid
import time
from werkzeug.utils import secure_filename

from framed.analysis.stage_timing import log_stage_done

from framed.analysis.vision import (
    run_full_analysis,
    load_echo_memory,
    save_echo_memory,
    update_echo_memory,
    ask_echo,
    generate_merged_critique,
    client
)

ALLOWED_EXTENSIONS = {"png","jpg","jpeg","webp","bmp","tiff"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _grounding_probe_enabled() -> bool:
    return os.getenv("ENABLE_DENSE_GROUNDING_PROBE", "false").lower() == "true"


def _validate_grounding_box(box: dict) -> bool:
    if not isinstance(box, dict):
        return False
    try:
        x, y, w, h = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    return x + w <= 1.000001 and y + h <= 1.000001


def _prepare_render_boxes(raw_boxes):
    render = []
    warnings = []
    if not isinstance(raw_boxes, list):
        return render, warnings
    for idx, box in enumerate(raw_boxes):
        box_id = f"g{idx}"
        if not _validate_grounding_box(box):
            warnings.append(f"grounding_box_invalid:{box_id}")
            continue
        entry = copy.deepcopy(box)
        entry["box_id"] = box_id
        entry["box_number"] = idx + 1
        render.append(entry)
    return render, warnings


def _grounding_state(visual_evidence: dict, result: dict) -> str:
    if bool(result.get("failed")) or result.get("error") is not None:
        return "error"
    if not _grounding_probe_enabled():
        return "disabled"
    grounding = visual_evidence.get("grounding")
    if grounding is None:
        return "unknown"
    if not isinstance(grounding, list):
        return "unknown"
    if len(grounding) == 0:
        return "empty"
    return "available"


def _tier_display(raw):
    if not isinstance(raw, str):
        return "Unavailable"
    mapping = {
        "licensed": "Licensed",
        "cautious": "Limited·cautious",
        "forbidden": "Restricted",
        "restricted": "Restricted",
    }
    return mapping.get(raw.lower(), raw)


def _build_claim_traces(visual_evidence: dict):
    traces = []
    tcl = visual_evidence.get("theme_claim_license")
    if not isinstance(tcl, dict):
        return traces
    reasons = [r for r in tcl.get("reasons") or [] if isinstance(r, str)]
    for claim in ("organic_growth", "reclamation", "weathered_stone", "tier"):
        raw_tier = tcl.get("tier") if claim == "tier" else tcl.get(claim)
        reason = reasons[0] if reasons else None
        paths = ["visual_evidence.theme_claim_license.reasons"]
        if claim == "organic_growth" and isinstance(visual_evidence.get("organic_growth"), dict):
            paths.append("visual_evidence.organic_growth")
        if claim == "reclamation" and isinstance(visual_evidence.get("organic_integration"), dict):
            paths.append("visual_evidence.organic_integration")
        if claim == "weathered_stone" and isinstance(visual_evidence.get("material_condition"), dict):
            paths.append("visual_evidence.material_condition")
        traces.append(
            {
                "claim": claim,
                "tier": _tier_display(raw_tier),
                "tier_raw": raw_tier if isinstance(raw_tier, str) else None,
                "reason": reason,
                "supporting_paths": paths,
                "relation_source": "theme_claim_license",
            }
        )
    return traces


def _collect_evidence_strings(visual_evidence: dict):
    out = []

    def walk(v):
        if isinstance(v, dict):
            for k, vv in v.items():
                if k == "evidence" and isinstance(vv, list):
                    out.extend(x for x in vv if isinstance(x, str))
                else:
                    walk(vv)
        elif isinstance(v, list):
            for item in v:
                walk(item)

    walk(visual_evidence)
    return out


def build_evidence_inspector(result: dict) -> dict:
    """Read-only evidence chain for UI — does not mutate `result`."""
    if not isinstance(result, dict):
        return {}
    visual_evidence = result.get("visual_evidence")
    if not isinstance(visual_evidence, dict):
        visual_evidence = {}
    intelligence = result.get("intelligence") if isinstance(result.get("intelligence"), dict) else {}
    recognition_raw = intelligence.get("recognition") if isinstance(intelligence.get("recognition"), dict) else {}

    scene_type = None
    scene_gate = visual_evidence.get("scene_gate")
    if isinstance(scene_gate, dict):
        scene_type = scene_gate.get("scene_type")

    grounding_raw = visual_evidence.get("grounding")
    grounding_list = copy.deepcopy(grounding_raw) if isinstance(grounding_raw, list) else []
    render_boxes, warnings = _prepare_render_boxes(grounding_list)

    theme_licenses = []
    tcl = visual_evidence.get("theme_claim_license")
    if isinstance(tcl, dict):
        theme_licenses.append(
            {
                "tier": tcl.get("tier"),
                "reasons": list(tcl.get("reasons") or []) if isinstance(tcl.get("reasons"), list) else [],
            }
        )

    critique_text = result.get("critique")
    critique = {
        "text": critique_text if isinstance(critique_text, str) else None,
        "status": "Available" if isinstance(critique_text, str) and critique_text.strip() else "Unavailable",
        "final_output_certified": False,
    }

    recognition = {"inference_status": "Inferred"}
    if isinstance(recognition_raw.get("what_i_see"), str):
        recognition["what_i_see"] = recognition_raw.get("what_i_see")
    if "confidence" in recognition_raw:
        recognition["confidence"] = recognition_raw.get("confidence")

    return {
        "recognition": recognition,
        "scene": {
            "scene_type": scene_type,
            "category": scene_type,
            "inference_status": "Inferred",
        },
        "evidence": _collect_evidence_strings(visual_evidence),
        "grounding": {
            "state": _grounding_state(visual_evidence, result),
            "boxes": grounding_list,
            "render_boxes": render_boxes,
            "inference_status": "Observed",
        },
        "theme_licenses": theme_licenses,
        "claim_traces": _build_claim_traces(visual_evidence),
        "critique": critique,
        "warnings": warnings,
        "provenance": {
            "grounding_probe_enabled": _grounding_probe_enabled(),
            "final_output_certified": False,
        },
    }


def clean_result_for_ui(result: dict) -> dict:
    """
    Presentation-only sanitizer: UI-focused view from canonical schema without mutating `result`.
    Omits internal error/debug structures
    - Reads from canonical schema structure
    - Provides evidence annotations (measured values) instead of interpretations
    """
    if not isinstance(result, dict):
        return {}

    # Check if this is canonical schema or legacy format
    is_canonical = "perception" in result and "metadata" in result
    
    # Extract semantic anchors (if present) - handle missing gracefully
    semantic_anchors = result.get("semantic_anchors", {}) if is_canonical else {}
    
    if is_canonical:
        # Read from canonical schema
        perception = result.get("perception", {})
        derived = result.get("derived", {})
        
        technical = perception.get("technical", {})
        composition = perception.get("composition", {})
        semantics = perception.get("semantics", {})
        color = perception.get("color", {})
        lighting = perception.get("lighting", {})
        genre = derived.get("genre", {})
        
        # Extract evidence values (measured facts, not interpretations)
        subject_framing = composition.get("subject_framing", {})
        
        ui_view = {
            # Semantic evidence
            "caption": semantics.get("caption") if semantics.get("available") else None,
            "tags": semantics.get("tags", []) if semantics.get("available") else [],
            "genre": genre.get("genre"),
            "subgenre": genre.get("subgenre"),
            
            # Evidence annotations (measured values)
            "evidence": {
                "technical": {
                    "brightness": technical.get("brightness") if technical.get("available") else None,
                    "contrast": technical.get("contrast") if technical.get("available") else None,
                    "sharpness": technical.get("sharpness") if technical.get("available") else None,
                },
                "composition": {
                    "symmetry": composition.get("symmetry") if composition.get("available") else None,
                    "subject_position": subject_framing.get("position") if composition.get("available") else None,
                    "subject_size": subject_framing.get("size") if composition.get("available") else None,
                },
                "color": {
                    "mood": color.get("mood") if color.get("available") else None,
                    "harmony_type": color.get("harmony", {}).get("harmony_type") if color.get("available") else None,
                },
                "lighting": {
                    "direction": lighting.get("direction") if lighting.get("available") else None,
                    "quality": lighting.get("quality") if lighting.get("available") else None,
                },
            },
            
            # Legacy fields (kept for backward compatibility, but deprecated)
            "emotional_mood": derived.get("emotional_mood"),
            "poetic_mood": None,  # Not in canonical schema yet
            "color_mood": color.get("mood") if color.get("available") else None,
            "lighting_direction": lighting.get("direction") if lighting.get("available") else None,
            "subject": None,  # Not in canonical schema yet
            "critique": result.get("critique"),  # Still added by downstream functions
            "remix_prompt": result.get("remix_prompt"),  # Still added by downstream functions
        }
    else:
        # Legacy format fallback (for backward compatibility during migration)
        clip = result.get("clip_description", {}) or {}
        summary = result.get("summary", {}) or {}

        ui_view = {
            "caption": clip.get("caption"),
            "tags": clip.get("tags"),
            "genre": result.get("genre") or summary.get("genre"),
            "subgenre": result.get("subgenre") or summary.get("subgenre"),
            "emotional_mood": result.get("emotional_mood") or summary.get("emotional_mood"),
            "poetic_mood": summary.get("poetic_mood"),
            "color_mood": result.get("color_mood"),
            "lighting_direction": result.get("lighting_direction"),
            "subject": summary.get("subject"),
            "critique": result.get("critique"),
            "remix_prompt": result.get("remix_prompt"),
            # Legacy format doesn't have structured evidence
            "evidence": None,
        }

    # Softly drop internal error metadata from the presentation layer
    if "errors" in ui_view:
        ui_view.pop("errors", None)

    ui_view["evidence_inspector"] = build_evidence_inspector(result)

    return {k: v for k, v in ui_view.items() if v}

main = Blueprint(
    "main",
    __name__,
    template_folder="../templates",
    static_folder="../static",
    static_url_path="/static",
)

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/upload")
def upload():
    return render_template("upload.html")

@main.post("/analyze")
def analyze():
    current_app.logger.info(f"FILES: {list(request.files.keys())}")
    current_app.logger.info(f"FORM: {dict(request.form)}")
    
    file = request.files.get("image")
    mentor_mode = request.form.get("mentor_mode", "Balanced Mentor")

    if not file or file.filename == "":
        current_app.logger.warning(f"No file uploaded. Available keys: {list(request.files.keys())}")
        return jsonify({"error": "No file received."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPEG, PNG, WEBP, TIFF, or BMP."}), 400

    from framed.analysis.vision import UPLOAD_DIR
    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    image_path = os.path.join(upload_dir, unique_name)

    try:
        file.save(image_path)

        photo_id = str(uuid.uuid4())
        t_request = time.perf_counter()
        t_pipeline = time.perf_counter()
        analysis_result = run_full_analysis(image_path, photo_id=photo_id, filename=safe_name)
        log_stage_done("run_full_analysis", t_request, t_pipeline)

        ui_view = clean_result_for_ui(analysis_result)
        response_payload = dict(analysis_result)

        t_critique = time.perf_counter()
        intelligence_output = analysis_result.get("intelligence", {})
        
        if intelligence_output and intelligence_output.get("recognition", {}).get("what_i_see"):
            # Use new expression layer (Model B)
            try:
                from framed.analysis.expression_layer import (
                    generate_poetic_critique,
                    apply_mentor_hierarchy,
                    integrate_self_correction,
                )
                from framed.analysis.temporal_memory import load_user_trajectory
                
                # Get user history for mentor hierarchy
                user_history = load_user_trajectory(user_id=photo_id)
                
                # Apply mentor hierarchy
                mentor_reasoning = intelligence_output.get("mentor", {})
                mentor_interventions = apply_mentor_hierarchy(mentor_reasoning, user_history)
                
                # Generate poetic critique
                critique = generate_poetic_critique(
                    intelligence_output=intelligence_output,
                    mentor_mode=mentor_mode,
                )
                
                # Integrate self-correction
                self_critique = intelligence_output.get("self_critique", {})
                critique = integrate_self_correction(critique, self_critique)
                
                response_payload["critique"] = critique
                if ui_view:
                    ui_view["critique"] = critique
                
                logger.info(f"Expression layer (Model B) completed: {len(critique)} characters")
                
            except Exception as e:
                current_app.logger.warning(f"Expression layer failed (non-fatal): {e}, falling back to legacy critique")
                # Fallback to legacy critique generation
                critique = generate_merged_critique(analysis_result, mentor_mode)
                response_payload["critique"] = critique
                if ui_view:
                    ui_view["critique"] = critique
        else:
            # Fallback to legacy critique generation (for backward compatibility)
            critique = generate_merged_critique(analysis_result, mentor_mode)
            response_payload["critique"] = critique
            if ui_view:
                ui_view["critique"] = critique
        
        try:
            from framed.analysis.critique_finalization import finalize_critique_with_reflection

            intelligence_output = analysis_result.get("intelligence", {})
            interpretive_conclusions = analysis_result.get("interpretive_conclusions", {})
            hitl_penalty = 0.0
            try:
                from framed.feedback.calibration import get_hitl_calibration
                hitl_penalty = get_hitl_calibration(None).get("mentor_drift_penalty", 0)
            except Exception:
                pass

            finalized = finalize_critique_with_reflection(
                critique,
                intelligence_output,
                interpretive_conclusions=interpretive_conclusions,
                analysis_result=analysis_result,
                mentor_mode=mentor_mode,
                hitl_mentor_drift_penalty=hitl_penalty,
            )
            if finalized.get("reflection_report"):
                critique = finalized["critique"]
                response_payload["critique"] = critique
                response_payload["reflection_report"] = finalized["reflection_report"]
                if ui_view:
                    ui_view["critique"] = critique
        except Exception as e:
            current_app.logger.warning(f"Reflection loop failed (non-fatal): {e}")

        log_stage_done("critique_expression_reflection", t_request, t_critique)

        if ui_view:
            response_payload["_ui"] = ui_view

        return jsonify(response_payload)
    except Exception as e:
        current_app.logger.exception("Analysis failed")
        return jsonify({"error": f"Internal error: {e}"}), 500
    finally:
        # Don’t accumulate files
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass

@main.post("/reset")
def reset():
    save_echo_memory([])
    return jsonify({"ok": True, "message": "History cleared"})

@main.post("/feedback")
def feedback_route():
    try:
        payload = request.get_json(force=True) or {}
        button = (payload.get("button") or payload.get("feedback_type") or "").strip()
        image_id = (payload.get("image_id") or "").strip()
        signature = (payload.get("signature") or payload.get("pattern_signature") or image_id).strip()
        correction = (payload.get("correction") or "").strip()
        excerpt = (payload.get("critique_excerpt") or "").strip()
        if not button:
            return jsonify({"error": "Missing feedback button"}), 400
        from framed.feedback.storage import append_ui_feedback

        ok = append_ui_feedback(image_id, button, signature, correction, excerpt)
        if not ok:
            return jsonify({"error": "Invalid feedback payload"}), 400
        return jsonify({"ok": True})
    except Exception as e:
        current_app.logger.exception("Feedback failed")
        return jsonify({"error": str(e)}), 500


@main.post("/ask-echo")
def ask_echo_route():
    try:
        payload = request.get_json(force=True) or {}
        question = payload.get("question","").strip()
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400
        mem = load_echo_memory()
        answer = ask_echo(question, mem, client)
        return jsonify({"answer": answer})
    except Exception as e:
        current_app.logger.exception("ECHO failed")
        return jsonify({"error": f"Echo error: {e}"}), 500
