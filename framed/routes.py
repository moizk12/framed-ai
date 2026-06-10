from flask import Blueprint, render_template, request, jsonify, current_app
import os, uuid, time
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
