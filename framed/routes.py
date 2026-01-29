# top of routes.py
from flask import Blueprint, render_template, request, jsonify, current_app
import os, uuid
from werkzeug.utils import secure_filename

from framed.analysis.vision import (
    run_full_analysis,
    load_echo_memory,
    save_echo_memory,
    update_echo_memory,
    ask_echo,
    generate_merged_critique,   # ✅ was missing
    client
)

ALLOWED_EXTENSIONS = {"png","jpg","jpeg","webp","bmp","tiff"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_result_for_ui(result: dict) -> dict:
    """
    Phase II: Reads from canonical schema structure.
    
    Presentation-only sanitizer that builds a gentle, UI-focused view
    from the canonical schema without mutating the core analysis output.

    - Never modifies `result` in-place
    - Omits internal error/debug structures
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
    static_url_path="/static"  # ✅ consistent static URL
)

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/upload")
def upload():
    return render_template("upload.html")

@main.post("/analyze")
def analyze():
    # Debug: Log incoming request data
    current_app.logger.info(f"FILES: {list(request.files.keys())}")
    current_app.logger.info(f"FORM: {dict(request.form)}")
    
    file = request.files.get("image")
    mentor_mode = request.form.get("mentor_mode", "Balanced Mentor")

    if not file or file.filename == "":
        current_app.logger.warning(f"No file uploaded. Available keys: {list(request.files.keys())}")
        return jsonify({"error": "No file was received. Try choosing an image and letting FRAMED breathe with it again."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "FRAMED can only read common image formats (JPEG, PNG, WEBP, TIFF, BMP)."}), 400

    # ✅ Use centralized runtime directory from vision.py
    from framed.analysis.vision import UPLOAD_DIR
    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    image_path = os.path.join(upload_dir, unique_name)

    try:
        file.save(image_path)
        
        # Generate photo_id and pass filename for canonical schema
        photo_id = str(uuid.uuid4())
        analysis_result = run_full_analysis(image_path, photo_id=photo_id, filename=safe_name)

        # Build a presentation-friendly view without mutating the core result
        ui_view = clean_result_for_ui(analysis_result)
        response_payload = dict(analysis_result)
        
        # === EXPRESSION LAYER (Model B) ===
        # Phase 3: Expression Layer - Transform intelligence output into poetic critique
        # Use new expression layer if intelligence output is available, otherwise fallback to legacy
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
        
        # Phase 5: Reflection loop (self-validation)
        try:
            from framed.analysis.reflection import reflect_on_critique
            
            # Prefer intelligence output over old interpretive conclusions
            intelligence_output = analysis_result.get("intelligence", {})
            interpretive_conclusions = analysis_result.get("interpretive_conclusions", {})
            
            # Use intelligence output if available, otherwise fallback to old conclusions
            if intelligence_output and intelligence_output.get("recognition", {}).get("what_i_see"):
                # Use intelligence output for reflection
                reflection = reflect_on_critique(critique, intelligence_output)
            elif interpretive_conclusions:
                # Fallback to old interpretive conclusions
                reflection = reflect_on_critique(critique, interpretive_conclusions)
            else:
                reflection = None
            
            if reflection:
                response_payload["reflection_report"] = reflection
                
                # Regenerate if quality is too low (once only)
                if reflection.get("requires_regeneration", False):
                    _logger = current_app.logger
                    _logger.warning(f"Reflection: Regenerating critique (quality: {reflection.get('quality_score', 0.0):.2f})")
                    critique = generate_merged_critique(analysis_result, mentor_mode)
                    response_payload["critique"] = critique
                    if ui_view:
                        ui_view["critique"] = critique
                    
                    # Re-check reflection
                    reflection = reflect_on_critique(critique, interpretive_conclusions)
                    response_payload["reflection_report"] = reflection
        except Exception as e:
            current_app.logger.warning(f"Reflection loop failed (non-fatal): {e}")
            # Don't fail the request - reflection is optional
        
        if ui_view:
            response_payload["_ui"] = ui_view
        
        # Note: update_echo_memory is now called inside run_full_analysis
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
    return jsonify({"ok": True, "message":"ECHO history cleared"})

@main.post("/ask-echo")
def ask_echo_route():
    try:
        payload = request.get_json(force=True) or {}
        question = payload.get("question","").strip()
        if not question:
            return jsonify({"error":"Missing 'question'"}), 400
        mem = load_echo_memory()
        answer = ask_echo(question, mem, client)
        return jsonify({"answer": answer})
    except Exception as e:
        current_app.logger.exception("ECHO failed")
        return jsonify({"error": f"ECHO error: {e}"}), 500
