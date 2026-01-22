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
        return jsonify({"error":"No file uploaded"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error":"Unsupported file type"}), 400

    # ✅ Use centralized runtime directory from vision.py
    from framed.analysis.vision import UPLOAD_DIR
    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    image_path = os.path.join(upload_dir, unique_name)

    try:
        file.save(image_path)
        analysis_result = run_full_analysis(image_path)
        # Re-generate critique only if the mentor mode is non-default
        if mentor_mode and mentor_mode != "Balanced Mentor":
            analysis_result["critique"] = generate_merged_critique(analysis_result, mentor_mode)
        update_echo_memory(analysis_result)  # keep the last 10
        return jsonify(analysis_result)
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
