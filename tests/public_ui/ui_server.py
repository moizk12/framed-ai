"""Frontend-only test server; it intentionally implements no API routes."""
from pathlib import Path

from flask import Flask, render_template, send_from_directory

ROOT = Path(__file__).parents[2]
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"), static_url_path="/static")
app.add_url_rule("/static/<path:filename>", "main.static", lambda filename: send_from_directory(ROOT / "static", filename))
app.add_url_rule("/", "main.index", lambda: render_template("index.html"))
app.add_url_rule("/upload", "main.upload", lambda: render_template("index.html"))
app.add_url_rule("/privacy", "main.privacy", lambda: render_template("privacy.html"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4173, debug=False)
