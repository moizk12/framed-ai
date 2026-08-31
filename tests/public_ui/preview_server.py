"""DEV/TEST-ONLY preview server.

Serves production templates and static assets, plus fixture-backed
/api/v1/analyses and /api/v1/feedback. Production runtime must never
import this module.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "static" / "fixtures"
PREVIEW_COOKIE = "framed_preview_scenario"
DEFAULT_SCENARIO = "success"
SCENARIOS = {
    "success": ("analysis-success.json", 200, 1.1),
    "empty-evidence": ("analysis-empty-evidence.json", 200, 1.1),
    "413": ("error-413.json", 413, 0.25),
    "429": ("error-429.json", 429, 0.25),
    "500": ("error-500.json", 500, 0.25),
    "503": ("error-503.json", 503, 0.25),
    "timeout": ("error-timeout.json", 504, 1.4),
}

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
    static_url_path="/static",
)
app.add_url_rule("/static/<path:filename>", "main.static", lambda filename: send_from_directory(ROOT / "static", filename))
app.add_url_rule("/", "main.index", lambda: render_template("index.html"))
app.add_url_rule("/upload", "main.upload", lambda: render_template("index.html"))
app.add_url_rule("/privacy", "main.privacy", lambda: render_template("privacy.html"))


def _requested_scenario() -> str:
    raw = (
        request.args.get("scenario")
        or request.cookies.get(PREVIEW_COOKIE)
        or os.environ.get("FRAMED_PREVIEW_SCENARIO")
        or DEFAULT_SCENARIO
    ).strip().lower()
    return raw if raw in SCENARIOS else DEFAULT_SCENARIO


def _fixture_payload(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@app.post("/api/v1/analyses")
def preview_analyses():
    filename, status, delay = SCENARIOS[_requested_scenario()]
    if os.environ.get("FRAMED_PREVIEW_NO_DELAY", "").lower() not in ("1", "true", "yes"):
        time.sleep(delay)
    return _fixture_payload(filename), status


@app.post("/api/v1/feedback")
def preview_feedback():
    return _fixture_payload("feedback-success.json"), 200


@app.after_request
def attach_preview_chrome(response):
    scenario = _requested_scenario()
    response.set_cookie(PREVIEW_COOKIE, scenario, httponly=False, samesite="Lax")
    content_type = response.headers.get("Content-Type", "")
    if response.status_code == 200 and "text/html" in content_type:
        html = response.get_data(as_text=True)
        if "</body>" in html and "data-preview-harness" not in html:
            html = html.replace("</body>", _preview_bar(scenario) + "</body>", 1)
            response.set_data(html)
            response.headers["Content-Length"] = str(len(response.get_data()))
    return response


def _preview_bar(current: str) -> str:
    options = "".join(
        f'<option value="{name}"{" selected" if name == current else ""}>{name}</option>'
        for name in SCENARIOS
    )
    return f"""
<aside data-preview-harness style="position:fixed;z-index:80;right:1rem;bottom:1rem;max-width:min(22rem,calc(100vw - 2rem));padding:.85rem 1rem;border:1px solid #20221e;background:#faf8f3;color:#10110f;font:650 .78rem/1.4 Inter,sans-serif;box-shadow:0 8px 24px rgba(16,17,15,.12);">
  <p style="margin:0 0 .45rem;letter-spacing:.08em;text-transform:uppercase;">Preview only</p>
  <p style="margin:0 0 .65rem;font-weight:400;color:#696b64;">Fixture API. Production still calls /api/v1/analyses.</p>
  <form method="get" action="/" style="display:flex;gap:.5rem;align-items:center;">
    <label for="preview-scenario">Scenario</label>
    <select id="preview-scenario" name="scenario" style="min-height:2.75rem;padding:.35rem .5rem;">{options}</select>
    <button type="submit" style="min-height:2.75rem;padding:.35rem .8rem;border:1px solid #10110f;background:#10110f;color:#faf8f3;cursor:pointer;">Use</button>
  </form>
</aside>
"""


if __name__ == "__main__":
    port = int(os.environ.get("FRAMED_PREVIEW_PORT", "4173"))
    app.run(host="127.0.0.1", port=port, debug=False)
