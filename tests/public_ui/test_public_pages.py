from pathlib import Path

import pytest

from framed import create_app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr("framed.analysis.vision.ensure_directories", lambda: None)
    return create_app({"TESTING": True}).test_client()


def test_public_homepage_has_one_h1_and_beta_journey(client):
    response = client.get("/")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert html.count("<h1") == 1
    assert "A visual-cognition companion for photographers" in html
    assert "Critique a photograph" in html
    assert "What FRAMED noticed" in html
    assert "/api/v1/analyses" not in html
    assert "AskECHO" not in html
    assert "mentor_mode\" value=\"balanced" in html
    assert "images/hero-photograph.jpg" in html
    assert "images/example-landscape.jpg" in html
    assert "images/preview-photograph.jpg" not in html
    assert "data-preview-harness" not in html


def test_upload_route_serves_same_cohesive_experience(client):
    response = client.get("/upload")
    assert response.status_code == 200
    assert "id=\"critique\"" in response.get_data(as_text=True)


def test_privacy_page_is_simple_get_only(client):
    response = client.get("/privacy")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "What happens to your photograph" in html
    assert "No public continuity" in html


def test_dynamic_renderers_do_not_use_html_injection_sinks():
    static_js = Path(__file__).parents[2] / "static" / "js"
    source = "\n".join(path.read_text(encoding="utf-8") for path in static_js.glob("*.js"))
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert forbidden not in source


def test_production_client_targets_only_future_public_contract():
    source = (Path(__file__).parents[2] / "static" / "js" / "analysis-client.js").read_text(encoding="utf-8")
    assert '"/api/v1/analyses"' in source
    assert '"/api/v1/feedback"' in source
    assert "/analyze" not in source
    assert "/feedback\"" not in source.replace('/api/v1/feedback"', "")
    assert "framed_preview_scenario" not in source
    assert "preview_server" not in source


def test_preview_server_is_isolated_from_production_app(client, monkeypatch):
    import importlib.util

    monkeypatch.setenv("FRAMED_PREVIEW_NO_DELAY", "1")
    spec = importlib.util.spec_from_file_location(
        "framed_public_ui_preview_server",
        Path(__file__).with_name("preview_server.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with module.app.test_client() as preview:
        home = preview.get("/")
        html = home.get_data(as_text=True)
        assert home.status_code == 200
        assert "data-preview-harness" in html
        success = preview.post("/api/v1/analyses")
        assert success.status_code == 200
        body = success.get_json()
        assert body["status"] == "complete"
        assert body["meta"]["contract_version"] == "1"
        oversized = preview.post("/api/v1/analyses", query_string={"scenario": "413"})
        assert oversized.status_code == 413
        feedback = preview.post("/api/v1/feedback", json={"analysis_id": "ana-91d4f2a8", "useful": True})
        assert feedback.status_code == 200

    missing = client.post("/api/v1/analyses")
    assert missing.status_code == 404
