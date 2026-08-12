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
