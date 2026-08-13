"""Dependency-light public-page checks for the frontend-only test server."""
from pathlib import Path

from ui_server import app


ROOT = Path(__file__).parents[2]


def require(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


with app.test_client() as client:
    home = client.get("/")
    home_html = home.get_data(as_text=True)
    require("homepage responds", home.status_code == 200)
    require("homepage has one h1", home_html.count("<h1") == 1)
    require("public promise is present", "A visual-cognition companion for photographers" in home_html)
    require("legacy controls are absent", all(term not in home_html for term in ("AskECHO", "Remix", "trajectory", "cognition_run_purpose", 'action="/reset"')))
    require("balanced mode is fixed", 'name="mentor_mode" value="balanced"' in home_html)
    require("hero image is independent", "images/hero-photograph.jpg" in home_html)
    require("example image is independent", "images/example-landscape.jpg" in home_html)
    require("preview image is not on the public page", "images/preview-photograph.jpg" not in home_html)
    require("upload route is cohesive", client.get("/upload").status_code == 200)
    privacy = client.get("/privacy")
    require("privacy route responds", privacy.status_code == 200 and b"No public continuity" in privacy.data)

source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "static" / "js").glob("*.js"))
require("no HTML injection sinks", all(sink not in source for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write")))
client_source = (ROOT / "static" / "js" / "analysis-client.js").read_text(encoding="utf-8")
require("versioned analysis endpoint", '"/api/v1/analyses"' in client_source)
require("versioned feedback endpoint", '"/api/v1/feedback"' in client_source)

print('{"passed":13,"suite":"public-ui-smoke"}')
