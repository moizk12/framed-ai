# Public UI tests

These tests keep frontend work independent of the future A2 implementation.

- `ui_server.py` serves pages and assets only; it intentionally implements no analysis or feedback API.
- `preview_server.py` is a DEV/TEST-ONLY interactive harness. It serves the same production templates and JavaScript, then answers `/api/v1/analyses` and `/api/v1/feedback` from `static/fixtures/`. Production runtime never imports it.
- `smoke_checks.py` verifies the page boundary, fixed Balanced mode, privacy route, versioned client endpoints, and absence of HTML-injection sinks.
- `browser_checks.js` intercepts `/api/v1/analyses` and `/api/v1/feedback` with reusable fixtures, exercises the complete journey and error states, and writes screenshots under `test-results/public-ui/`.
- `test_public_pages.py` provides equivalent Flask assertions for the repository's normal pytest environment.

Production browser code always calls the versioned public endpoints. There is no production demo-backend path.

Interactive preview (not used by CI):

```text
python tests/public_ui/preview_server.py
http://127.0.0.1:4173/
```

Switch fixture scenarios with `?scenario=success` (or `empty-evidence`, `413`, `429`, `500`, `503`, `timeout`) or the preview-only control on the page.
