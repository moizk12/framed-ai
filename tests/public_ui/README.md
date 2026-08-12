# Public UI tests

These tests keep frontend work independent of the future A2 implementation. The test server serves pages and assets only; it intentionally implements no analysis or feedback API.

- `smoke_checks.py` verifies the page boundary, fixed Balanced mode, privacy route, versioned client endpoints, and absence of HTML-injection sinks.
- `browser_checks.js` intercepts `/api/v1/analyses` and `/api/v1/feedback` with reusable fixtures, exercises the complete journey and error states, and writes screenshots under `test-results/public-ui/`.
- `test_public_pages.py` provides equivalent Flask assertions for the repository's normal pytest environment.

Production browser code always calls the versioned public endpoints. There is no demo-backend path.
