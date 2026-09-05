# Public Body verification — 2026-09-05

## Baseline

- Canonical base: `a188e001d4325a30dc15f5c3864ba3d96ebcfe8c` (`origin/main`, PR #10 merged).
- Focused branch: `public-body/video-ready`, isolated under `C:/Dev/FRAMED/active/public-body-video-ready`.
- Original inspected Track A checkout: `3dd9d93b74eff67b7099d36dfb1741620f342e5d`.
- HF public Space metadata reports running Docker revision `1c959b2bb3f02dc716662205a6ddb696e46ebbe8`, CPU basic.
  Its `/version` returned 404. Its Dockerfile uses unlocked requirements, root execution,
  and a 120-second Gunicorn setting; it is not this canonical public runtime.
- PR #9 remains open at `13b9264da7313bb9bb6061c3d828c4793dbc2795`; no Slice B files changed.

## Implemented and checked

- Captured cloud Responses/Chat/fallback clients carry actual image bytes for recognition;
  later reasoning and Model B remain text only. No paid API used.
- Public reasoning excludes temporal/trajectory/past-error layers, research context,
  calibration/rules retrieval, caches and cognitive/legacy persistence. Behavioral tests
  execute the public pipeline with forbidden research hooks instrumented.
- Strict recognition shape validation; empty expression fails rather than generating
  fallback critique; provider errors are sanitized at the public API; no successful
  analysis record on failure. Public expression has an explicit standalone contract.
- Four existing deterministic signals; unavailable values omitted; heuristic grounding
  hidden while preserving disabled/empty API states.
- Shared admission limit, six attempts per ten minutes by default; probes/static unaffected.
- Browser 300 seconds, worker 360 seconds, configurable with validation. Browser abort
  does not cancel Python work; gthread timeout is not a per-request execution deadline.
- Serialized process-level model loading, serving-worker preload and inference prerequisites
  in production readiness. Readiness cannot certify quota or every upstream response.
- Three distinct existing photographs; provenance in `static/images/ASSETS.md`.
- Current privacy/storage copy and concise Research/Public Body README.
- Four classified disposable placeholder runs removed; frozen/claim-supporting evidence preserved.

## Local acceptance evidence

- Full pytest: **207 passed**, 11 dependency/deprecation warnings (46.91 seconds).
- Browser harness: **62 checks passed**, 320/375/768/1280 CSS pixels.
  Landing, upload, loading, success, measured signals, evidence, limitations, feedback,
  429 and 503 states at every width; additional validation, keyboard, reduced-motion,
  cancellation, retry, XSS, malformed-response and error checks.
- Screenshots visually inspected; empty evidence-grid cells corrected. Browser fixtures
  test presentation. Real local inference is separately measured below.
- Bandit public-path scan: no new actionable issue; existing denied development-secret
  comparison and two legacy-route exception-pass findings remain outside this change.
- New modules/test/benchmark lint checked; no global Ruff cleanup.

Real local inference used Qwen2.5-VL recognition and `qwen/qwen3-14b` text expression
through LM Studio. Same immutable image SHA256:
`c5e082789de0f59bb4883f16fa1f33fdb27cb3e282a0ba9e6b8d7393801b7b7c`.

- Process-cold: **58.70 s**, complete public critique.
- Warm: **52.16 s**, complete public critique.
- Existing disk model caches retained. These are local-machine timings, not HF CPU-basic
  or empty-volume download timings. YOLO/CLIP initialized once and reused.
- Reproduce: `python scripts/benchmark_public.py static/images/example-landscape.jpg`.
- Local outputs: ignored `test-results/inference-latency.json`,
  `test-results/public-ui/browser-checks.json`, screenshots and `public-bandit.json`.

## Deployment boundary

No HF credential was available to this task. No remote preview or production Space
was modified. Docker Desktop/WSL are absent locally; the existing GitHub CI workflow
is the Linux/PostgreSQL/container acceptance path. Its status is reported with the
final branch SHA, not inferred from local tests.

README contains the exact Docker build/run commands and required configuration.
For HF, create a separate Docker preview Space, export the clean branch HEAD, add
`sdk: docker` and `app_port: 7860` to its README YAML, and upload that build context.
Set `FRAMED_BUILD_SHA` to that HEAD and `FRAMED_VERSION=public-beta`; HF passes these
Space variables to the declared Docker build arguments. Set PostgreSQL/secret/provider
configuration using Space Settings, never in the source tree. A local-only LM Studio
URL is not reachable from HF. Keep the research database separate.

After credentials and preview infrastructure exist, the upload action is:

```sh
hf upload "$PREVIEW_SPACE" "$EXPORTED_CLEAN_HEAD_CONTEXT" . --repo-type space
```

Post-deploy: `/version` must equal the uploaded canonical HEAD; `/health` and `/ready`
must succeed; multipart `/api/v1/analyses` must return a nonempty critique and real
measured signals. Verify feedback and sanitized failure/capacity states at desktop
and mobile widths. Do not treat a build-only CI success as deployed inference proof.
