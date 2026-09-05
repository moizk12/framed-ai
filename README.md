# FRAMED

FRAMED investigates whether one persistent artificial visual mind can develop
measurable, cumulative, transferable competence through its own visual lifetime.
This **Developmental Visual Intelligence (DVI)** thesis is a research question,
not a demonstrated result. Photography is the first developmental world.

## FRAMED Research

The research programme separates experience, provisional memory, deliberation,
feedback, evaluation and controlled promotion. Local, open-weight, self-hostable,
provider-replaceable intelligence is an architectural requirement.

Built and merged:

- **Slice A persistent cognition**, frozen at `slice-a-persistent-cognition-v1`:
  provenance-preserving experience storage and retrieval affecting later
  deliberation, with replay and isolation checks in `framed/cognition/tests`.
- **Multimodal Model A**, merged in [PR #10](https://github.com/moizk12/framed-ai/pull/10):
  generic OpenAI-compatible image input, including the local Qwen2.5-VL path.
  Only recognition receives the photograph. Later deliberation and Model B
  receive text. Provider contract tests live in `tests/test_local_openai_multimodal.py`
  and `tests/test_public_body.py`.
- The public beta/API and its separate persistence boundary.

**Slice B controlled learning** remains separate work in
[PR #9](https://github.com/moizk12/framed-ai/pull/9), outside this public-body change.
Cumulative development, general transfer, and the full DVI thesis remain to be
established by controlled experiments. Historical calibration reports are bounded
observations, not evidence that the whole thesis is proven.

There are no AGI, consciousness or sentience claims. Research claims must stay
within their evidence, and public and research cognition remain isolated.

## FRAMED Public Body

The photography beta is one public surface of the programme:

`image → measured perception → Model A recognition → standalone text reasoning → Model B critique`

Upload one JPEG, PNG or WebP (12 MiB default), receive a Balanced Mentor critique,
inspect evidence and limitations, and optionally attach feedback. The Flask UI
uses server-rendered HTML, CSS and progressive vanilla JavaScript.

- Measured Signals expose existing brightness, tonal spread, edge response and
  green-pixel measurements. They are not quality scores or semantic conclusions.
- Heuristic grounding stays internal; unavailable grounding does not create an
  empty evidence row. Disabled and enabled-but-empty remain distinct in the API.
- Public requests neither retrieve nor write research memory, personal trajectory,
  legacy learning or expression caches. Errors never become successful critiques.
- Shared process-level rate limiting defaults to six admission attempts per ten
  minutes. It protects `/api/v1/analyses`, including invalid attempts, without
  trusting spoofable forwarded IP headers. Limits reset on process restart; use
  one worker/replica for this beta. Multiple replicas need shared admission control.
- The public database stores analysis IDs/timestamps and feedback, not raw uploads
  or critique text. Temporary uploads are deleted on request completion where
  possible. See the public `/privacy` page for interruption, logging and provider limits.

## Run and verify

Use Python 3.11 on Linux for the production dependency lock:

```sh
python -m venv .venv
. .venv/bin/activate
pip install --require-hashes -r requirements.lock
pip install pytest
pytest -q
```

Set `DATABASE_URL` (PostgreSQL) and a `SECRET_KEY` of at least 32 characters.
Set `FRAMED_ENV=production`, `FRAMED_PUBLIC_BETA_ONLY=true`,
`FRAMED_COGNITION_V1=false`, and immutable `FRAMED_BUILD_SHA`/`FRAMED_VERSION`.
For local models set `FRAMED_LOCAL_BASE_URL` to the reachable OpenAI-compatible
`/v1` endpoint, `FRAMED_LOCAL_MODEL_A` to a vision-capable model ID and
`FRAMED_LOCAL_MODEL_B` to a text model ID. `FRAMED_STRICT_LOCAL=true` enforces the
local path. No OpenAI account or funded key is required for local providers.
Cloud providers are optional; their image contract is tested with captured clients.

```sh
python -m framed.public_migrations
python run.py
```

`pipeline.py` in `framed/analysis` is canonical orchestration. `vision.py` remains a
legacy compatibility surface for older research callers; the public API imports
`pipeline.py` directly. Do not use the legacy `/analyze` route for public traffic.

The isolated fixture preview is `python tests/public_ui/preview_server.py` at
`http://127.0.0.1:4173`; it is explicitly not live inference. Browser regression
checks are in `tests/public_ui/browser_checks.js`.

## Container / Hugging Face preview

The Docker image installs `requirements.lock`, runs migrations, and starts one
Gunicorn process. Build a known clean commit; unknown build metadata is rejected:

```sh
SHA=$(git rev-parse HEAD)
docker build --build-arg FRAMED_VERSION=public-beta --build-arg FRAMED_BUILD_SHA="$SHA" -t framed-public-beta:"$SHA" .
docker run --rm --env-file /secure/path/framed-preview.env -p 7860:7860 framed-public-beta:"$SHA"
```

The environment file needs PostgreSQL, the secret key and reachable provider
settings above. In a container, `localhost` means the container itself; point
local/self-hosted providers at an actually reachable host. Use a dedicated preview
PostgreSQL database, separate from research. `FRAMED_DATA_DIR=/data/framed` holds
model/cache files. Provision persistent cache storage for predictable starts.

For a **separate HF Docker preview Space**, upload this clean commit's build
context, set Space SDK to Docker and port to 7860, and supply the same environment
variables/secrets. Set the Space variable `FRAMED_BUILD_SHA` to the canonical SHA;
[HF passes variables as Docker build arguments](https://huggingface.co/docs/hub/en/spaces-sdks-docker).
The HF
Space repository revision is a different identifier from the canonical build SHA.
Do not overwrite the production Space as part of preview verification.

Gunicorn preloads YOLO/CLIP once per serving worker (`PUBLIC_PRELOAD_MODELS=true`).
Loading is serialized to avoid duplicate initialization on concurrent cold calls.
`/health` is liveness; production `/ready` requires database, loaded vision models
and available configured providers. It is a prerequisite check, not proof that
an upstream funded account will accept its next inference request.

Budgets are centralized in `framed/public_runtime.py` and `gunicorn.conf.py`:
`PUBLIC_ANALYSIS_TIMEOUT_SECONDS=300`, `PUBLIC_WORKER_TIMEOUT_SECONDS=360`
(minimum 30-second margin), `PUBLIC_RATE_LIMIT=6`, `PUBLIC_RATE_WINDOW_SECONDS=600`.
The browser receives the configured budget from HTML. Browser abort does not
cancel Python inference; Gunicorn gthread timeout is worker-health supervision,
not a hard per-request deadline. Prewarm and measure on deployment hardware.

Post-deploy checks (replace URL with the actual preview URL):

```sh
curl -fsS "$PREVIEW_URL/version"  # build_sha must equal SHA above
curl -fsS "$PREVIEW_URL/health"
curl -fsS "$PREVIEW_URL/ready"
curl -fsS -F image=@static/images/example-landscape.jpg "$PREVIEW_URL/api/v1/analyses"
```

Inspect desktop/mobile upload, loading, critique, measured signals, limitations,
feedback, sanitized provider failures and capacity rejection. Tests alone do not
prove live readiness. CI runs pytest, PostgreSQL bootstrap and the production
container build. Public photograph provenance is in `static/images/ASSETS.md`.
