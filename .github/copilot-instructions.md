## Copilot / AI Agent Instructions for FRAMED

Purpose: Give short, actionable guidance so an AI coding agent can be productive immediately.

- **Big picture:** FRAMED is a Flask-based image-analysis mentor combining pixel-level analysis (OpenCV, YOLOv8) with semantic embeddings (CLIP) and an LLM-based reasoning layer. The HTTP routes live in `framed/routes.py` and call the central analysis pipeline in `framed.analysis.vision` which orchestrates `run_full_analysis` and then the intelligence layers.

- **Key files to read first:**
  - [framed/analysis/intelligence_core.py](framed/analysis/intelligence_core.py#L1-L20) — the 7-layer reasoning design and the critical invariant about memory vs. LLM.
  - [framed/analysis/llm_provider.py](framed/analysis/llm_provider.py#L1-L80) — `LocalOpenAICompatProvider` (LM Studio default), `OpenAIProvider` (optional cloud), `MODEL_CONFIGS`, `call_model_a` / `call_model_b`.
  - [framed/routes.py](framed/routes.py#L1-L200) — HTTP endpoints and the presentation sanitiser `clean_result_for_ui`.
  - [scripts/run_single_image_once.py](scripts/run_single_image_once.py#L1-L120) — example CLI to run the full pipeline on one image (useful for quick integration tests).
  - [config.py](config.py#L1-L40) — canonical environment flags used across the app.

- **Critical conventions and constraints (must follow):**
  - NEVER put learning or memory updates into LLM prompts. All learning/state updates must happen in memory modules (`framed/analysis/temporal_memory.py`, `framed/analysis/learning_system.py`). See the top of `intelligence_core.py` for the explanation.
  - Two-model pattern: Model A = Reasoning (structured JSON expected), Model B = Expression (poetic critique). See `llm_provider.py` (MODEL_A_TYPE / MODEL_B_TYPE).
  - Prompts often expect JSON objects. Use the helper `_safe_parse_layer_json` in `intelligence_core.py` as reference for robust parsing (handle markdown/code fences and partial outputs).
  - Canonical schema: analysis results expose `perception`, `derived`, `intelligence`, etc. UI code reads `perception`/`derived`; prefer updating the canonical schema before changing UI sanitizers (`clean_result_for_ui`).

- **LM Studio (default local):** Start LM Studio, load a model (e.g. Qwen2.5-VL-7B-Instruct), enable local server (default **http://localhost:1234**). Set `FRAMED_LOCAL_BASE_URL` (default `http://localhost:1234/v1`), `FRAMED_LOCAL_API_KEY` (default `lm-studio`). Defaults: `FRAMED_MODEL_A`=`FRAMED_MODEL_B`=`LOCAL_QWEN25_VL_7B`. Swap to Gemma: set both to `LOCAL_GEMMA4_E4B` or override ids with `FRAMED_LOCAL_MODEL_A` / `FRAMED_LOCAL_MODEL_B`. `FRAMED_STRICT_LOCAL=true` fails fast if the server or model id is missing (no silent placeholder).
- **Environment & feature flags:**
  - `OPENAI_API_KEY` — optional; only used when `FRAMED_MODEL_A`/`B` select `GPT_5_2` / `GPT_5_MINI`.
  - `FRAMED_MODEL_A` / `FRAMED_MODEL_B` — config keys in `MODEL_CONFIGS` (default local Qwen).
  - `DEEPFACE_ENABLE` — optional DeepFace.
  - `FRAMED_MAX_REGENERATIONS` — reflection loop (see `routes.py`).

- **Developer workflows & commands (examples):**
  - Local dev (venv + run):
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    python run.py
    ```
  - Quick, non-interactive pipeline run for one image (useful for debugging):
    ```bash
    python scripts/run_single_image_once.py --image_path stress_test_master/dataset_v2/ambiguous/v2_ambiguous_002.jpg
    ```
  - Tests: `pytest` (see `tests/test_health.py`, `framed/tests/`).
  - Local A/B (two models, five tasks): `python scripts/ab_test_local_models.py --image path/to.jpg` (writes under `eval_outputs/`).

- **How to change/replace LLM implementations:**
  - Modify `MODEL_CONFIGS` in `framed/analysis/llm_provider.py` to add a provider entry; implement a provider class (subclass `LLMProvider`) and wire it into `create_provider()`.
  - Keep the model interface stable: callers expect `call(prompt, system_prompt, max_tokens, temperature, response_format)` returning a dict with `content`, `usage`, `model`, and optional `error`.
  - Respect rate-limits and retry settings defined near the top of `llm_provider.py` (MAX_RETRIES, FALLBACK_MODEL_*).

- **Patterns for modifications and tests:**
  - Make small, local changes to `intelligence_core.py` or prompt templates; use `scripts/run_single_image_once.py` to validate behavior and inspect `confidence`, `alternatives_count`, and `critique_preview` in the printed JSON.
  - `PLACEHOLDER` exists only if explicitly selected; default path is LM Studio via `LocalOpenAICompatProvider`.

- **When editing output schema or UI sanitizers:**
  - Update the canonical schema first (analysis pipeline), then update `clean_result_for_ui` in `[framed/routes.py](framed/routes.py#L1-L200)` to maintain backwards compatibility. The UI code explicitly expects evidence vs. interpretation separation.

- **What to avoid:**
  - Do not embed persistent side effects inside LLM prompts. Memory updates must be explicit Python calls and saved via memory modules.
  - Don’t change model-selection globals without adjusting environment docs; CI and HuggingFace deployment rely on the Dockerfile and `requirements.txt`.

- **Next step:** If you'd like, I can add a short provider template (stub) for a new LLM service or create a PR checklist for swapping models safely.
