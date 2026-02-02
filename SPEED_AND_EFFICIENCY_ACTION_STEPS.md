# FRAMED — Speed & Efficiency Action Steps

**Goal:** Make FRAMED faster and more efficient on upload (e.g. Hugging Face) **without compromising behavior or losing progress.**

**Implemented (2026-01-29):**
- **Step 1 — Expression cache:** Key = `(intelligence_output, mentor_mode, EXPRESSION_CACHE_VERSION, HITL calibration mtime)`. Cache in `expression_cache/`. Bump `EXPRESSION_CACHE_VERSION` or change HITL calibration to invalidate.
- **Step 2 — Parallel perception:** CLIP, YOLO, NIMA, color, lighting, objects, lines, tonal, subject_emotion, clutter, visual_evidence run in `ThreadPoolExecutor(max_workers=PERCEPTION_MAX_WORKERS)`. Derived steps (interpret_visual_features, infer_emotion, detect_genre) run in parallel. Set `FRAMED_PERCEPTION_WORKERS` (default 4) to cap GPU load.
- **Step 3 — Combined layers 2–7:** Feature flag `FRAMED_COMBINED_LAYERS_2_7` (default: true). One Model A call for layers 2–7; strict schema validation; fallback to 6 separate calls on failure. Recognition is read-only evidence; combined call does not modify it.
- **Step 4 — Token and regeneration:** Expression `max_tokens` 1500. Regeneration cap: `FRAMED_MAX_REGENERATIONS` (default 1). Set to 2 to allow a second regeneration attempt.

---

## Where Time Goes Today

| Stage | What happens | Approx. cost |
|-------|----------------|--------------|
| **Vision (perception)** | CLIP, YOLO, NIMA, color, lighting, symmetry, etc. — all **sequential** | 5–15+ s (model loads + inference) |
| **Intelligence core** | **7 sequential LLM calls** (Model A) — Layers 1–7 | 7 × round-trip (dominant) |
| **Expression** | 1 LLM call (Model B) | 1 × round-trip |
| **Reflection** | Rule-based (no LLM), then up to 2 regeneration loops | 0 LLM + up to 2× (expression + reflection) |
| **Cache** | By file hash; critique **not** cached — expression runs every time even when analysis is cached | Repeat uploads save vision+intelligence only |

**Bottlenecks:** (1) Seven Model A calls in a row. (2) Perception steps run one after another. (3) No expression cache, so Model B runs on every request even when intelligence is cached.

---

## Tier 1 — Highest Impact, No Compromise

### 1. Combine intelligence core into fewer LLM calls (largest win)

**Current:** 7 sequential Model A calls (recognition → meta_cognition → temporal → emotion → continuity → mentor → self_critique).

**Change:** Keep Layer 1 (recognition) as-is. **Combine Layers 2–7 into one structured Model A call** that returns a single JSON with: `meta_cognition`, `temporal`, `emotion`, `continuity`, `mentor`, `self_critique`. Same schema, same inputs (recognition + temporal_memory + user_history), one round-trip instead of six.

- **Preserves:** All 7 layer outputs; expression and reflection still consume the same structure.
- **Risk:** One very structured prompt; needs clear JSON schema and validation. Fallback: if combined call fails or malforms, retry with current 6-call sequence.
- **Rough gain:** ~6× fewer Model A round-trips for the “tail” of the pipeline.

**Alternative (smaller step):** Combine only 2+3+4 (meta_cognition, temporal, emotion) into one call, and 5+6+7 (continuity, mentor, self_critique) into another. That gives 1 + 1 + 1 + 1 = 4 Model A calls instead of 7.

---

### 2. Cache expression by (intelligence_hash, mentor_mode)

**Current:** Critique is computed in `routes.py` after `run_full_analysis`. Even when analysis is cached (same image, same file hash), expression and reflection run every time.

**Change:** Before calling Model B, compute a stable hash of `(intelligence_output, mentor_mode)` (e.g. SHA256 of canonical JSON). If a cache entry exists for that key, return the cached critique and skip Model B. Optionally run reflection on the cached critique (cheap, no LLM). Store cache in the same area as analysis cache (or a dedicated `expression_cache/`), with same versioning/invalidation as analysis cache.

- **Preserves:** Same critique for same intelligence + mode; behavior unchanged.
- **Gain:** Repeat requests (same image or same analysis cache hit) skip Model B entirely.

---

### 3. Parallelize perception in `vision.py`

**Current:** CLIP, then CLIP inventory, then NIMA, then color, color_harmony, YOLO, lines/symmetry, lighting, tonal range, subject_emotion, etc. — all in sequence.

**Change:** Run independent steps in parallel. For example:
- **Group A (all need only image path):** `get_clip_description`, `get_clip_inventory`, `analyze_color`, `analyze_color_harmony`, `detect_objects_and_framing`, `analyze_lines_and_symmetry`, `analyze_lighting_direction`, `analyze_tonal_range`, `analyze_subject_emotion`, `extract_visual_features`.
- Use `concurrent.futures.ThreadPoolExecutor` (or asyncio if the stack is async). Wait for all, then build `result` from the outputs as today. Keep `safe_analyze` so one failure doesn’t kill the rest.

- **Preserves:** Same inputs and outputs; only order and concurrency change.
- **Gain:** Perception stage time ≈ slowest of the parallel tasks instead of sum of all (often 2–4× faster).

**Note:** NIMA and CLIP may share a GPU; cap concurrency (e.g. 2–4 workers) to avoid OOM. Optionally run “heavy” (CLIP, YOLO, NIMA) in parallel and “light” (color, lines, etc.) after or in a second parallel batch.

---

## Tier 2 — Good Impact, Low Risk

### 4. Ensure analysis cache is used on HF

**Current:** Cache key is file hash; `disable_cache` is not set in routes, so cache is used when the same image bytes are seen again. On HF, the first upload of an image is always cold.

**Change:**
- Ensure cache directory is writable and, if possible, persistent (HF Spaces can have ephemeral disk; document “first run per image is slow”).
- In routes, explicitly pass `disable_cache=False` (or read from config) so caching is clearly the default for production.
- Optionally add a short doc or comment: “Cache is by content hash; identical image re-upload is fast.”

---

### 5. Cap reflection regeneration

**Current:** Up to 2 regeneration attempts (expression + reflection again each time).

**Change:** Reduce to **1** regeneration by default (configurable). Or keep 2 but only trigger the second if quality_score &lt; 0.4 (first regeneration already attempted). Lowers worst-case latency with minimal impact on quality.

---

### 6. Slightly shorter prompts / token caps

**Current:** Large prompts and high `max_tokens` in intelligence and expression.

**Change:** Trim redundant instructions; set `max_completion_tokens` (or equivalent) to the minimum needed per layer (e.g. 400–800 for recognition, 600 for meta_cognition, etc.). Same structure, fewer tokens → faster and cheaper.

---

## Tier 3 — Optional / Later

### 7. Progressive response (UX only)

Return quickly with “Analyzing…” or a job id, then poll or stream. Doesn’t reduce server work but improves perceived speed. Only if the HF app supports background jobs or SSE.

### 8. Lighter model for downstream layers (optional)

If you ever split Layer 2–7 into one or two calls, you could use a smaller/cheaper model for the “tail” (e.g. continuity, mentor, self_critique) with a fallback to the main model. Document as optional and A/B test; slight behavior change possible.

### 9. Expression streaming

Stream Model B output token-by-token so the user sees the critique as it’s generated. Same total time, better perceived responsiveness.

---

## Suggested order of implementation

1. **Expression cache** — Small, localized change; no change to intelligence or perception; big win on repeat uploads and cached analysis.
2. **Parallelize perception** — Confined to `vision.py`; same API and schema; clear latency win.
3. **Combine Layers 2–7** — Biggest win; do after 1–2, with a fallback to current 7-call path if the combined call fails or returns invalid schema.
4. **Tier 2 items** — Cache behavior on HF, regeneration cap, token caps — as you have time.

---

## What not to do (preserve progress)

- **Don’t** remove or bypass the plausibility gate, confidence governor, or HITL integration.
- **Don’t** change the 7-layer *schema* consumed by expression and reflection; only change *how* those layers are produced (fewer calls, same structure).
- **Don’t** drop reflection or regeneration without a product decision; at most reduce regeneration count or tighten thresholds.
- **Don’t** parallelize the intelligence layers themselves (2–7 depend on previous outputs); only combine them into fewer calls.

This keeps FRAMED’s behavior and architecture intact while making it faster and more efficient end-to-end.
