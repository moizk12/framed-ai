# Calibration Protocol — Step-by-Step

**Status:** Steps 8.2–8.4 and Phase 9 completed (2026-02-01).  
Phase 10–11 next: lock HITL limits, choose next frontier.

Run these commands **in the terminal where you set your API key** (or use `.env`).

---

## API Key Verification

**Option A — Terminal session (you did this):**
```powershell
$env:OPENAI_API_KEY = "sk-your-key"
```

**Option B — Persistent via .env:**
1. Copy `.env.example` to `.env`
2. Edit `.env` and set `OPENAI_API_KEY=sk-your-key`
3. Verify: `python scripts/check_api_key.py`  
   Should print: `OK: OPENAI_API_KEY is set (XX chars)`

---

## Calibration Micro-Set (10–15 images)

Located at: `calibration_micro_set/`

| Category   | Images                  | Intent                       |
|-----------|-------------------------|------------------------------|
| ambiguous | 3 (abstract, conceptual, minimal) | Clearly ambiguous         |
| architecture | 3 (building, facade, structure) | Overconfident failure risks |
| artistic  | 1                        | Mentor-tone risk             |
| mixed     | 3 (complex, mixed, scene) | Mixed-signal scenes        |
| nature    | 2 (landscape, tree)      | Emotionally subtle           |
| interiors | 1 (room)                 | Emotionally subtle           |
| portraits | 1                        | Emotionally subtle           |

**Total: 14 images.** Stress belief formation, not accuracy.  
**Not** the stress-test dataset.

---

## Step 8.2 — Run FRAMED normally (expression ON)

Let FRAMED speak. Do not guide it yet.

```powershell
cd C:\Users\moizk\Downloads\framed-clean
python scripts/run_calibration_protocol.py step_8_2
```

Or directly:
```powershell
python -m framed.tests.test_intelligence_pipeline --dataset_path calibration_micro_set --shuffle --seed 42 --disable_cache
```

**Save the run directory** (e.g. `framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS`) for Step 8.3 (extract signatures) and Step 8.4 (comparison).

---

## Step 8.3 — Inject HITL feedback (surgically)

For each image, give **at most ONE** correction.  
Sparse feedback > dense feedback.  
Use only: `overconfidence`, `missed_alternative`, `emphasis_misaligned`, `mentor_failure`.

**Get pattern signatures** from the run_8_2 output:

```powershell
python scripts/extract_hitl_signatures.py framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS
```

This prints `image_id,pattern_signature` for each image. Use the signature in HITL submit.

**Suggested mapping** (adjust after you see run_8_2 outputs). Replace `SIG` with the signature from `extract_hitl_signatures.py`:

| Image ID              | Feedback Type        | Example CLI |
|-----------------------|----------------------|-------------|
| ambiguous_sample_abstract_001 | missed_alternative | `python -m framed.feedback.submit -i ambiguous_sample_abstract_001 -t missed_alternative -s SIG -a "reflection vs texture"` |
| ambiguous_sample_conceptual_003 | overconfidence | `python -m framed.feedback.submit -i ambiguous_sample_conceptual_003 -t overconfidence -s SIG -c -0.15` |
| ambiguous_sample_minimal_002 | overconfidence | `python -m framed.feedback.submit -i ambiguous_sample_minimal_002 -t overconfidence -s SIG` |
| architecture_sample_facade_003 | overconfidence | `python -m framed.feedback.submit -i architecture_sample_facade_003 -t overconfidence -s SIG` |
| artistic_sample_artistic_001 | mentor_failure | `python -m framed.feedback.submit -i artistic_sample_artistic_001 -t mentor_failure -s SIG -r "generic guidance"` |
| mixed_sample_complex_002 | emphasis_misaligned | `python -m framed.feedback.submit -i mixed_sample_complex_002 -t emphasis_misaligned -s SIG -d emotional_weighting` |

**Minimal injection (3–5 images):** Pick 3–5 images from run_8_2 that most need calibration. One feedback each.

---

## Step 8.4 — Re-run the same images

Compare: confidence deltas, hypothesis count changes, language softening, mentor sharpness.

```powershell
python scripts/run_calibration_protocol.py step_8_4 --run_dir framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS
```

Or:
```powershell
python -m framed.tests.test_intelligence_pipeline --dataset_path calibration_micro_set --shuffle --seed 42 --disable_cache
```

**Compare runs:**
```powershell
python scripts/compare_calibration_runs.py framed/tests/test_runs/run_8_2 framed/tests/test_runs/run_8_4
```

**Success check:**
- If FRAMED changes **globally** (all images) → something is wrong.
- If FRAMED changes **locally** (only signature-matched images) → correct.

---

## Phase 9 — HITL stability test ✅ DONE

Fresh 20–30 image batch. **No HITL feedback.**

```powershell
python scripts/run_calibration_protocol.py phase_9
```

Or:
```powershell
python -m framed.tests.test_intelligence_pipeline --dataset_path stress_test_master/dataset_v1_ext --max_images 25 --shuffle --seed 99 --disable_cache
```

**Completed:** run_2026_02_01_120045 (25 images, 0 failures, avg confidence 0.62).

**Dataset v2:** 36 new images in `stress_test_master/dataset_v2` (run `python scripts/download_dataset_v2.py` to refresh).

**Verify:**
- No confidence collapse ✓
- No generic hedging ✓
- No mentor paralysis ✓  
**Success =** FRAMED is humbler without becoming timid.

---

## Phase 10 — Lock the learning boundary ← YOU ARE HERE

See **HITL_LIMITS.md** — already created.  
This is FRAMED’s constitution for HITL.

---

## Phase 11 — Decide the next frontier (choose ONE)

After HITL stabilizes, pick **one** direction:

| Option | Direction |
|--------|-----------|
| **A** | Perceptual depth — better visual primitives (texture, geometry, light logic) |
| **B** | Mentor intelligence — long-arc guidance, "I've seen you do this before…" |
| **C** | Temporal selfhood — FRAMED narrates its own evolution |

**Do not do all three.**  
Document your choice in `NEXT_FRONTIER.md` when ready.
