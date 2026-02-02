# Calibration Protocol — Step-by-Step

**Status:** Steps 8.2–8.4 and Phase 9 completed. Phase 10 locked (HITL_LIMITS.md).  
Dataset v2: 36 images in `stress_test_master/dataset_v2`.  
**How we know results are right:** See **VALIDATION_AND_KNOWING.md** (manual pass + micro loop + metrics).

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

## Dataset (real photos only)

Use **dataset v2** only. Located at: `stress_test_master/dataset_v2/`

| Category     | Images (v2_*) | Intent                       |
|-------------|---------------|------------------------------|
| ambiguous   | 4             | Clearly ambiguous             |
| architecture| 5             | Overconfident failure risks   |
| artistic    | 4             | Mentor-tone risk              |
| mixed       | 4             | Mixed-signal scenes           |
| nature      | 5             | Emotionally subtle            |
| interiors   | 4             | Emotionally subtle            |
| portraits   | 5             | Emotionally subtle            |
| street      | 5             | Street/urban                  |

**Total: 36 images.** Real photos. No sample/solid-color images.

---

## Step 8.2 — Run FRAMED normally (expression ON)

Let FRAMED speak. Do not guide it yet.

```powershell
cd C:\Users\moizk\Downloads\framed-clean
python scripts/run_calibration_protocol.py step_8_2
```

Or directly:
```powershell
python -m framed.tests.test_intelligence_pipeline --dataset_path stress_test_master/dataset_v2 --shuffle --seed 42 --disable_cache
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
| ambiguous_v2_ambiguous_001 | missed_alternative | `python -m framed.feedback.submit -i ambiguous_v2_ambiguous_001 -t missed_alternative -s SIG -a "reflection vs texture"` |
| ambiguous_v2_ambiguous_002 | overconfidence | `python -m framed.feedback.submit -i ambiguous_v2_ambiguous_002 -t overconfidence -s SIG -c -0.15` |
| architecture_v2_architecture_001 | overconfidence | `python -m framed.feedback.submit -i architecture_v2_architecture_001 -t overconfidence -s SIG` |
| artistic_v2_artistic_001 | mentor_failure | `python -m framed.feedback.submit -i artistic_v2_artistic_001 -t mentor_failure -s SIG -r "generic guidance"` |
| mixed_v2_mixed_001 | emphasis_misaligned | `python -m framed.feedback.submit -i mixed_v2_mixed_001 -t emphasis_misaligned -s SIG -d emotional_weighting` |

**Minimal injection (3–5 images):** Pick 3–5 images from run_8_2 that most need calibration. One feedback each.

---

## Step 8.4 — Re-run the same images

Compare: confidence deltas, hypothesis count changes, language softening, mentor sharpness.

```powershell
python scripts/run_calibration_protocol.py step_8_4 --run_dir framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS
```

Or:
```powershell
python -m framed.tests.test_intelligence_pipeline --dataset_path stress_test_master/dataset_v2 --shuffle --seed 42 --disable_cache
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
python -m framed.tests.test_intelligence_pipeline --dataset_path stress_test_master/dataset_v2 --max_images 25 --shuffle --seed 99 --disable_cache
```

**Completed:** run_2026_02_01_120045 (25 images, 0 failures, avg confidence 0.62).

**Dataset v2:** 36 images in `stress_test_master/dataset_v2` (refresh: `python scripts/download_dataset_v2.py`).

**Verify:**
- No confidence collapse ✓
- No generic hedging ✓
- No mentor paralysis ✓  
**Success =** FRAMED is humbler without becoming timid.

---

## A. Manual reading pass (small but deep)

Pick 10–15 images only. Expression ON. Real GPT-5.2 + GPT-5-mini.

For each output, answer:
1. Did it hedge when it should have?
2. Did it surprise me in a way that felt earned?
3. Did it sound like a mentor, not a summarizer?

If any answer is **no** → submit HITL feedback. This is where FRAMED becomes yours.

```powershell
python scripts/run_manual_reading_pass.py
```

Open **MANUAL_READING_PASS_REPORT.md** in the run directory. Answer the 3 questions per image; use the printed pattern signatures to submit HITL when needed.

**How we know results are right:** See **VALIDATION_AND_KNOWING.md** — you validate by sample (this pass), behavior (micro loop), and metrics (big runs).

---

## B. HITL-heavy micro loop

Not big tests. Intimate ones: analyze image → read critique → submit 1 HITL note → re-run same image.

You're validating: Does FRAMED change *how* it thinks (confidence, tone), not just what it says?

```powershell
python scripts/run_hitl_micro_loop.py --image_path stress_test_master/dataset_v2/ambiguous/v2_ambiguous_001.jpg
```

Or by image_id:
```powershell
python scripts/run_hitl_micro_loop.py --dataset_path stress_test_master/dataset_v2 --image_id ambiguous_v2_ambiguous_001
```

After HITL, re-run the same image and check: confidence delta, critique change, hypothesis count. If it works on 2–3 images, you've succeeded.

---

## Phase 10 — Lock the learning boundary ✅

See **HITL_LIMITS.md**. It contains: Max influence radius, No confidence increases, Signature-scoped only, No prompt mutation, No retroactive belief rewriting. This is FRAMED's constitution for HITL. Code must enforce it.

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
