# Test Run Report — Dataset v2 (Real Photos)

**Date:** 2026-02-02  
**Dataset:** `stress_test_master/dataset_v2` only (36 real photos).  
**Purpose:** Baseline after invalidating previous tests; all runs use dataset v2.

---

## 1. Pipeline run (metrics)

**Command run:**
```powershell
python -m framed.tests.test_intelligence_pipeline --dataset_path stress_test_master/dataset_v2 --max_images 8 --shuffle --seed 42 --disable_cache
```

**Run directory:** `framed/tests/test_runs/run_2026_02_02_071744`

| Metric | Value |
|--------|--------|
| **Total images** | 8 |
| **Completed** | 8 |
| **Failed** | 0 |
| **Pass** | Yes (no failures) |
| **Elapsed** | 7m 23s |
| **Images/hour** | ~65 |

**Intelligence health (metrics.json):**
| Metric | Value |
|--------|--------|
| Average confidence | 0.61 |
| Confidence std | 0.11 |
| Uncertainty acknowledged % | 100 |
| Multiple hypotheses % | 50 |
| Unresolved disagreement % | 75 |
| Hallucination rate | 0 |
| Overconfidence rate | 0 |
| Reflection regeneration % | 0 |
| New patterns stored | 8 |

**Failure list:** `failures.json` = [] (empty).

**Summary:** Pipeline completes on dataset v2 with zero hard failures. Confidence is moderate (0.61 avg); uncertainty and multiple hypotheses are present. Expression layer returned empty critique on some images (Model B); others got full critiques.

---

## 2. Manual reading pass (sample — needs your review)

**Command run:**
```powershell
python scripts/run_manual_reading_pass.py --dataset_path stress_test_master/dataset_v2 --max_images 5
```

**Report location:** `framed/tests/test_runs/manual_pass_2026_02_02_072529/MANUAL_READING_PASS_REPORT.md`

**What was done:** The script ran on 5 images and wrote the report with primary interpretation, critique (or “no critique”), pattern signature, and the three review questions per image.

**What you need to do:** Open that report and, for each image, answer:

1. Did it hedge when it should have? (YES / NO)  
2. Did it surprise me in a way that felt earned? (YES / NO)  
3. Did it sound like a mentor, not a summarizer? (YES / NO)  

If any answer is **NO** → submit HITL using the `pattern_signature` in the report, e.g.:

```powershell
python -m framed.feedback.submit -i IMAGE_ID -t TYPE -s SIGNATURE
```

I cannot answer these questions; they require your (or a human) judgment.

---

## 3. HITL micro loop (behavior check)

**Command run (one image, no HITL submitted):**
```powershell
python scripts/run_hitl_micro_loop.py --image_path stress_test_master/dataset_v2/ambiguous/v2_ambiguous_001.jpg
```
(Enter was piped so the “after” run executed without interactive HITL.)

**Result:** Script ran correctly. Before/after were identical (confidence 0.60, empty critique) because no HITL was submitted.

**What you need to do for full validation:** Run the micro loop **interactively** for 2–3 images:

1. Run: `python scripts/run_hitl_micro_loop.py --image_path stress_test_master/dataset_v2/ambiguous/v2_ambiguous_001.jpg`  
2. Read the “Before” output; if you want to correct, submit HITL (e.g. overconfidence, missed_alternative) using the printed pattern signature.  
3. Press Enter to re-run the same image.  
4. Compare “After”: confidence and tone should move in the right direction for that image only.

Repeat for 1–2 more images (e.g. `v2_street_001.jpg`, `v2_nature_001.jpg`).

---

## 4. Observations

- **CLIP meta-tensor errors:** Some parallel CLIP loads log “Cannot copy out of meta tensor”; other workers succeed and the pipeline continues. Vision still produces evidence (e.g. green_coverage, condition, Places365 when CLIP works).
- **Empty critiques:** Model B sometimes returns an empty critique (e.g. street, interiors, ambiguous in these runs). When it does not (e.g. nature_v2_nature_001, mixed_v2_mixed_004, artistic_v2_artistic_001), critiques are long and mentor-style.
- **Combined layers 2–7:** Logs show “Combined layers 2–7 completed (single Model A call)” — the combined Model A path is active.

---

## 5. Commands for future runs

| Test | Command |
|------|--------|
| Pipeline (more images) | `python -m framed.tests.test_intelligence_pipeline --dataset_path stress_test_master/dataset_v2 --max_images 15 --shuffle --seed 42 --disable_cache` |
| Manual pass (larger sample) | `python scripts/run_manual_reading_pass.py --dataset_path stress_test_master/dataset_v2 --max_images 10` |
| HITL micro loop | `python scripts/run_hitl_micro_loop.py --image_path stress_test_master/dataset_v2/ambiguous/v2_ambiguous_001.jpg` |

All tests use **dataset v2** only.
