# How We Know the Results Are Right

You can't validate *every* image without looking. You validate by **sample**, **behavior**, and **metrics**.

---

## 1. Manual reading pass (sample)

**What:** Review 10–15 images using three questions per image:

1. Did it hedge when it should have?
2. Did it surprise me in a way that felt earned?
3. Did it sound like a mentor, not a summarizer?

**How:** Run the manual pass; open the generated report. For each image, answer YES/NO. If any answer is **NO** → submit HITL feedback.

**Why this works:** This is your **quality gate**. The sample is small enough to do by hand and diverse enough (categories, ambiguity) to catch systematic issues (e.g. overconfident on ambiguous, generic tone).

**Command:**
```powershell
python scripts/run_manual_reading_pass.py --dataset_path stress_test_master/dataset_v2 --max_images 5
```
Use `--max_images 5` for a quick run; 10–15 for full sample. Report: `framed/tests/test_runs/manual_pass_YYYY_MM_DD_HHMMSS/MANUAL_READING_PASS_REPORT.md`

---

## 2. HITL micro loop (behavior)

**What:** For 2–3 images: run FRAMED → read critique → submit one HITL note → re-run the *same* image → compare before/after.

**How:** Use the micro-loop script. Check: Does confidence move in the right direction? Does mentor tone tighten instead of flatten? Do alternatives appear where you said "missed alternative"?

**Why this works:** You're validating that **FRAMED changes how it thinks**, not just what it says. If after HITL the same image gets lower confidence or more hedged language, the learning boundary is working.

**Command:**
```powershell
python scripts/run_hitl_micro_loop.py --image_path stress_test_master/dataset_v2/ambiguous/v2_ambiguous_001.jpg
```

---

## 3. Big runs (metrics + spot-check)

**What:** For large test runs (e.g. 25+ images), you don't review every output. You use **metrics** and **flags**.

**Metrics:**
- Average confidence (per run or per category)
- Uncertainty acknowledged %
- Multiple hypotheses %
- Failure rate

**Flags:** The manual-pass report and pipeline can flag:
- Ambiguous image with confidence > 0.7 → review hedging
- Very high confidence (> 0.85) → check if alternatives were considered

**Why this works:** You spot-check only **flagged** items. If metrics stay in expected ranges and no flags fire, you have indirect evidence that the run is sane. If something is off, flags tell you *which* images to open.

---

## Summary

| Question | How you know |
|----------|----------------|
| Are results right for *this* image? | Manual pass: you answer the 3 questions for 10–15 images; any NO → HITL. |
| Does FRAMED respond to HITL? | Micro loop: before/after on 2–3 images; confidence and tone should move in the right direction. |
| Are big runs trustworthy? | Metrics + flags; spot-check only flagged or outlier images. |

**You don't need to look at every image.** You need a **calibrated sample** (manual pass), **proof of behavior** (micro loop), and **metrics + flags** for scale.
