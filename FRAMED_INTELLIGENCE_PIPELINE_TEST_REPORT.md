# FRAMED Intelligence Pipeline — Test Report

**Date:** 2026-01-29  
**Status:** Ready for run  
**Datasets:** `stress_test_master/dataset_v2` only (36 real photos)  
**Canonical run:** `run_YYYY_MM_DD_HHMMSS` (after test completes)

---

## Overview

FRAMED's 7-layer reasoning engine is stress-tested on categorized datasets (architecture, interiors, street, portraits, nature, mixed, ambiguous, artistic). The pipeline uses Model A (gpt-5.2) for reasoning and Model B (gpt-5-mini) for expression.

---

## Run Command

```powershell
cd C:\Users\moizk\Downloads\framed-clean

python -m framed.tests.test_intelligence_pipeline `
  --dataset_path stress_test_master/dataset_v2 `
  --max_images 25 --shuffle --seed 42 --disable_cache
```

**Flags:**
- `--disable_cache` — Force fresh reasoning/expression (no cache)
- `--disable_expression` — Skip expression layer (reasoning only)
- `--max_images N` — Limit to N images

---

## Output Location

```
framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS/
├── summary.json      # Elapsed time, config, pass/fail
├── metrics.json      # Intelligence health, failure metrics
├── failures.json     # [] if clean
└── raw/              # Per-image condensed results
```

---

## Datasets

Use **dataset_v2** only (real photos). No sample/solid-color images.

```powershell
python scripts/download_dataset_v2.py
```

36 images. Categories: architecture, interiors, street, portraits, nature, ambiguous, mixed, artistic.
