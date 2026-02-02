# Regression: Scene Gate

This directory contains a tiny, stable regression set to ensure the **scene gate**
never regresses back into “weathered stone everywhere”.

## Images

Located in `framed/tests/regression_scene_gate/images/`:

- `interior_001.jpg` — interior scene with textured surfaces
- `landscape_001.jpg` — outdoor wide landscape (rocks + greenery)
- `abstract_001.jpg` — abstract / artistic image with texture
- `portrait_001.jpg` — people-centric (busy background)
- `surface_closeup_001.jpg` — true close-up texture crop (surface study)

## What we assert (routing only)

- `scene_type`
- `is_surface_study`
- For non-surface scenes: organic/material “aging” signals are **suppressed** (not primary).

We intentionally do **not** assert critique quality here.

## Run

```powershell
python -m unittest framed.tests.test_regression_scene_gate
```

This test sets `FRAMED_ENABLE_INTELLIGENCE_CORE=false` to ensure it does not spend API credits.

