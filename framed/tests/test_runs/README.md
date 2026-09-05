# FRAMED Intelligence Pipeline Test Runs

Test output is written here, e.g. `run_2026_01_31_HHMMSS/`.

Each run contains:
- `summary.json` — Config, elapsed time, pass/fail
- `metrics.json` — Intelligence health, failure metrics, run metadata
- `failures.json` — Failed image IDs (empty if clean)
- `raw/` — Per-image condensed results

**Recent runs:** run_2026_02_01_112130 (8.2), run_2026_02_01_120033 (8.4), run_2026_02_01_120045 (Phase 9).  
**Datasets:** Use stress_test_master/dataset_v2 only (36 real photos).

## Public-body artifact audit (2026-09-05)

A — preserved: canonical datasets, frozen regression fixtures, Slice A evidence
and the Slice B worktree (outside this branch).

B — preserved: manual reading passes, documented calibration 8.2/8.4 and Phase 9
runs, dataset-v2 report runs, and other historical runs where claim support is
uncertain. In particular `run_2026_02_01_112130` contains placeholder outputs but
is a documented calibration baseline; it is retained as historical evidence,
not presented as successful model critique.

C/D — removed from the tracked presentation tree: the four one-image exploratory
placeholder/debug runs `run_2026_02_01_101409`, `run_2026_02_01_105549`,
`run_2026_02_01_110235`, `run_2026_02_01_110551`. Their summaries show one-image
transient execution, and their reports contain mock placeholder critique; no
specific external report or frozen test was found to depend on these runs.
They remain recoverable in Git history. New transient runs are ignored.
