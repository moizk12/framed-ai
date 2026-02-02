# HITL Limits — FRAMED Constitution

**Phase 10 — One-time hardening.** This document locks the learning boundary.  
It defines the learning boundary for Human-in-the-Loop (HITL) feedback and is part of FRAMED's constitution, not just code.

---

## 1. Max influence radius

- HITL feedback is **signature-scoped only**.
- One-off feedback for image A does **not** change behavior for image B.
- Calibration is localized, not systemic.
- If feedback lacks a pattern signature → it is rejected (fail fast).

---

## 2. No confidence increases

- **No HITL signal may push confidence up.**
- HITL is for humility, breadth, and alignment — not assertiveness.
- If someone implies "be more confident," that signal is ignored.
- Positive `confidence_delta_hint` values are clamped to zero.

---

## 3. Signature-scoped only

- `pattern_signature` is mandatory at ingestion time.
- Feedback without a valid signature is rejected.
- Prevents global drift from one-off corrections.
- Keeps calibration localized to similar evidence patterns.

---

## 4. No prompt mutation

- HITL feedback never changes LLM prompts.
- Learning happens in memory (temporal, calibration, self-assessment).
- Model weights, system prompts, and instruction text remain unchanged.
- Humans shape **how** FRAMED decides, not **what** it is told to say.

---

## 5. No retroactive belief rewriting

- HITL does not rewrite past conclusions.
- It does not change stored interpretations.
- It biases **future** reasoning: confidence caps, ambiguity sensitivity, hypothesis branching, mentor tone.
- Past outputs remain as-is.

---

## Summary

| Rule | Meaning |
|------|---------|
| Max influence radius | Signature-scoped; no global drift |
| No confidence increases | HITL = humility, not assertiveness |
| Signature-scoped only | Reject feedback without signature |
| No prompt mutation | Memory learns; prompts stay fixed |
| No retroactive rewriting | Bias future, not past |

---

## Code enforcement (Phase 10)

| Rule | Where enforced |
|------|----------------|
| **Signature-scoped only** | `framed/feedback/storage.py`: `submit_feedback()` rejects if `pattern_signature` missing (returns `False`). `framed/feedback/calibration.py`: `ingest_hitl_feedback()` skips entries without `pattern_signature` (fail fast). |
| **No confidence increases** | `framed/feedback/calibration.py`: `_clamp_no_confidence_increase(val)` clamps to `min(val, 0.0)`; `confidence_delta_hint` and heuristic adjustments are passed through it before applying to `confidence_ceiling_adjustment`. |
| **No prompt mutation** | HITL is consumed only in calibration/memory (e.g. `get_hitl_calibration()`, governor bias, reflection). No code path writes HITL into system prompts or instruction text. |
| **No retroactive rewriting** | Calibration affects future reasoning (confidence caps, ambiguity sensitivity, mentor tone). Stored past outputs and interpretations are not rewritten by HITL. |

*This document is the learning boundary. Code must enforce it.*
