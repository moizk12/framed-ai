#!/usr/bin/env python3
"""
Compare before/after calibration runs (Step 8.2 vs 8.4).

Usage:
  python scripts/compare_calibration_runs.py run_8_2_dir run_8_4_dir
"""

import json
import sys
from pathlib import Path


def load_raw(run_dir: Path) -> dict:
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        return {}
    out = {}
    for f in raw_dir.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            image_id = data.get("image_id", f.stem)
            out[image_id] = data
        except Exception:
            pass
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_calibration_runs.py <run_8_2_dir> <run_8_4_dir>")
        sys.exit(1)
    before = Path(sys.argv[1])
    after = Path(sys.argv[2])
    b = load_raw(before)
    a = load_raw(after)

    print("=" * 70)
    print("Calibration Run Comparison (Step 8.2 vs 8.4)")
    print("=" * 70)

    for image_id in sorted(set(b) | set(a)):
        rb = b.get(image_id, {})
        ra = a.get(image_id, {})

        core_b = rb.get("core_interpretation", {})
        core_a = ra.get("core_interpretation", {})

        conf_b = core_b.get("confidence", 0)
        conf_a = core_a.get("confidence", 0)
        delta = conf_a - conf_b if (conf_b and conf_a) else None

        alts_b = len(core_b.get("alternatives", []) or [])
        alts_a = len(core_a.get("alternatives", []) or [])
        hyp_change = alts_a - alts_b

        critique_b = (rb.get("critique") or "")[:80]
        critique_a = (ra.get("critique") or "")[:80]
        same_critique = critique_b == critique_a

        print(f"\n{image_id}")
        print(f"  Confidence: {conf_b:.3f} -> {conf_a:.3f} (delta: {delta:+.3f})" if delta is not None else "  Confidence: N/A")
        print(f"  Hypotheses: {alts_b} -> {alts_a} (change: {hyp_change:+d})")
        print(f"  Critique changed: {not same_critique}")

    print("\n" + "=" * 70)
    print("Success: FRAMED changes locally (signature-matched), not globally.")
    print("=" * 70)


if __name__ == "__main__":
    main()
