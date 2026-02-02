#!/usr/bin/env python3
"""
Run FRAMED on one image and print a compact summary as JSON.

Usage (from repo root):
  python scripts/run_single_image_once.py --image_path stress_test_master/dataset_v2/ambiguous/v2_ambiguous_002.jpg

This is a non-interactive alternative to the HITL micro loop:
  1) Run once (BEFORE) and inspect confidence/primary/critique.
  2) Submit HITL (if needed).
  3) Run again (AFTER) and compare.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description="Run one image once and print compact JSON summary")
    parser.add_argument("--image_path", required=True, help="Path to a single image (jpg/png)")
    parser.add_argument("--disable_cache", action="store_true", default=True, help="Disable cache (default True)")
    args = parser.parse_args()

    p = Path(args.image_path)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        raise SystemExit(f"Image not found: {p}")

    # category from parent dir name, image_id = category_stem
    category = p.parent.name
    image_id = f"{category}_{p.stem}"
    record = {"image_id": image_id, "image_path": str(p), "category": category}

    from framed.tests.test_intelligence_pipeline import IntelligencePipelineTester

    config = {
        "dataset_path": str(p),
        "disable_expression": False,
        "disable_cache": args.disable_cache,
    }
    tester = IntelligencePipelineTester(config)
    result = tester._process_single(record)

    core = result.get("core_interpretation", {}) or {}
    summary = {
        "image_id": result.get("image_id", image_id),
        "image_path": str(p),
        "category": result.get("category", category),
        "pattern_signature": result.get("pattern_signature", ""),
        "confidence": core.get("confidence", 0.0),
        "uncertainty_acknowledged": core.get("uncertainty_acknowledged", False),
        "alternatives_count": len(core.get("alternatives") or []),
        "primary": (core.get("primary") or "").strip(),
        "critique_empty": (result.get("critique") or "").strip() == "" or (result.get("critique") or "").strip().startswith("[Critique generation returned empty"),
        "critique_preview": ((result.get("critique") or "").strip()[:500]),
        "visual_evidence": result.get("visual_evidence", {}),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

