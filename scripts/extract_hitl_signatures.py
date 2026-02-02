#!/usr/bin/env python3
"""
Extract pattern_signature from a test run for HITL feedback injection.

Usage:
  python scripts/extract_hitl_signatures.py framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS
"""

import json
import sys
from pathlib import Path


def _compute_signature_from_raw(data: dict) -> str:
    """Compute pattern signature from visual_evidence when not stored."""
    try:
        from framed.analysis.temporal_memory import create_pattern_signature
        ve = data.get("visual_evidence", {})
        if not ve:
            return ""
        semantic = {
            "objects": data.get("perception", {}).get("composition", {}).get("objects", []) or [],
            "tags": data.get("perception", {}).get("semantics", {}).get("tags", []) or [],
            "caption_keywords": (data.get("perception", {}).get("semantics", {}).get("caption", "") or "").split()[:20],
        }
        return create_pattern_signature(ve, semantic)
    except Exception:
        return ""


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_hitl_signatures.py <run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        print(f"Raw dir not found: {raw_dir}")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    print("image_id,pattern_signature\n" + "-" * 60)
    for f in sorted(raw_dir.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            sig = data.get("pattern_signature", "")
            if not sig:
                sig = _compute_signature_from_raw(data)
            image_id = data.get("image_id", f.stem)
            if sig:
                print(f"{image_id},{sig}")
            else:
                print(f"{image_id},(not found)")
        except Exception as e:
            print(f"{f.stem},(error: {e})")


if __name__ == "__main__":
    main()
