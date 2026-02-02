#!/usr/bin/env python3
"""
HITL-heavy micro loop (B): one image → analyze → read critique → submit 1 HITL → re-run → compare.

Validates: Does FRAMED change how it thinks (confidence, tone), not just what it says?

Usage:
  python scripts/run_hitl_micro_loop.py --image_path stress_test_master/dataset_v2/ambiguous/v2_ambiguous_001.jpg
  python scripts/run_hitl_micro_loop.py --dataset_path stress_test_master/dataset_v2 --image_id ambiguous_v2_ambiguous_001
"""

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

for _d in (_root, Path.cwd()):
    _p = _d / ".env"
    if _p.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_p)
            break
        except ImportError:
            break


def _run_one(record: dict, config: dict) -> dict:
    from framed.tests.test_intelligence_pipeline import IntelligencePipelineTester
    tester = IntelligencePipelineTester(config)
    return tester._process_single(record)


def main():
    parser = argparse.ArgumentParser(description="HITL micro loop: run one image → HITL → re-run → compare")
    parser.add_argument("--image_path", help="Path to single image (e.g. stress_test_master/dataset_v2/ambiguous/v2_ambiguous_001.jpg)")
    parser.add_argument("--dataset_path", help="Dataset root (used with --image_id to resolve path)")
    parser.add_argument("--image_id", help="Image ID (e.g. ambiguous_sample_abstract_001); used with --dataset_path")
    parser.add_argument("--disable_cache", action="store_true", default=True, help="Disable cache")
    args = parser.parse_args()

    if args.image_path:
        p = Path(args.image_path)
        if not p.is_absolute():
            p = _root / p
        if not p.exists():
            print(f"Image not found: {p}")
            sys.exit(1)
        # category from parent dir name, image_id = category_stem
        category = p.parent.name
        image_id = f"{category}_{p.stem}"
        record = {"image_id": image_id, "image_path": str(p), "category": category}
    elif args.dataset_path and args.image_id:
        # Resolve path: dataset_path/category/rest.jpg where image_id = category_rest
        base = Path(args.dataset_path)
        if not base.is_absolute():
            base = _root / base
        if not base.exists():
            print(f"Dataset not found: {base}")
            sys.exit(1)
        # image_id is like "ambiguous_sample_abstract_001" -> category=ambiguous, stem=sample_abstract_001
        parts = args.image_id.split("_", 1)
        category = parts[0] if len(parts) > 1 else parts[0]
        stem = parts[1] if len(parts) > 1 else args.image_id
        # Find file: category/sample_abstract_001.jpg or category/<stem>.jpg
        cat_dir = base / category
        if not cat_dir.exists():
            print(f"Category dir not found: {cat_dir}")
            sys.exit(1)
        found = None
        for ext in (".jpg", ".jpeg", ".png"):
            f = cat_dir / f"{stem}{ext}"
            if f.exists():
                found = f
                break
        if not found:
            # Try any file whose stem matches
            for f in cat_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.stem in args.image_id:
                    found = f
                    break
        if not found:
            print(f"Image not found for id {args.image_id} in {cat_dir}")
            sys.exit(1)
        record = {"image_id": args.image_id, "image_path": str(found), "category": category}
    else:
        print("Provide either --image_path or --dataset_path and --image_id")
        sys.exit(1)

    config = {
        "dataset_path": str(record.get("image_path", "")),
        "disable_expression": False,
        "disable_cache": args.disable_cache,
    }

    print("=" * 60)
    print("HITL Micro Loop — Before")
    print("=" * 60)
    print(f"Image: {record['image_id']} ({record['image_path']})")
    print("")

    before = _run_one(record, config)
    core_b = before.get("core_interpretation", {})
    conf_b = core_b.get("confidence", 0)
    primary_b = (core_b.get("primary") or "").strip()
    alts_b = core_b.get("alternatives", []) or []
    critique_b = (before.get("critique") or "").strip()
    sig = before.get("pattern_signature", "")

    print("**Confidence:**", conf_b)
    print("**Primary:**", primary_b[:200] + "..." if len(primary_b) > 200 else primary_b)
    print("**Alternatives:**", len(alts_b), alts_b[:3])
    print("")
    print("**Critique:**")
    print(critique_b[:800] + "..." if len(critique_b) > 800 else critique_b)
    print("")
    print("**Pattern signature:**", sig or "(not found)")
    print("")
    print("Submit HITL if needed:")
    print(f"  python -m framed.feedback.submit -i {record['image_id']} -t TYPE -s \"{sig}\"")
    print("")
    input("Press Enter to re-run the same image (after HITL or skip)... ")

    after = _run_one(record, config)
    core_a = after.get("core_interpretation", {})
    conf_a = core_a.get("confidence", 0)
    primary_a = (core_a.get("primary") or "").strip()
    alts_a = core_a.get("alternatives", []) or []
    critique_a = (after.get("critique") or "").strip()

    delta_conf = conf_a - conf_b if (conf_b is not None and conf_a is not None) else None
    hyp_change = len(alts_a) - len(alts_b)
    critique_changed = critique_b != critique_a

    print("")
    print("=" * 60)
    print("HITL Micro Loop — After (comparison)")
    print("=" * 60)
    print(f"Confidence: {conf_b:.3f} → {conf_a:.3f}" + (f" (delta: {delta_conf:+.3f})" if delta_conf is not None else ""))
    print(f"Hypotheses: {len(alts_b)} → {len(alts_a)} (change: {hyp_change:+d})")
    print(f"Critique changed: {critique_changed}")
    print("")
    if critique_changed:
        print("**New critique (first 500 chars):**")
        print((critique_a[:500] + "...") if len(critique_a) > 500 else critique_a)
    print("")
    print("Success check: FRAMED should change locally (confidence/tone in right direction), not globally.")


if __name__ == "__main__":
    main()
