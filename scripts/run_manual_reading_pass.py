#!/usr/bin/env python3
"""
Manual reading pass (A): 10–15 images, expression ON, real models.

Runs the pipeline and generates a review report so you can answer, per image:
  1. Did it hedge when it should have?
  2. Did it surprise me in a way that felt earned?
  3. Did it sound like a mentor, not a summarizer?
If any answer is "no" → submit HITL feedback.

Usage:
  python scripts/run_manual_reading_pass.py
  python scripts/run_manual_reading_pass.py --dataset_path stress_test_master/dataset_v2 --max_images 12
"""

import argparse
import json
import sys
from pathlib import Path

# Project root
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Load .env
for _d in (_root, Path.cwd()):
    _p = _d / ".env"
    if _p.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_p)
            break
        except ImportError:
            break


def _flag_review(category: str, confidence: float, primary: str) -> list:
    """Suggest review flags: e.g. ambiguous + high confidence."""
    flags = []
    if category == "ambiguous" and confidence > 0.7:
        flags.append("⚠ Ambiguous image with confidence > 0.7 — check hedging")
    if confidence > 0.85:
        flags.append("⚠ High confidence — check if alternatives were considered")
    if not primary or len(primary) < 10:
        flags.append("⚠ Very short primary interpretation — check depth")
    return flags


def main():
    parser = argparse.ArgumentParser(description="Run manual reading pass (10–15 images) and generate review report")
    parser.add_argument("--dataset_path", default="stress_test_master/dataset_v2", help="Dataset path (real photos)")
    parser.add_argument("--max_images", type=int, default=15, help="Max images (10–15 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_dir", default=None, help="Output run directory (default: test_runs/run_YYYY_MM_DD_HHMMSS)")
    parser.add_argument("--disable_cache", action="store_true", default=True, help="Disable cache (default True)")
    args = parser.parse_args()

    from framed.tests.test_intelligence_pipeline import IntelligencePipelineTester, load_dataset
    from framed.tests.reporting import save_run
    from framed.tests.metrics import compute_metrics
    from datetime import datetime, timezone

    dataset_path = _root / args.dataset_path
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    records = load_dataset(
        str(dataset_path),
        shuffle=True,
        seed=args.seed,
        max_images=args.max_images,
    )
    if not records:
        print("No images found in dataset")
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else _root / "framed" / "tests" / "test_runs" / f"manual_pass_{ts}"
    run_dir = run_dir if run_dir.is_absolute() else _root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)

    config = {
        "dataset_path": str(dataset_path),
        "max_images": args.max_images,
        "shuffle": True,
        "seed": args.seed,
        "disable_expression": False,
        "disable_cache": args.disable_cache,
        "run_dir": str(run_dir),
    }
    tester = IntelligencePipelineTester(config)
    results = []
    for rec in records:
        r = tester._process_single(rec)
        results.append(r)
        vid = (r.get("image_id") or "unknown").replace("/", "_").replace("\\", "_")
        out = {k: v for k, v in r.items() if k != "full_analysis"}
        with open(run_dir / "raw" / f"{vid}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)

    metrics = compute_metrics(results)
    run_metadata = {
        "elapsed_ms": 0,
        "elapsed_seconds": 0,
        "elapsed_human": "N/A",
        "run_id": run_dir.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    save_run(run_dir, config, results, metrics, run_metadata)

    # Generate manual reading pass report (3 questions + review flags)
    report_path = run_dir / "MANUAL_READING_PASS_REPORT.md"
    lines = [
        "# Manual Reading Pass — Review Report",
        "",
        f"**Run:** {run_dir.name}",
        f"**Images:** {len(results)}",
        "",
        "For each image, read the critique (and optionally look at the image), then answer:",
        "",
        "1. **Did it hedge when it should have?** (YES / NO)",
        "2. **Did it surprise me in a way that felt earned?** (YES / NO)",
        "3. **Did it sound like a mentor, not a summarizer?** (YES / NO)",
        "",
        "If any answer is **NO** → submit HITL feedback (use the pattern_signature below).",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(results, 1):
        image_id = r.get("image_id", "?")
        category = r.get("category", "?")
        failed = r.get("failed", False)
        core = r.get("core_interpretation", {})
        confidence = core.get("confidence", 0)
        primary = (core.get("primary") or "").strip()
        alts = core.get("alternatives", []) or []
        critique = (r.get("critique") or "").strip()
        sig = r.get("pattern_signature", "")

        flags = _flag_review(category, confidence, primary)

        lines.append(f"## {i}. {image_id}")
        lines.append("")
        lines.append(f"- **Category:** {category}  |  **Confidence:** {confidence:.2f}")
        if flags:
            for f in flags:
                lines.append(f"- {f}")
        lines.append("")
        lines.append("**Primary:** " + (primary or "(none)"))
        if alts:
            lines.append("**Alternatives:** " + ", ".join(str(a) for a in alts[:5]))
        lines.append("")
        lines.append("**Critique:**")
        lines.append("")
        lines.append("> " + critique.replace("\n", "\n> ") if critique else "> *(no critique)*")
        lines.append("")
        lines.append("**Pattern signature (for HITL):** `" + (sig or "(not found)") + "`")
        lines.append("")
        lines.append("### Your review")
        lines.append("")
        lines.append("- [ ] 1. Hedged when it should have?  YES / NO")
        lines.append("- [ ] 2. Surprise felt earned?  YES / NO")
        lines.append("- [ ] 3. Mentor, not summarizer?  YES / NO")
        lines.append("")
        lines.append("If any NO → HITL: `python -m framed.feedback.submit -i " + image_id + " -t TYPE -s " + (sig or "SIGNATURE") + "`")
        lines.append("")
        lines.append("---")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 60)
    print("Manual reading pass complete")
    print("=" * 60)
    print(f"Run dir: {run_dir}")
    print(f"Review report: {report_path}")
    print("")
    print("Next: Open MANUAL_READING_PASS_REPORT.md, answer the 3 questions per image.")
    print("For any 'NO', submit HITL: python -m framed.feedback.submit -i IMAGE_ID -t TYPE -s SIGNATURE")
    print("Signatures are in the report (and in raw/*.json).")
    print("Extract all: python scripts/extract_hitl_signatures.py " + str(run_dir))


if __name__ == "__main__":
    main()
