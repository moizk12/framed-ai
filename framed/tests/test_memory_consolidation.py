#!/usr/bin/env python3
"""IC_0013 — memory consolidation smoke + manifest injection tests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Stub framed package so tests skip Flask app factory in framed/__init__.py
import types

if "framed" not in sys.modules:
    _framed_pkg = types.ModuleType("framed")
    _framed_pkg.__path__ = [str(_project_root / "framed")]
    sys.modules["framed"] = _framed_pkg
if "framed.analysis" not in sys.modules:
    _analysis_pkg = types.ModuleType("framed.analysis")
    _analysis_pkg.__path__ = [str(_project_root / "framed" / "analysis")]
    sys.modules["framed.analysis"] = _analysis_pkg

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _isolated_data_dir() -> str:
    d = tempfile.mkdtemp(prefix="framed_ic0013_")
    os.environ["FRAMED_DATA_DIR"] = d
    return d


def _import_consolidation():
    """Import after FRAMED_DATA_DIR is set (runtime_paths reads env at import)."""
    for mod in (
        "framed.analysis.runtime_paths",
        "framed.analysis.interpretive_memory",
        "framed.analysis.temporal_memory",
        "framed.analysis.echo_memory",
        "framed.analysis.memory_consolidation",
    ):
        sys.modules.pop(mod, None)
    from framed.analysis.memory_consolidation import run_consolidation_pass
    return run_consolidation_pass


def run_dry_run() -> dict:
    data_dir = _isolated_data_dir()
    run_consolidation_pass = _import_consolidation()
    report = run_consolidation_pass(dry_run=True)
    out = asdict(report)
    out["framed_data_dir"] = data_dir
    logger.info("Dry-run OK in %.3fs", report.duration_sec)
    return out


def run_manifest(manifest_path: Path, output_dir: Path) -> dict:
    data_dir = _isolated_data_dir()
    run_consolidation_pass = _import_consolidation()
    from framed.analysis import interpretive_memory as im
    before = {
        "interpretive": len(im.load_memory()),
        "unconsolidated": len(im.list_unconsolidated_entries()),
        "rules": len(im.get_active_rules()),
    }

    report = run_consolidation_pass(correction_manifest=manifest_path, dry_run=False)
    after = {
        "interpretive": len(im.load_memory()),
        "unconsolidated": len(im.list_unconsolidated_entries()),
        "rules": len(im.get_active_rules()),
    }

    result = {
        "report": asdict(report),
        "before": before,
        "after": after,
        "framed_data_dir": data_dir,
        "rules_sample": im.get_active_rules()[:5],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"ic_0013_consolidation_{ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote results: %s", out_file)
    logger.info(
        "Promoted %d rules in %.3fs",
        len(report.promoted_rules),
        report.duration_sec,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="IC_0013 memory consolidation test")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test without writes")
    parser.add_argument("--correction-manifest", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\Users\moizk\Music\FRAMED_AGI_Research_Starter\FRAMED_AGI_Research_Starter\local_lab\archdaemon\status\ic_0013_results",
        help="Directory for JSON result artifacts",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dry_run:
        result = run_dry_run()
        print(json.dumps(result, indent=2))
        return

    if args.correction_manifest:
        manifest = Path(args.correction_manifest)
        if not manifest.exists():
            logger.error("Manifest not found: %s", manifest)
            sys.exit(1)
        result = run_manifest(manifest, output_dir)
        print(json.dumps({"status": "ok", "promoted": len(result["report"]["promoted_rules"])}, indent=2))
        if result["report"]["duration_sec"] > 60:
            logger.warning("FAIL metric: consolidation exceeded 60s")
            sys.exit(2)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
