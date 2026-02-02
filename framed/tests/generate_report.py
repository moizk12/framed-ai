#!/usr/bin/env python3
"""
Generate a comprehensive test report from an existing run directory.

Usage:
    python -m framed.tests.generate_report --run_dir framed/tests/test_runs/run_2026_02_01_014405

If --run_dir is omitted, uses the most recent run in framed/tests/test_runs/.
"""

import argparse
import sys
from pathlib import Path

# Add project root
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from framed.tests.reporting import generate_comprehensive_report


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive test report from run directory")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to test run directory (e.g. framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS). Default: most recent run",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        test_runs = Path(__file__).parent / "test_runs"
        if not test_runs.exists():
            print("Error: No test_runs directory found. Specify --run_dir explicitly.")
            sys.exit(1)
        runs = sorted([d for d in test_runs.iterdir() if d.is_dir() and d.name.startswith("run_")], key=lambda x: x.name, reverse=True)
        if not runs:
            print("Error: No run directories found. Specify --run_dir explicitly.")
            sys.exit(1)
        run_dir = runs[0]
        print(f"Using most recent run: {run_dir}")
    else:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)

    try:
        report_path = generate_comprehensive_report(run_dir)
        print(f"Report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
