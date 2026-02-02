#!/usr/bin/env python3
"""
Calibration Protocol Runner

Step 8.2 / 8.4: Run FRAMED on calibration micro-set (expression ON).
Phase 9: Run stability test on fresh batch (no HITL).

Usage:
  python scripts/run_calibration_protocol.py step_8_2
  python scripts/run_calibration_protocol.py step_8_4 --run_dir path/to/run_8_2
  python scripts/run_calibration_protocol.py phase_9
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_SET = PROJECT_ROOT / "stress_test_master" / "dataset_v2"
STRESS_SET = PROJECT_ROOT / "stress_test_master" / "dataset_v2"


def run_step_8_2():
    """Run FRAMED normally (expression ON) on calibration micro-set."""
    cmd = [
        sys.executable, "-m", "framed.tests.test_intelligence_pipeline",
        "--dataset_path", str(CALIBRATION_SET),
        "--shuffle", "--seed", "42",
        "--disable_cache",
    ]
    print("Step 8.2 — Run FRAMED normally (expression ON)")
    print(f"  Dataset: {CALIBRATION_SET}")
    print(f"  Command: {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def run_step_8_4(run_dir: str = None):
    """Re-run same images after HITL feedback (for comparison)."""
    cmd = [
        sys.executable, "-m", "framed.tests.test_intelligence_pipeline",
        "--dataset_path", str(CALIBRATION_SET),
        "--shuffle", "--seed", "42",
        "--disable_cache",
    ]
    print("Step 8.4 — Re-run same images (post-HITL)")
    print(f"  Dataset: {CALIBRATION_SET}")
    if run_dir:
        print(f"  Compare with: {run_dir}\n")
    print(f"  Command: {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def run_phase_9():
    """Phase 9 — HITL stability test: fresh 20–30 image batch, NO HITL feedback."""
    cmd = [
        sys.executable, "-m", "framed.tests.test_intelligence_pipeline",
        "--dataset_path", str(STRESS_SET),
        "--max_images", "25",
        "--shuffle", "--seed", "99",
        "--disable_cache",
    ]
    print("Phase 9 — HITL stability test (fresh batch, no feedback)")
    print(f"  Dataset: {STRESS_SET} (25 images)")
    print(f"  Command: {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run calibration protocol steps")
    parser.add_argument("step", choices=["step_8_2", "step_8_4", "phase_9"])
    parser.add_argument("--run_dir", help="For step_8_4: path to run_8_2 for comparison")
    args = parser.parse_args()

    if args.step == "step_8_2":
        return run_step_8_2()
    elif args.step == "step_8_4":
        return run_step_8_4(args.run_dir)
    elif args.step == "phase_9":
        return run_phase_9()
    return 1


if __name__ == "__main__":
    sys.exit(main())
