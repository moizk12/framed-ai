#!/usr/bin/env python3
"""
Dataset verification for FRAMED stress testing.

Usage:
  python -m framed.tests.datasets --verify stress_test_master/dataset_v2

Exits with no output if zero hash collisions. Raises on overlap.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Set


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def verify_zero_overlap(dataset_path: Path) -> bool:
    """Verify no SHA256 overlap across category folders. Returns True if clean."""
    all_hashes: Dict[str, List[str]] = {}
    categories = ["architecture", "interiors", "street", "portraits", "nature", "mixed", "ambiguous", "artistic"]
    for cat in categories:
        cat_path = dataset_path / cat
        if not cat_path.exists():
            continue
        for f in cat_path.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                h = sha256_file(f)
                if h not in all_hashes:
                    all_hashes[h] = []
                all_hashes[h].append(str(f.relative_to(dataset_path)))
    overlaps = [v for v in all_hashes.values() if len(v) > 1]
    if overlaps:
        print("ERROR: SHA256 overlap detected - same image in multiple folders:", overlaps, file=sys.stderr)
        sys.exit(1)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", required=True, help="Dataset path to verify")
    args = parser.parse_args()
    verify_zero_overlap(Path(args.verify))
    # No output = success (by design)


if __name__ == "__main__":
    main()
