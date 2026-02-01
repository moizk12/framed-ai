#!/usr/bin/env python3
"""
Download 36 new images into dataset_v2 — distinct from dataset_v1_ext.

Uses Lorem Picsum (picsum.photos) for reproducible, unique images.
No API key required. Images are distributed across categories.

Usage:
  python scripts/download_dataset_v2.py
  python scripts/download_dataset_v2.py --out stress_test_master/dataset_v2
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None
    print("Install requests: pip install requests")
    sys.exit(1)

# 36 unique seeds — one per image. Each seed produces a distinct image from Lorem Picsum.
# Distributed: arch 5, interiors 4, street 5, nature 5, portraits 5, ambiguous 4, mixed 4, artistic 4
SEEDS = [
    ("architecture", "v2_arch_001"), ("architecture", "v2_arch_002"), ("architecture", "v2_arch_003"),
    ("architecture", "v2_arch_004"), ("architecture", "v2_arch_005"),
    ("interiors", "v2_int_001"), ("interiors", "v2_int_002"), ("interiors", "v2_int_003"),
    ("interiors", "v2_int_004"),
    ("street", "v2_str_001"), ("street", "v2_str_002"), ("street", "v2_str_003"),
    ("street", "v2_str_004"), ("street", "v2_str_005"),
    ("nature", "v2_nat_001"), ("nature", "v2_nat_002"), ("nature", "v2_nat_003"),
    ("nature", "v2_nat_004"), ("nature", "v2_nat_005"),
    ("portraits", "v2_por_001"), ("portraits", "v2_por_002"), ("portraits", "v2_por_003"),
    ("portraits", "v2_por_004"), ("portraits", "v2_por_005"),
    ("ambiguous", "v2_amb_001"), ("ambiguous", "v2_amb_002"), ("ambiguous", "v2_amb_003"),
    ("ambiguous", "v2_amb_004"),
    ("mixed", "v2_mix_001"), ("mixed", "v2_mix_002"), ("mixed", "v2_mix_003"),
    ("mixed", "v2_mix_004"),
    ("artistic", "v2_art_001"), ("artistic", "v2_art_002"), ("artistic", "v2_art_003"),
    ("artistic", "v2_art_004"),
]

SIZE = "800/600"


def download_image(category: str, seed: str, dest_dir: Path, idx: int) -> bool:
    url = f"https://picsum.photos/seed/{seed}/{SIZE}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"v2_{category}_{idx:03d}.jpg"
    try:
        r = requests.get(url, timeout=15, allow_redirects=True)
        r.raise_for_status()
        dest_path.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  Failed {category}/{seed}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download 36 new images into dataset_v2")
    parser.add_argument("--out", default="stress_test_master/dataset_v2", help="Output directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading 36 images to {out_dir}...")
    indices = {c: 0 for c in set(cat for cat, _ in SEEDS)}
    counts = {c: 0 for c in set(cat for cat, _ in SEEDS)}
    for category, seed in SEEDS:
        indices[category] += 1
        if download_image(category, seed, out_dir / category, indices[category]):
            counts[category] += 1

    total = sum(counts.values())
    report = {"categories": counts, "total": total, "source": "picsum.photos", "version": "v2"}
    (out_dir / "DATASET_REPORT.json").write_text(json.dumps(report, indent=2))

    print(f"\nDone: {total} images in {out_dir}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    print(f"\nReport: {out_dir / 'DATASET_REPORT.json'}")


if __name__ == "__main__":
    main()
