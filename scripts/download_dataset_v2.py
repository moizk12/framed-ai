#!/usr/bin/env python3
"""
Download 36 category-matched real photos into dataset_v2. Use this dataset only (no sample/solid-color images).

Uses curated Unsplash URLs so each folder contains images that actually match
the category (architecture, interiors, street, nature, portraits, ambiguous,
mixed, artistic). No API key required.

Usage:
  python scripts/download_dataset_v2.py
  python scripts/download_dataset_v2.py --out stress_test_master/dataset_v2
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None
    print("Install requests: pip install requests")
    sys.exit(1)

# Curated Unsplash image URLs — each URL matches its category
# Format: category -> list of (url, optional_desc). Add ?w=800&q=80 for size.
CURATED_URLS = {
    "architecture": [
        "https://images.unsplash.com/photo-1487958449943-2429e8be8625",  # building
        "https://images.unsplash.com/photo-1511818966892-d7d671e672a2",  # architecture
        "https://images.unsplash.com/photo-1514525253161-7a46d19cd819",  # structure
        "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00",    # building
        "https://images.unsplash.com/photo-1497366216548-37526070297c",  # office building
    ],
    "interiors": [
        "https://images.unsplash.com/photo-1493809842364-78817add7ffb",  # living room
        "https://images.unsplash.com/photo-1524758631624-e2822e304c36",  # room
        "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2",    # interior
        "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85",  # indoor
    ],
    "street": [
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000",  # street
        "https://images.unsplash.com/photo-1514565131-fce0801e5785",    # urban
        "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df",  # city
        "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b",  # cityscape
        "https://images.unsplash.com/photo-1444723121867-7a241cacace9",  # street scene
    ],
    "nature": [
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e",  # forest
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",  # landscape
        "https://images.unsplash.com/photo-1448375240586-882707db888b",  # nature
        "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05",  # mountains
        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",  # ocean/beach
    ],
    "portraits": [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d",  # portrait
        "https://images.unsplash.com/photo-1494790108377-be9c29b29330",  # person
        "https://images.unsplash.com/photo-1500648767791-00dcc994a43e",  # face
        "https://images.unsplash.com/photo-1534528741775-53994a69daeb",  # woman portrait
        "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d",  # man portrait
    ],
    "ambiguous": [
        "https://images.unsplash.com/photo-1519681393784-d120267933ba",  # foggy mountains
        "https://images.unsplash.com/photo-1470770841072-f978cf4d019e",  # foggy landscape
        "https://images.unsplash.com/photo-1557683316-973673baf926",     # abstract gradient
        "https://images.unsplash.com/photo-1579546929518-9e396f3cc809",  # blur/abstract
    ],
    "mixed": [
        "https://images.unsplash.com/photo-1511895426328-dc8714191300",  # person in scene
        "https://images.unsplash.com/photo-1529156069898-49953e39b3ac",  # people environment
        "https://images.unsplash.com/photo-1511632765486-a01980e01a18",  # person + nature
        "https://images.unsplash.com/photo-1522202176988-66273c2fd55f",  # people working
    ],
    "artistic": [
        "https://images.unsplash.com/photo-1541961017774-22349e4a1262",  # conceptual
        "https://images.unsplash.com/photo-1452587925148-ce544e77e70d",  # artistic
        "https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d",    # creative
        "https://images.unsplash.com/photo-1460661419201-fd4cecdf8a8b",  # abstract art
    ],
}

SIZE_PARAMS = "?w=800&h=600&fit=crop&q=80"


def download_image(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    full_url = f"{url}{SIZE_PARAMS}" if "?" not in url else f"{url}&w=800&h=600&fit=crop&q=80"
    try:
        r = requests.get(full_url, timeout=20, allow_redirects=True)
        r.raise_for_status()
        dest_path.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  Failed {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download 36 category-matched images into dataset_v2")
    parser.add_argument("--out", default="stress_test_master/dataset_v2", help="Output directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading 36 category-matched images to {out_dir}...")
    print("(Each folder will contain images that actually match the category)\n")
    counts = {}
    for category, urls in CURATED_URLS.items():
        dest_sub = out_dir / category
        dest_sub.mkdir(parents=True, exist_ok=True)
        n = 0
        for i, url in enumerate(urls, 1):
            dest_path = dest_sub / f"v2_{category}_{i:03d}.jpg"
            if download_image(url, dest_path):
                n += 1
                print(f"  {category}/{dest_path.name}")
        counts[category] = n

    total = sum(counts.values())
    report = {
        "categories": counts,
        "total": total,
        "source": "unsplash.com (curated, category-matched)",
        "version": "v2",
    }
    (out_dir / "DATASET_REPORT.json").write_text(json.dumps(report, indent=2))

    print(f"\nDone: {total} images in {out_dir}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    print(f"\nReport: {out_dir / 'DATASET_REPORT.json'}")


if __name__ == "__main__":
    main()
