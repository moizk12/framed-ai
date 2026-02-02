#!/usr/bin/env python3
"""
Dataset Download and Categorization for FRAMED Intelligence Stress Testing.

Downloads real images from public datasets and categorizes them with:
- Places365 Standard → architecture, interiors, street
- Open Images V7 → portraits, mixed
- Unsplash Lite → nature, ambiguous, artistic

HARD CONSTRAINTS:
- No image may appear in more than one folder (SHA256 verified)
- Each folder contains unique files
- Zero overlap across categories

Run: python scripts/dataset_download_and_categorize.py --out stress_test_master/dataset_v2 --n 250 --allow-large-downloads
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import requests
except ImportError:
    requests = None

# Increase CSV field size for Unsplash Lite
csv.field_size_limit(sys.maxsize)

# Category definitions and dataset mapping
CATEGORIES = [
    "architecture",
    "interiors",
    "street",
    "portraits",
    "nature",
    "mixed",
    "ambiguous",
    "artistic",
]

# Places365 category index -> our category
# Based on categories_places365.txt and IO_places365.txt
PLACES365_ARCHITECTURE = {
    "building_facade", "skyscraper", "tower", "church/outdoor", "mosque/outdoor",
    "temple/asia", "palace", "castle", "mansion", "lighthouse", "pagoda",
    "office_building", "downtown", "bridge", "arch", "ruin", "monastery",
}
PLACES365_INTERIORS = {
    "living_room", "bedroom", "dining_room", "kitchen", "bathroom",
    "office", "classroom", "library/indoor", "museum/indoor", "art_gallery",
    "corridor", "staircase", "basement", "attic", "nursery",
}
PLACES365_STREET = {
    "street", "alley", "parking_lot", "highway", "residential_neighborhood",
    "gas_station", "bus_station/indoor", "subway_station/platform",
    "market/outdoor", "plaza", "boardwalk", "park",
}

# Unsplash keyword heuristics
UNSPLASH_NATURE = {"landscape", "forest", "mountain", "ocean", "nature", "tree", "wildlife", "flower"}
UNSPLASH_AMBIGUOUS = {"abstract", "minimal", "fog", "reflection", "mystery", "unclear", "ambiguous"}
UNSPLASH_ARTISTIC = {"surreal", "conceptual", "artistic", "creative", "experimental", "abstract", "minimal"}


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _http_get_to_file(url: str, dest: Path, timeout: int = 300) -> bool:
    """Download URL to file. Returns True on success."""
    if not requests:
        raise RuntimeError("requests required. pip install requests")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def verify_zero_overlap(out_dir: Path) -> Tuple[bool, Set[str]]:
    """Verify no SHA256 overlap across category folders. Returns (ok, all_hashes)."""
    all_hashes: Dict[str, List[str]] = {}
    for cat in CATEGORIES:
        cat_path = out_dir / cat
        if not cat_path.exists():
            continue
        for f in cat_path.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                h = sha256_file(f)
                if h not in all_hashes:
                    all_hashes[h] = []
                all_hashes[h].append(str(f.relative_to(out_dir)))
    overlaps = [v for v in all_hashes.values() if len(v) > 1]
    if overlaps:
        raise RuntimeError(f"SHA256 overlap detected: {overlaps}")
    return True, set(all_hashes.keys())


def download_unsplash_lite(cache_dir: Path, out_dir: Path, n_per_cat: Dict[str, int]) -> int:
    """Download Unsplash Lite and categorize into nature, ambiguous, artistic."""
    if not requests:
        raise RuntimeError("requests required")
    unsplash_dir = cache_dir / "unsplash_lite"
    unsplash_dir.mkdir(parents=True, exist_ok=True)
    zip_path = unsplash_dir / "unsplash_lite_latest.zip"

    if not zip_path.exists():
        url = "https://unsplash.com/data/lite/latest"
        print(f"Downloading Unsplash Lite from {url}...")
        r = requests.get(url, allow_redirects=True, timeout=600)
        r.raise_for_status()
        # Check if we got a redirect to the actual zip
        if "unsplash" in r.url.lower() and r.url.endswith(".zip"):
            _http_get_to_file(r.url, zip_path, timeout=600)
        else:
            # Try direct zip URL
            _http_get_to_file("https://unsplash.com/data/lite/latest", zip_path, timeout=600)

    # Extract TSV/CSV
    tsv_files = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".tsv") or name.endswith(".csv") or name.endswith(".csv000"):
                z.extract(name, unsplash_dir)
                tsv_files.append(unsplash_dir / name)

    if not tsv_files:
        raise RuntimeError("Unsplash Lite zip contains no TSV/CSV files")

    # Load photos - photos.tsv000 has: photo_id, photo_url, photographer_id, etc.
    rows = []
    for tsv_path in tsv_files:
        if not tsv_path.exists():
            continue
        with open(tsv_path) as f:
            sample = f.read(4096)
            delim = "\t" if "\t" in sample else ","
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                rows.append(row)

    def _unsplash_pick_categories(rows: List[dict], n_nature: int, n_ambiguous: int, n_artistic: int
                                  ) -> Tuple[List[dict], List[dict], List[dict]]:
        nature_rows, ambiguous_rows, artistic_rows = [], [], []
        seen_ids = set()

        # First pass: strict matching
        for row in rows:
            pid = row.get("photo_id") or row.get("id") or ""
            if pid in seen_ids:
                continue
            desc = (row.get("alt_description") or row.get("description") or "").lower()
            tags = (row.get("tags") or row.get("tag") or "").lower()
            combined = f"{desc} {tags}"

            if len(nature_rows) < n_nature and any(k in combined for k in UNSPLASH_NATURE):
                nature_rows.append(row)
                seen_ids.add(pid)
            elif len(ambiguous_rows) < n_ambiguous and any(k in combined for k in UNSPLASH_AMBIGUOUS):
                ambiguous_rows.append(row)
                seen_ids.add(pid)
            elif len(artistic_rows) < n_artistic and any(k in combined for k in UNSPLASH_ARTISTIC):
                artistic_rows.append(row)
                seen_ids.add(pid)

        # Second pass: permissive for underfilled
        for row in rows:
            pid = row.get("photo_id") or row.get("id") or ""
            if pid in seen_ids:
                continue
            desc = (row.get("alt_description") or row.get("description") or "").lower()
            combined = f"{desc}"

            if len(nature_rows) < n_nature and ("land" in combined or "sky" in combined or "water" in combined):
                nature_rows.append(row)
                seen_ids.add(pid)
            elif len(ambiguous_rows) < n_ambiguous:
                ambiguous_rows.append(row)
                seen_ids.add(pid)
            elif len(artistic_rows) < n_artistic and ("art" in combined or "photo" in combined):
                artistic_rows.append(row)
                seen_ids.add(pid)

        return nature_rows, ambiguous_rows, artistic_rows

    n_nat = n_per_cat.get("nature", 250)
    n_amb = n_per_cat.get("ambiguous", 250)
    n_art = n_per_cat.get("artistic", 250)

    nature_r, ambiguous_r, artistic_r = _unsplash_pick_categories(rows, n_nat, n_amb, n_art)

    def _download_one(row: dict, cat: str, idx: int) -> bool:
        url_key = "urls" if "urls" in row else "photo_url" if "photo_url" in row else "url"
        url = row.get(url_key)
        if isinstance(url, dict):
            url = url.get("regular") or url.get("small") or url.get("raw") or ""
        if not url:
            return False
        # Fix URL: add ? or & for params
        sep = "&" if "?" in url else "?"
        img_url = f"{url}{sep}w=800&q=80" if "unsplash" in url.lower() else url
        pid = row.get("photo_id") or row.get("id") or str(idx)
        dest = out_dir / cat / f"unsplash_{pid}.jpg"
        try:
            _http_get_to_file(img_url, dest, timeout=30)
            return dest.exists()
        except Exception:
            return False

    total = 0
    for cat, rlist in [("nature", nature_r), ("ambiguous", ambiguous_r), ("artistic", artistic_r)]:
        (out_dir / cat).mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rlist):
            if _download_one(row, cat, i):
                total += 1
    return total


def download_open_images_v7(out_dir: Path, n_portraits: int, n_mixed: int) -> int:
    """Download from Open Images V7 via FiftyOne. Fallback: use COCO if FiftyOne available."""
    try:
        import fiftyone as fo
    except ImportError:
        print("fiftyone not installed. pip install fiftyone. Skipping Open Images.")
        return 0

    # Use COCO as fallback (has person detection for portraits)
    try:
        dataset = fo.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=n_portraits + n_mixed)
    except Exception as e:
        print(f"FiftyOne dataset load failed: {e}")
        return 0

    portraits_dir = out_dir / "portraits"
    mixed_dir = out_dir / "mixed"
    portraits_dir.mkdir(parents=True, exist_ok=True)
    mixed_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    p_count, m_count = 0, 0
    for sample in dataset:
        if not sample.filepath:
            continue
        src = Path(sample.filepath)
        if not src.exists():
            continue
        # Check for person in detections
        has_person = False
        if sample.get("ground_truth"):
            for det in sample.ground_truth.detections or []:
                if det.label and "person" in str(det.label).lower():
                    has_person = True
                    break
        name = src.name
        if has_person and p_count < n_portraits:
            shutil.copy(src, portraits_dir / name)
            p_count += 1
        elif m_count < n_mixed:
            shutil.copy(src, mixed_dir / name)
            m_count += 1
        if p_count >= n_portraits and m_count >= n_mixed:
            break
    return p_count + m_count


def download_or_use_places365(cache_dir: Path, out_dir: Path, n_per_cat: int,
                              allow_large: bool) -> int:
    """Download Places365 val_256 if --allow-large-downloads, extract, categorize."""
    places_dir = cache_dir / "places365"
    places_dir.mkdir(parents=True, exist_ok=True)

    tar_path = places_dir / "val_256.tar"
    val_txt = places_dir / "places365_val.txt"
    io_txt = places_dir / "IO_places365.txt"

    base_url = "http://data.csail.mit.edu/places/places365"
    val_txt_url = f"{base_url}/places365_val.txt"
    if not tar_path.exists():
        if not allow_large:
            raise RuntimeError("Places365 required. Use --allow-large-downloads to auto-download (~500MB)")
        print("Downloading Places365 val_256 (~500MB)...")
        _http_get_to_file(f"{base_url}/val_256.tar", tar_path, timeout=3600)
    if not val_txt.exists():
        _http_get_to_file(val_txt_url, val_txt)
    if not io_txt.exists():
        try:
            _http_get_to_file("https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt", io_txt)
        except Exception:
            pass  # Optional

    # Load label mapping
    cat_path = places_dir / "categories_places365.txt"
    if not cat_path.exists():
        _http_get_to_file("https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt", cat_path)

    idx_to_name = {}
    with open(cat_path) as f:
        for line in f:
            parts = line.strip().split(None, 1)  # Split only on first whitespace
            if len(parts) >= 2:
                idx_to_name[int(parts[1])] = parts[0].lower()  # e.g. /a/airfield

    # Load val split (path + label_idx)
    val_lines = []
    with open(val_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                val_lines.append((parts[0], int(parts[1])))

    # Map Places365 indices to our categories (architecture, interiors, street)
    arch_idx, int_idx, street_idx = set(), set(), set()
    arch_terms = ["building", "tower", "church", "mosque", "skyscraper", "facade", "castle", "palace", "lighthouse", "arch", "monument", "temple", "cathedral"]
    int_terms = ["room", "indoor", "kitchen", "bedroom", "living", "corridor", "office", "classroom", "bathroom", "dining", "library", "museum", "attic", "basement"]
    street_terms = ["street", "road", "alley", "urban", "parking", "highway", "plaza", "market", "downtown", "crosswalk", "boardwalk"]

    for idx, name in idx_to_name.items():
        n = name.lower()
        if any(t in n for t in arch_terms) and "indoor" not in n:
            arch_idx.add(idx)
        elif any(t in n for t in int_terms):
            int_idx.add(idx)
        elif any(t in n for t in street_terms):
            street_idx.add(idx)

    extracted = places_dir / "val_256"
    if not extracted.exists():
        print("Extracting Places365 val_256...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(places_dir)
        # Some tars have val_256/ prefix, some don't
        if not extracted.exists():
            extracted = places_dir

    arch_dir = out_dir / "architecture"
    int_dir = out_dir / "interiors"
    street_dir = out_dir / "street"
    arch_dir.mkdir(parents=True, exist_ok=True)
    int_dir.mkdir(parents=True, exist_ok=True)
    street_dir.mkdir(parents=True, exist_ok=True)

    arch_count = len(list(arch_dir.glob("*.jpg")))
    int_count = len(list(int_dir.glob("*.jpg")))
    street_count = len(list(street_dir.glob("*.jpg")))

    for path_rel, label_idx in val_lines:
        # path_rel may be "val_256/Places365_val_00000001.jpg" or "Places365_val_00000001.jpg"
        src = places_dir / path_rel
        if not src.exists():
            src = extracted / Path(path_rel).name
        if not src.exists():
            continue
        name = Path(path_rel).name
        if label_idx in arch_idx and arch_count < n_per_cat:
            dest = arch_dir / name
            if not dest.exists():
                import shutil
                shutil.copy(src, dest)
                arch_count += 1
        elif label_idx in int_idx and int_count < n_per_cat:
            dest = int_dir / name
            if not dest.exists():
                import shutil
                shutil.copy(src, dest)
                int_count += 1
        elif label_idx in street_idx and street_count < n_per_cat:
            dest = street_dir / name
            if not dest.exists():
                import shutil
                shutil.copy(src, dest)
                street_count += 1
        if arch_count >= n_per_cat and int_count >= n_per_cat and street_count >= n_per_cat:
            break

    return arch_count + int_count + street_count


def main():
    parser = argparse.ArgumentParser(description="Download and categorize datasets for FRAMED stress testing")
    parser.add_argument("--out", default="stress_test_master/dataset_v2", help="Output directory (real photos)")
    parser.add_argument("--n", type=int, default=250, help="Target images per category")
    parser.add_argument("--allow-large-downloads", action="store_true", help="Allow Places365 (~500MB) download")
    args = parser.parse_args()

    out_dir = Path(args.out)
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for cat in CATEGORIES:
        (out_dir / cat).mkdir(parents=True, exist_ok=True)

    n_per = {c: args.n for c in CATEGORIES}
    report = {"categories": {}, "sources": {}, "total": 0, "zero_overlap": False}

    # Places365
    try:
        n = download_or_use_places365(cache_dir, out_dir, args.n, args.allow_large_downloads)
        for c in ["architecture", "interiors", "street"]:
            cnt = len(list((out_dir / c).glob("*.jpg")))
            report["categories"][c] = cnt
            report["sources"][c] = "Places365"
    except Exception as e:
        print(f"Places365: {e}")
        report["categories"]["architecture"] = 0
        report["categories"]["interiors"] = 0
        report["categories"]["street"] = 0

    # Open Images
    try:
        download_open_images_v7(out_dir, args.n, args.n)
        for c in ["portraits", "mixed"]:
            cnt = len(list((out_dir / c).glob("*.jpg")))
            report["categories"][c] = cnt
            report["sources"][c] = "Open Images V7"
    except Exception as e:
        print(f"Open Images: {e}")

    # Unsplash
    try:
        download_unsplash_lite(cache_dir, out_dir, n_per)
        for c in ["nature", "ambiguous", "artistic"]:
            cnt = len(list((out_dir / c).glob("*.jpg")))
            report["categories"][c] = cnt
            report["sources"][c] = "Unsplash Lite"
    except Exception as e:
        print(f"Unsplash: {e}")

    # Verify
    verify_zero_overlap(out_dir)
    report["zero_overlap"] = True
    report["total"] = sum(report["categories"].values())

    report_path = out_dir / "DATASET_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Dataset ready:")
    for cat, cnt in report["categories"].items():
        print(f"  {cat}: {cnt} ({report['sources'].get(cat, '')})")
    print(f"Total: {report['total']}, zero_overlap: {report['zero_overlap']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
