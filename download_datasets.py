"""
Download and organize datasets for FRAMED testing.

This script downloads *sample* images from various sources and organizes them
into the required directory structure for testing.

What it supports out of the box (no auth required):
- COCO (via fiftyone, if installed)
- Unsplash *sample* photos (direct image URLs; not Unsplash Lite)
- LFW (Labeled Faces in the Wild) full dataset (HTTP download + extract)

What it does NOT auto-download (too large / requires special access):
- SUN397 (37GB extracted)
- Visual Genome images (15GB+)
- Unsplash Lite (often distributed via Kaggle / requires auth)

You can still use this script to ensure your `stress_test_master/` has real images
in each category folder immediately, and then layer the huge datasets later.
"""

import os
import sys
import json
import shutil
import argparse
import tarfile
import urllib.request
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available - will create minimal test structure")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("requests not available - skipping URL downloads")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def _download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)


def download_lfw_full(output_dir: str) -> int:
    """
    Download and extract LFW (13,233 images) and copy into portraits folder.

    Uses plain HTTP download and Python's tarfile to extract (Windows-friendly).
    """
    output_base = Path(output_dir)
    portraits_dir = output_base / "portraits"
    portraits_dir.mkdir(parents=True, exist_ok=True)

    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    tgz_path = output_base / "lfw.tgz"
    extracted_dir = output_base / "lfw"

    if not extracted_dir.exists():
        if not tgz_path.exists():
            print(f"Downloading LFW (~180MB) -> {tgz_path}")
            try:
                _download_file(lfw_url, tgz_path)
            except Exception as e:
                print(f"Failed to download LFW: {e}")
                return 0

        print(f"Extracting {tgz_path} -> {output_base}")
        try:
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(output_base)
        except Exception as e:
            print(f"Failed to extract LFW archive: {e}")
            return 0

    copied = 0
    if extracted_dir.exists():
        print(f"Copying LFW images -> {portraits_dir}")
        for img_file in extracted_dir.rglob("*.jpg"):
            try:
                dst = portraits_dir / img_file.name
                if not dst.exists():
                    shutil.copy(img_file, dst)
                copied += 1
                if copied % 1000 == 0:
                    print(f"  Copied {copied} images...")
            except Exception:
                pass

    print(f"LFW ready: {copied} images in {portraits_dir}")
    return copied


def download_unsplash_sample(output_dir: str, category: str, count: int = 10):
    """
    Download sample images from Unsplash (using their API or direct URLs).
    
    Note: For production use, you'll need an Unsplash API key.
    This function uses sample URLs for demonstration.
    """
    if not REQUESTS_AVAILABLE or not PIL_AVAILABLE:
        return 0
    
    output_path = Path(output_dir) / category
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample Unsplash image URLs (these are public domain examples)
    # In production, use Unsplash API: https://unsplash.com/developers
    sample_urls = {
        "architecture": [
            "https://images.unsplash.com/photo-1487958449943-2429e8be8625",  # Building
            "https://images.unsplash.com/photo-1511818966892-d7d671e672a2",  # Architecture
            "https://images.unsplash.com/photo-1514525253161-7a46d19cd819",  # Structure
        ],
        "nature": [
            "https://images.unsplash.com/photo-1441974231531-c6227db76b6e",  # Forest
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",  # Landscape
            "https://images.unsplash.com/photo-1448375240586-882707db888b",  # Nature
        ],
        "street": [
            "https://images.unsplash.com/photo-1449824913935-59a10b8d2000",  # Street
            "https://images.unsplash.com/photo-1514565131-fce0801e5785",  # Urban
            "https://images.unsplash.com/photo-1449824913935-59a10b8d2000",  # City
        ],
        "portraits": [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d",  # Portrait
            "https://images.unsplash.com/photo-1494790108377-be9c29b29330",  # Person
            "https://images.unsplash.com/photo-1500648767791-00dcc994a43e",  # Face
        ],
    }
    
    urls = sample_urls.get(category, [])
    downloaded = 0
    
    for i, url in enumerate(urls[:count], 1):
        try:
            from io import BytesIO
            # Add query params for direct download
            download_url = f"{url}?w=800&q=80"
            response = requests.get(download_url, timeout=10)
            response.raise_for_status()
            
            # Verify it's an image
            img = Image.open(BytesIO(response.content))
            
            # Save
            filename = f"unsplash_{category}_{i:03d}.jpg"
            img.save(output_path / filename, "JPEG")
            downloaded += 1
            print(f"Downloaded {category}/{filename}")
        
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    return downloaded


def download_coco_sample(output_dir: str, max_samples: int = 50):
    """
    Download COCO dataset sample using fiftyone (if available).
    """
    try:
        import fiftyone as fo
        
        print(f"Loading COCO dataset (max_samples={max_samples})...")
        dataset = fo.zoo.load_zoo_dataset(
            "coco-2017",
            split="validation",
            max_samples=max_samples
        )
        
        output_path = Path(output_dir) / "mixed"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying {len(dataset)} images to {output_path}...")
        for i, sample in enumerate(dataset, 1):
            src_path = Path(sample.filepath)
            dst_path = output_path / src_path.name
            shutil.copy(src_path, dst_path)
            if i % 10 == 0:
                print(f"Copied {i}/{len(dataset)} images...")
        
        print(f"COCO dataset prepared: {output_path}")
        return len(dataset)
    
    except ImportError:
        print("fiftyone not installed. Install with: pip install fiftyone")
        print("Skipping COCO dataset download.")
        return 0
    except Exception as e:
        print(f"COCO dataset download failed: {e}")
        return 0


def create_sample_images(output_dir: str):
    """
    Create sample test images using PIL (for testing without internet).
    """
    if not PIL_AVAILABLE:
        print("PIL not available - cannot create sample images")
        return 0
    
    output_path = Path(output_dir)
    
    categories = {
        "architecture": ["building", "structure", "facade"],
        "nature": ["forest", "landscape", "tree"],
        "street": ["street", "urban", "city"],
        "portraits": ["person", "face", "portrait"],
        "ambiguous": ["abstract", "minimal", "conceptual"],
        "mixed": ["mixed", "complex", "scene"]
    }
    
    created = 0
    for category, keywords in categories.items():
        cat_path = output_path / category
        cat_path.mkdir(parents=True, exist_ok=True)
        
        for i, keyword in enumerate(keywords, 1):
            # Create a simple colored image
            img = Image.new('RGB', (800, 600), color=(100 + i*50, 150 + i*30, 200 - i*20))
            filename = f"sample_{keyword}_{i:03d}.jpg"
            img.save(cat_path / filename, "JPEG")
            created += 1
    
    print(f"Created {created} sample images")
    return created


def main():
    """Main function to download and organize datasets."""
    parser = argparse.ArgumentParser(description="Download/prepare FRAMED test datasets (samples + LFW).")
    parser.add_argument("--output_dir", default="./test_dataset", help="Target dataset directory (e.g. stress_test_master)")
    parser.add_argument("--coco_max", type=int, default=0, help="Download COCO sample via fiftyone (0 disables)")
    parser.add_argument("--unsplash_per_category", type=int, default=10, help="Unsplash sample images per category")
    parser.add_argument("--download_lfw", action="store_true", help="Download + extract LFW into portraits/")
    args = parser.parse_args()

    print("=" * 80)
    print("FRAMED Dataset Download and Organization")
    print("=" * 80)

    # Set output directory
    output_dir = args.output_dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create category directories
    categories = ["architecture", "street", "nature", "portraits", "ambiguous", "mixed"]
    for category in categories:
        (output_path / category).mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_path.absolute()}")
    
    coco_count = 0
    if args.coco_max and args.coco_max > 0:
        print("\n--- Option 1: COCO Dataset ---")
        coco_count = download_coco_sample(output_dir, max_samples=args.coco_max)
    
    # Option 2: Download Unsplash samples
    print("\n--- Option 2: Unsplash Samples ---")
    unsplash_total = 0
    for category in ["architecture", "nature", "street", "portraits"]:
        count = download_unsplash_sample(output_dir, category, count=args.unsplash_per_category)
        unsplash_total += count

    # Option 3: LFW full dataset
    lfw_count = 0
    if args.download_lfw:
        print("\n--- Option 3: LFW Full Dataset ---")
        lfw_count = download_lfw_full(output_dir)
    
    # Option 3: Create sample images (fallback)
    if coco_count == 0 and unsplash_total == 0 and lfw_count == 0:
        print("\n--- Option 4: Creating Sample Images (Fallback) ---")
        create_sample_images(output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("Dataset Preparation Summary")
    print("=" * 80)
    
    total_images = 0
    for category in categories:
        cat_path = output_path / category
        image_files = list(cat_path.glob("*.jpg")) + list(cat_path.glob("*.png"))
        count = len(image_files)
        total_images += count
        print(f"{category:15s}: {count:3d} images")
    
    print(f"\nTotal images: {total_images}")
    print(f"\nDataset ready at: {output_path.absolute()}")
    print("\nNext step: Run tests with:")
    print(f"  python -m framed.tests.test_intelligence_pipeline --dataset_path {output_dir} --max_images 10")


if __name__ == "__main__":
    main()
