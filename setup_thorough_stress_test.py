"""
Comprehensive Stress Test Dataset Setup for FRAMED

Downloads and organizes multiple datasets for thorough testing:
- COCO 2017 (Mixed/General)
- SUN397 (Architecture & Street)
- Visual Genome (Ambiguous)
- LFW (Portraits)
- Unsplash Lite (Aesthetic)

Organizes them into the required directory structure.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
from typing import Optional

def run_command(cmd: list, description: str, check: bool = True) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Warning: {result.stderr}")
        return result.returncode == 0
    except FileNotFoundError as e:
        print(f"Error: Command not found. {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_dependencies():
    """Check if required tools are available."""
    print("\n" + "="*80)
    print("Checking Dependencies")
    print("="*80)
    
    dependencies = {
        "python": ["python", "--version"],
        "pip": ["python", "-m", "pip", "--version"],
    }
    
    optional = {
        "wget": ["wget", "--version"],
        "curl": ["curl", "--version"],
    }
    
    all_ok = True
    for name, cmd in dependencies.items():
        if not run_command(cmd, f"Check {name}", check=False):
            print(f"[X] {name} not found")
            all_ok = False
        else:
            print(f"[OK] {name} available")
    
    for name, cmd in optional.items():
        if run_command(cmd, f"Check {name}", check=False):
            print(f"[OK] {name} available")
        else:
            print(f"[!] {name} not found (optional)")
    
    return all_ok

def install_fiftyone():
    """Install fiftyone for COCO dataset."""
    print("\n" + "="*80)
    print("Installing fiftyone")
    print("="*80)
    
    try:
        import fiftyone as fo
        print("[OK] fiftyone already installed")
        return True
    except ImportError:
        print("Installing fiftyone...")
        return run_command(
            ["python", "-m", "pip", "install", "fiftyone"],
            "Install fiftyone"
        )

def download_coco_dataset(output_dir: str, max_samples: int = 5000, split: str = "validation"):
    """
    Download COCO dataset using fiftyone.
    
    Args:
        output_dir: Output directory for organized images
        max_samples: Maximum number of images (default: 5000 for validation set)
        split: Dataset split ("validation" or "train")
    """
    print("\n" + "="*80)
    print(f"Downloading COCO 2017 Dataset ({split}, max_samples={max_samples})")
    print("="*80)
    
    try:
        import fiftyone as fo
        
        print(f"Loading COCO dataset (split={split}, max_samples={max_samples})...")
        print("Note: This may take a while. First-time download is ~1GB for validation set.")
        
        dataset = fo.zoo.load_zoo_dataset(
            "coco-2017",
            split=split,
            max_samples=max_samples
        )
        
        output_path = Path(output_dir) / "mixed"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {len(dataset)} images to {output_path}...")
        copied = 0
        for i, sample in enumerate(dataset, 1):
            try:
                src_path = Path(sample.filepath)
                if src_path.exists():
                    dst_path = output_path / src_path.name
                    shutil.copy(src_path, dst_path)
                    copied += 1
                    if i % 100 == 0:
                        print(f"  Copied {i}/{len(dataset)} images...")
            except Exception as e:
                print(f"  Warning: Failed to copy {sample.filepath}: {e}")
        
        print(f"\n[OK] COCO dataset prepared: {copied} images in {output_path}")
        return copied
    
    except ImportError:
        print("[X] fiftyone not installed. Install with: pip install fiftyone")
        return 0
    except Exception as e:
        print(f"[X] COCO dataset download failed: {e}")
        return 0

def download_lfw_dataset(output_dir: str):
    """
    Download LFW (Labeled Faces in the Wild) dataset for portraits.
    
    Args:
        output_dir: Output directory for portraits
    """
    print("\n" + "="*80)
    print("Downloading LFW Dataset (Portraits)")
    print("="*80)
    
    output_path = Path(output_dir) / "portraits"
    output_path.mkdir(parents=True, exist_ok=True)
    
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    lfw_tgz = output_path.parent / "lfw.tgz"
    lfw_extracted = output_path.parent / "lfw"
    
    # Check if already downloaded
    if lfw_extracted.exists():
        print("[OK] LFW already extracted")
    elif lfw_tgz.exists():
        print("Extracting LFW archive...")
        import tarfile
        with tarfile.open(lfw_tgz, "r:gz") as tar:
            tar.extractall(output_path.parent)
    else:
        print(f"Downloading LFW from {lfw_url}...")
        print("Note: This requires wget or curl. File size: ~180MB")
        print("[SKIP] LFW download skipped (requires manual download)")
        print(f"   URL: {lfw_url}")
        print(f"   Save to: {lfw_tgz}")
        return 0
    
    # Copy images to portraits folder
    if lfw_extracted.exists():
        print(f"Copying images from {lfw_extracted} to {output_path}...")
        copied = 0
        for img_file in lfw_extracted.rglob("*.jpg"):
            try:
                dst_path = output_path / img_file.name
                shutil.copy(img_file, dst_path)
                copied += 1
                if copied % 100 == 0:
                    print(f"  Copied {copied} images...")
            except Exception as e:
                print(f"  Warning: Failed to copy {img_file}: {e}")
        
        print(f"\n[OK] LFW dataset prepared: {copied} images in {output_path}")
        return copied
    
    return 0

def organize_sun397_dataset(source_dir: str, output_dir: str, max_per_category: int = 50):
    """
    Organize SUN397 dataset into architecture and street categories.
    
    Args:
        source_dir: Path to SUN397 extracted directory
        output_dir: Output directory for organized images
        max_per_category: Maximum images per category (to avoid overwhelming)
    """
    print("\n" + "="*80)
    print("Organizing SUN397 Dataset")
    print("="*80)
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"[X] SUN397 directory not found: {source_dir}")
        print("Download instructions:")
        print("  wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz")
        print("  tar -xvzf SUN397.tar.gz")
        return 0
    
    arch_path = Path(output_dir) / "architecture"
    street_path = Path(output_dir) / "street"
    arch_path.mkdir(parents=True, exist_ok=True)
    street_path.mkdir(parents=True, exist_ok=True)
    
    # Architecture-related categories
    arch_keywords = [
        "abbey", "arch", "architecture", "building", "cathedral", "church",
        "facade", "monument", "palace", "temple", "tower", "structure"
    ]
    
    # Street-related categories
    street_keywords = [
        "street", "road", "sidewalk", "urban", "city", "alley", "avenue",
        "boulevard", "crosswalk", "highway", "intersection"
    ]
    
    arch_count = 0
    street_count = 0
    
    print("Scanning SUN397 categories...")
    for category_dir in source_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name.lower()
        
        # Check if architecture-related
        is_arch = any(keyword in category_name for keyword in arch_keywords)
        is_street = any(keyword in category_name for keyword in street_keywords)
        
        if is_arch and arch_count < max_per_category * len(arch_keywords):
            for img_file in category_dir.glob("*.jpg"):
                if arch_count >= max_per_category * len(arch_keywords):
                    break
                try:
                    dst_path = arch_path / f"{category_dir.name}_{img_file.name}"
                    shutil.copy(img_file, dst_path)
                    arch_count += 1
                except Exception:
                    pass
        
        if is_street and street_count < max_per_category * len(street_keywords):
            for img_file in category_dir.glob("*.jpg"):
                if street_count >= max_per_category * len(street_keywords):
                    break
                try:
                    dst_path = street_path / f"{category_dir.name}_{img_file.name}"
                    shutil.copy(img_file, dst_path)
                    street_count += 1
                except Exception:
                    pass
    
    print(f"\n[OK] SUN397 organized: {arch_count} architecture, {street_count} street images")
    return arch_count + street_count

def organize_visual_genome(source_dir: str, output_dir: str, max_samples: int = 1000):
    """
    Organize Visual Genome dataset into ambiguous category.
    
    Args:
        source_dir: Path to Visual Genome images directory
        output_dir: Output directory for ambiguous images
        max_samples: Maximum number of images to copy
    """
    print("\n" + "="*80)
    print("Organizing Visual Genome Dataset (Ambiguous)")
    print("="*80)
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"[X] Visual Genome directory not found: {source_dir}")
        print("Download instructions:")
        print("  wget https://cs.stanford.edu/people/rak248/VG_100K/images.zip")
        print("  wget https://cs.stanford.edu/people/rak248/VG_100K/images2.zip")
        print("  unzip images.zip")
        print("  unzip images2.zip")
        return 0
    
    output_path = Path(output_dir) / "ambiguous"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying up to {max_samples} images from Visual Genome...")
    copied = 0
    for img_file in source_path.glob("*.jpg"):
        if copied >= max_samples:
            break
        try:
            dst_path = output_path / img_file.name
            shutil.copy(img_file, dst_path)
            copied += 1
            if copied % 100 == 0:
                print(f"  Copied {copied} images...")
        except Exception as e:
            print(f"  Warning: Failed to copy {img_file}: {e}")
    
    print(f"\n[OK] Visual Genome organized: {copied} images in {output_path}")
    return copied

def create_stress_test_structure(base_dir: str = "stress_test_master"):
    """Create the master stress test directory structure."""
    print("\n" + "="*80)
    print("Creating Stress Test Directory Structure")
    print("="*80)
    
    base_path = Path(base_dir)
    categories = ["architecture", "street", "nature", "portraits", "ambiguous", "mixed"]
    
    for category in categories:
        (base_path / category).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created {base_path / category}")
    
    return str(base_path)

def generate_dataset_summary(output_dir: str):
    """Generate a summary of the prepared dataset."""
    print("\n" + "="*80)
    print("Dataset Summary")
    print("="*80)
    
    output_path = Path(output_dir)
    categories = ["architecture", "street", "nature", "portraits", "ambiguous", "mixed"]
    
    summary = {}
    total = 0
    
    for category in categories:
        cat_path = output_path / category
        if cat_path.exists():
            image_files = list(cat_path.glob("*.jpg")) + list(cat_path.glob("*.png"))
            count = len(image_files)
            summary[category] = count
            total += count
            print(f"{category:15s}: {count:6d} images")
        else:
            summary[category] = 0
            print(f"{category:15s}: {0:6d} images")
    
    print(f"\n{'Total':15s}: {total:6d} images")
    
    # Save summary to JSON
    summary_file = output_path / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "categories": summary,
            "total": total,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n[OK] Summary saved to: {summary_file}")
    return summary

def main():
    """Main function to set up thorough stress test."""
    print("="*80)
    print("FRAMED Comprehensive Stress Test Dataset Setup")
    print("="*80)
    
    # Configuration
    STRESS_TEST_DIR = "stress_test_master"
    COCO_SAMPLES = 5000  # Validation set size
    LFW_ENABLED = True
    SUN397_ENABLED = False  # Set to True if you have SUN397 downloaded
    VG_ENABLED = False  # Set to True if you have Visual Genome downloaded
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n[X] Required dependencies missing. Please install them first.")
        return
    
    # Step 2: Create directory structure
    stress_test_path = create_stress_test_structure(STRESS_TEST_DIR)
    
    # Step 3: Install fiftyone
    if not install_fiftyone():
        print("\n[!] Warning: fiftyone installation failed. COCO download will be skipped.")
    
    # Step 4: Download COCO dataset
    coco_count = download_coco_dataset(stress_test_path, max_samples=COCO_SAMPLES, split="validation")
    
    # Step 5: Download LFW dataset (if enabled)
    lfw_count = 0
    if LFW_ENABLED:
        lfw_count = download_lfw_dataset(stress_test_path)
    
    # Step 6: Organize SUN397 (if enabled and available)
    sun_count = 0
    if SUN397_ENABLED:
        sun_source = input("Enter path to SUN397 extracted directory (or press Enter to skip): ").strip()
        if sun_source:
            sun_count = organize_sun397_dataset(sun_source, stress_test_path, max_per_category=50)
    
    # Step 7: Organize Visual Genome (if enabled and available)
    vg_count = 0
    if VG_ENABLED:
        vg_source = input("Enter path to Visual Genome images directory (or press Enter to skip): ").strip()
        if vg_source:
            vg_count = organize_visual_genome(vg_source, stress_test_path, max_samples=1000)
    
    # Step 8: Generate summary
    summary = generate_dataset_summary(stress_test_path)
    
    # Final instructions
    print("\n" + "="*80)
    print("Setup Complete!")
    print("="*80)
    print(f"\nStress test dataset ready at: {Path(stress_test_path).absolute()}")
    print(f"Total images: {summary.get('total', 0)}")
    print("\nNext steps:")
    print(f"1. Run stress test:")
    print(f"   python -m framed.tests.test_intelligence_pipeline \\")
    print(f"       --dataset_path {stress_test_path} \\")
    print(f"       --shuffle --seed 42")
    print("\n2. For a quick test (100 images):")
    print(f"   python -m framed.tests.test_intelligence_pipeline \\")
    print(f"       --dataset_path {stress_test_path} \\")
    print(f"       --max_images 100 --shuffle --seed 42")
    print("\n3. For full stress test (all images, disable expression to save cost):")
    print(f"   python -m framed.tests.test_intelligence_pipeline \\")
    print(f"       --dataset_path {stress_test_path} \\")
    print(f"       --shuffle --seed 42 --disable_expression")

if __name__ == "__main__":
    main()
