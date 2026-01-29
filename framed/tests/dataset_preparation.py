"""
Dataset Preparation Scripts for FRAMED Intelligence Pipeline Tests

Helper scripts to download and organize datasets for testing.
Supports COCO, Unsplash, and other common datasets.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def prepare_coco_dataset(
    output_dir: str = "./test_dataset",
    max_samples: int = 100,
    split: str = "validation"
) -> str:
    """
    Download and organize COCO dataset for testing.
    
    Requires: pip install fiftyone
    
    Args:
        output_dir: Output directory for organized images
        max_samples: Maximum number of images to download
        split: Dataset split ("validation" or "train")
    
    Returns:
        Path to organized dataset
    """
    try:
        import fiftyone as fo
    except ImportError:
        raise ImportError(
            "fiftyone is required for COCO dataset. Install with: pip install fiftyone"
        )
    
    logger.info(f"Loading COCO dataset (split={split}, max_samples={max_samples})")
    
    # Load COCO dataset
    dataset = fo.zoo.load_zoo_dataset(
        "coco-2017",
        split=split,
        max_samples=max_samples
    )
    
    # Create output directory structure
    output_path = Path(output_dir)
    mixed_dir = output_path / "mixed"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    logger.info(f"Copying {len(dataset)} images to {mixed_dir}")
    for sample in dataset:
        src_path = Path(sample.filepath)
        dst_path = mixed_dir / src_path.name
        shutil.copy(src_path, dst_path)
    
    logger.info(f"COCO dataset prepared: {mixed_dir}")
    return str(mixed_dir)


def prepare_unsplash_dataset(
    image_urls: List[str],
    output_dir: str = "./test_dataset",
    category: str = "mixed"
) -> str:
    """
    Download Unsplash images from URLs and organize them.
    
    Args:
        image_urls: List of Unsplash image URLs
        output_dir: Output directory for organized images
        category: Category folder name
    
    Returns:
        Path to organized dataset
    """
    try:
        import requests
        from PIL import Image
        from io import BytesIO
    except ImportError:
        raise ImportError(
            "requests and Pillow are required. Install with: pip install requests pillow"
        )
    
    output_path = Path(output_dir) / category
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {len(image_urls)} images from Unsplash")
    
    downloaded = 0
    for i, url in enumerate(image_urls, 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Verify it's an image
            img = Image.open(BytesIO(response.content))
            
            # Save with unique name
            filename = f"unsplash_{i:04d}.jpg"
            img.save(output_path / filename, "JPEG")
            
            downloaded += 1
            logger.debug(f"Downloaded {i}/{len(image_urls)}: {filename}")
        
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
    
    logger.info(f"Downloaded {downloaded}/{len(image_urls)} images to {output_path}")
    return str(output_path)


def organize_images_by_keywords(
    source_dir: str,
    output_dir: str = "./test_dataset",
    keyword_mapping: Optional[dict] = None
) -> str:
    """
    Organize images into categories based on filename keywords.
    
    Args:
        source_dir: Source directory with images
        output_dir: Output directory for organized images
        keyword_mapping: Dict mapping keywords to categories:
            {
                "building": "architecture",
                "person": "portraits",
                "forest": "nature",
                ...
            }
    
    Returns:
        Path to organized dataset
    """
    if keyword_mapping is None:
        # Default keyword mapping
        keyword_mapping = {
            "building": "architecture",
            "structure": "architecture",
            "facade": "architecture",
            "architectural": "architecture",
            "person": "portraits",
            "portrait": "portraits",
            "face": "portraits",
            "people": "portraits",
            "forest": "nature",
            "landscape": "nature",
            "nature": "nature",
            "tree": "nature",
            "street": "street",
            "urban": "street",
            "city": "street",
            "road": "street",
        }
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create category directories
    categories = set(keyword_mapping.values())
    for category in categories:
        (output_path / category).mkdir(parents=True, exist_ok=True)
    
    # Also create "ambiguous" and "mixed" directories
    (output_path / "ambiguous").mkdir(parents=True, exist_ok=True)
    (output_path / "mixed").mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp', '.tif'}
    
    organized = 0
    for image_file in source_path.iterdir():
        if not image_file.is_file():
            continue
        
        if image_file.suffix.lower() not in image_extensions:
            continue
        
        # Check filename for keywords
        filename_lower = image_file.stem.lower()
        category = None
        
        for keyword, cat in keyword_mapping.items():
            if keyword in filename_lower:
                category = cat
                break
        
        # Default to "mixed" if no keyword matches
        if category is None:
            category = "mixed"
        
        # Copy to category directory
        dst_path = output_path / category / image_file.name
        shutil.copy(image_file, dst_path)
        organized += 1
    
    logger.info(f"Organized {organized} images into {output_path}")
    return str(output_path)


def create_correction_file_template(
    image_id: str,
    output_path: str,
    framed_interpretation: str = "example interpretation"
):
    """
    Create a template correction file for human-in-the-loop feedback.
    
    Args:
        image_id: Image identifier
        output_path: Path to save correction file
        framed_interpretation: What FRAMED said (example)
    """
    template = {
        "image_id": image_id,
        "framed_interpretation": framed_interpretation,
        "user_feedback": "This is ivy",  # Example correction
        "confidence_adjustment": 0.15  # Positive = increase confidence, negative = decrease
    }
    
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Created correction file template: {output_file}")


if __name__ == "__main__":
    # Example usage
    print("FRAMED Dataset Preparation Scripts")
    print("\nAvailable functions:")
    print("1. prepare_coco_dataset() - Download COCO dataset")
    print("2. prepare_unsplash_dataset() - Download Unsplash images")
    print("3. organize_images_by_keywords() - Organize images by filename")
    print("4. create_correction_file_template() - Create correction file template")
    print("\nSee EXECUTION_GUIDE.md for detailed usage instructions.")
