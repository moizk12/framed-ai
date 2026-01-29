"""
Dataset Handling for FRAMED Intelligence Pipeline Tests

Handles loading images from structured directories and optional ground truth.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ImageRecord:
    """Single image record with metadata."""
    image_id: str
    image_path: str
    category: str
    ground_truth: Optional[Dict] = None
    human_correction: Optional[Dict] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_path: str
    max_images: Optional[int] = None
    shuffle: bool = True
    seed: Optional[int] = None
    categories: Optional[List[str]] = None


def load_dataset(config: DatasetConfig) -> List[ImageRecord]:
    """
    Load images from structured directory.
    
    Expected structure:
    IMAGE_ROOT/
    ├── architecture/
    ├── street/
    ├── nature/
    ├── portraits/
    ├── ambiguous/
    └── mixed/
    
    Args:
        config: Dataset configuration
    
    Returns:
        List of ImageRecord objects
    """
    dataset_path = Path(config.dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp', '.tif'}
    
    records = []
    
    # Determine which categories to load
    if config.categories:
        categories = config.categories
    else:
        # Auto-detect categories from directory structure
        categories = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not categories:
        raise ValueError(f"No categories found in dataset path: {dataset_path}")
    
    # Load images from each category
    for category in categories:
        category_path = dataset_path / category
        
        if not category_path.exists() or not category_path.is_dir():
            continue
        
        # Find all images in category
        image_files = [f for f in category_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            continue  # Skip empty categories
        
        for image_file in image_files:
            image_id = f"{category}_{image_file.stem}"
            
            # Try to load ground truth if available
            ground_truth_path = image_file.with_suffix('.json')
            ground_truth = None
            human_correction = None
            
            if ground_truth_path.exists():
                try:
                    with open(ground_truth_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Check if it's a human correction file
                        if "framed_interpretation" in data and "user_feedback" in data:
                            # Human correction file format
                            human_correction = data
                        else:
                            # Ground truth file format
                            if validate_ground_truth(data):
                                ground_truth = data
                except (json.JSONDecodeError, IOError) as e:
                    # File exists but is invalid - log and continue
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Invalid ground truth/correction file {ground_truth_path}: {e}"
                    )
            
            records.append(ImageRecord(
                image_id=image_id,
                image_path=str(image_file.absolute()),  # Use absolute path
                category=category,
                ground_truth=ground_truth,
                human_correction=human_correction
            ))
    
    if not records:
        raise ValueError(f"No images found in dataset path: {dataset_path}")
    
    # Shuffle if requested
    if config.shuffle:
        if config.seed is not None:
            random.seed(config.seed)
        random.shuffle(records)
    
    # Limit number of images
    if config.max_images:
        records = records[:config.max_images]
    
    return records


def validate_ground_truth(ground_truth: Dict) -> bool:
    """
    Validate ground truth structure.
    
    Expected structure:
    {
        "image_id": "abc123",
        "known_ambiguity": true,
        "expected_uncertainty": true,
        "forbidden_claims": ["ivy", "religious structure"],
        "allowed_interpretations": ["green-painted facade", "decorative surface"]
    }
    
    Args:
        ground_truth: Ground truth dictionary
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(ground_truth, dict):
        return False
    
    # Check for required fields (at least some structure)
    if "image_id" not in ground_truth:
        return False
    
    return True


def get_category_distribution(records: List[ImageRecord]) -> Dict[str, int]:
    """
    Get distribution of images across categories.
    
    Args:
        records: List of ImageRecord objects
    
    Returns:
        Dictionary mapping category to count
    """
    distribution = {}
    for record in records:
        distribution[record.category] = distribution.get(record.category, 0) + 1
    return distribution
