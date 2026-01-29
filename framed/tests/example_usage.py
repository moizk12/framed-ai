"""
Example usage of FRAMED Intelligence Pipeline Stress Test

This script demonstrates how to use the test harness programmatically.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from framed.tests.test_intelligence_pipeline import IntelligencePipelineTester
from framed.tests.datasets import DatasetConfig, load_dataset


def example_basic_test():
    """Basic test example."""
    print("=" * 80)
    print("Example 1: Basic Test")
    print("=" * 80)
    
    config = {
        "dataset_path": "/path/to/images",  # Update this path
        "max_images": 10,
        "shuffle": True,
        "seed": 42,
        "disable_expression": False,
    }
    
    tester = IntelligencePipelineTester(config)
    summary, metrics, pass_fail = tester.run_tests()
    
    print(f"\nResults:")
    print(f"  Passed: {pass_fail['passed']}")
    print(f"  Failures: {len(pass_fail['failures'])}")
    print(f"  Warnings: {len(pass_fail['warnings'])}")


def example_dataset_inspection():
    """Example of inspecting dataset before testing."""
    print("=" * 80)
    print("Example 2: Dataset Inspection")
    print("=" * 80)
    
    dataset_config = DatasetConfig(
        dataset_path="/path/to/images",  # Update this path
        max_images=None,
        shuffle=False,
    )
    
    records = load_dataset(dataset_config)
    
    print(f"\nLoaded {len(records)} images")
    
    # Get category distribution
    from collections import Counter
    categories = Counter(r.category for r in records)
    
    print("\nCategory distribution:")
    for category, count in categories.items():
        print(f"  {category}: {count}")
    
    # Show first few images
    print("\nFirst 5 images:")
    for record in records[:5]:
        print(f"  {record.image_id}: {record.image_path}")


def example_single_image_test():
    """Example of testing a single image."""
    print("=" * 80)
    print("Example 3: Single Image Test")
    print("=" * 80)
    
    from framed.tests.datasets import ImageRecord
    
    # Create a single image record
    image_record = ImageRecord(
        image_id="test_image_001",
        image_path="/path/to/image.jpg",  # Update this path
        category="architecture",
    )
    
    config = {
        "dataset_path": "/path/to/images",  # Not used for single image
        "disable_expression": True,  # Skip critique generation
    }
    
    tester = IntelligencePipelineTester(config)
    result = tester.test_single_image(image_record)
    
    print(f"\nTest result for {image_record.image_id}:")
    print(f"  Confidence: {result.get('core_interpretation', {}).get('confidence', 0.0):.2f}")
    print(f"  Reflection score: {result.get('reflection_diagnostics', {}).get('reflection_score', 0.0):.2f}")
    print(f"  Hallucination detected: {result.get('evidence_alignment', {}).get('hallucination_detected', False)}")


def example_metrics_analysis():
    """Example of analyzing metrics after test run."""
    print("=" * 80)
    print("Example 4: Metrics Analysis")
    print("=" * 80)
    
    import json
    from pathlib import Path
    
    # Load metrics from a previous run
    run_dir = "framed/tests/test_runs/run_2026_01_24_010000"  # Update this path
    metrics_path = Path(run_dir) / "metrics.json"
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("\nIntelligence Health:")
        ih = metrics.get("intelligence_health", {})
        print(f"  Average confidence: {ih.get('average_confidence', 0.0):.2f}")
        print(f"  Uncertainty acknowledged: {ih.get('uncertainty_acknowledged_percent', 0.0):.2f}%")
        
        print("\nFailure Metrics:")
        fm = metrics.get("failure_metrics", {})
        print(f"  Hallucination rate: {fm.get('hallucination_rate', 0.0):.2f}%")
        print(f"  Overconfidence rate: {fm.get('overconfidence_rate', 0.0):.2f}%")
        
        print("\nLearning Metrics:")
        lm = metrics.get("learning_metrics", {})
        print(f"  Memory growth rate: {lm.get('memory_growth_rate', 0.0):.2f}%")
        print(f"  Correction effectiveness: {lm.get('correction_effectiveness', 0.0):.2f}%")
    else:
        print(f"Metrics file not found: {metrics_path}")


if __name__ == "__main__":
    print("FRAMED Intelligence Pipeline Test - Example Usage")
    print("\nNote: Update paths in examples before running")
    print("\nAvailable examples:")
    print("  1. example_basic_test() - Run basic test")
    print("  2. example_dataset_inspection() - Inspect dataset")
    print("  3. example_single_image_test() - Test single image")
    print("  4. example_metrics_analysis() - Analyze metrics")
    print("\nUncomment the example you want to run:")
    
    # Uncomment to run examples:
    # example_basic_test()
    # example_dataset_inspection()
    # example_single_image_test()
    # example_metrics_analysis()
