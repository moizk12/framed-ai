# Dataset Preparation Guide for FRAMED Stress Tests

This guide shows you how to download and organize datasets for comprehensive FRAMED testing.

---

## 🎯 Recommended Datasets

### 1. COCO Dataset (Mixed/General)

**Why:** Industry standard for complex scene understanding.

**Download:**

```bash
# Install fiftyone
pip install fiftyone

# Use the preparation script
python -c "
from framed.tests.dataset_preparation import prepare_coco_dataset
prepare_coco_dataset(
    output_dir='./test_dataset',
    max_samples=100,
    split='validation'
)
"
```

**Or manually:**

```python
import fiftyone as fo
import shutil
from pathlib import Path

# Load COCO dataset
dataset = fo.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=100)

# Export to your structured directory
output_base = Path("./test_dataset/mixed")
output_base.mkdir(parents=True, exist_ok=True)

for sample in dataset:
    shutil.copy(sample.filepath, output_base / Path(sample.filepath).name)
```

---

### 2. Unsplash Dataset (High Quality, Aesthetic)

**Why:** Excellent for testing Expression Layer - aesthetically complex images.

**Download:**

```bash
# Option 1: Use Unsplash API (requires API key)
# See: https://unsplash.com/developers

# Option 2: Download manually from Unsplash.com
# Search terms:
# - Architecture: "Modernist Building", "Architectural Facade"
# - Nature: "Macro Forest", "Landscape Photography"
# - Portraits: "Portrait Photography", "Human Face"
# - Ambiguous: "Minimalist Abstract", "Conceptual Art"
```

**Organize:**

```bash
# Create folders
mkdir -p test_dataset/{architecture,nature,portraits,ambiguous}

# Move downloaded images to appropriate folders
mv ~/Downloads/building_*.jpg test_dataset/architecture/
mv ~/Downloads/forest_*.jpg test_dataset/nature/
mv ~/Downloads/person_*.jpg test_dataset/portraits/
mv ~/Downloads/abstract_*.jpg test_dataset/ambiguous/
```

---

### 3. Visual Genome (Ambiguous/Out-of-Context)

**Why:** Perfect for testing uncertainty acknowledgment and hallucination detection.

**Download:**

```bash
# Visual Genome dataset
# See: https://visualgenome.org/api/v0/api_home.html

# Look for "Out of Context" images - objects in unexpected places
# Example: A cow in a living room
# This forces FRAMED to trigger reflection loop
```

---

## 📁 Dataset Structure

### Required Structure

```
test_dataset/
├── architecture/
│   ├── img001.jpg
│   ├── img001.json  (optional ground truth)
│   └── ...
├── street/
│   ├── img001.jpg
│   └── ...
├── nature/
├── portraits/
├── ambiguous/
└── mixed/
```

### Ground Truth Files (Optional)

For each image, create a JSON file with the same name:

**`img001.json`:**

```json
{
  "image_id": "img001",
  "known_ambiguity": true,
  "expected_uncertainty": true,
  "forbidden_claims": ["ivy", "religious structure"],
  "allowed_interpretations": ["green-painted facade", "decorative surface"]
}
```

### Human Correction Files (Optional)

**`img001.json` (correction format):**

```json
{
  "image_id": "img001",
  "framed_interpretation": "painted surface",
  "user_feedback": "This is ivy",
  "confidence_adjustment": 0.15
}
```

**Note:** The test automatically detects which format is used.

---

## 🔧 Preparation Scripts

### Quick Setup Script

```python
# prepare_dataset.py
from framed.tests.dataset_preparation import (
    prepare_coco_dataset,
    organize_images_by_keywords,
    create_correction_file_template
)

# Option 1: Download COCO
prepare_coco_dataset(
    output_dir="./test_dataset",
    max_samples=100
)

# Option 2: Organize existing images
organize_images_by_keywords(
    source_dir="./raw_images",
    output_dir="./test_dataset"
)

# Option 3: Create correction file template
create_correction_file_template(
    image_id="img001",
    output_path="./test_dataset/architecture/img001.json"
)
```

---

## ✅ Verification

After preparing your dataset:

```bash
# Check structure
ls -la test_dataset/
ls -la test_dataset/architecture/

# Count images per category
find test_dataset -name "*.jpg" -o -name "*.png" | wc -l

# Verify test can read dataset
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 5 \
    --disable_expression
```

---

## 📊 Dataset Recommendations by Test Goal

### Test Hallucination Detection
- **Use:** Visual Genome "Out of Context" images
- **Why:** Forces FRAMED to detect contradictions

### Test Uncertainty Handling
- **Use:** Ambiguous/abstract images
- **Why:** Tests uncertainty acknowledgment

### Test Reflection Loop
- **Use:** Images with known issues (add ground truth)
- **Why:** Validates reflection catches errors

### Test Expression Layer
- **Use:** Unsplash aesthetic images
- **Why:** Tests poetic critique quality

### Test Learning System
- **Use:** Repeated images with corrections
- **Why:** Tests confidence calibration

---

## 🚀 Quick Start

```bash
# 1. Create directory structure
mkdir -p test_dataset/{architecture,street,nature,portraits,ambiguous,mixed}

# 2. Download images (choose one):
# Option A: COCO
pip install fiftyone
python -c "from framed.tests.dataset_preparation import prepare_coco_dataset; prepare_coco_dataset()"

# Option B: Manual download from Unsplash
# Download images and organize into folders

# 3. Verify
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 10 \
    --disable_expression
```

---

**Ready to test FRAMED with real-world datasets!**
