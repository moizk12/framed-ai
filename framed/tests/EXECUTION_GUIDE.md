# FRAMED Intelligence Pipeline Test - Execution Guide

## 📋 Prerequisites

1. **Python Environment**
   - Python 3.11+
   - All dependencies from `requirements.txt` installed

2. **Dataset Preparation**
   - Organize images in category folders
   - Optional: Add ground truth JSON files

3. **Environment Variables** (Optional)
   - `FRAMED_USE_INTELLIGENCE_CORE=true` (default: true)
   - `FRAMED_DISABLE_EXPRESSION=true` (to save cost)
   - `FRAMED_LOG_LEVEL=DEBUG` (for detailed logs)
   - `FRAMED_DATA_DIR=/path/to/data` (for runtime data)

---

## 🚀 Quick Start

### Step 1: Prepare Dataset

Create a directory structure:

```bash
mkdir -p test_dataset/{architecture,street,nature,portraits,ambiguous,mixed}
```

Copy images into appropriate folders:

```bash
# Example
cp architecture_images/*.jpg test_dataset/architecture/
cp street_images/*.jpg test_dataset/street/
# ... etc
```

### Step 2: Run Basic Test

```bash
# From project root
cd framed-clean

# Basic test (10 images)
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 10
```

### Step 3: Review Results

Results are saved to `framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS/`:

```bash
# View summary
cat framed/tests/test_runs/run_*/summary.json | python -m json.tool

# View metrics
cat framed/tests/test_runs/run_*/metrics.json | python -m json.tool

# View failures
cat framed/tests/test_runs/run_*/failures.json | python -m json.tool
```

---

## 📊 Full Test Execution

### Option 1: Small Test (Development)

```bash
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 50 \
    --shuffle \
    --seed 42 \
    --disable_expression
```

**Use case:** Quick iteration during development

### Option 2: Medium Test (Validation)

```bash
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 200 \
    --shuffle \
    --seed 42
```

**Use case:** Validate changes before deployment

### Option 3: Full Test (Benchmark)

```bash
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --shuffle \
    --seed 42
```

**Use case:** Complete benchmark on full dataset

### Option 4: Category-Specific Test

```bash
# Test only architecture images
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 100
```

Then manually filter results by category in the output.

---

## 🔧 Advanced Options

### Disable Expression Layer (Save Cost)

```bash
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 100 \
    --disable_expression
```

**Why:** Expression layer (Model B) generates critiques, which costs tokens. Disable for faster, cheaper testing.

### Custom Output Directory

```bash
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 100 \
    --run_dir ./custom_results/run_001
```

### Reproducible Tests (Fixed Seed)

```bash
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 100 \
    --shuffle \
    --seed 42
```

**Why:** Same seed = same image order = reproducible results

---

## 📁 Dataset Structure

### Required Structure

```
test_dataset/
├── architecture/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── street/
│   ├── img001.jpg
│   └── ...
├── nature/
├── portraits/
├── ambiguous/
└── mixed/
```

### Optional Ground Truth

For each image, you can add a JSON file with the same name:

```
architecture/
├── img001.jpg
├── img001.json  (optional ground truth)
├── img002.jpg
└── ...
```

**Ground Truth Format:**

```json
{
  "image_id": "img001",
  "known_ambiguity": true,
  "expected_uncertainty": true,
  "forbidden_claims": ["ivy", "religious structure"],
  "allowed_interpretations": ["green-painted facade", "decorative surface"]
}
```

---

## 📊 Understanding Results

### Summary (`summary.json`)

```json
{
  "test_config": {...},
  "total_images": 100,
  "completed": 100,
  "failed": 5,
  "pass_fail_report": {
    "passed": true,
    "failures": [],
    "warnings": [...]
  }
}
```

**Key Fields:**
- `passed`: `true` if all hard rules pass
- `failures`: List of hard rule violations
- `warnings`: List of soft warnings

### Metrics (`metrics.json`)

```json
{
  "intelligence_health": {
    "average_confidence": 0.75,
    "confidence_variance": 0.05,
    "uncertainty_acknowledged_percent": 25.0,
    "multiple_hypotheses_percent": 40.0
  },
  "failure_metrics": {
    "hallucination_rate": 2.0,
    "overconfidence_rate": 1.5,
    "reflection_failure_escape_rate": 5.0,
    "mentor_drift_frequency": 3.0
  },
  "learning_metrics": {
    "memory_growth_rate": 80.0,
    "correction_effectiveness": 10.0,
    "average_confidence_adjustment": -0.05
  }
}
```

**Key Metrics:**
- `hallucination_rate`: Should be < 5%
- `overconfidence_rate`: Should be < 3%
- `reflection_failure_escape_rate`: Should be < 10%
- `uncertainty_acknowledged_percent`: Should be > 20% for diverse images

### Failures (`failures.json`)

List of images that failed tests, with detailed diagnostics for debugging.

### Raw Logs (`raw/image_id.json`)

Complete test result for each image, including:
- Core interpretation
- Evidence alignment
- Reflection diagnostics
- Learning impact
- Mentor integrity
- Category validation
- Full analysis (limited)

---

## 🎯 Pass/Fail Criteria

### Hard Rules (Immediate Fail)

- ❌ **Hallucination rate > 5%** - FRAMED invents facts
- ❌ **Overconfidence without evidence > 3%** - Claims certainty when uncertain
- ❌ **Reflection loop misses > 10%** - Reflection fails to catch errors
- ❌ **Memory reinforces incorrect interpretations** - Learning corrupts

### Warnings (Soft Failures)

- ⚠️ **Confidence rarely drops below 0.6** - Too confident
- ⚠️ **Alternatives not generated in ambiguous images** - Too single-minded
- ⚠️ **Mentor becomes repetitive** - Losing mentor voice

---

## 🔍 Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root
cd framed-clean

# Verify Python path
python -c "import sys; print(sys.path)"

# Run from project root
python -m framed.tests.test_intelligence_pipeline --dataset_path ./test_dataset --max_images 10
```

### No Images Found

```bash
# Check dataset structure
ls -la test_dataset/
ls -la test_dataset/architecture/

# Verify image extensions are supported
# Supported: .jpg, .jpeg, .png, .webp, .tiff, .bmp, .tif
```

### Memory Issues

```bash
# Use smaller test size
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 10

# Disable expression layer
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 10 \
    --disable_expression
```

### Slow Performance

```bash
# Disable expression layer (saves time and cost)
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 50 \
    --disable_expression
```

---

## 📈 Comparing Results

### Compare Two Test Runs

```bash
# Run 1
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 100 \
    --run_dir ./results/baseline

# Make changes to FRAMED...

# Run 2
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 100 \
    --run_dir ./results/after_changes

# Compare metrics
diff <(cat ./results/baseline/metrics.json | python -m json.tool) \
     <(cat ./results/after_changes/metrics.json | python -m json.tool)
```

---

## 🧪 Example Workflows

### Workflow 1: Development Testing

```bash
# Quick test during development
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 10 \
    --disable_expression \
    --shuffle \
    --seed 42
```

### Workflow 2: Pre-Deployment Validation

```bash
# Comprehensive test before deployment
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 200 \
    --shuffle \
    --seed 42
```

### Workflow 3: Full Benchmark

```bash
# Complete benchmark on full dataset
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --shuffle \
    --seed 42
```

### Workflow 4: Category-Specific Analysis

```bash
# Test specific category
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path ./test_dataset \
    --max_images 100

# Then filter results by category
cat framed/tests/test_runs/run_*/raw/*.json | \
    python -c "import sys, json; \
    data = [json.loads(line) for line in sys.stdin if line.strip()]; \
    arch = [r for r in data if r.get('category') == 'architecture']; \
    print(f'Architecture: {len(arch)} images')"
```

---

## 📝 Best Practices

1. **Start Small**: Test with 10-50 images first
2. **Use Seeds**: Always use `--seed` for reproducibility
3. **Disable Expression**: Use `--disable_expression` during development
4. **Check Failures**: Always review `failures.json` after tests
5. **Compare Metrics**: Track metrics over time
6. **Add Ground Truth**: Annotate ambiguous images for better validation

---

## 🆘 Getting Help

If tests fail:

1. **Check Summary**: Review `summary.json` for pass/fail status
2. **Review Failures**: Check `failures.json` for specific issues
3. **Inspect Raw Logs**: Look at `raw/image_id.json` for detailed diagnostics
4. **Check Logs**: Review console output for errors
5. **Verify Dataset**: Ensure images are valid and readable

---

## ✅ Verification Checklist

Before running tests, verify:

- [ ] Dataset directory exists and has images
- [ ] Images are in correct category folders
- [ ] Image files are readable (not corrupted)
- [ ] Python environment has all dependencies
- [ ] Environment variables are set (if needed)
- [ ] Output directory is writable

After tests, verify:

- [ ] Summary shows expected number of images processed
- [ ] Pass/fail report shows `passed: true`
- [ ] Metrics are within acceptable ranges
- [ ] No critical failures in failures.json
- [ ] Raw logs exist for all images

---

**Ready to test FRAMED's intelligence!**
