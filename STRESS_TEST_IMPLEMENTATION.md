# FRAMED Stress Test Implementation Summary

**Date:** 2026-01-24  
**Status:** ✅ Complete

---

## 🎯 Overview

A comprehensive, repeatable intelligence evaluation harness for FRAMED that systematically tests:
- Interpretive accuracy
- Uncertainty handling
- Reflection effectiveness
- Learning sanity
- Mentor integrity

---

## 📁 Files Created

### Core Test Files

1. **`framed/tests/test_intelligence_pipeline.py`** (Main Test Harness)
   - Main test runner
   - Executes full pipeline for each image
   - Extracts structured diagnostics
   - Computes metrics
   - Generates reports

2. **`framed/tests/datasets.py`** (Dataset Handling)
   - Loads images from structured directories
   - Supports optional ground truth files
   - Handles shuffling and sampling

3. **`framed/tests/metrics.py`** (Metrics Computation)
   - Intelligence health metrics
   - Failure metrics
   - Learning metrics
   - Aggregate computation

4. **`framed/tests/reporting.py`** (Report Generation)
   - Creates run directories
   - Saves summaries, metrics, failures
   - Generates pass/fail reports
   - Saves raw logs

### Documentation

5. **`framed/tests/README.md`** (User Guide)
   - Quick start guide
   - Command line options
   - Output structure
   - Pass/fail criteria
   - Troubleshooting

6. **`framed/tests/example_usage.py`** (Examples)
   - Basic test example
   - Dataset inspection
   - Single image test
   - Metrics analysis

---

## 🔍 Test Pipeline Stages

For each image, the test executes:

1. **Visual Evidence Extraction**
   - `extract_visual_features()` - HSV, texture, spatial analysis

2. **Full Analysis**
   - `analyze_image()` - Semantic signals (CLIP/YOLO) + Intelligence Core

3. **Core Interpretation Extraction**
   - Primary conclusion
   - Confidence
   - Alternatives
   - Uncertainty acknowledgment

4. **Evidence Alignment**
   - Visual evidence usage
   - Semantic conflicts
   - Evidence priority respect
   - Hallucination detection

5. **Expression Layer** (Optional)
   - `generate_poetic_critique()` - Can be disabled to save cost

6. **Reflection Loop**
   - `reflect_on_critique()` - Self-validation
   - Regeneration if needed

7. **Reflection Diagnostics**
   - Reflection score
   - Failure types
   - Regeneration status

8. **Learning Impact**
   - Memory updates
   - Confidence adjustments
   - Pattern storage
   - Evolution entries

9. **Mentor Integrity**
   - Mentor drift
   - Flattery detection
   - Instruction creep
   - Question quality

---

## 📊 Metrics Computed

### Intelligence Health Metrics
- Average confidence
- Confidence variance/standard deviation
- % cases with uncertainty acknowledged
- % cases with multiple hypotheses

### Failure Metrics
- Hallucination rate (%)
- Overconfidence rate (%)
- Reflection failure escape rate (%)
- Mentor drift frequency (%)

### Learning Metrics
- Memory growth rate (%)
- Correction effectiveness (%)
- Average confidence adjustment
- New patterns stored
- Evolution entries added

---

## 🚨 Pass/Fail Criteria

### Immediate Fail If:
- **Hallucination rate > 5%** - FRAMED invents facts
- **Overconfidence without evidence > 3%** - Claims certainty when uncertain
- **Reflection loop misses > 10%** - Reflection fails to catch errors
- **Memory reinforces incorrect interpretations** - Learning corrupts

### Warning If:
- **Confidence rarely drops below 0.6** - Too confident
- **Alternatives not generated in ambiguous images** - Too single-minded
- **Mentor becomes repetitive** - Losing mentor voice

---

## 📁 Output Structure

```
framed/tests/test_runs/
└── run_2026_01_24_010000/
    ├── summary.json          # Test summary and pass/fail report
    ├── metrics.json          # Aggregate metrics
    ├── failures.json         # List of failures
    └── raw/                  # Individual image results
        ├── image_001.json
        ├── image_002.json
        └── ...
```

---

## 🚀 Usage

### Command Line

```bash
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path /path/to/images \
    --max_images 100 \
    --shuffle \
    --seed 42 \
    --disable_expression
```

### Programmatic

```python
from framed.tests.test_intelligence_pipeline import IntelligencePipelineTester

config = {
    "dataset_path": "/path/to/images",
    "max_images": 100,
    "shuffle": True,
    "seed": 42,
    "disable_expression": False,
}

tester = IntelligencePipelineTester(config)
summary, metrics, pass_fail = tester.run_tests()
```

---

## 📋 Dataset Structure

```
IMAGE_ROOT/
├── architecture/
│   ├── image1.jpg
│   ├── image1.json  (optional ground truth)
│   └── ...
├── street/
├── nature/
├── portraits/
├── ambiguous/
└── mixed/
```

### Ground Truth Format (Optional)

```json
{
  "image_id": "image1",
  "known_ambiguity": true,
  "expected_uncertainty": true,
  "forbidden_claims": ["ivy", "religious structure"],
  "allowed_interpretations": ["green-painted facade", "decorative surface"]
}
```

---

## 🔧 Configuration

### Command Line Options
- `--dataset_path` (required): Path to image dataset
- `--max_images`: Maximum images to test
- `--shuffle`: Shuffle images before testing
- `--seed`: Random seed for shuffling
- `--disable_expression`: Disable expression layer (saves cost)
- `--run_dir`: Output directory (default: auto-generated)

### Environment Variables
- `FRAMED_USE_INTELLIGENCE_CORE=true` - Use intelligence core
- `FRAMED_DISABLE_EXPRESSION=true` - Disable expression layer
- `FRAMED_LOG_LEVEL=DEBUG` - Set log level
- `FRAMED_REFLECTION_STRICT=true` - Strict reflection mode

---

## ✅ Features Implemented

- ✅ Structured dataset loading
- ✅ Full pipeline execution
- ✅ Core interpretation extraction
- ✅ Evidence alignment diagnostics
- ✅ Reflection loop diagnostics
- ✅ Learning impact tracking
- ✅ Mentor integrity checks
- ✅ Aggregate metrics computation
- ✅ Pass/fail report generation
- ✅ Raw log saving
- ✅ Failure tracking
- ✅ Ground truth support (optional)
- ✅ Expression layer toggle
- ✅ Comprehensive documentation

---

## 🎯 Next Steps

1. **Run Baseline Test**
   - Establish baseline metrics
   - Identify current performance

2. **Add Ground Truth**
   - Annotate images with expected behavior
   - Validate epistemic behavior

3. **Iterate**
   - Make changes to FRAMED
   - Re-run tests
   - Compare metrics

4. **Fix Failures**
   - Address failures systematically
   - Verify fixes with tests

5. **CI/CD Integration**
   - Add to CI/CD pipeline
   - Run on every commit
   - Track metrics over time

---

## 📚 Related Documents

- `FRAMED_CONSTITUTION.md` - Core principles
- `CRITICAL_IMPROVEMENTS_SUMMARY.md` - Recent improvements
- `IMPLEMENTATION_GAP_ANALYSIS.md` - Gap analysis

---

## 🔍 Example Output

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
    "warnings": [
      "Average confidence 0.82 is high - may indicate insufficient uncertainty acknowledgment"
    ]
  }
}
```

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

---

**Status:** ✅ **Stress test harness complete and ready for use.**

This test suite is your proof that FRAMED is thinking, not performing.
