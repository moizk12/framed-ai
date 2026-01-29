# FRAMED Intelligence Pipeline Stress Test

This is a comprehensive, repeatable intelligence evaluation harness for FRAMED.

## Purpose

To systematically evaluate:
- **Interpretive accuracy** - Does FRAMED see what's actually there?
- **Uncertainty handling** - Does FRAMED acknowledge when it's uncertain?
- **Reflection effectiveness** - Does the reflection loop catch mistakes?
- **Learning sanity** - Does memory learn correctly without corrupting?
- **Mentor integrity** - Does FRAMED maintain its mentor philosophy?

## Quick Start

### 1. Prepare Dataset

Organize your images in a structured directory:

```
IMAGE_ROOT/
├── architecture/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── street/
│   ├── image1.jpg
│   └── ...
├── nature/
├── portraits/
├── ambiguous/
└── mixed/
```

### 2. Run Tests

```bash
# Basic test run
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path /path/to/images \
    --max_images 100

# Full test run with options
python -m framed.tests.test_intelligence_pipeline \
    --dataset_path /path/to/images \
    --max_images 1000 \
    --shuffle \
    --seed 42 \
    --disable_expression
```

### 3. Review Results

Results are saved to `framed/tests/test_runs/run_YYYY_MM_DD_HHMMSS/`:

- `summary.json` - Test summary and pass/fail report
- `metrics.json` - Aggregate metrics
- `failures.json` - List of failures
- `raw/` - Individual image results

## Command Line Options

- `--dataset_path` (required): Path to image dataset directory
- `--max_images`: Maximum number of images to test (default: all)
- `--shuffle`: Shuffle images before testing
- `--seed`: Random seed for shuffling
- `--disable_expression`: Disable expression layer to save cost
- `--run_dir`: Output directory for results (default: auto-generated)

## Output Structure

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

### Failures (`failures.json`)

List of images that failed tests, with detailed diagnostics.

### Raw Logs (`raw/image_id.json`)

Complete test result for each image, including:
- Core interpretation
- Evidence alignment
- Reflection diagnostics
- Learning impact
- Mentor integrity
- Full analysis result

## Pass/Fail Criteria

### Immediate Fail If:

- **Hallucination rate > 5%** - FRAMED invents facts not in evidence
- **Overconfidence without evidence > 3%** - FRAMED claims certainty when uncertain
- **Reflection loop misses > 10%** - Reflection fails to catch detectable errors
- **Memory reinforces incorrect interpretations** - Learning corrupts understanding

### Warning If:

- **Confidence rarely drops below 0.6** - FRAMED may be too confident
- **Alternatives not generated in ambiguous images** - FRAMED may be too single-minded
- **Mentor becomes repetitive** - FRAMED may be losing mentor voice

## Ground Truth (Optional)

You can provide ground truth files alongside images:

`image1.jpg` → `image1.json`

```json
{
  "image_id": "image1",
  "known_ambiguity": true,
  "expected_uncertainty": true,
  "forbidden_claims": ["ivy", "religious structure"],
  "allowed_interpretations": ["green-painted facade", "decorative surface"]
}
```

This helps validate epistemic behavior, not just correctness.

## Environment Variables

The test respects these environment variables:

- `FRAMED_USE_INTELLIGENCE_CORE=true` - Use intelligence core (default: true)
- `FRAMED_DISABLE_EXPRESSION=true` - Disable expression layer
- `FRAMED_LOG_LEVEL=DEBUG` - Set log level
- `FRAMED_REFLECTION_STRICT=true` - Strict reflection mode

## Free Datasets

Recommended free datasets for testing:

1. **COCO Dataset** - Common Objects in Context
   - Download: https://cocodataset.org/
   - Categories: street, nature, portraits, architecture

2. **Open Images Dataset** - Google
   - Download: https://storage.googleapis.com/openimages/web/index.html
   - Categories: mixed, diverse

3. **Unsplash Dataset** - Photography
   - Download: https://unsplash.com/data
   - Categories: nature, portraits, architecture

4. **Flickr30k** - Scene descriptions
   - Download: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
   - Categories: street, nature, mixed

## Example Usage

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

print(f"Passed: {pass_fail['passed']}")
print(f"Failures: {pass_fail['failures']}")
```

## Integration with CI/CD

This test suite is designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run FRAMED Intelligence Tests
  run: |
    python -m framed.tests.test_intelligence_pipeline \
      --dataset_path ./test_images \
      --max_images 50 \
      --shuffle \
      --seed 42
```

## Troubleshooting

### Import Errors

Make sure you're running from the project root:
```bash
cd framed-clean
python -m framed.tests.test_intelligence_pipeline ...
```

### Memory Issues

If testing large datasets, consider:
- Using `--max_images` to limit test size
- Using `--disable_expression` to reduce memory usage
- Testing in batches

### Slow Performance

- Use `--disable_expression` to skip critique generation
- Reduce `--max_images` for faster iteration
- Use smaller image resolutions

## Next Steps

1. **Run baseline test** - Establish baseline metrics
2. **Add ground truth** - Annotate images with expected behavior
3. **Iterate** - Make changes and re-run tests
4. **Compare** - Compare metrics across runs
5. **Fix failures** - Address failures systematically

---

**This test suite is your proof that FRAMED is thinking, not performing.**
