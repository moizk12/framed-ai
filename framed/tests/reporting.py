"""
Report Generation for FRAMED Intelligence Pipeline Tests

Generates structured reports from test results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def create_run_directory(base_path: str = "framed/tests/test_runs") -> str:
    """
    Create a new run directory with timestamp.
    
    Args:
        base_path: Base path for test runs
    
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = Path(base_path) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create raw logs directory
    (run_dir / "raw").mkdir(exist_ok=True)
    
    return str(run_dir)


def save_summary(run_dir: str, summary: Dict[str, Any]):
    """
    Save summary.json to run directory.
    
    Args:
        run_dir: Run directory path
        summary: Summary dictionary
    """
    summary_path = Path(run_dir) / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def save_metrics(run_dir: str, metrics: Dict[str, Any]):
    """
    Save metrics.json to run directory.
    
    Args:
        run_dir: Run directory path
        metrics: Metrics dictionary
    """
    metrics_path = Path(run_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def save_failures(run_dir: str, failures: List[Dict[str, Any]]):
    """
    Save failures.json to run directory.
    
    Args:
        run_dir: Run directory path
        failures: List of failure dictionaries
    """
    failures_path = Path(run_dir) / "failures.json"
    with open(failures_path, 'w') as f:
        json.dump(failures, f, indent=2, default=str)


def save_raw_log(run_dir: str, image_id: str, result: Dict[str, Any]):
    """
    Save raw log for a single image.
    
    Args:
        run_dir: Run directory path
        image_id: Image identifier
        result: Test result dictionary
    """
    raw_dir = Path(run_dir) / "raw"
    log_path = raw_dir / f"{image_id}.json"
    with open(log_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)


def generate_pass_fail_report(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate pass/fail report based on hard rules.
    
    Hard Rules (Immediate Fail):
    - Hallucination rate > 5%
    - Overconfidence without evidence > 3%
    - Reflection loop misses > 10% of detectable failures
    - Memory reinforces incorrect interpretations
    
    Warnings:
    - Confidence rarely drops below 0.6
    - Alternatives not generated in ambiguous images
    - Mentor becomes repetitive
    
    Args:
        metrics: Computed metrics dictionary
    
    Returns:
        Dictionary with pass/fail status and warnings
    """
    failure_metrics = metrics.get("failure_metrics", {})
    intelligence_health = metrics.get("intelligence_health", {})
    
    failures = []
    warnings = []
    
    # Hard Rules - Immediate Fail
    hallucination_rate = failure_metrics.get("hallucination_rate", 0.0)
    if hallucination_rate > 5.0:
        failures.append(f"Hallucination rate {hallucination_rate:.2f}% exceeds 5% threshold")
    
    overconfidence_rate = failure_metrics.get("overconfidence_rate", 0.0)
    if overconfidence_rate > 3.0:
        failures.append(f"Overconfidence rate {overconfidence_rate:.2f}% exceeds 3% threshold")
    
    reflection_failure_escape_rate = failure_metrics.get("reflection_failure_escape_rate", 0.0)
    if reflection_failure_escape_rate > 10.0:
        failures.append(f"Reflection failure escape rate {reflection_failure_escape_rate:.2f}% exceeds 10% threshold")
    
    # Warnings
    avg_confidence = intelligence_health.get("average_confidence", 0.85)
    if avg_confidence > 0.8:
        warnings.append(f"Average confidence {avg_confidence:.2f} is high - may indicate insufficient uncertainty acknowledgment")
    
    uncertainty_acknowledged_percent = intelligence_health.get("uncertainty_acknowledged_percent", 0.0)
    if uncertainty_acknowledged_percent < 20.0:
        warnings.append(f"Uncertainty acknowledged in only {uncertainty_acknowledged_percent:.2f}% of cases - may be too confident")
    
    multiple_hypotheses_percent = intelligence_health.get("multiple_hypotheses_percent", 0.0)
    if multiple_hypotheses_percent < 30.0:
        warnings.append(f"Multiple hypotheses generated in only {multiple_hypotheses_percent:.2f}% of cases - may be too single-minded")
    
    mentor_drift_frequency = failure_metrics.get("mentor_drift_frequency", 0.0)
    if mentor_drift_frequency > 5.0:
        warnings.append(f"Mentor drift frequency {mentor_drift_frequency:.2f}% - mentor philosophy may be drifting")
    
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "timestamp": datetime.now().isoformat(),
    }
