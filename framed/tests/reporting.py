"""
Reporting for FRAMED intelligence pipeline stress tests.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def generate_comprehensive_report(run_dir: Path) -> Path:
    """
    Generate a comprehensive markdown report from a test run.
    Includes summary, metrics, and per-image results with full critiques.
    Returns the path to the generated report file.
    """
    run_dir = Path(run_dir)
    raw_dir = run_dir / "raw"

    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw results directory not found: {raw_dir}")

    with open(summary_path) as f:
        summary = json.load(f)
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    lines = [
        "# FRAMED Intelligence Pipeline — Comprehensive Test Report",
        "",
        f"**Run ID:** {summary.get('run_id', run_dir.name)}",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Total images:** {summary.get('total_images', 0)}",
        f"- **Completed:** {summary.get('completed', 0)}",
        f"- **Failed:** {summary.get('failed', 0)}",
        f"- **Passed:** {'Yes' if summary.get('pass_fail_report', {}).get('passed', False) else 'No'}",
        f"- **Duration:** {summary.get('elapsed_human', 'N/A')}",
        f"- **Throughput:** {summary.get('images_per_hour', 0)} images/hour",
        f"- **Started:** {summary.get('started_at', 'N/A')}",
        f"- **Ended:** {summary.get('ended_at', 'N/A')}",
        "",
    ]

    # Config
    config = summary.get("test_config", {})
    lines.extend([
        "### Test Configuration",
        "",
        f"- Dataset: `{config.get('dataset_path', 'N/A')}`",
        f"- Shuffle: {config.get('shuffle', False)}, Seed: {config.get('seed', 'N/A')}",
        f"- Expression layer: {'enabled' if not config.get('disable_expression', False) else 'disabled'}",
        f"- Cache: {'enabled' if not config.get('disable_cache', False) else 'disabled'}",
        "",
    ])

    # Pass/fail warnings
    pf = summary.get("pass_fail_report", {})
    if pf.get("warnings"):
        lines.extend(["### Warnings", ""])
        for w in pf["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    # Metrics
    ih = metrics.get("intelligence_health", {})
    fm = metrics.get("failure_metrics", {})
    lm = metrics.get("learning_metrics", {})
    lines.extend([
        "## Metrics",
        "",
        "### Intelligence Health",
        f"- Average confidence: {ih.get('average_confidence', 0):.2f}",
        f"- Confidence std: {ih.get('confidence_std', 0):.3f}",
        f"- Uncertainty acknowledged: {ih.get('uncertainty_acknowledged_percent', 0):.1f}%",
        f"- Multiple hypotheses: {ih.get('multiple_hypotheses_percent', 0):.1f}%",
        "",
        "### Failure Metrics",
        f"- Hallucination rate: {fm.get('hallucination_rate', 0):.1%}",
        f"- Overconfidence rate: {fm.get('overconfidence_rate', 0):.1%}",
        f"- Mentor drift: {fm.get('mentor_drift_frequency', 0):.1%}",
        "",
        "### Learning",
        f"- New patterns stored: {lm.get('new_patterns_stored', 0)}",
        f"- Evolution entries: {lm.get('evolution_entries_added', 0)}",
        "",
        "---",
        "",
        "## Per-Image Results",
        "",
    ])

    # Load all raw results, sort by image_id
    raw_files = sorted(raw_dir.glob("*.json"))
    for i, p in enumerate(raw_files, 1):
        with open(p) as f:
            r = json.load(f)
        img_id = r.get("image_id", p.stem)
        cat = r.get("category", "")
        failed = r.get("failed", False)
        err = r.get("error")

        lines.append(f"### {i}. {img_id}")
        lines.append("")
        lines.append(f"- **Category:** {cat}")
        lines.append(f"- **Status:** {'Failed' if failed else 'OK'}")

        if failed and err:
            lines.append(f"- **Error:** {err}")
            lines.append("")
            continue

        # Core interpretation
        ci = r.get("core_interpretation", {})
        conf = ci.get("confidence", 0)
        primary = ci.get("primary", "")
        lines.append(f"- **Confidence:** {conf:.2f}")
        lines.append(f"- **Primary interpretation:** {primary}")
        lines.append("")

        # Visual evidence summary
        ve = r.get("visual_evidence", {})
        if ve:
            og = ve.get("organic_growth", {})
            mc = ve.get("material_condition", {})
            oi = ve.get("organic_integration", {})
            lines.append("**Visual evidence:**")
            lines.append(f"- Green coverage: {og.get('green_coverage', 0):.3f} (salience: {og.get('salience', 'N/A')})")
            lines.append(f"- Material condition: {mc.get('condition', 'N/A')} (conf: {mc.get('confidence', 0):.2f})")
            lines.append(f"- Integration: {oi.get('relationship', 'N/A')} ({oi.get('integration_level', 'N/A')})")
            lines.append("")

        # Critique
        critique = r.get("critique")
        if critique:
            lines.append("**Mentor critique (gpt-5-mini):**")
            lines.append("")
            lines.append("> " + critique.replace("\n", "\n> "))
            lines.append("")
        else:
            lines.append("*No critique generated.*")
            lines.append("")

        # Reflection diagnostics
        rd = r.get("reflection_diagnostics", {})
        if rd:
            q = rd.get("quality_score")
            req = rd.get("requires_regeneration", False)
            q_str = f"{q:.2f}" if q is not None else "N/A"
            lines.append(f"**Reflection:** quality={q_str}, requires_regeneration={req}")
            lines.append("")

        lines.append("---")
        lines.append("")

    report_path = run_dir / "COMPREHENSIVE_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Comprehensive report written to {report_path}")
    return report_path


def write_pass_fail_report(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate pass/fail report with warnings."""
    failures = []
    warnings = []

    avg_conf = metrics.get("intelligence_health", {}).get("average_confidence", 0.0)
    if avg_conf > 0.8:
        warnings.append(f"Average confidence {avg_conf} is high - may indicate insufficient uncertainty acknowledgment")

    uncert = metrics.get("intelligence_health", {}).get("uncertainty_acknowledged_percent", 0.0)
    if uncert < 5.0:
        warnings.append(f"Uncertainty acknowledged in only {uncert}% of cases - may be too confident")

    multi = metrics.get("intelligence_health", {}).get("multiple_hypotheses_percent", 0.0)
    if multi < 5.0:
        warnings.append(f"Multiple hypotheses generated in only {multi}% of cases - may be too single-minded")

    for r in results:
        if r.get("failed"):
            failures.append(r.get("image_id", "unknown"))

    report = {
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return report


def save_run(
    output_dir: Path,
    config: Dict[str, Any],
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    run_metadata: Dict[str, Any],
) -> None:
    """Save test run to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    failures = [r for r in results if r.get("failed")]
    pass_report = write_pass_fail_report(results, metrics, output_dir)

    summary = {
        "test_config": config,
        "total_images": len(results),
        "completed": len([r for r in results if not r.get("failed")]),
        "failed": len(failures),
        "pass_fail_report": pass_report,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if run_metadata:
        summary.update(run_metadata)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    metrics_with_meta = dict(metrics)
    if run_metadata:
        metrics_with_meta["run_metadata"] = run_metadata
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_with_meta, f, indent=2, default=str)

    with open(output_dir / "failures.json", "w") as f:
        json.dump([{"image_id": r.get("image_id"), "error": r.get("error")} for r in failures], f, indent=2)

    for r in results:
        vid = (r.get("image_id") or "unknown").replace("/", "_").replace("\\", "_")
        if vid:
            p = raw_dir / f"{vid}.json"
            # Omit full_analysis when condensed to keep files small
            out = {k: v for k, v in r.items() if k != "full_analysis"}
            with open(p, "w") as f:
                json.dump(out, f, indent=2, default=str)

    # Generate comprehensive markdown report
    try:
        generate_comprehensive_report(output_dir)
    except Exception as e:
        logger.warning(f"Could not generate comprehensive report: {e}")
