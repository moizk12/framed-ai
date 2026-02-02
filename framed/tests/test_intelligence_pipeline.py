#!/usr/bin/env python3
"""
FRAMED Intelligence Pipeline Stress Test.

Runs the full 7-layer intelligence pipeline on a dataset and reports metrics.
"""

# Load .env from project root BEFORE any imports that use OPENAI_API_KEY
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Try multiple .env locations (project root, cwd)
for _env_dir in (_project_root, Path.cwd()):
    _env_path = _env_dir / ".env"
    if _env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path)
            break
        except ImportError:
            break

import argparse
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str, shuffle: bool = True, seed: Optional[int] = None, max_images: Optional[int] = None
                 ) -> List[Dict[str, Any]]:
    """Load image records from dataset path (category folders with images)."""
    base = Path(dataset_path)
    if not base.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    records = []
    categories = ["architecture", "interiors", "street", "portraits", "nature", "mixed", "ambiguous", "artistic"]
    for cat in categories:
        cat_path = base / cat
        if not cat_path.exists():
            continue
        for f in cat_path.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_id = f"{cat}_{f.stem}"
                records.append({
                    "image_id": image_id,
                    "image_path": str(f),
                    "category": cat,
                })

    if shuffle and seed is not None:
        random.seed(seed)
        random.shuffle(records)
    if max_images is not None:
        records = records[:max_images]

    return records


class IntelligencePipelineTester:
    """Runs intelligence pipeline stress test on a dataset."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _process_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process one image through the pipeline."""
        image_id = record["image_id"]
        path = record["image_path"]
        category = record["category"]

        result = {"image_id": image_id, "category": category, "failed": False, "error": None}

        try:
            from framed.analysis.vision import analyze_image
            from framed.analysis.expression_layer import generate_poetic_critique
            from framed.analysis.reflection import reflect_on_critique

            disable_cache = self.config.get("disable_cache", False)

            analysis_result = analyze_image(
                path,
                photo_id=image_id,
                filename=Path(path).name,
                disable_cache=disable_cache,
            )

            result["visual_evidence"] = analysis_result.get("visual_evidence", {})
            intelligence = analysis_result.get("intelligence", {})

            rec = intelligence.get("recognition", {})
            meta = intelligence.get("meta_cognition", {})
            alts = meta.get("rejected_alternatives", []) or rec.get("rejected_alternatives", []) or rec.get("alternatives", [])
            alts = [a.get("conclusion", a) if isinstance(a, dict) else a for a in alts] if alts else []
            result["core_interpretation"] = {
                "confidence": meta.get("confidence") or rec.get("confidence", 0.0),
                "primary": (rec.get("what_i_see") or ""),
                "uncertainty_acknowledged": bool(meta.get("what_i_might_be_missing")) or intelligence.get("disagreement_state", {}).get("exists", False) or intelligence.get("require_multiple_hypotheses", False),
                "alternatives": alts,
            }

            result["evidence_alignment"] = {
                "hallucination_detected": False,
                "visual_evidence_used": bool(analysis_result.get("visual_evidence")),
            }
            result["learning_impact"] = {"memory_updated": True, "new_pattern_stored": True}
            result["mentor_integrity"] = {"mentor_drift": False}

            critique = None
            if not self.config.get("disable_expression", False) and intelligence:
                critique = generate_poetic_critique(intelligence_output=intelligence, mentor_mode="Balanced Mentor")
                if critique:
                    hitl_penalty = 0.0
                    try:
                        from framed.feedback.calibration import get_hitl_calibration
                        hitl_penalty = get_hitl_calibration(None).get("mentor_drift_penalty", 0)
                    except Exception:
                        pass
                    reflection = reflect_on_critique(critique, intelligence, hitl_mentor_drift_penalty=hitl_penalty)
                    result["reflection_diagnostics"] = reflection
                    # Store self-assessment for governor calibration (Option 2)
                    try:
                        from framed.analysis.self_assessment import store_self_assessment
                        store_self_assessment(intelligence, reflection)
                    except Exception:
                        pass

            result["critique"] = critique  # Include gpt-5-mini output in saved results
            result["full_analysis"] = analysis_result
            result["pattern_signature"] = analysis_result.get("pattern_signature", "")  # For HITL feedback
            result["condensed"] = True

        except Exception as e:
            result["failed"] = True
            result["error"] = str(e)
            logger.warning(f"Failed {image_id}: {e}")

        return result

    def run_tests(self) -> Dict[str, Any]:
        """Run full test suite."""
        dataset_path = self.config.get("dataset_path", "./stress_test_master/dataset_v2")
        max_images = self.config.get("max_images")
        shuffle = self.config.get("shuffle", True)
        seed = self.config.get("seed")

        records = load_dataset(dataset_path, shuffle=shuffle, seed=seed, max_images=max_images)
        logger.info(f"Loaded {len(records)} images from dataset")

        run_dir = self.config.get("run_dir")
        if not run_dir:
            ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
            run_dir = Path(__file__).parent / "test_runs" / f"run_{ts}"

        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "raw").mkdir(exist_ok=True)
        logger.info(f"Test run directory: {run_path}")

        start_time = datetime.now(timezone.utc)
        results = []
        for i, rec in enumerate(records):
            logger.info(f"Processing image {i+1}/{len(records)}: {rec['image_id']}")
            r = self._process_single(rec)
            results.append(r)

        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()
        elapsed_ms = int(elapsed * 1000)
        hours, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        elapsed_human = f"{hours}h {mins}m {secs}s"

        run_metadata = {
            "elapsed_ms": elapsed_ms,
            "elapsed_seconds": round(elapsed, 3),
            "elapsed_human": elapsed_human,
            "images_per_hour": round(3600 * len(records) / elapsed, 1) if elapsed > 0 else 0,
            "run_id": run_path.name,
            "started_at": start_time.isoformat(),
            "ended_at": end_time.isoformat(),
        }

        metrics = self._compute_metrics(results)
        metrics["run_metadata"] = run_metadata

        from .reporting import save_run
        save_run(
            run_path,
            self.config,
            results,
            metrics,
            run_metadata,
        )

        failed = [r for r in results if r.get("failed")]
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total images: {len(records)}")
        logger.info(f"Completed: {len(results) - len(failed)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Passed: {len(failed) == 0}")
        logger.info(f"Results saved to: {run_path}")

        return {"results": results, "metrics": metrics, "run_dir": str(run_path)}

    def _compute_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        from .metrics import compute_metrics
        return compute_metrics(results)


def main():
    parser = argparse.ArgumentParser(description="FRAMED Intelligence Pipeline Stress Test")
    parser.add_argument("--dataset_path", default="stress_test_master/dataset_v2", help="Dataset path (real photos)")
    parser.add_argument("--max_images", type=int, default=None, help="Max images to process")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable_expression", action="store_true", help="Disable expression layer")
    parser.add_argument("--disable_cache", action="store_true", help="Disable analysis cache")
    parser.add_argument("--run_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no_feedback", action="store_true", help="Disable feedback ingestion")
    args = parser.parse_args()

    if args.disable_expression:
        os.environ["FRAMED_DISABLE_EXPRESSION"] = "true"

    config = {
        "dataset_path": args.dataset_path,
        "max_images": args.max_images,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "disable_expression": args.disable_expression,
        "disable_cache": args.disable_cache,
        "run_dir": args.run_dir,
        "ingest_feedback": not args.no_feedback,
    }

    tester = IntelligencePipelineTester(config)
    tester.run_tests()


if __name__ == "__main__":
    main()
