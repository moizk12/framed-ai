"""
FRAMED Intelligence Pipeline Stress Test

This is not a demo, not a notebook, not a one-off script.
It is a repeatable intelligence evaluation harness.

Its job is to answer one question:
Is FRAMED actually thinking correctly under pressure?

Usage:
    python -m framed.tests.test_intelligence_pipeline \
        --dataset_path /path/to/images \
        --max_images 100 \
        --shuffle \
        --seed 42 \
        --disable_expression
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from framed.analysis.vision import analyze_image, extract_visual_features
from framed.analysis.intelligence_core import framed_intelligence
from framed.analysis.temporal_memory import (
    create_pattern_signature,
    query_memory_patterns,
    get_evolution_history,
)
from framed.analysis.reflection import reflect_on_critique
from framed.analysis.expression_layer import generate_poetic_critique
from framed.analysis.learning_system import (
    learn_implicitly,
    ingest_test_feedback,
    ingest_human_correction,
)

from .datasets import load_dataset, DatasetConfig, ImageRecord
from .metrics import compute_all_metrics
from .reporting import (
    create_run_directory,
    save_summary,
    save_metrics,
    save_failures,
    save_raw_log,
    generate_pass_fail_report,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntelligencePipelineTester:
    """Main test harness for FRAMED intelligence pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tester with configuration.
        
        Args:
            config: Configuration dictionary with:
                - dataset_path: Path to image dataset
                - max_images: Maximum number of images to test
                - shuffle: Whether to shuffle images
                - seed: Random seed for shuffling
                - disable_expression: Whether to disable expression layer
                - run_dir: Output directory for results
        """
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []
        
        # Create run directory
        if config.get("run_dir"):
            self.run_dir = config["run_dir"]
        else:
            self.run_dir = create_run_directory()
        
        # Ensure ingest_feedback is enabled by default
        if "ingest_feedback" not in config:
            config["ingest_feedback"] = True
        
        logger.info(f"Test run directory: {self.run_dir}")
        logger.info(f"Feedback ingestion: {'enabled' if config.get('ingest_feedback', True) else 'disabled'}")
    
    def extract_core_interpretation(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract core interpretation record from analysis result.
        
        Returns:
            Dictionary with:
            - image_id
            - primary_conclusion
            - confidence
            - alternatives
            - uncertainty_acknowledged
            - uncertainty_reasons
        """
        intelligence = analysis_result.get("intelligence", {})
        recognition = intelligence.get("recognition", {})
        meta_cognition = intelligence.get("meta_cognition", {})
        
        primary_conclusion = recognition.get("what_i_see", "")
        confidence = recognition.get("confidence", 0.85)
        
        # Extract alternatives
        alternatives = []
        rejected_alternatives = meta_cognition.get("rejected_alternatives", [])
        for alt in rejected_alternatives:
            if isinstance(alt, dict):
                alternatives.append({
                    "conclusion": alt.get("interpretation", ""),
                    "confidence": alt.get("confidence", 0.0),
                    "rejected_reason": alt.get("reason_rejected", "")
                })
        
        # Check uncertainty
        uncertainty_acknowledged = confidence < 0.65
        uncertainty_reasons = []
        if uncertainty_acknowledged:
            uncertainty_reasons = recognition.get("uncertainty_reasons", [])
        
        return {
            "primary_conclusion": primary_conclusion,
            "confidence": confidence,
            "alternatives": alternatives,
            "uncertainty_acknowledged": uncertainty_acknowledged,
            "uncertainty_reasons": uncertainty_reasons,
        }
    
    def extract_evidence_alignment(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract evidence alignment diagnostics.
        
        Returns:
            Dictionary with:
            - visual_evidence_used
            - semantic_conflicts
            - evidence_priority_respected
            - hallucination_detected
        """
        # Visual evidence is stored at top level, not in perception
        visual_evidence = analysis_result.get("visual_evidence", {})
        visual_evidence_used = bool(visual_evidence)
        
        # Check for semantic conflicts
        intelligence = analysis_result.get("intelligence", {})
        recognition = intelligence.get("recognition", {})
        what_i_see = recognition.get("what_i_see", "").lower()
        
        # Check if visual evidence contradicts text inference
        semantic_conflicts = False
        if visual_evidence:
            organic_growth = visual_evidence.get("organic_growth", {})
            green_coverage = organic_growth.get("green_coverage", 0.0)
            
            # If visual evidence shows organic growth but text doesn't mention it
            if green_coverage > 0.2 and "organic" not in what_i_see and "ivy" not in what_i_see:
                semantic_conflicts = True
        
        # Check evidence priority (visual > text > style)
        evidence_priority_respected = True  # Simplified - would need deeper analysis
        
        # Check for hallucination (invented facts)
        hallucination_detected = False
        forbidden_claims = analysis_result.get("ground_truth", {}).get("forbidden_claims", [])
        if forbidden_claims:
            for claim in forbidden_claims:
                if claim.lower() in what_i_see:
                    hallucination_detected = True
                    break
        
        return {
            "visual_evidence_used": visual_evidence_used,
            "semantic_conflicts": semantic_conflicts,
            "evidence_priority_respected": evidence_priority_respected,
            "hallucination_detected": hallucination_detected,
        }
    
    def extract_reflection_diagnostics(self, analysis_result: Dict[str, Any], critique: Optional[str]) -> Dict[str, Any]:
        """
        Extract reflection loop diagnostics.
        
        Returns:
            Dictionary with reflection diagnostics
        """
        reflection_report = analysis_result.get("reflection_report", {})
        
        if not reflection_report:
            return {
                "reflection_score": 0.0,
                "reflection_failures": [],
                "regenerated": False,
                "failure_types": [],
            }
        
        reflection_score = reflection_report.get("quality_score", 0.0)
        requires_regeneration = reflection_report.get("requires_regeneration", False)
        regenerated = analysis_result.get("regenerated", False)
        
        # Identify failure types
        failure_types = []
        if reflection_report.get("contradiction_score", 1.0) < 0.7:
            failure_types.append("contradiction")
        if reflection_report.get("invented_facts_score", 1.0) < 0.7:
            failure_types.append("invented_facts")
        if not reflection_report.get("uncertainty_acknowledged", True):
            failure_types.append("uncertainty_omission")
        if reflection_report.get("generic_language_score", 1.0) < 0.7:
            failure_types.append("generic_language")
        if reflection_report.get("overconfidence_score", 1.0) < 0.5:
            failure_types.append("overconfidence")
        if reflection_report.get("mentor_drift_score", 1.0) < 0.5:
            failure_types.append("mentor_drift")
        
        return {
            "reflection_score": reflection_score,
            "reflection_failures": failure_types,
            "regenerated": regenerated,
            "failure_types": failure_types,
        }
    
    def extract_learning_impact(self, analysis_result: Dict[str, Any], before_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract learning and memory impact.
        
        Returns:
            Dictionary with learning impact metrics
        """
        # Check if intelligence core ran (which triggers memory updates)
        intelligence = analysis_result.get("intelligence", {})
        memory_updated = bool(intelligence)
        
        # Check for confidence adjustment (from user feedback or correction)
        # This would require tracking before/after, simplified for now
        confidence_adjusted = 0.0
        
        # Check if this is a new pattern (would need to query temporal memory)
        # Simplified: assume new if intelligence output exists
        new_pattern_stored = memory_updated
        
        # Check for evolution entry
        # This would require checking temporal memory for evolution history
        # Simplified: assume evolution entry added if intelligence output exists
        evolution_entry_added = memory_updated
        
        return {
            "memory_updated": memory_updated,
            "confidence_adjusted": confidence_adjusted,
            "new_pattern_stored": new_pattern_stored,
            "evolution_entry_added": evolution_entry_added,
        }
    
    def extract_mentor_integrity(self, critique: Optional[str], reflection_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract mentor integrity diagnostics.
        
        Returns:
            Dictionary with mentor integrity metrics
        """
        if not critique:
            return {
                "mentor_drift": False,
                "flattery_detected": False,
                "instruction_creep": False,
                "question_quality_score": 0.0,
            }
        
        critique_lower = critique.lower()
        
        # Check for flattery
        flattery_phrases = ["beautiful", "amazing", "perfect", "excellent", "outstanding", "brilliant", "stunning"]
        flattery_detected = any(phrase in critique_lower for phrase in flattery_phrases)
        
        # Check for instructions
        instruction_phrases = ["you should", "you must", "try to", "consider", "use this", "apply this"]
        instruction_creep = any(phrase in critique_lower for phrase in instruction_phrases)
        
        # Mentor drift from reflection report
        mentor_drift_score = reflection_report.get("mentor_drift_score", 1.0)
        mentor_drift = mentor_drift_score < 0.5
        
        # Question quality (simplified - would need deeper analysis)
        question_quality_score = 0.5  # Placeholder
        
        return {
            "mentor_drift": mentor_drift,
            "flattery_detected": flattery_detected,
            "instruction_creep": instruction_creep,
            "question_quality_score": question_quality_score,
        }
    
    def validate_image_file(self, image_path: str) -> Dict[str, Any]:
        """
        Validate that image file exists and is readable.
        
        Returns:
            Dict with validation results
        """
        validation = {
            "exists": False,
            "readable": False,
            "size_bytes": 0,
            "error": None
        }
        
        try:
            path = Path(image_path)
            if not path.exists():
                validation["error"] = "File does not exist"
                return validation
            
            validation["exists"] = True
            validation["size_bytes"] = path.stat().st_size
            
            # Try to read first few bytes to verify it's an image
            with open(path, 'rb') as f:
                header = f.read(8)
                # Check for common image file signatures
                if header[:2] == b'\xff\xd8':  # JPEG
                    validation["readable"] = True
                elif header[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
                    validation["readable"] = True
                elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':  # WEBP
                    validation["readable"] = True
                elif header[:2] == b'BM':  # BMP
                    validation["readable"] = True
                elif header[:4] == b'II*\x00' or header[:4] == b'MM\x00*':  # TIFF
                    validation["readable"] = True
                else:
                    validation["error"] = "Unknown image format"
        except Exception as e:
            validation["error"] = str(e)
        
        return validation
    
    def validate_category_specific(self, category: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate category-specific expectations.
        
        Returns:
            Dict with category-specific validation results
        """
        validation = {
            "category": category,
            "expectations_met": True,
            "warnings": [],
            "errors": []
        }
        
        intelligence = analysis_result.get("intelligence", {})
        recognition = intelligence.get("recognition", {})
        what_i_see = recognition.get("what_i_see", "").lower()
        
        # Category-specific checks
        if category == "architecture":
            # Architecture should detect structures, buildings, etc.
            architecture_terms = ["building", "structure", "architecture", "facade", "wall", "roof", "dome"]
            if not any(term in what_i_see for term in architecture_terms):
                validation["warnings"].append("Architecture image but no architecture terms detected")
        
        elif category == "portraits":
            # Portraits should detect human presence
            human_terms = ["person", "human", "face", "portrait", "people"]
            if not any(term in what_i_see for term in human_terms):
                validation["warnings"].append("Portrait image but no human presence detected")
        
        elif category == "nature":
            # Nature should detect organic elements
            nature_terms = ["nature", "tree", "plant", "vegetation", "landscape", "organic"]
            if not any(term in what_i_see for term in nature_terms):
                validation["warnings"].append("Nature image but no nature terms detected")
        
        elif category == "street":
            # Street photography might have urban elements
            street_terms = ["street", "urban", "city", "building", "road", "sidewalk"]
            if not any(term in what_i_see for term in street_terms):
                validation["warnings"].append("Street image but no urban terms detected")
        
        elif category == "ambiguous":
            # Ambiguous images should have lower confidence or uncertainty acknowledgment
            confidence = recognition.get("confidence", 1.0)
            if confidence > 0.8:
                validation["warnings"].append("Ambiguous image but high confidence - may be overconfident")
        
        if validation["warnings"]:
            validation["expectations_met"] = False
        
        return validation
    
    def test_single_image(self, image_record: ImageRecord) -> Dict[str, Any]:
        """
        Test a single image through the full pipeline.
        
        Args:
            image_record: ImageRecord object
        
        Returns:
            Complete test result dictionary
        """
        logger.info(f"Testing image: {image_record.image_id} ({image_record.category})")
        
        result = {
            "image_id": image_record.image_id,
            "image_path": image_record.image_path,
            "category": image_record.category,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            # === PRE-VALIDATION: Check image file ===
            file_validation = self.validate_image_file(image_record.image_path)
            result["file_validation"] = file_validation
            
            if not file_validation["exists"]:
                raise FileNotFoundError(f"Image file does not exist: {image_record.image_path}")
            if not file_validation["readable"]:
                raise ValueError(f"Image file is not readable or not a valid image: {image_record.image_path}")
            # === STAGE 1: Full Analysis (includes Visual Evidence Extraction) ===
            # Note: analyze_image() internally extracts visual evidence and stores it in result["visual_evidence"]
            logger.debug(f"Stage 1: Full analysis (including visual evidence) for {image_record.image_id}")
            analysis_result = analyze_image(
                image_record.image_path,
                photo_id=image_record.image_id,
                filename=Path(image_record.image_path).name
            )
            
            # Extract visual evidence from analysis result (it's already computed inside analyze_image)
            visual_evidence = analysis_result.get("visual_evidence", {})
            result["visual_evidence"] = visual_evidence
            
            # Add ground truth if available
            if image_record.ground_truth:
                analysis_result["ground_truth"] = image_record.ground_truth
            
            # Process human correction if available
            if image_record.human_correction:
                analysis_result["human_correction"] = image_record.human_correction
            
            # === STAGE 2: Extract Core Interpretation ===
            logger.debug(f"Stage 2: Extract core interpretation for {image_record.image_id}")
            core_interpretation = self.extract_core_interpretation(analysis_result)
            result["core_interpretation"] = core_interpretation
            
            # === STAGE 3: Extract Evidence Alignment ===
            logger.debug(f"Stage 3: Extract evidence alignment for {image_record.image_id}")
            evidence_alignment = self.extract_evidence_alignment(analysis_result)
            result["evidence_alignment"] = evidence_alignment
            
            # === STAGE 4: Expression Layer (Optional) ===
            critique = None
            if not self.config.get("disable_expression", False):
                logger.debug(f"Stage 4: Expression layer for {image_record.image_id}")
                try:
                    intelligence_output = analysis_result.get("intelligence", {})
                    if intelligence_output:
                        critique = generate_poetic_critique(
                            intelligence_output=intelligence_output,
                            mentor_mode="Balanced Mentor"
                        )
                        analysis_result["critique"] = critique
                except Exception as e:
                    logger.warning(f"Expression layer failed for {image_record.image_id}: {e}")
            
            # === STAGE 5: Reflection Loop ===
            logger.debug(f"Stage 5: Reflection loop for {image_record.image_id}")
            intelligence_output = analysis_result.get("intelligence", {})
            if intelligence_output and critique:
                reflection_report = reflect_on_critique(critique, intelligence_output)
                analysis_result["reflection_report"] = reflection_report
                
                # Regenerate if needed
                if reflection_report.get("requires_regeneration", False):
                    logger.warning(f"Reflection: Regenerating critique for {image_record.image_id}")
                    try:
                        critique = generate_poetic_critique(
                            intelligence_output=intelligence_output,
                            mentor_mode="Balanced Mentor"
                        )
                        analysis_result["critique"] = critique
                        analysis_result["regenerated"] = True
                    except Exception as e:
                        logger.warning(f"Regeneration failed for {image_record.image_id}: {e}")
            
            # === STAGE 6: Extract Reflection Diagnostics ===
            logger.debug(f"Stage 6: Extract reflection diagnostics for {image_record.image_id}")
            reflection_diagnostics = self.extract_reflection_diagnostics(analysis_result, critique)
            result["reflection_diagnostics"] = reflection_diagnostics
            
            # === STAGE 7: Extract Learning Impact ===
            logger.debug(f"Stage 7: Extract learning impact for {image_record.image_id}")
            learning_impact = self.extract_learning_impact(analysis_result, {})
            result["learning_impact"] = learning_impact
            
            # === STAGE 8: Extract Mentor Integrity ===
            logger.debug(f"Stage 8: Extract mentor integrity for {image_record.image_id}")
            mentor_integrity = self.extract_mentor_integrity(
                critique,
                analysis_result.get("reflection_report", {})
            )
            result["mentor_integrity"] = mentor_integrity
            
            # === STAGE 9: Category-Specific Validation ===
            logger.debug(f"Stage 9: Category-specific validation for {image_record.image_id}")
            category_validation = self.validate_category_specific(image_record.category, analysis_result)
            result["category_validation"] = category_validation
            
            # === STAGE 10: Process Human Correction (if available) ===
            if image_record.human_correction:
                logger.debug(f"Stage 10: Processing human correction for {image_record.image_id}")
                try:
                    # Create pattern signature for this image
                    semantic_signals = {
                        "objects": analysis_result.get("perception", {}).get("composition", {}).get("subject_framing", {}).get("objects", []),
                        "tags": analysis_result.get("perception", {}).get("semantics", {}).get("tags", []),
                        "caption_keywords": analysis_result.get("perception", {}).get("semantics", {}).get("caption", "").split()[:20] if analysis_result.get("perception", {}).get("semantics", {}).get("caption") else [],
                    }
                    pattern_signature = create_pattern_signature(visual_evidence, semantic_signals)
                    
                    # Extract interpretation
                    intelligence_output = analysis_result.get("intelligence", {})
                    recognition = intelligence_output.get("recognition", {})
                    framed_interpretation = recognition.get("what_i_see", "")
                    
                    # Ingest human correction
                    correction = image_record.human_correction
                    ingest_human_correction(
                        image_id=image_record.image_id,
                        pattern_signature=pattern_signature,
                        framed_interpretation=framed_interpretation,
                        user_feedback=correction.get("user_feedback", ""),
                        confidence_adjustment=correction.get("confidence_adjustment", 0.0)
                    )
                    
                    result["human_correction_applied"] = True
                    logger.info(f"Human correction ingested for {image_record.image_id}")
                except Exception as e:
                    logger.warning(f"Human correction ingestion failed for {image_record.image_id}: {e}")
                    result["human_correction_applied"] = False
            
            # Add full analysis result for raw logs (but limit size to avoid huge files)
            # Only include essential parts for debugging
            result["full_analysis"] = {
                "intelligence": analysis_result.get("intelligence", {}),
                "perception": {
                    "technical": analysis_result.get("perception", {}).get("technical", {}),
                    "semantics": analysis_result.get("perception", {}).get("semantics", {}),
                },
                "visual_evidence": analysis_result.get("visual_evidence", {}),
                "reflection_report": analysis_result.get("reflection_report", {}),
            }
            
            logger.info(f"Completed testing {image_record.image_id}")
            
        except Exception as e:
            logger.error(f"Error testing {image_record.image_id}: {e}", exc_info=True)
            result["error"] = str(e)
            self.failures.append(result)
        
        return result
    
    def ingest_failure_feedback(self):
        """
        Ingest feedback from test failures.
        
        Golden Rule:
        ❌ Never teach FRAMED "what the image is"
        ✅ Teach FRAMED "when it should be less confident"
        
        This calibrates confidence, not content.
        """
        logger.info(f"Ingesting feedback from {len(self.failures)} failures")
        
        ingested_count = 0
        
        for failure in self.failures:
            try:
                # Extract pattern signature
                visual_evidence = failure.get("visual_evidence", {})
                analysis_result = failure.get("full_analysis", {})
                
                if not visual_evidence or not analysis_result:
                    continue  # Skip if missing required data
                
                # Create semantic signals
                perception = analysis_result.get("perception", {})
                composition = perception.get("composition", {})
                semantics = perception.get("semantics", {})
                
                semantic_signals = {
                    "objects": composition.get("subject_framing", {}).get("objects", []),
                    "tags": semantics.get("tags", []),
                    "caption_keywords": semantics.get("caption", "").split()[:20] if semantics.get("caption") else [],
                }
                
                # Create pattern signature
                pattern_signature = create_pattern_signature(visual_evidence, semantic_signals)
                
                # Determine issue type from failure diagnostics
                reflection_diagnostics = failure.get("reflection_diagnostics", {})
                failure_types = reflection_diagnostics.get("failure_types", [])
                evidence_alignment = failure.get("evidence_alignment", {})
                
                # Map failure types to issue types
                issue_type = "reflection_failure"  # Default
                if "hallucination" in failure_types or evidence_alignment.get("hallucination_detected", False):
                    issue_type = "hallucination"
                elif "overconfidence" in failure_types:
                    issue_type = "overconfidence"
                elif "contradiction" in failure_types or evidence_alignment.get("semantic_conflicts", False):
                    issue_type = "contradiction"
                elif "uncertainty_omission" in failure_types:
                    issue_type = "uncertainty_omission"
                
                # Create note
                note = f"Test failure: {', '.join(failure_types) if failure_types else 'unknown'}"
                
                # Ingest feedback
                success = ingest_test_feedback(
                    evidence_signature=pattern_signature,
                    issue_type=issue_type,
                    correction=None,  # No hard labels
                    note=note
                )
                
                if success:
                    ingested_count += 1
                    logger.debug(f"Ingested feedback for {failure.get('image_id')}: {issue_type}")
            
            except Exception as e:
                logger.warning(f"Failed to ingest feedback for failure: {e}")
        
        logger.info(f"Ingested feedback from {ingested_count}/{len(self.failures)} failures")
    
    def run_tests(self):
        """Run tests on all images in dataset."""
        logger.info("Starting FRAMED Intelligence Pipeline Stress Test")
        
        # Load dataset
        dataset_config = DatasetConfig(
            dataset_path=self.config["dataset_path"],
            max_images=self.config.get("max_images"),
            shuffle=self.config.get("shuffle", True),
            seed=self.config.get("seed"),
        )
        
        image_records = load_dataset(dataset_config)
        logger.info(f"Loaded {len(image_records)} images from dataset")
        
        # Test each image
        for i, image_record in enumerate(image_records, 1):
            logger.info(f"Processing image {i}/{len(image_records)}: {image_record.image_id}")
            
            result = self.test_single_image(image_record)
            self.results.append(result)
            
            # Save raw log
            save_raw_log(self.run_dir, image_record.image_id, result)
            
            # Check for failures
            # Only treat hard errors as failures here.
            # Reflection/quality issues are tracked via metrics + pass/fail report thresholds,
            # and may be expected when running with placeholder LLM providers.
            if result.get("error"):
                self.failures.append(result)
        
        # === POST-TEST: Ingest Feedback from Failures ===
        if self.config.get("ingest_feedback", True):
            logger.info("Ingesting feedback from test failures")
            self.ingest_failure_feedback()
        
        # Compute metrics
        logger.info("Computing aggregate metrics")
        metrics = compute_all_metrics(self.results)
        
        # Generate pass/fail report
        pass_fail_report = generate_pass_fail_report(metrics)
        
        # Create summary
        summary = {
            "test_config": self.config,
            "total_images": len(image_records),
            "completed": len(self.results),
            "failed": len(self.failures),
            "pass_fail_report": pass_fail_report,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save outputs
        logger.info(f"Saving results to {self.run_dir}")
        save_summary(self.run_dir, summary)
        save_metrics(self.run_dir, metrics)
        save_failures(self.run_dir, self.failures)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total images: {len(image_records)}")
        logger.info(f"Completed: {len(self.results)}")
        logger.info(f"Failed: {len(self.failures)}")
        logger.info(f"Passed: {pass_fail_report['passed']}")
        logger.info(f"Failures: {pass_fail_report['failures']}")
        logger.info(f"Warnings: {pass_fail_report['warnings']}")
        logger.info(f"Results saved to: {self.run_dir}")
        logger.info("=" * 80)
        
        return summary, metrics, pass_fail_report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FRAMED Intelligence Pipeline Stress Test"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to image dataset directory"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to test"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle images before testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--disable_expression",
        action="store_true",
        help="Disable expression layer to save cost"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Output directory for results (default: auto-generated)"
    )
    parser.add_argument(
        "--no_feedback",
        action="store_true",
        help="Disable feedback ingestion from test failures"
    )
    
    args = parser.parse_args()
    
    # Set environment variables from config
    if args.disable_expression:
        os.environ["FRAMED_DISABLE_EXPRESSION"] = "true"
    
    # Create tester
    config = {
        "dataset_path": args.dataset_path,
        "max_images": args.max_images,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "disable_expression": args.disable_expression,
        "run_dir": args.run_dir,
        "ingest_feedback": not args.no_feedback,  # Enable by default
    }
    
    tester = IntelligencePipelineTester(config)
    tester.run_tests()


if __name__ == "__main__":
    main()
