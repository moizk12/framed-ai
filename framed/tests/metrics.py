"""
Metrics Computation for FRAMED Intelligence Pipeline Tests

Computes aggregate metrics from test results.
"""

from typing import Dict, List, Any
from collections import defaultdict
import statistics


class IntelligenceMetrics:
    """Aggregate intelligence health metrics."""
    
    def __init__(self):
        self.confidences = []
        self.uncertainty_acknowledged_count = 0
        self.multiple_hypotheses_count = 0
        self.total_images = 0
    
    def add_result(self, result: Dict[str, Any]):
        """Add a single test result."""
        self.total_images += 1
        
        # Extract confidence
        intelligence = result.get("intelligence", {})
        recognition = intelligence.get("recognition", {})
        confidence = recognition.get("confidence", 0.85)
        self.confidences.append(confidence)
        
        # Check uncertainty acknowledgment
        if result.get("uncertainty_acknowledged", False):
            self.uncertainty_acknowledged_count += 1
        
        # Check multiple hypotheses
        meta_cognition = intelligence.get("meta_cognition", {})
        alternatives = meta_cognition.get("rejected_alternatives", [])
        if len(alternatives) > 0:
            self.multiple_hypotheses_count += 1
    
    def compute(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        if not self.confidences:
            return {
                "average_confidence": 0.0,
                "confidence_variance": 0.0,
                "uncertainty_acknowledged_percent": 0.0,
                "multiple_hypotheses_percent": 0.0,
            }
        
        return {
            "average_confidence": statistics.mean(self.confidences),
            "confidence_variance": statistics.variance(self.confidences) if len(self.confidences) > 1 else 0.0,
            "confidence_std": statistics.stdev(self.confidences) if len(self.confidences) > 1 else 0.0,
            "uncertainty_acknowledged_percent": (self.uncertainty_acknowledged_count / self.total_images) * 100,
            "multiple_hypotheses_percent": (self.multiple_hypotheses_count / self.total_images) * 100,
        }


class FailureMetrics:
    """Aggregate failure metrics."""
    
    def __init__(self):
        self.hallucination_count = 0
        self.overconfidence_count = 0
        self.reflection_failure_escape_count = 0
        self.mentor_drift_count = 0
        self.total_images = 0
    
    def add_result(self, result: Dict[str, Any]):
        """Add a single test result."""
        self.total_images += 1
        
        # Check for hallucination
        if result.get("hallucination_detected", False):
            self.hallucination_count += 1
        
        # Check for overconfidence
        reflection = result.get("reflection_report", {})
        overconfidence_score = reflection.get("overconfidence_score", 1.0)
        if overconfidence_score < 0.5:
            self.overconfidence_count += 1
        
        # Check reflection failure escape
        if reflection.get("requires_regeneration", False) and not result.get("regenerated", False):
            self.reflection_failure_escape_count += 1
        
        # Check mentor drift
        mentor_drift_score = reflection.get("mentor_drift_score", 1.0)
        if mentor_drift_score < 0.5:
            self.mentor_drift_count += 1
    
    def compute(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        if self.total_images == 0:
            return {
                "hallucination_rate": 0.0,
                "overconfidence_rate": 0.0,
                "reflection_failure_escape_rate": 0.0,
                "mentor_drift_frequency": 0.0,
            }
        
        return {
            "hallucination_rate": (self.hallucination_count / self.total_images) * 100,
            "overconfidence_rate": (self.overconfidence_count / self.total_images) * 100,
            "reflection_failure_escape_rate": (self.reflection_failure_escape_count / self.total_images) * 100,
            "mentor_drift_frequency": (self.mentor_drift_count / self.total_images) * 100,
        }


class LearningMetrics:
    """Aggregate learning metrics."""
    
    def __init__(self):
        self.memory_updates = 0
        self.confidence_adjustments = []
        self.new_patterns_stored = 0
        self.evolution_entries_added = 0
        self.total_images = 0
    
    def add_result(self, result: Dict[str, Any]):
        """Add a single test result."""
        self.total_images += 1
        
        learning = result.get("learning_impact", {})
        
        if learning.get("memory_updated", False):
            self.memory_updates += 1
        
        confidence_adjusted = learning.get("confidence_adjusted", 0.0)
        if confidence_adjusted != 0.0:
            self.confidence_adjustments.append(confidence_adjusted)
        
        if learning.get("new_pattern_stored", False):
            self.new_patterns_stored += 1
        
        if learning.get("evolution_entry_added", False):
            self.evolution_entries_added += 1
    
    def compute(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        if self.total_images == 0:
            return {
                "memory_growth_rate": 0.0,
                "correction_effectiveness": 0.0,
                "average_confidence_adjustment": 0.0,
            }
        
        avg_confidence_adjustment = (
            statistics.mean(self.confidence_adjustments) if self.confidence_adjustments else 0.0
        )
        
        return {
            "memory_growth_rate": (self.memory_updates / self.total_images) * 100,
            "correction_effectiveness": (len(self.confidence_adjustments) / self.total_images) * 100,
            "average_confidence_adjustment": avg_confidence_adjustment,
            "new_patterns_stored": self.new_patterns_stored,
            "evolution_entries_added": self.evolution_entries_added,
        }


def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute all aggregate metrics from test results.
    
    Args:
        results: List of test result dictionaries
    
    Returns:
        Dictionary with all computed metrics
    """
    intelligence_metrics = IntelligenceMetrics()
    failure_metrics = FailureMetrics()
    learning_metrics = LearningMetrics()
    
    for result in results:
        intelligence_metrics.add_result(result)
        failure_metrics.add_result(result)
        learning_metrics.add_result(result)
    
    return {
        "intelligence_health": intelligence_metrics.compute(),
        "failure_metrics": failure_metrics.compute(),
        "learning_metrics": learning_metrics.compute(),
        "total_images": len(results),
    }
