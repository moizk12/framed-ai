"""
Test imports for new intelligence architecture modules
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all new intelligence modules can be imported"""
    errors = []
    
    print("Testing Intelligence Architecture Imports...")
    print("=" * 60)
    
    # Test Phase 0: LLM Provider
    try:
        from framed.analysis.llm_provider import (
            call_model_a,
            call_model_b,
            get_llm_provider,
        )
        print("✅ Phase 0: LLM Provider - All imports successful")
    except Exception as e:
        errors.append(f"Phase 0 (LLM Provider): {e}")
        print(f"❌ Phase 0: LLM Provider - Import failed: {e}")
    
    # Test Phase 1: Intelligence Core
    try:
        from framed.analysis.intelligence_core import (
            framed_intelligence,
            reason_about_recognition,
            reason_about_thinking,
            reason_about_evolution,
            reason_about_feeling,
            reason_about_trajectory,
            reason_about_mentorship,
            reason_about_past_errors,
        )
        print("✅ Phase 1: Intelligence Core - All imports successful")
    except Exception as e:
        errors.append(f"Phase 1 (Intelligence Core): {e}")
        print(f"❌ Phase 1: Intelligence Core - Import failed: {e}")
    
    # Test Phase 2: Temporal Memory
    try:
        from framed.analysis.temporal_memory import (
            create_pattern_signature,
            store_interpretation,
            query_memory_patterns,
            track_user_trajectory,
            format_temporal_memory_for_intelligence,
        )
        print("✅ Phase 2: Temporal Memory - All imports successful")
    except Exception as e:
        errors.append(f"Phase 2 (Temporal Memory): {e}")
        print(f"❌ Phase 2: Temporal Memory - Import failed: {e}")
    
    # Test Phase 3: Expression Layer
    try:
        from framed.analysis.expression_layer import (
            generate_poetic_critique,
            apply_mentor_hierarchy,
            integrate_self_correction,
        )
        print("✅ Phase 3: Expression Layer - All imports successful")
    except Exception as e:
        errors.append(f"Phase 3 (Expression Layer): {e}")
        print(f"❌ Phase 3: Expression Layer - Import failed: {e}")
    
    # Test Phase 4: Learning System
    try:
        from framed.analysis.learning_system import (
            recognize_patterns,
            learn_implicitly,
            calibrate_explicitly,
        )
        print("✅ Phase 4: Learning System - All imports successful")
    except Exception as e:
        errors.append(f"Phase 4 (Learning System): {e}")
        print(f"❌ Phase 4: Learning System - Import failed: {e}")
    
    # Test Integration Points
    try:
        from framed.analysis.vision import analyze_image, run_full_analysis
        print("✅ Integration: vision.py imports successful")
    except Exception as e:
        errors.append(f"Integration (vision.py): {e}")
        print(f"❌ Integration: vision.py - Import failed: {e}")
    
    try:
        from framed.routes import main
        print("✅ Integration: routes.py imports successful")
    except Exception as e:
        errors.append(f"Integration (routes.py): {e}")
        print(f"❌ Integration: routes.py - Import failed: {e}")
    
    print("=" * 60)
    
    if errors:
        print(f"\n❌ {len(errors)} import error(s) found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
