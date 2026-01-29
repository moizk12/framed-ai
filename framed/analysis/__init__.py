"""
FRAMED Analysis Module
Computer vision and AI analysis tools

⚠️ CRITICAL FILE - Makes 'analysis' a proper Python package
"""

from .vision import (
    analyze_image,
    run_full_analysis,
    load_echo_memory,
    save_echo_memory,
    update_echo_memory,
    ask_echo,
    generate_merged_critique,
    generate_remix_prompt,
    ensure_directories
)
from .schema import (
    create_empty_analysis_result,
    validate_schema,
    normalize_to_schema
)

# Intelligence Core (Phase 1)
from .intelligence_core import framed_intelligence

# Temporal Memory (Phase 2)
from .temporal_memory import (
    create_pattern_signature,
    store_interpretation,
    query_memory_patterns,
    get_evolution_history,
    format_evolution_history_for_prompt,
    track_user_trajectory,
    format_temporal_memory_for_intelligence,
)

# Expression Layer (Phase 3)
from .expression_layer import (
    generate_poetic_critique,
    apply_mentor_hierarchy,
    integrate_self_correction,
)

# Learning System (Phase 4)
from .learning_system import (
    recognize_patterns,
    learn_implicitly,
    calibrate_explicitly,
    ingest_test_feedback,
    ingest_human_correction,
)

__all__ = [
    'analyze_image',
    'run_full_analysis',
    'load_echo_memory',
    'save_echo_memory',
    'update_echo_memory',
    'ask_echo',
    'generate_merged_critique',
    'generate_remix_prompt',
    'ensure_directories',
    # Intelligence Core
    'framed_intelligence',
    # Temporal Memory
    'create_pattern_signature',
    'store_interpretation',
    'query_memory_patterns',
    'get_evolution_history',
    'format_evolution_history_for_prompt',
    'track_user_trajectory',
    'format_temporal_memory_for_intelligence',
    # Expression Layer
    'generate_poetic_critique',
    'apply_mentor_hierarchy',
    'integrate_self_correction',
    # Learning System
    'recognize_patterns',
    'learn_implicitly',
    'calibrate_explicitly',
    'ingest_test_feedback',
    'ingest_human_correction',
]