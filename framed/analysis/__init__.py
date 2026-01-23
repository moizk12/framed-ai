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

__all__ = [
    'analyze_image',
    'run_full_analysis',
    'load_echo_memory',
    'save_echo_memory',
    'update_echo_memory',
    'ask_echo',
    'generate_merged_critique',
    'generate_remix_prompt',
    'ensure_directories'
]