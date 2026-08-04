"""Dry-run legacy memory import — Slice A does not mutate production legacy stores."""

from __future__ import annotations

from typing import Any, Dict, List


def dry_run_legacy_import(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report what would be imported without writing."""
    return {
        "schema": "legacy_import_dry_run_v1",
        "would_import_count": len(candidates),
        "note": "Slice A defers semantic promotion and legacy migration.",
        "candidates": candidates[:10],
    }
