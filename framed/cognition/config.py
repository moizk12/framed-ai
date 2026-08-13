"""Cognition feature flags and paths."""

from __future__ import annotations

import os
from pathlib import Path

from framed.analysis.runtime_paths import BASE_DATA_DIR


def cognition_enabled() -> bool:
    return os.getenv("FRAMED_COGNITION_V1", "").lower() in ("1", "true", "yes")


def cognition_data_dir() -> Path:
    root = os.getenv("FRAMED_COGNITION_DIR") or os.path.join(BASE_DATA_DIR, "cognition")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    (path / "artefacts").mkdir(parents=True, exist_ok=True)
    return path


def cognition_db_path() -> Path:
    return cognition_data_dir() / "cognition_ledger.sqlite3"


def identity_store_path() -> Path:
    return cognition_data_dir() / "identity.json"
