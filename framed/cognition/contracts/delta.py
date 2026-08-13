"""Field-level deliberation delta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class DeliberationDelta:
    field_changed: str
    baseline_value: Any
    memory_condition_value: Any
    source_memory_refs: List[str]
    mechanism: str
    reason: str
