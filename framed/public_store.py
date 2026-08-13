"""Process-local public beta state, isolated from research persistence."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock


class PublicBetaStore:
    """Minimal replaceable boundary for analysis/feedback association.

    This store is intentionally process-local. It is not continuity, durable
    product persistence, or an adapter to any research database.
    """

    def __init__(self) -> None:
        self._analysis_ids: set[str] = set()
        self._feedback: list[dict] = []
        self._lock = Lock()

    def record_analysis(self, analysis_id: str) -> None:
        with self._lock:
            self._analysis_ids.add(analysis_id)

    def has_analysis(self, analysis_id: str) -> bool:
        with self._lock:
            return analysis_id in self._analysis_ids

    def record_feedback(self, feedback: dict) -> None:
        with self._lock:
            self._feedback.append(deepcopy(feedback))

    def feedback_for(self, analysis_id: str) -> list[dict]:
        with self._lock:
            return [deepcopy(item) for item in self._feedback if item["analysis_id"] == analysis_id]
