"""Episode open/close helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from framed.cognition.ledger.sqlite_store import CognitionLedger, get_ledger


def project_envelope(episode_id: str, ledger: Optional[CognitionLedger] = None) -> Dict[str, Any]:
    """Project a close snapshot envelope from append-only events."""
    ledger = ledger or get_ledger()
    events = ledger.get_episode_events(episode_id)
    return {
        "schema": "episode_envelope_v1",
        "episode_id": episode_id,
        "event_count": len(events),
        "events": [{"event_type": e["event_type"], "sequence_num": e["sequence_num"]} for e in events],
    }
