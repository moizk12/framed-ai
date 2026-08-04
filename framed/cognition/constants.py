"""Slice A cognition integrity constants."""

from __future__ import annotations

# Retrieval / context bounds
MAX_MEMORY_REFS = 5
MAX_CHARS_PER_REF = 500
MAX_TOTAL_COGNITION_BLOCK_CHARS = 2000

# Ledger concurrency
APPEND_EVENT_MAX_RETRIES = 8
APPEND_EVENT_RETRY_BASE_MS = 5

# Replay bundle
REPLAY_BUNDLE_SCHEMA = "replay_bundle_v1"
PERCEPTION_SNAPSHOT_SCHEMA = "perception_snapshot_v1"
