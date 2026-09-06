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
FROZEN_DELIBERATION_INPUT_SCHEMA = "frozen_deliberation_input_v1"
GOVERNANCE_POLICY_VERSION = "slice_a_provisional_confidence_v1"

# Slice B controlled learning (additive; does not replace Slice A schemas)
OUTCOME_SCHEMA = "outcome_v1"
UPDATE_PROPOSAL_SCHEMA = "update_proposal_v1"
PROPOSAL_EVALUATION_SCHEMA = "proposal_evaluation_v1"
PROMOTION_DECISION_SCHEMA = "promotion_decision_v1"
ROLLBACK_RECORD_SCHEMA = "rollback_record_v1"
BELIEF_POLICY_VERSION = "slice_b_promote_episode_belief_v1"
PROPOSAL_GENERATOR_ID = "proposal_generator"
ALLOWED_PROMOTION_AUTHORITIES = frozenset({"human", "testdaemon"})
FORBIDDEN_PROMOTION_AUTHORITIES = frozenset({"model", "self", "proposal_generator"})
PROMOTABLE_VERDICTS = frozenset({"useful", "correction"})
