"""Cognitive run modes and purposes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

# Run purposes eligible to become retrieval memories (indexed on close).
RETRIEVAL_ELIGIBLE_PURPOSES = frozenset({"live", "demo_seed"})

# Purposes recorded in ledger but never indexed for ordinary retrieval.
RETRIEVAL_INELIGIBLE_PURPOSES = frozenset(
    {"baseline", "control", "replay", "memory_enabled", "migration", "failed", "diagnostic"}
)


class RunMode(str, Enum):
    BASELINE = "baseline"
    MEMORY_ENABLED = "memory_enabled"
    CONTROL = "control"
    REPLAY = "replay"


class RunPurpose(str, Enum):
    LIVE = "live"
    BASELINE = "baseline"
    MEMORY_ENABLED = "memory_enabled"
    CONTROL = "control"
    REPLAY = "replay"
    MIGRATION = "migration"
    DEMO_SEED = "demo_seed"
    FAILED = "failed"
    DIAGNOSTIC = "diagnostic"


class SameAssetPolicy(str, Enum):
    EXCLUDE = "exclude"
    ALLOW_RELATED_REVISION = "allow_related_revision"
    ALLOW_REPLAY = "allow_replay"


def purpose_from_mode(mode: RunMode, *, explicit: Optional[RunPurpose] = None) -> RunPurpose:
    if explicit is not None:
        return explicit
    mapping = {
        RunMode.BASELINE: RunPurpose.BASELINE,
        RunMode.CONTROL: RunPurpose.CONTROL,
        RunMode.REPLAY: RunPurpose.REPLAY,
        RunMode.MEMORY_ENABLED: RunPurpose.LIVE,
    }
    return mapping.get(mode, RunPurpose.LIVE)


VALID_MODE_PURPOSE_PAIRS = frozenset(
    {
        (RunMode.BASELINE, RunPurpose.BASELINE),
        (RunMode.CONTROL, RunPurpose.CONTROL),
        (RunMode.REPLAY, RunPurpose.REPLAY),
        (RunMode.MEMORY_ENABLED, RunPurpose.LIVE),
        (RunMode.MEMORY_ENABLED, RunPurpose.MEMORY_ENABLED),
        (RunMode.MEMORY_ENABLED, RunPurpose.DEMO_SEED),
        (RunMode.MEMORY_ENABLED, RunPurpose.DIAGNOSTIC),
    }
)


def validate_mode_purpose(mode: RunMode, purpose: RunPurpose) -> None:
    if (mode, purpose) not in VALID_MODE_PURPOSE_PAIRS:
        raise ValueError(f"Incompatible RunMode {mode.value!r} and RunPurpose {purpose.value!r}")


def is_retrieval_eligible(purpose: RunPurpose) -> bool:
    return purpose.value in RETRIEVAL_ELIGIBLE_PURPOSES


@dataclass
class CognitiveRun:
    run_id: str
    episode_id: str
    mode: RunMode
    run_purpose: RunPurpose
    state_version_id: str
    context_fingerprint: Optional[str]
    retrieval_enabled: bool
    model_provenance: Dict[str, Any]
    prompt_provenance: Dict[str, Any]
    started_at: str
    completed_at: Optional[str] = None
    baseline_run_id: Optional[str] = None
    comparison_group_id: Optional[str] = None
    retrieval_eligible: bool = False
