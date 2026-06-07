"""Merge memory stores and promote correction rules from eval manifests."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runtime_paths import BASE_DATA_DIR

logger = logging.getLogger(__name__)

CONSOLIDATION_LOG_PATH = os.path.join(BASE_DATA_DIR, "consolidation_log.json")


@dataclass
class ConsolidationReport:
    timestamp: str
    dry_run: bool
    source_entries: Dict[str, int] = field(default_factory=dict)
    merged_groups: int = 0
    promoted_rules: List[Dict[str, Any]] = field(default_factory=list)
    contradictions_resolved: int = 0
    stale_entries_marked: int = 0
    echo_promoted: int = 0
    temporal_patterns_consolidated: int = 0
    duration_sec: float = 0.0
    errors: List[str] = field(default_factory=list)


def _load_correction_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    slots: List[Dict[str, Any]] = []
    for _cat, items in data.get("categories", {}).items():
        if isinstance(items, list):
            for slot in items:
                if slot.get("correction_note"):
                    slots.append(slot)
    return slots


def _merge_interpretive_duplicates(dry_run: bool) -> tuple[int, int]:
    from . import interpretive_memory as im

    memory = im.load_memory()
    if not memory:
        return 0, 0

    groups: Dict[str, List[int]] = {}
    for idx, entry in enumerate(memory):
        if entry.get("status") == "consolidated":
            continue
        sig = json.dumps(entry.get("pattern_signature", {}), sort_keys=True)
        groups.setdefault(sig, []).append(idx)

    merged = 0
    contradictions = 0
    for _sig, indices in groups.items():
        if len(indices) < 2:
            continue
        entries = [memory[i] for i in indices]
        interpretations = {e.get("chosen_interpretation") for e in entries if e.get("chosen_interpretation")}
        if len(interpretations) > 1:
            contradictions += len(interpretations) - 1
        keeper = max(indices, key=lambda i: memory[i].get("timestamp", ""))
        for i in indices:
            if i == keeper:
                memory[i]["status"] = "semantic_summary"
                memory[i]["consolidated_at"] = datetime.now(timezone.utc).isoformat()
            else:
                memory[i]["status"] = "consolidated"
                memory[i]["superseded_by_index"] = keeper
                merged += 1

    if not dry_run and (merged or contradictions):
        im.save_memory(memory)
    return merged, contradictions


def run_consolidation_pass(
    *,
    correction_manifest: Optional[Path] = None,
    dry_run: bool = False,
) -> ConsolidationReport:
    from . import echo_memory as em
    from . import interpretive_memory as im
    from . import temporal_memory as tm

    start = time.perf_counter()
    report = ConsolidationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dry_run=dry_run,
    )

    report.source_entries = {
        "interpretive": len(im.load_memory()),
        "temporal_patterns": len(tm.load_temporal_memory().get("patterns", {})),
        "echo": len(em.load_echo_memory()),
        "unconsolidated_interpretive": len(im.list_unconsolidated_entries()),
    }

    promoted_ids: set[str] = set()

    if correction_manifest and correction_manifest.exists():
        for slot in _load_correction_manifest(correction_manifest):
            note = slot.get("correction_note", "")
            failure_mode = slot.get("expected_failure_mode") or "general"
            image_id = slot.get("id", "")
            if image_id in promoted_ids:
                continue
            pattern_sig = {
                "eval_slot_id": image_id,
                "category_file": slot.get("file"),
                "failure_mode": failure_mode,
            }
            if dry_run:
                report.promoted_rules.append({
                    "failure_mode": failure_mode,
                    "correction_note": note,
                    "image_id": image_id,
                    "dry_run": True,
                })
                promoted_ids.add(image_id)
            elif im.promote_correction_rule(failure_mode, note, pattern_sig, image_id=image_id):
                report.promoted_rules.append({
                    "failure_mode": failure_mode,
                    "correction_note": note,
                    "image_id": image_id,
                })
                promoted_ids.add(image_id)
                em.store_correction_echo(image_id, note, failure_mode)

    for cand in em.extract_promotion_candidates():
        image_id = cand.get("image_id") or ""
        if image_id in promoted_ids:
            continue
        if dry_run:
            report.echo_promoted += 1
            promoted_ids.add(image_id)
            continue
        if im.promote_correction_rule(
            cand.get("failure_mode", "general"),
            cand.get("correction_note", ""),
            cand.get("pattern_signature", {}),
            image_id=image_id or None,
        ):
            report.echo_promoted += 1
            promoted_ids.add(image_id)

    memory = tm.load_temporal_memory()
    for signature in list(memory.get("patterns", {}).keys()):
        result = tm.consolidate_pattern_history(signature, dry_run=dry_run)
        if result.get("consolidated"):
            report.temporal_patterns_consolidated += 1
            report.contradictions_resolved += int(result.get("disagreements_resolved", 0))

    merged, contradictions = _merge_interpretive_duplicates(dry_run)
    report.merged_groups = merged
    report.contradictions_resolved += contradictions
    report.stale_entries_marked = merged

    report.duration_sec = round(time.perf_counter() - start, 3)

    if not dry_run:
        os.makedirs(os.path.dirname(CONSOLIDATION_LOG_PATH), exist_ok=True)
        with open(CONSOLIDATION_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("Wrote consolidation log: %s", CONSOLIDATION_LOG_PATH)

    return report
