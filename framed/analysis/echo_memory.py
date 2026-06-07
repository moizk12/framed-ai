"""ECHO memory persistence (JSON list on disk)."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from .runtime_paths import ECHO_MEMORY_PATH

PROMOTION_TAGS = frozenset({"correction", "moiz_rule"})
ECHO_KEEP = 10


def load_echo_memory() -> List[Dict[str, Any]]:
    if os.path.exists(ECHO_MEMORY_PATH):
        try:
            with open(ECHO_MEMORY_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def tag_echo_entry(entry: Dict[str, Any], tag: str) -> Dict[str, Any]:
    tags = entry.setdefault("tags", [])
    if tag not in tags:
        tags.append(tag)
    return entry


def extract_promotion_candidates() -> List[Dict[str, Any]]:
    candidates = []
    for entry in load_echo_memory():
        tags = set(entry.get("tags", []))
        if tags & PROMOTION_TAGS:
            candidates.append({
                "image_id": entry.get("image_id"),
                "correction_note": entry.get("correction_note", ""),
                "failure_mode": entry.get("failure_mode", "general"),
                "pattern_signature": entry.get("pattern_signature", {}),
            })
    return candidates


def store_correction_echo(image_id: str, correction_note: str, failure_mode: str) -> None:
    entry = tag_echo_entry({
        "image_id": image_id,
        "correction_note": correction_note,
        "failure_mode": failure_mode,
        "pattern_signature": {"eval_slot_id": image_id, "failure_mode": failure_mode},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, "moiz_rule")
    tag_echo_entry(entry, "correction")
    memory = load_echo_memory()
    memory.append(entry)
    save_echo_memory(memory)


def save_echo_memory(memory: List[Dict[str, Any]]) -> None:
    def convert(o):
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    promoted = [e for e in memory if set(e.get("tags", [])) & PROMOTION_TAGS]
    regular = [e for e in memory if e not in promoted]
    trimmed = regular[-ECHO_KEEP:]
    combined = promoted + trimmed
    seen = set()
    deduped = []
    for e in combined:
        key = json.dumps(e, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    with open(ECHO_MEMORY_PATH, "w") as f:
        json.dump(deduped, f, indent=2, default=convert)


def update_echo_memory(photo_data: Dict[str, Any]) -> None:
    memory = load_echo_memory()
    memory.append(photo_data)
    save_echo_memory(memory)
