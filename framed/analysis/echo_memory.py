"""ECHO memory persistence (JSON list on disk)."""

import json
import os
from typing import Any, Dict, List

from .runtime_paths import ECHO_MEMORY_PATH


def load_echo_memory() -> List[Dict[str, Any]]:
    if os.path.exists(ECHO_MEMORY_PATH):
        try:
            with open(ECHO_MEMORY_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


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

    memory = memory[-10:]
    with open(ECHO_MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2, default=convert)


def update_echo_memory(photo_data: Dict[str, Any]) -> None:
    memory = load_echo_memory()
    memory.append(photo_data)
    save_echo_memory(memory)

