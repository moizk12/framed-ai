"""Persist actor_id and workspace_id across sessions."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Tuple

from framed.cognition.config import identity_store_path


def load_or_create_identity() -> Tuple[str, str]:
    path: Path = identity_store_path()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["actor_id"], data["workspace_id"]
    actor_id = str(uuid.uuid4())
    workspace_id = str(uuid.uuid4())
    path.write_text(
        json.dumps({"actor_id": actor_id, "workspace_id": workspace_id}, indent=2),
        encoding="utf-8",
    )
    return actor_id, workspace_id


def get_identity() -> Dict[str, str]:
    actor_id, workspace_id = load_or_create_identity()
    return {"actor_id": actor_id, "workspace_id": workspace_id}
