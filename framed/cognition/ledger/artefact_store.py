"""Canonical JSON hashing and artefact storage."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from framed.cognition.config import cognition_data_dir


def canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def artefact_hash(obj: Any) -> str:
    payload = canonical_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def asset_id_from_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtefactStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (cognition_data_dir() / "artefacts")
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, schema_name: str, schema_version: str, obj: Any) -> str:
        digest = artefact_hash(obj)
        rel = f"{digest[:2]}/{digest}.json"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical_json_dumps(obj).encode("utf-8")
        path.write_bytes(raw)
        return digest, rel, len(raw)

    def get(self, digest: str) -> Dict[str, Any]:
        path = self.root / digest[:2] / f"{digest}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
