"""SQLite append-only cognition ledger."""

from __future__ import annotations

import json
import gc
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from framed.cognition.config import cognition_db_path
from framed.cognition.contracts.memory import MemoryReference, ScoreComponents
from framed.cognition.contracts.runs import CognitiveRun, RunMode, RunPurpose, is_retrieval_eligible
from framed.cognition.ledger.artefact_store import ArtefactStore


class CognitionLedger:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or cognition_db_path()
        self.artefacts = ArtefactStore()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _ensure_schema(self) -> None:
        migrations_dir = Path(__file__).parent / "migrations"
        with self._connect() as conn:
            conn.executescript((migrations_dir / "001_initial.sql").read_text(encoding="utf-8"))
            row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = int(row["v"]) if row and row["v"] is not None else 0
            if current < 1:
                conn.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (1, ?)",
                    (ArtefactStore.utc_now(),),
                )
                current = 1
            if current < 2:
                for stmt in (migrations_dir / "002_run_purpose.sql").read_text(encoding="utf-8").split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            conn.execute(stmt)
                        except sqlite3.OperationalError as exc:
                            if "duplicate column" not in str(exc).lower():
                                raise
                conn.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (2, ?)",
                    (ArtefactStore.utc_now(),),
                )
                conn.execute(
                    """
                    UPDATE cognitive_runs
                    SET run_purpose = CASE mode
                        WHEN 'baseline' THEN 'baseline'
                        WHEN 'control' THEN 'control'
                        WHEN 'replay' THEN 'replay'
                        ELSE 'migration'
                    END
                    WHERE run_purpose = 'migration' AND mode IN ('baseline', 'control', 'replay')
                    """
                )
                conn.execute(
                    """
                    UPDATE cognitive_runs
                    SET retrieval_eligible = CASE
                        WHEN run_purpose IN ('live', 'demo_seed') THEN 1
                        ELSE 0
                    END
                    """
                )
                conn.execute(
                    """
                    DELETE FROM retrieval_index
                    WHERE episode_id IN (
                        SELECT episode_id FROM cognitive_runs
                        WHERE run_purpose NOT IN ('live', 'demo_seed')
                    )
                    """
                )

    def put_artefact(self, schema_name: str, schema_version: str, obj: Any) -> str:
        digest, rel, byte_len = self.artefacts.put(schema_name, schema_version, obj)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO artefacts
                (artefact_hash, schema_name, schema_version, relative_path, byte_length, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (digest, schema_name, schema_version, rel, byte_len, ArtefactStore.utc_now()),
            )
        return digest

    def ensure_demo_states(self, workspace_id: str) -> Tuple[str, str]:
        """Create baseline + memory_enabled demo states if missing."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT state_version_id, label FROM cognitive_state_versions WHERE workspace_id=?",
                (workspace_id,),
            ).fetchall()
            by_label = {r["label"]: r["state_version_id"] for r in rows}
        baseline_id = by_label.get("state_baseline")
        memory_id = by_label.get("state_memory_enabled")
        if not baseline_id:
            baseline_id = str(uuid.uuid4())
            snap = {
                "schema": "state_snapshot_v1",
                "label": "state_baseline",
                "retrieval_policy_version": "v1",
                "retrieval_enabled": False,
                "memory_visibility_horizon": "1970-01-01T00:00:00Z",
                "index_generation": 1,
                "cutoff_score": 0.7,
                "same_asset_policy": "exclude",
                "allowed_lifecycle_states": ["closed"],
                "allowed_epistemic_states": ["inferred", "provisional"],
            }
            snap_hash = self.put_artefact("state_snapshot", "v1", snap)
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO cognitive_state_versions
                    (state_version_id, workspace_id, parent_version_id, label, created_at, is_active, snapshot_artefact_hash)
                    VALUES (?, ?, NULL, 'state_baseline', ?, 0, ?)
                    """,
                    (baseline_id, workspace_id, ArtefactStore.utc_now(), snap_hash),
                )
        if not memory_id:
            memory_id = str(uuid.uuid4())
            snap = {
                "schema": "state_snapshot_v1",
                "label": "state_memory_enabled",
                "retrieval_policy_version": "v1",
                "retrieval_enabled": True,
                "memory_visibility_horizon": "9999-12-31T23:59:59Z",
                "index_generation": 1,
                "cutoff_score": 0.7,
                "same_asset_policy": "exclude",
                "allowed_lifecycle_states": ["closed"],
                "allowed_epistemic_states": ["inferred", "provisional"],
            }
            snap_hash = self.put_artefact("state_snapshot", "v1", snap)
            with self._connect() as conn:
                # First install: activate memory state once at creation
                conn.execute(
                    "UPDATE cognitive_state_versions SET is_active=0 WHERE workspace_id=?",
                    (workspace_id,),
                )
                conn.execute(
                    """
                    INSERT INTO cognitive_state_versions
                    (state_version_id, workspace_id, parent_version_id, label, created_at, is_active, snapshot_artefact_hash)
                    VALUES (?, ?, ?, 'state_memory_enabled', ?, 1, ?)
                    """,
                    (memory_id, workspace_id, baseline_id, ArtefactStore.utc_now(), snap_hash),
                )
        return baseline_id, memory_id

    def get_active_state(self, workspace_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM cognitive_state_versions
                WHERE workspace_id=? AND is_active=1 LIMIT 1
                """,
                (workspace_id,),
            ).fetchone()
            if row is None:
                _, mem = self.ensure_demo_states(workspace_id)
                row = conn.execute(
                    "SELECT * FROM cognitive_state_versions WHERE state_version_id=?",
                    (mem,),
                ).fetchone()
            snap = self.artefacts.get(row["snapshot_artefact_hash"])
            return {"state_version_id": row["state_version_id"], "label": row["label"], "snapshot": snap}

    def activate_state(self, workspace_id: str, label: str) -> str:
        with self._connect() as conn:
            conn.execute(
                "UPDATE cognitive_state_versions SET is_active=0 WHERE workspace_id=?",
                (workspace_id,),
            )
            row = conn.execute(
                """
                SELECT state_version_id FROM cognitive_state_versions
                WHERE workspace_id=? AND label=? LIMIT 1
                """,
                (workspace_id, label),
            ).fetchone()
            if row is None:
                self.ensure_demo_states(workspace_id)
                row = conn.execute(
                    "SELECT state_version_id FROM cognitive_state_versions WHERE workspace_id=? AND label=?",
                    (workspace_id, label),
                ).fetchone()
            conn.execute(
                "UPDATE cognitive_state_versions SET is_active=1 WHERE state_version_id=?",
                (row["state_version_id"],),
            )
            return row["state_version_id"]

    def open_episode(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        asset_id: str,
        goal_type: str,
        goal_instance_id: Optional[str],
        state_version_id: str,
        asset_filename: Optional[str] = None,
        source_kind: str = "live",
    ) -> str:
        episode_id = str(uuid.uuid4())
        now = ArtefactStore.utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes
                (episode_id, workspace_id, actor_id, asset_id, goal_type, goal_instance_id,
                 status, source_kind, asset_filename, created_at, state_version_id)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    workspace_id,
                    actor_id,
                    asset_id,
                    goal_type,
                    goal_instance_id,
                    source_kind,
                    asset_filename,
                    now,
                    state_version_id,
                ),
            )
        return episode_id

    def create_run(self, run: CognitiveRun) -> None:
        eligible = is_retrieval_eligible(run.run_purpose)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cognitive_runs
                (run_id, episode_id, mode, run_purpose, baseline_run_id, comparison_group_id,
                 state_version_id, context_fingerprint, retrieval_enabled, retrieval_eligible,
                 model_provenance_json, prompt_provenance_json, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.episode_id,
                    run.mode.value,
                    run.run_purpose.value,
                    run.baseline_run_id,
                    run.comparison_group_id,
                    run.state_version_id,
                    run.context_fingerprint,
                    1 if run.retrieval_enabled else 0,
                    1 if eligible else 0,
                    json.dumps(run.model_provenance),
                    json.dumps(run.prompt_provenance),
                    run.started_at,
                    run.completed_at,
                ),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cognitive_runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def complete_run(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE cognitive_runs SET completed_at=? WHERE run_id=?",
                (ArtefactStore.utc_now(), run_id),
            )

    def _next_sequence(self, conn: sqlite3.Connection, episode_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(sequence_num), 0) AS m FROM episode_events WHERE episode_id=?",
            (episode_id,),
        ).fetchone()
        return int(row["m"]) + 1

    def append_event(
        self,
        *,
        episode_id: str,
        run_id: str,
        event_type: str,
        payload: Dict[str, Any],
        artefact_hash: Optional[str] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        now = ArtefactStore.utc_now()
        with self._connect() as conn:
            seq = self._next_sequence(conn, episode_id)
            conn.execute(
                """
                INSERT INTO episode_events
                (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    episode_id,
                    run_id,
                    event_type,
                    seq,
                    now,
                    artefact_hash,
                    json.dumps(payload, sort_keys=True),
                ),
            )
        return event_id

    def close_episode(
        self,
        episode_id: str,
        *,
        run_id: str,
        run_purpose: RunPurpose,
        scene_signature: str,
        category_signature: str,
        goal_type: str,
        goal_instance_id: Optional[str],
        final_fingerprint: Optional[str],
        perception_artefact_hash: Optional[str],
    ) -> None:
        now = ArtefactStore.utc_now()
        with self._connect() as conn:
            ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
            conn.execute(
                """
                UPDATE episodes SET status='closed', closed_at=?, final_fingerprint=?, perception_artefact_hash=?
                WHERE episode_id=?
                """,
                (now, final_fingerprint, perception_artefact_hash, episode_id),
            )
            if is_retrieval_eligible(run_purpose):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO retrieval_index
                    (episode_id, workspace_id, actor_id, asset_id, scene_signature, category_signature,
                     goal_type, goal_instance_id, recorded_at, closed_at, source_run_id, run_purpose)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        ep["workspace_id"],
                        ep["actor_id"],
                        ep["asset_id"],
                        scene_signature,
                        category_signature,
                        goal_type,
                        goal_instance_id,
                        ep["created_at"],
                        now,
                        run_id,
                        run_purpose.value,
                    ),
                )

    def list_retrieval_candidates(self, query_workspace: str, query_actor: str) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT ri.*, e.status
                FROM retrieval_index ri
                JOIN episodes e ON e.episode_id = ri.episode_id
                WHERE ri.workspace_id=? AND ri.actor_id=? AND e.status='closed'
                """,
                (query_workspace, query_actor),
            ).fetchall()

    def store_memory_reference(self, run_id: str, ref: MemoryReference, target_episode_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_references
                (memory_ref_id, run_id, target_episode_id, source_episode_id, source_event_id,
                 source_run_id, source_asset_id, source_run_purpose, eligibility_decision,
                 ref_type, epistemic_status, lifecycle_status, memory_role, trust_level,
                 category_score, scene_score, goal_score, relation_score, recency_score, final_score,
                 contamination_flags_json, match_reason, artefact_hash, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ref.memory_ref_id,
                    run_id,
                    target_episode_id,
                    ref.source_episode_id,
                    ref.source_event_id,
                    ref.source_run_id,
                    ref.source_asset_id,
                    ref.source_run_purpose,
                    ref.eligibility_decision,
                    ref.memory_role,
                    ref.epistemic_status,
                    ref.lifecycle_status,
                    ref.memory_role,
                    ref.trust_level,
                    ref.scores.category_score,
                    ref.scores.scene_score,
                    ref.scores.goal_score,
                    ref.scores.relation_score,
                    ref.scores.recency_score,
                    ref.scores.final_score,
                    json.dumps(list(ref.scores.contamination_flags)),
                    ref.match_reason,
                    ref.artefact_hash,
                    ArtefactStore.utc_now(),
                ),
            )

    def get_episode_events(self, episode_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episode_events WHERE episode_id=? ORDER BY sequence_num",
                (episode_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_baseline_snapshot_by_run_id(self, baseline_run_id: str) -> Optional[Dict[str, Any]]:
        run = self.get_run(baseline_run_id)
        if not run or run.get("run_purpose") != RunPurpose.BASELINE.value:
            return None
        for ev in reversed(self.get_episode_events(run["episode_id"])):
            if ev["event_type"] == "deliberation_snapshot":
                return json.loads(ev["payload_json"])
        return None

    def find_compatible_baseline_snapshot(
        self,
        workspace_id: str,
        asset_id: str,
        perception_artefact_hash: str,
        baseline_run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find baseline deliberation snapshot by explicit run link or asset match."""
        if baseline_run_id:
            snap = self.get_baseline_snapshot_by_run_id(baseline_run_id)
            if snap:
                return snap
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT cr.run_id, cr.episode_id
                FROM cognitive_runs cr
                JOIN episodes e ON e.episode_id = cr.episode_id
                WHERE e.workspace_id=? AND e.asset_id=? AND cr.run_purpose='baseline'
                ORDER BY cr.started_at DESC
                """,
                (workspace_id, asset_id),
            ).fetchall()
        for row in rows:
            events = self.get_episode_events(row["episode_id"])
            pe_hash = None
            for ev in events:
                if ev["event_type"] == "perception_completed":
                    pe_hash = json.loads(ev["payload_json"]).get("perception_artefact_hash")
                    break
            if pe_hash != perception_artefact_hash:
                continue
            for ev in reversed(events):
                if ev["event_type"] == "deliberation_snapshot":
                    payload = json.loads(ev["payload_json"])
                    payload["perception_artefact_hash"] = pe_hash
                    return payload
        return None

    def store_deliberation_link(
        self,
        *,
        episode_id: str,
        run_id: str,
        snapshot: Dict[str, Any],
        link_type: str,
    ) -> str:
        return self.append_event(
            episode_id=episode_id,
            run_id=run_id,
            event_type=f"deliberation_{link_type}",
            payload=snapshot,
        )

    def export_replay_bundle(self, run_ids: List[str]) -> Dict[str, Any]:
        with self._connect() as conn:
            runs = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM cognitive_runs WHERE run_id IN ({','.join('?'*len(run_ids))})",
                    run_ids,
                ).fetchall()
            ]
            episode_ids = list({r["episode_id"] for r in runs})
            episodes = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM episodes WHERE episode_id IN ({','.join('?'*len(episode_ids))})",
                    episode_ids,
                ).fetchall()
            ]
            events = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM episode_events WHERE episode_id IN ({','.join('?'*len(episode_ids))})",
                    episode_ids,
                ).fetchall()
            ]
            refs = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM memory_references WHERE run_id IN ({','.join('?'*len(run_ids))})",
                    run_ids,
                ).fetchall()
            ]
            artefacts = [dict(r) for r in conn.execute("SELECT * FROM artefacts").fetchall()]
        return {
            "schema": "replay_bundle_v1",
            "runs": runs,
            "episodes": episodes,
            "events": events,
            "memory_references": refs,
            "artefact_manifest": artefacts,
        }


_ledger: Optional[CognitionLedger] = None


def get_ledger() -> CognitionLedger:
    global _ledger
    if _ledger is None:
        _ledger = CognitionLedger()
    return _ledger


def reset_ledger(db_path: Optional[Path] = None) -> CognitionLedger:
    """Reset singleton ledger (tests/demo isolation)."""
    global _ledger
    _ledger = CognitionLedger(db_path=db_path)
    return _ledger


def clear_ledger() -> None:
    """Release the singleton ledger so temp stores can be cleaned up."""
    global _ledger
    _ledger = None


def release_ledger_store(db_path: Path) -> None:
    """Release the singleton and checkpoint a store for safe temp cleanup."""
    clear_ledger()
    gc.collect()
    if db_path.exists():
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA journal_mode=DELETE")
