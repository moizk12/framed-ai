"""SQLite append-only cognition ledger."""

from __future__ import annotations

import json
import gc
import random
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from framed.cognition.config import cognition_db_path
from framed.cognition.constants import APPEND_EVENT_MAX_RETRIES, APPEND_EVENT_RETRY_BASE_MS, REPLAY_BUNDLE_SCHEMA
from framed.cognition.contracts.memory import MemoryReference
from framed.cognition.contracts.runs import CognitiveRun, RunPurpose, is_retrieval_eligible
from framed.cognition.ledger.artefact_store import ArtefactStore, artefact_hash as compute_artefact_hash


class CognitionLedger:
    def __init__(self, db_path: Optional[Path] = None, artefact_root: Optional[Path] = None) -> None:
        self.db_path = db_path or cognition_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_artefact_root = artefact_root or (self.db_path.parent / "artefacts")
        self.artefacts = ArtefactStore(root=resolved_artefact_root)
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
            if current < 3:
                for stmt in (migrations_dir / "003_integrity.sql").read_text(encoding="utf-8").split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            conn.execute(stmt)
                        except sqlite3.OperationalError as exc:
                            if "duplicate column" not in str(exc).lower():
                                raise
                conn.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (3, ?)",
                    (ArtefactStore.utc_now(),),
                )
            if current < 4:
                conn.executescript((migrations_dir / "004_controlled_learning.sql").read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (4, ?)",
                    (ArtefactStore.utc_now(),),
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

    def ensure_initial_states(self, workspace_id: str) -> Tuple[str, str]:
        """Create baseline (default active) and memory-enabled (inactive) states if missing."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT state_version_id, label, is_active FROM cognitive_state_versions WHERE workspace_id=?",
                (workspace_id,),
            ).fetchall()
            by_label = {r["label"]: r["state_version_id"] for r in rows}
            has_active = any(int(r["is_active"]) for r in rows)
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
                    VALUES (?, ?, NULL, 'state_baseline', ?, ?, ?)
                    """,
                    (baseline_id, workspace_id, ArtefactStore.utc_now(), 1 if not has_active else 0, snap_hash),
                )
                has_active = has_active or not by_label
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
                conn.execute(
                    """
                    INSERT INTO cognitive_state_versions
                    (state_version_id, workspace_id, parent_version_id, label, created_at, is_active, snapshot_artefact_hash)
                    VALUES (?, ?, ?, 'state_memory_enabled', ?, 0, ?)
                    """,
                    (memory_id, workspace_id, baseline_id, ArtefactStore.utc_now(), snap_hash),
                )
        return baseline_id, memory_id

    def ensure_demo_states(self, workspace_id: str) -> Tuple[str, str]:
        """Backward-compatible alias for initial state bootstrap."""
        return self.ensure_initial_states(workspace_id)

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
            baseline_id, _ = self.ensure_initial_states(workspace_id)
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM cognitive_state_versions WHERE state_version_id=?",
                    (baseline_id,),
                ).fetchone()
        snap = self.artefacts.get(row["snapshot_artefact_hash"])
        return {"state_version_id": row["state_version_id"], "label": row["label"], "snapshot": snap}

    def activate_state(self, workspace_id: str, label: str) -> str:
        self.ensure_initial_states(workspace_id)
        with self._connect() as conn:
            target = conn.execute(
                """
                SELECT state_version_id FROM cognitive_state_versions
                WHERE workspace_id=? AND label=? LIMIT 1
                """,
                (workspace_id, label),
            ).fetchone()
            if target is None:
                raise ValueError(f"Unknown cognitive state label: {label}")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE cognitive_state_versions SET is_active=0 WHERE workspace_id=?",
                (workspace_id,),
            )
            conn.execute(
                "UPDATE cognitive_state_versions SET is_active=1 WHERE state_version_id=?",
                (target["state_version_id"],),
            )
            conn.commit()
            return target["state_version_id"]

    def get_state_version(self, state_version_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cognitive_state_versions WHERE state_version_id=?",
                (state_version_id,),
            ).fetchone()
        if row is None:
            return None
        snap = self.artefacts.get(row["snapshot_artefact_hash"])
        return {
            "state_version_id": row["state_version_id"],
            "workspace_id": row["workspace_id"],
            "parent_version_id": row["parent_version_id"],
            "label": row["label"],
            "created_at": row["created_at"],
            "is_active": int(row["is_active"]),
            "snapshot_artefact_hash": row["snapshot_artefact_hash"],
            "snapshot": snap,
        }

    def create_state_version(
        self,
        *,
        workspace_id: str,
        parent_version_id: Optional[str],
        label: str,
        snapshot: Dict[str, Any],
    ) -> str:
        state_version_id = str(uuid.uuid4())
        snap_hash = self.put_artefact("state_snapshot", "v1", snapshot)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cognitive_state_versions
                (state_version_id, workspace_id, parent_version_id, label, created_at, is_active, snapshot_artefact_hash)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (state_version_id, workspace_id, parent_version_id, label, ArtefactStore.utc_now(), snap_hash),
            )
        return state_version_id

    def activate_state_version(self, workspace_id: str, state_version_id: str) -> str:
        with self._connect() as conn:
            target = conn.execute(
                """
                SELECT state_version_id FROM cognitive_state_versions
                WHERE workspace_id=? AND state_version_id=? LIMIT 1
                """,
                (workspace_id, state_version_id),
            ).fetchone()
            if target is None:
                raise ValueError(f"Unknown cognitive state version: {state_version_id}")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE cognitive_state_versions SET is_active=0 WHERE workspace_id=?",
                (workspace_id,),
            )
            conn.execute(
                "UPDATE cognitive_state_versions SET is_active=1 WHERE state_version_id=?",
                (state_version_id,),
            )
            conn.commit()
            return state_version_id

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        return dict(row) if row else None

    def list_runs_for_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cognitive_runs WHERE episode_id=? ORDER BY started_at",
                (episode_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_indexed_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM retrieval_index WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_runs_retrieving_episode(self, source_episode_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT cr.*
                FROM cognitive_runs cr
                JOIN memory_references mr ON mr.run_id = cr.run_id
                WHERE mr.source_episode_id=?
                ORDER BY cr.started_at
                """,
                (source_episode_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_outcome(
        self,
        *,
        outcome_id: str,
        workspace_id: str,
        source_episode_id: str,
        source_run_id: str,
        kind: str,
        verdict: str,
        created_by: str,
        artefact_hash: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO outcomes
                (outcome_id, workspace_id, source_episode_id, source_run_id, kind, verdict,
                 created_by, artefact_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome_id,
                    workspace_id,
                    source_episode_id,
                    source_run_id,
                    kind,
                    verdict,
                    created_by,
                    artefact_hash,
                    created_at,
                ),
            )

    def get_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM outcomes WHERE outcome_id=?", (outcome_id,)).fetchone()
        return dict(row) if row else None

    def insert_proposal(
        self,
        *,
        proposal_id: str,
        workspace_id: str,
        base_state_version_id: str,
        outcome_id: str,
        kind: str,
        created_by: str,
        artefact_hash: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO update_proposals
                (proposal_id, workspace_id, base_state_version_id, outcome_id, kind,
                 created_by, artefact_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    workspace_id,
                    base_state_version_id,
                    outcome_id,
                    kind,
                    created_by,
                    artefact_hash,
                    created_at,
                ),
            )

    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM update_proposals WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        try:
            record["payload"] = self.artefacts.get(record["artefact_hash"])
        except (FileNotFoundError, OSError):
            record["payload"] = None
        return record

    def insert_evaluation(
        self,
        *,
        evaluation_id: str,
        proposal_id: str,
        status: str,
        artefact_hash: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO proposal_evaluations
                (evaluation_id, proposal_id, status, artefact_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (evaluation_id, proposal_id, status, artefact_hash, created_at),
            )

    def get_latest_evaluation(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM proposal_evaluations
                WHERE proposal_id=?
                ORDER BY created_at DESC, evaluation_id DESC
                LIMIT 1
                """,
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        try:
            record["payload"] = self.artefacts.get(record["artefact_hash"])
        except (FileNotFoundError, OSError):
            record["payload"] = None
        return record

    def insert_decision(
        self,
        *,
        decision_id: str,
        proposal_id: str,
        evaluation_id: Optional[str],
        action: str,
        authority_kind: str,
        actor_id: str,
        resulting_state_version_id: Optional[str],
        artefact_hash: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO promotion_decisions
                (decision_id, proposal_id, evaluation_id, action, authority_kind, actor_id,
                 resulting_state_version_id, artefact_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    proposal_id,
                    evaluation_id,
                    action,
                    authority_kind,
                    actor_id,
                    resulting_state_version_id,
                    artefact_hash,
                    created_at,
                ),
            )

    def get_decision_for_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM promotion_decisions WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        try:
            record["payload"] = self.artefacts.get(record["artefact_hash"])
        except (FileNotFoundError, OSError):
            record["payload"] = None
        return record

    def get_decision_by_resulting_state(self, state_version_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM promotion_decisions
                WHERE resulting_state_version_id=? AND action='accept'
                LIMIT 1
                """,
                (state_version_id,),
            ).fetchone()
        return dict(row) if row else None

    def insert_rollback(
        self,
        *,
        rollback_id: str,
        workspace_id: str,
        from_state_version_id: str,
        to_state_version_id: str,
        authority_kind: str,
        actor_id: str,
        artefact_hash: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rollback_records
                (rollback_id, workspace_id, from_state_version_id, to_state_version_id,
                 authority_kind, actor_id, artefact_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rollback_id,
                    workspace_id,
                    from_state_version_id,
                    to_state_version_id,
                    authority_kind,
                    actor_id,
                    artefact_hash,
                    created_at,
                ),
            )

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

    def create_run(self, run: CognitiveRun, *, provenance_manifest: Optional[Dict[str, Any]] = None) -> None:
        eligible = is_retrieval_eligible(run.run_purpose)
        manifest_json = json.dumps(provenance_manifest) if provenance_manifest else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cognitive_runs
                (run_id, episode_id, mode, run_purpose, baseline_run_id, comparison_group_id,
                 state_version_id, context_fingerprint, retrieval_enabled, retrieval_eligible,
                 model_provenance_json, prompt_provenance_json, started_at, completed_at,
                 provenance_manifest_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    manifest_json,
                ),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cognitive_runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def complete_run(self, run_id: str, *, failure_code: Optional[str] = None, failure_stage: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cognitive_runs
                SET completed_at=?, failure_code=COALESCE(?, failure_code), failure_stage=COALESCE(?, failure_stage)
                WHERE run_id=?
                """,
                (ArtefactStore.utc_now(), failure_code, failure_stage, run_id),
            )

    def fail_episode(
        self,
        episode_id: str,
        *,
        failure_code: str,
        failure_message: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET status='failed', closed_at=?, failure_code=?, failure_message=?
                WHERE episode_id=?
                """,
                (ArtefactStore.utc_now(), failure_code, failure_message, episode_id),
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
        payload_json = json.dumps(payload, sort_keys=True)
        last_exc: Optional[Exception] = None
        for attempt in range(APPEND_EVENT_MAX_RETRIES):
            try:
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
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
                            payload_json,
                        ),
                    )
                    conn.commit()
                    return event_id
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    time.sleep((APPEND_EVENT_RETRY_BASE_MS / 1000.0) * (1 + random.random()) * (attempt + 1))
                    continue
                raise
        raise last_exc or RuntimeError("append_event failed after retries")

    def finalize_run_atomic(
        self,
        *,
        episode_id: str,
        run_id: str,
        run_purpose: RunPurpose,
        scene_signature: str,
        category_signature: str,
        goal_type: str,
        goal_instance_id: Optional[str],
        final_fingerprint: Optional[str],
        perception_artefact_hash: Optional[str],
        deliberation_snapshot: Dict[str, Any],
        deliberation_snapshot_hash: str,
        experience_closed_payload: Dict[str, Any],
        delta_payload: Optional[Dict[str, Any]] = None,
        baseline_link_payload: Optional[Dict[str, Any]] = None,
        frozen_input: Optional[Dict[str, Any]] = None,
        frozen_input_hash: Optional[str] = None,
        frozen_input_rel: Optional[str] = None,
        frozen_input_len: Optional[int] = None,
        deliberation_snapshot_rel: Optional[str] = None,
        deliberation_snapshot_len: Optional[int] = None,
    ) -> str:
        """Atomically persist finalization events, close episode, index retrieval, complete run."""
        now = ArtefactStore.utc_now()
        snapshot_event_id = str(uuid.uuid4())
        baseline_digest: Optional[str] = None
        baseline_rel: Optional[str] = None
        baseline_len: Optional[int] = None
        delta_digest: Optional[str] = None
        delta_rel: Optional[str] = None
        delta_len: Optional[int] = None

        # Write payload artefact files before the DB transaction.
        # DB rows for these artefacts are inserted inside the transaction so the ledger
        # never points at missing/partial artefacts.
        if frozen_input is not None and frozen_input_hash is None:
            frozen_input_hash, frozen_input_rel, frozen_input_len = self.artefacts.put(
                "frozen_deliberation_input", "v1", frozen_input
            )
        if deliberation_snapshot_rel is None:
            deliberation_snapshot_hash, deliberation_snapshot_rel, deliberation_snapshot_len = self.artefacts.put(
                "deliberation_snapshot", "v1", deliberation_snapshot
            )
        if baseline_link_payload is not None:
            baseline_digest, baseline_rel, baseline_len = self.artefacts.put(
                "deliberation_baseline_record", "v1", baseline_link_payload
            )
        if delta_payload is not None:
            delta_digest, delta_rel, delta_len = self.artefacts.put(
                "deliberation_delta", "v1", delta_payload
            )

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            seq = self._next_sequence(conn, episode_id)
            if frozen_input is not None and frozen_input_hash is not None:
                assert frozen_input_rel is not None and frozen_input_len is not None
                conn.execute(
                    """
                    INSERT OR IGNORE INTO artefacts
                    (artefact_hash, schema_name, schema_version, relative_path, byte_length, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        frozen_input_hash,
                        "frozen_deliberation_input",
                        "v1",
                        frozen_input_rel,
                        frozen_input_len,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO episode_events
                    (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                    VALUES (?, ?, ?, 'frozen_deliberation_input_recorded', ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        episode_id,
                        run_id,
                        seq,
                        now,
                        frozen_input_hash,
                        json.dumps({"frozen_deliberation_input_hash": frozen_input_hash}, sort_keys=True),
                    ),
                )
                seq += 1
            if deliberation_snapshot_rel is not None and deliberation_snapshot_len is not None:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO artefacts
                    (artefact_hash, schema_name, schema_version, relative_path, byte_length, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        deliberation_snapshot_hash,
                        "deliberation_snapshot",
                        "v1",
                        deliberation_snapshot_rel,
                        deliberation_snapshot_len,
                        now,
                    ),
                )
            conn.execute(
                """
                INSERT INTO episode_events
                (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                VALUES (?, ?, ?, 'deliberation_snapshot', ?, ?, ?, ?)
                """,
                (
                    snapshot_event_id,
                    episode_id,
                    run_id,
                    seq,
                    now,
                    deliberation_snapshot_hash,
                    json.dumps(deliberation_snapshot, sort_keys=True),
                ),
            )
            if baseline_link_payload is not None:
                seq += 1
                assert baseline_digest is not None and baseline_rel is not None and baseline_len is not None
                conn.execute(
                    """
                    INSERT OR IGNORE INTO artefacts
                    (artefact_hash, schema_name, schema_version, relative_path, byte_length, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        baseline_digest,
                        "deliberation_baseline_record",
                        "v1",
                        baseline_rel,
                        baseline_len,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO episode_events
                    (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                    VALUES (?, ?, ?, 'deliberation_baseline_record', ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        episode_id,
                        run_id,
                        seq,
                        now,
                        baseline_digest,
                        json.dumps(baseline_link_payload, sort_keys=True),
                    ),
                )
            if delta_payload is not None:
                seq += 1
                assert delta_digest is not None and delta_rel is not None and delta_len is not None
                conn.execute(
                    """
                    INSERT OR IGNORE INTO artefacts
                    (artefact_hash, schema_name, schema_version, relative_path, byte_length, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        delta_digest,
                        "deliberation_delta",
                        "v1",
                        delta_rel,
                        delta_len,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO episode_events
                    (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                    VALUES (?, ?, ?, 'deliberation_delta', ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        episode_id,
                        run_id,
                        seq,
                        now,
                        delta_digest,
                        json.dumps(delta_payload, sort_keys=True),
                    ),
                )
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
            seq += 1
            conn.execute(
                """
                INSERT INTO episode_events
                (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                VALUES (?, ?, ?, 'experience_closed', ?, ?, NULL, ?)
                """,
                (
                    str(uuid.uuid4()),
                    episode_id,
                    run_id,
                    seq,
                    now,
                    json.dumps(experience_closed_payload, sort_keys=True),
                ),
            )
            conn.execute(
                "UPDATE cognitive_runs SET completed_at=? WHERE run_id=?",
                (now, run_id),
            )
            conn.commit()
        return snapshot_event_id

    def fail_run_atomic(
        self,
        *,
        episode_id: str,
        run_id: str,
        error_code: str,
        safe_message: str,
        stage: str,
        internal_exception_type: Optional[str],
        run_purpose: str,
        _fail_inject_at: Optional[str] = None,
    ) -> str:
        """Atomically mark run/episode failed without indexing."""
        now = ArtefactStore.utc_now()
        event_id = str(uuid.uuid4())
        payload = {
            "error_code": error_code,
            "safe_message": safe_message,
            "stage": stage,
            "internal_exception_type": internal_exception_type,
            "run_purpose": run_purpose,
        }
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if _fail_inject_at == "before_event":
                raise RuntimeError("injected_fail:before_event")
            seq = self._next_sequence(conn, episode_id)
            conn.execute(
                """
                INSERT INTO episode_events
                (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                VALUES (?, ?, ?, 'run_failed', ?, ?, NULL, ?)
                """,
                (event_id, episode_id, run_id, seq, now, json.dumps(payload, sort_keys=True)),
            )
            if _fail_inject_at == "after_event":
                raise RuntimeError("injected_fail:after_event")
            conn.execute(
                """
                UPDATE episodes
                SET status='failed', closed_at=?, failure_code=?, failure_message=?
                WHERE episode_id=?
                """,
                (now, error_code, safe_message, episode_id),
            )
            if _fail_inject_at == "after_episode_update":
                raise RuntimeError("injected_fail:after_episode_update")
            if _fail_inject_at == "before_run_update":
                raise RuntimeError("injected_fail:before_run_update")
            conn.execute(
                """
                UPDATE cognitive_runs
                SET completed_at=?, failure_code=?, failure_stage=?
                WHERE run_id=?
                """,
                (now, error_code, stage, run_id),
            )
            conn.execute("DELETE FROM retrieval_index WHERE episode_id=?", (episode_id,))
            if _fail_inject_at == "before_commit":
                raise RuntimeError("injected_fail:before_commit")
            conn.commit()
        return event_id

    def open_run_atomic(
        self,
        *,
        episode_id: str,
        workspace_id: str,
        actor_id: str,
        asset_id: str,
        goal_type: str,
        goal_instance_id: Optional[str],
        state_version_id: str,
        asset_filename: Optional[str],
        source_kind: str,
        run: CognitiveRun,
        provenance_manifest: Optional[Dict[str, Any]],
        perception_hash: str,
        perception_rel: str,
        perception_len: int,
        experience_opened_payload: Dict[str, Any],
        retrieval_performed_payload: Optional[Dict[str, Any]] = None,
        memory_refs: Optional[List[MemoryReference]] = None,
        _fail_inject_at: Optional[str] = None,
    ) -> None:
        """Atomically create episode+run+opening events+memory refs. Artefact file must already exist."""
        now = ArtefactStore.utc_now()
        eligible = is_retrieval_eligible(run.run_purpose)
        manifest_json = json.dumps(provenance_manifest) if provenance_manifest else None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if _fail_inject_at == "before_episode":
                raise RuntimeError("injected_fail:before_episode")
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
            if _fail_inject_at == "before_run":
                raise RuntimeError("injected_fail:before_run")
            conn.execute(
                """
                INSERT INTO cognitive_runs
                (run_id, episode_id, mode, run_purpose, baseline_run_id, comparison_group_id,
                 state_version_id, context_fingerprint, retrieval_enabled, retrieval_eligible,
                 model_provenance_json, prompt_provenance_json, started_at, completed_at,
                 provenance_manifest_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    episode_id,
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
                    run.started_at or now,
                    run.completed_at,
                    manifest_json,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO artefacts
                (artefact_hash, schema_name, schema_version, relative_path, byte_length, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (perception_hash, "perception_snapshot", "v1", perception_rel, perception_len, now),
            )
            if _fail_inject_at == "before_experience_opened":
                raise RuntimeError("injected_fail:before_experience_opened")
            seq = 1
            conn.execute(
                """
                INSERT INTO episode_events
                (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                VALUES (?, ?, ?, 'experience_opened', ?, ?, NULL, ?)
                """,
                (
                    str(uuid.uuid4()),
                    episode_id,
                    run.run_id,
                    seq,
                    now,
                    json.dumps(experience_opened_payload, sort_keys=True),
                ),
            )
            if _fail_inject_at == "before_perception_completed":
                raise RuntimeError("injected_fail:before_perception_completed")
            seq += 1
            conn.execute(
                """
                INSERT INTO episode_events
                (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                VALUES (?, ?, ?, 'perception_completed', ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    episode_id,
                    run.run_id,
                    seq,
                    now,
                    perception_hash,
                    json.dumps({"perception_artefact_hash": perception_hash}, sort_keys=True),
                ),
            )
            if retrieval_performed_payload is not None:
                if _fail_inject_at == "before_retrieval_event":
                    raise RuntimeError("injected_fail:before_retrieval_event")
                seq += 1
                conn.execute(
                    """
                    INSERT INTO episode_events
                    (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
                    VALUES (?, ?, ?, 'retrieval_performed', ?, ?, NULL, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        episode_id,
                        run.run_id,
                        seq,
                        now,
                        json.dumps(retrieval_performed_payload, sort_keys=True),
                    ),
                )
            if memory_refs:
                if _fail_inject_at == "before_memory_refs":
                    raise RuntimeError("injected_fail:before_memory_refs")
                for ref in memory_refs:
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
                            run.run_id,
                            episode_id,
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
                            now,
                        ),
                    )
            if _fail_inject_at == "before_commit":
                raise RuntimeError("injected_fail:before_commit")
            conn.commit()

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
        if not run_ids:
            return {
                "schema": REPLAY_BUNDLE_SCHEMA,
                "runs": [],
                "episodes": [],
                "events": [],
                "memory_references": [],
                "state_versions": [],
                "retrieval_index": [],
                "artefact_manifest": [],
                "artefact_payloads": [],
                "expected_hashes": {"runs": {}},
            }

        run_placeholders = ",".join("?" * len(run_ids))

        with self._connect() as conn:
            runs = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM cognitive_runs WHERE run_id IN ({run_placeholders})",
                    run_ids,
                ).fetchall()
            ]
            refs = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM memory_references WHERE run_id IN ({run_placeholders})",
                    run_ids,
                ).fetchall()
            ]

            run_episode_ids = {r["episode_id"] for r in runs}
            source_episode_ids = {r["source_episode_id"] for r in refs}
            episode_ids = list(run_episode_ids | source_episode_ids)

            episode_placeholders = ",".join("?" * len(episode_ids)) if episode_ids else "NULL"
            episodes = (
                [dict(r) for r in conn.execute(
                    f"SELECT * FROM episodes WHERE episode_id IN ({episode_placeholders})",
                    episode_ids,
                ).fetchall()]
                if episode_ids
                else []
            )
            events = (
                [dict(r) for r in conn.execute(
                    f"""
                    SELECT * FROM episode_events
                    WHERE episode_id IN ({episode_placeholders})
                    ORDER BY episode_id, sequence_num
                    """,
                    episode_ids,
                ).fetchall()]
                if episode_ids
                else []
            )

            workspace_ids = list({e["workspace_id"] for e in episodes})
            state_versions = (
                [dict(r) for r in conn.execute(
                    f"""
                    SELECT * FROM cognitive_state_versions
                    WHERE workspace_id IN ({','.join('?' * len(workspace_ids))})
                    """,
                    workspace_ids,
                ).fetchall()]
                if workspace_ids
                else []
            )

            retrieval_index = (
                [dict(r) for r in conn.execute(
                    f"""
                    SELECT * FROM retrieval_index
                    WHERE episode_id IN ({episode_placeholders})
                    """,
                    episode_ids,
                ).fetchall()]
                if episode_ids
                else []
            )
            all_index = {row["episode_id"]: row for row in retrieval_index}

            # Enrich MemoryReference with deliberation snapshot-derived values.
            snapshot_payload_by_event_id: Dict[str, Any] = {}
            for ev in events:
                if ev.get("event_type") != "deliberation_snapshot":
                    continue
                if not ev.get("event_id") or not ev.get("payload_json"):
                    continue
                try:
                    snapshot_payload_by_event_id[ev["event_id"]] = json.loads(ev["payload_json"])
                except json.JSONDecodeError:
                    continue

            for ref in refs:
                src_payload = snapshot_payload_by_event_id.get(ref.get("source_event_id"))
                if not src_payload:
                    continue
                ref["scene_signature"] = src_payload.get("scene_signature")
                ref["category_signature"] = src_payload.get("category_signature")
                ref["hypothesis_summary"] = src_payload.get("primary_hypothesis")
                ref["confidence_at_source"] = src_payload.get("confidence")

        reachable_hashes = self._collect_reachable_artefact_hashes(
            runs=runs,
            episodes=episodes,
            events=events,
            refs=refs,
            state_versions=state_versions,
        )
        with self._connect() as conn:
            if reachable_hashes:
                artefacts = [
                    dict(r)
                    for r in conn.execute(
                        f"SELECT * FROM artefacts WHERE artefact_hash IN ({','.join('?'*len(reachable_hashes))})",
                        list(reachable_hashes),
                    ).fetchall()
                ]
            else:
                artefacts = []

        artefact_payloads: List[Dict[str, Any]] = []
        artefact_payload_by_digest: Dict[str, Any] = {}
        for row in artefacts:
            digest = row["artefact_hash"]
            try:
                payload = self.artefacts.get(digest)
                artefact_payloads.append({"artefact_hash": digest, "payload": payload})
                artefact_payload_by_digest[digest] = payload
            except (FileNotFoundError, OSError):
                continue

        active_state_versions = [sv for sv in state_versions if int(sv.get("is_active", 0)) == 1]
        active_state_snapshots = [
            {
                "state_version_id": sv.get("state_version_id"),
                "label": sv.get("label"),
                "snapshot_artefact_hash": sv.get("snapshot_artefact_hash"),
                "snapshot": artefact_payload_by_digest.get(sv.get("snapshot_artefact_hash")),
            }
            for sv in active_state_versions
        ]

        expected_hashes: Dict[str, Any] = {"runs": {}}
        for run in runs:
            run_events = [e for e in events if e["run_id"] == run["run_id"]]
            snap_hash = next(
                (e.get("artefact_hash") for e in run_events if e.get("event_type") == "deliberation_snapshot"),
                None,
            )
            frozen_hash = next(
                (
                    e.get("artefact_hash")
                    for e in run_events
                    if e.get("event_type") == "frozen_deliberation_input_recorded"
                ),
                None,
            )
            delta_hashes = sorted(
                [
                    e.get("artefact_hash")
                    for e in run_events
                    if e.get("event_type") == "deliberation_delta" and e.get("artefact_hash")
                ]
            )
            snap_payload = None
            for e in run_events:
                if e.get("event_type") == "deliberation_snapshot" and e.get("payload_json"):
                    try:
                        snap_payload = json.loads(e["payload_json"])
                    except json.JSONDecodeError:
                        snap_payload = None
                    break
            confidence_provenance = (snap_payload or {}).get("confidence_provenance") or {}
            confidence_provenance_hash = (
                compute_artefact_hash(confidence_provenance) if confidence_provenance else None
            )
            retrieval_as_of = None
            try:
                manifest = json.loads(run.get("provenance_manifest_json") or "{}")
                retrieval_as_of = manifest.get("retrieval_as_of")
            except (TypeError, json.JSONDecodeError):
                retrieval_as_of = None
            if not retrieval_as_of:
                for e in run_events:
                    if e.get("event_type") == "retrieval_performed" and e.get("payload_json"):
                        try:
                            retrieval_as_of = json.loads(e["payload_json"]).get("retrieval_as_of")
                        except json.JSONDecodeError:
                            retrieval_as_of = None
                        break
            if not retrieval_as_of:
                retrieval_as_of = run.get("started_at")
            expected_hashes["runs"][run["run_id"]] = {
                "deliberation_snapshot_hash": snap_hash,
                "deliberation_delta_hashes": delta_hashes,
                "frozen_deliberation_input_hash": frozen_hash,
                "confidence_provenance_hash": confidence_provenance_hash,
                "confidence_provenance": confidence_provenance,
                "context_fingerprint": run.get("context_fingerprint"),
                "retrieval_as_of": retrieval_as_of,
                "perception_artefact_hash": next(
                    (
                        json.loads(e["payload_json"]).get("perception_artefact_hash")
                        for e in run_events
                        if e.get("event_type") == "perception_completed" and e.get("payload_json")
                    ),
                    None,
                ),
                "state_version_id": run.get("state_version_id"),
            }

        return {
            "schema": REPLAY_BUNDLE_SCHEMA,
            "runs": runs,
            "episodes": episodes,
            "events": events,
            "memory_references": refs,
            "state_versions": state_versions,
            "active_state_snapshots": active_state_snapshots,
            "retrieval_index": list(all_index.values()),
            "artefact_manifest": artefacts,
            "artefact_payloads": artefact_payloads,
            "expected_hashes": expected_hashes,
        }

    @staticmethod
    def _collect_reachable_artefact_hashes(
        *,
        runs: List[Dict[str, Any]],
        episodes: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        refs: List[Dict[str, Any]],
        state_versions: List[Dict[str, Any]],
    ) -> Set[str]:
        hashes: Set[str] = set()
        for ev in events:
            if ev.get("artefact_hash"):
                hashes.add(ev["artefact_hash"])
        for ep in episodes:
            if ep.get("perception_artefact_hash"):
                hashes.add(ep["perception_artefact_hash"])
        for ref in refs:
            if ref.get("artefact_hash"):
                hashes.add(ref["artefact_hash"])
        for sv in state_versions:
            if sv.get("snapshot_artefact_hash"):
                hashes.add(sv["snapshot_artefact_hash"])
        return hashes


_ledger: Optional[CognitionLedger] = None


def get_ledger() -> CognitionLedger:
    global _ledger
    if _ledger is None:
        _ledger = CognitionLedger()
    return _ledger


def reset_ledger(db_path: Optional[Path] = None, artefact_root: Optional[Path] = None) -> CognitionLedger:
    """Reset singleton ledger (tests/demo isolation)."""
    global _ledger
    if db_path is not None and artefact_root is None:
        artefact_root = db_path.parent / "artefacts"
    _ledger = CognitionLedger(db_path=db_path, artefact_root=artefact_root)
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
