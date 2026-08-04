from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from framed.cognition.contracts.memory import RetrievalQuery
from framed.cognition.contracts.runs import CognitiveRun, RunMode, RunPurpose
from framed.cognition.ledger.artefact_store import artefact_hash
from framed.cognition.ledger.sqlite_store import CognitionLedger
from framed.cognition.retrieval.service import retrieve_memories


def _exec_sql_script(conn: sqlite3.Connection, path: Path) -> None:
    conn.executescript(path.read_text(encoding="utf-8"))


@pytest.fixture
def migration_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(tmp_path))
    return tmp_path


def _seed_old_schema_db(db_path: Path) -> None:
    migrations_dir = db_path.parent / "migrations_src"
    source_dir = Path(__file__).resolve().parents[1] / "ledger" / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_initial.sql").write_text((source_dir / "001_initial.sql").read_text(encoding="utf-8"), encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _exec_sql_script(conn, migrations_dir / "001_initial.sql")
        conn.execute("INSERT INTO schema_version(version, applied_at) VALUES (1, '2026-01-01T00:00:00Z')")
        conn.execute(
            """
            INSERT INTO cognitive_state_versions
            (state_version_id, workspace_id, parent_version_id, label, created_at, is_active, snapshot_artefact_hash)
            VALUES ('state-1', 'ws', NULL, 'state_memory_enabled', '2026-01-01T00:00:00Z', 1, 'snap')
            """
        )
        conn.execute(
            """
            INSERT INTO artefacts
            (artefact_hash, schema_name, schema_version, relative_path, byte_length, created_at)
            VALUES ('snap', 'state_snapshot', 'v1', 'aa/snap.json', 2, '2026-01-01T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO episodes
            (episode_id, workspace_id, actor_id, asset_id, goal_type, goal_instance_id, status, source_kind, asset_filename, created_at, closed_at, state_version_id, final_fingerprint)
            VALUES ('ep-live', 'ws', 'actor', 'asset-live', 'critique', NULL, 'closed', 'live', 'live.jpg', '2026-01-01T00:00:00Z', '2026-01-01T00:05:00Z', 'state-1', 'fp1')
            """
        )
        conn.execute(
            """
            INSERT INTO episodes
            (episode_id, workspace_id, actor_id, asset_id, goal_type, goal_instance_id, status, source_kind, asset_filename, created_at, closed_at, state_version_id, final_fingerprint)
            VALUES ('ep-base', 'ws', 'actor', 'asset-base', 'critique', NULL, 'closed', 'live', 'base.jpg', '2026-01-01T00:10:00Z', '2026-01-01T00:15:00Z', 'state-1', 'fp2')
            """
        )
        conn.execute(
            """
            INSERT INTO cognitive_runs
            (run_id, episode_id, mode, state_version_id, context_fingerprint, retrieval_enabled, model_provenance_json, prompt_provenance_json, started_at, completed_at)
            VALUES ('run-live', 'ep-live', 'memory_enabled', 'state-1', NULL, 1, '{}', '{}', '2026-01-01T00:00:00Z', '2026-01-01T00:05:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO cognitive_runs
            (run_id, episode_id, mode, state_version_id, context_fingerprint, retrieval_enabled, model_provenance_json, prompt_provenance_json, started_at, completed_at)
            VALUES ('run-base', 'ep-base', 'baseline', 'state-1', NULL, 0, '{}', '{}', '2026-01-01T00:10:00Z', '2026-01-01T00:15:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO retrieval_index
            (episode_id, workspace_id, actor_id, asset_id, scene_signature, category_signature, goal_type, goal_instance_id, recorded_at, closed_at)
            VALUES ('ep-live', 'ws', 'actor', 'asset-live', 'interior_scene', 'cluttered_room_weak_composition', 'critique', NULL, '2026-01-01T00:00:00Z', '2026-01-01T00:05:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO retrieval_index
            (episode_id, workspace_id, actor_id, asset_id, scene_signature, category_signature, goal_type, goal_instance_id, recorded_at, closed_at)
            VALUES ('ep-base', 'ws', 'actor', 'asset-base', 'interior_scene', 'cluttered_room_weak_composition', 'critique', NULL, '2026-01-01T00:10:00Z', '2026-01-01T00:15:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO episode_events
            (event_id, episode_id, run_id, event_type, sequence_num, recorded_at, artefact_hash, payload_json)
            VALUES ('evt-1', 'ep-live', 'run-live', 'deliberation_snapshot', 1, '2026-01-01T00:04:00Z', NULL, '{"primary_hypothesis":"legacy"}')
            """
        )


def test_fresh_install_upgrade_and_retrieval(migration_env):
    db_path = migration_env / "fresh.sqlite3"
    ledger = CognitionLedger(db_path=db_path)

    with ledger._connect() as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 2
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    ws, actor = str(uuid.uuid4()), str(uuid.uuid4())
    baseline_id, _ = ledger.ensure_demo_states(ws)
    eid = ledger.open_episode(
        workspace_id=ws,
        actor_id=actor,
        asset_id="fresh-asset",
        goal_type="critique",
        goal_instance_id=None,
        state_version_id=baseline_id,
    )
    run = CognitiveRun(
        run_id=str(uuid.uuid4()),
        episode_id=eid,
        mode=RunMode.MEMORY_ENABLED,
        run_purpose=RunPurpose.LIVE,
        state_version_id=baseline_id,
        context_fingerprint=None,
        retrieval_enabled=True,
        model_provenance={},
        prompt_provenance={},
        started_at="2026-01-01T00:00:00Z",
        retrieval_eligible=True,
    )
    ledger.create_run(run)
    snap = {"primary_hypothesis": "fresh", "confidence": 0.5}
    snap_hash = ledger.put_artefact("deliberation_snapshot", "v1", snap)
    ledger.append_event(
        episode_id=eid,
        run_id=run.run_id,
        event_type="deliberation_snapshot",
        payload=snap,
        artefact_hash=snap_hash,
    )
    ledger.close_episode(
        eid,
        run_id=run.run_id,
        run_purpose=RunPurpose.LIVE,
        scene_signature="interior_scene",
        category_signature="cluttered_room_weak_composition",
        goal_type="critique",
        goal_instance_id=None,
        final_fingerprint=artefact_hash({"episode": eid}),
        perception_artefact_hash=snap_hash,
    )
    ledger.complete_run(run.run_id)

    result = retrieve_memories(
        RetrievalQuery(
            workspace_id=ws,
            actor_id=actor,
            asset_id="query-asset",
            goal_type="critique",
            goal_instance_id=None,
            scene_signature="interior_scene",
            category_signature="cluttered_room_weak_composition",
        ),
        ledger=ledger,
        state_snapshot=ledger.get_active_state(ws)["snapshot"],
    )
    assert len(result.references) == 1
    assert result.references[0].source_episode_id == eid


def test_upgrade_from_001_is_fail_closed_and_preserves_history(migration_env):
    db_path = migration_env / "upgrade.sqlite3"
    _seed_old_schema_db(db_path)

    ledger = CognitionLedger(db_path=db_path)

    with ledger._connect() as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 2
        upgraded_live = conn.execute(
            "SELECT run_purpose, retrieval_eligible FROM cognitive_runs WHERE run_id='run-live'"
        ).fetchone()
        upgraded_base = conn.execute(
            "SELECT run_purpose, retrieval_eligible FROM cognitive_runs WHERE run_id='run-base'"
        ).fetchone()
        assert upgraded_live["run_purpose"] == "migration"
        assert upgraded_live["retrieval_eligible"] == 0
        assert upgraded_base["run_purpose"] == "baseline"
        assert upgraded_base["retrieval_eligible"] == 0
        remaining_index = {
            row["episode_id"] for row in conn.execute("SELECT episode_id FROM retrieval_index").fetchall()
        }
        assert "ep-live" not in remaining_index
        assert "ep-base" not in remaining_index

    events = ledger.get_episode_events("ep-live")
    assert events and events[0]["event_id"] == "evt-1"

    result = retrieve_memories(
        RetrievalQuery(
            workspace_id="ws",
            actor_id="actor",
            asset_id="query-asset",
            goal_type="critique",
            goal_instance_id=None,
            scene_signature="interior_scene",
            category_signature="cluttered_room_weak_composition",
        ),
        ledger=ledger,
        state_snapshot={"retrieval_enabled": True, "cutoff_score": 0.7},
    )
    assert not result.references


def test_reopen_upgraded_database_does_not_reapply_migration(migration_env):
    db_path = migration_env / "reopen.sqlite3"
    _seed_old_schema_db(db_path)
    CognitionLedger(db_path=db_path)
    reopened = CognitionLedger(db_path=db_path)

    with reopened._connect() as conn:
        versions = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        assert [row["version"] for row in versions] == [1, 2]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO cognitive_state_versions
                (state_version_id, workspace_id, parent_version_id, label, created_at, is_active, snapshot_artefact_hash)
                VALUES ('state-2', 'ws', NULL, 'state_baseline', '2026-01-01T00:00:00Z', 1, 'snap')
                """
            )
