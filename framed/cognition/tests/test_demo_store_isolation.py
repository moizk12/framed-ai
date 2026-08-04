from __future__ import annotations

import json
from pathlib import Path

import pytest

from framed.cognition.demo.slice_a_e1_e2 import run_slice_a_demo


@pytest.fixture
def cognition_env(monkeypatch, tmp_path):
    production_dir = tmp_path / "production_cognition"
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(production_dir))
    return production_dir


def test_two_default_demo_runs_are_clean_and_equivalent(cognition_env):
    first = run_slice_a_demo()
    second = run_slice_a_demo()

    assert first["status"] == "PASS"
    assert second["status"] == "PASS"
    assert first["temporary_store"] is True
    assert second["temporary_store"] is True
    assert len(first["selected_memory_reference_ids"]) == 1
    assert len(second["selected_memory_reference_ids"]) == 1
    assert first["delta_count"] == second["delta_count"]
    assert first["rollback_retrieval_count"] == second["rollback_retrieval_count"] == 0
    assert {r["rejection_reason"] for r in first["baseline_rejections"]} == {"excluded_by_experiment"}
    assert {r["rejection_reason"] for r in second["baseline_rejections"]} == {"excluded_by_experiment"}


def test_second_default_run_does_not_retrieve_first_run_memories(cognition_env):
    first = run_slice_a_demo()
    second = run_slice_a_demo()

    assert second["selected_source_episode_ids"] == [second["e1_episode_id"]]
    assert first["e1_episode_id"] not in second["selected_source_episode_ids"]
    assert first["selected_memory_reference_ids"][0] != second["selected_memory_reference_ids"][0]


def test_default_demo_does_not_touch_configured_production_cognition_dir(cognition_env):
    run_slice_a_demo()

    assert not cognition_env.exists() or not any(cognition_env.iterdir())


def test_explicit_non_empty_directory_without_permission_fails_safely(cognition_env, tmp_path):
    supplied = tmp_path / "demo_store"
    supplied.mkdir()
    (supplied / "existing.sqlite3").write_text("busy", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        run_slice_a_demo(cognition_dir=supplied)

    assert (supplied / "existing.sqlite3").exists()


def test_reset_store_resets_only_supplied_demo_directory(cognition_env, tmp_path):
    supplied = tmp_path / "demo_store"
    supplied.mkdir()
    (supplied / "stale.txt").write_text("stale", encoding="utf-8")
    untouched = tmp_path / "untouched"
    untouched.mkdir()
    (untouched / "keep.txt").write_text("keep", encoding="utf-8")

    report = run_slice_a_demo(cognition_dir=supplied, reset_store=True)

    assert report["status"] == "PASS"
    assert not (supplied / "stale.txt").exists()
    assert (untouched / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (supplied / "cognition_ledger.sqlite3").exists()


def test_keep_store_retains_temporary_store_and_reports_path(cognition_env):
    report = run_slice_a_demo(keep_store=True)

    kept_path = Path(report["kept_store_path"])
    assert report["status"] == "PASS"
    assert kept_path.exists()
    assert (kept_path / "cognition_ledger.sqlite3").exists()
    assert Path(report["evidence_dir"], "slice_a_demo_report.json").exists()
