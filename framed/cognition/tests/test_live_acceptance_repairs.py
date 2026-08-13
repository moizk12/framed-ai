"""Targeted coverage for Slice A live-acceptance defects D1 and D2."""

from __future__ import annotations

import base64
import io
import os
import socket
import sys
from pathlib import Path

import pytest

from framed.cognition.contracts.runs import (
    RunMode,
    RunPurpose,
    resolve_mode_purpose,
    validate_mode_purpose,
)
from framed.cognition.demo.server_process import ManagedServer
from framed.cognition.integration.pipeline_hook import begin_cognition_run, finalize_cognition_run
from framed.cognition.ledger.sqlite_store import reset_ledger


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

TINY_SERVER = """
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = os.environ.get("FRAMED_COGNITION_DIR", "").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(os.environ["TINY_PORT"])), Handler).serve_forever()
"""


@pytest.fixture
def cognition_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMED_COGNITION_V1", "true")
    monkeypatch.setenv("FRAMED_COGNITION_DIR", str(tmp_path))
    ledger = reset_ledger(tmp_path / "test.sqlite3")
    yield ledger, tmp_path
    reset_ledger()


def _result():
    return {"visual_evidence": {"scene_gate": {"scene_type": "interior_scene", "signals": {}}}}


def _intel(text="room"):
    return {"recognition": {"what_i_see": text, "confidence": 0.4}}


def _open_count(ledger) -> int:
    with ledger._connect() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM episodes WHERE status='open'").fetchone()["c"]


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.parametrize(
    "purpose,mode",
    [
        (RunPurpose.BASELINE, RunMode.BASELINE),
        (RunPurpose.CONTROL, RunMode.CONTROL),
        (RunPurpose.REPLAY, RunMode.REPLAY),
        (RunPurpose.LIVE, RunMode.MEMORY_ENABLED),
        (RunPurpose.MEMORY_ENABLED, RunMode.MEMORY_ENABLED),
    ],
)
def test_omitted_mode_is_inferred_from_purpose(purpose, mode):
    resolved_mode, resolved_purpose = resolve_mode_purpose(None, purpose)
    assert resolved_mode is mode
    assert resolved_purpose is purpose


def test_explicit_incompatible_mode_purpose_still_rejected():
    with pytest.raises(ValueError, match="Incompatible"):
        resolve_mode_purpose(RunMode.MEMORY_ENABLED, RunPurpose.BASELINE)
    with pytest.raises(ValueError, match="Incompatible"):
        validate_mode_purpose(RunMode.CONTROL, RunPurpose.LIVE)


def test_begin_baseline_without_mode_creates_ineligible_run(cognition_env, tmp_path):
    ledger, _ = cognition_env
    img = tmp_path / "shot.jpg"
    img.write_bytes(b"baseline-bytes")
    session = begin_cognition_run(
        result=_result(),
        image_path=str(img),
        asset_filename="shot.jpg",
        run_purpose=RunPurpose.BASELINE,
    )
    assert session is not None
    assert session.run_mode is RunMode.BASELINE
    assert session.run_purpose is RunPurpose.BASELINE
    assert session.memory_reference_ids == []
    finalize_cognition_run(session, _result(), _intel("baseline"))
    with ledger._connect() as conn:
        row = conn.execute(
            "SELECT mode, run_purpose, retrieval_eligible FROM cognitive_runs WHERE run_id=?",
            (session.run_id,),
        ).fetchone()
        indexed = conn.execute("SELECT COUNT(*) AS c FROM retrieval_index").fetchone()["c"]
    assert row["mode"] == "baseline"
    assert row["run_purpose"] == "baseline"
    assert row["retrieval_eligible"] == 0
    assert indexed == 0
    assert _open_count(ledger) == 0


def test_begin_control_without_mode_is_ineligible(cognition_env, tmp_path):
    ledger, _ = cognition_env
    img = tmp_path / "ctrl.jpg"
    img.write_bytes(b"control-bytes")
    session = begin_cognition_run(
        result=_result(),
        image_path=str(img),
        asset_filename="ctrl.jpg",
        run_purpose=RunPurpose.CONTROL,
    )
    assert session is not None
    assert session.run_mode is RunMode.CONTROL
    assert session.run_purpose is RunPurpose.CONTROL
    finalize_cognition_run(session, _result(), _intel("street"))
    with ledger._connect() as conn:
        row = conn.execute(
            "SELECT mode, run_purpose, retrieval_eligible FROM cognitive_runs WHERE run_id=?",
            (session.run_id,),
        ).fetchone()
    assert row["mode"] == "control"
    assert row["run_purpose"] == "control"
    assert row["retrieval_eligible"] == 0
    assert _open_count(ledger) == 0


def test_incompatible_begin_leaves_no_open_episode(cognition_env, tmp_path):
    ledger, _ = cognition_env
    img = tmp_path / "bad.jpg"
    img.write_bytes(b"bad-bytes")
    with pytest.raises(ValueError, match="Incompatible"):
        begin_cognition_run(
            result=_result(),
            image_path=str(img),
            asset_filename="bad.jpg",
            run_mode=RunMode.MEMORY_ENABLED,
            run_purpose=RunPurpose.BASELINE,
        )
    assert _open_count(ledger) == 0
    with ledger._connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM cognitive_runs").fetchone()["c"] == 0


def test_http_baseline_and_control_create_real_runs(cognition_env, monkeypatch):
    ledger, _ = cognition_env
    monkeypatch.setattr("framed.analysis.vision.ensure_directories", lambda: None)

    def fake_run_full_analysis(image_path, cognition_run_purpose=None, **kwargs):
        purpose = RunPurpose(cognition_run_purpose) if cognition_run_purpose else None
        result = _result()
        session = begin_cognition_run(
            result=result,
            image_path=image_path,
            asset_filename=os.path.basename(image_path),
            run_purpose=purpose,
            baseline_run_id=kwargs.get("baseline_run_id"),
            comparison_group_id=kwargs.get("comparison_group_id"),
        )
        assert session is not None
        finalize_cognition_run(session, result, _intel(cognition_run_purpose or "live"))
        result["intelligence"] = {"recognition": {"confidence": 0.4}}
        return result

    monkeypatch.setattr("framed.routes.run_full_analysis", fake_run_full_analysis)
    from framed import create_app

    client = create_app({"TESTING": True}).test_client()
    baseline = client.post(
        "/analyze",
        data={
            "mentor_mode": "Balanced Mentor",
            "cognition_run_purpose": "baseline",
            "image": (io.BytesIO(PNG), "shot.png"),
        },
    )
    assert baseline.status_code == 200
    baseline_body = baseline.get_json()
    prov = baseline_body.get("cognition_provenance") or {}
    assert prov.get("run_id")
    assert prov.get("run_mode") == "baseline"
    assert prov.get("run_purpose") == "baseline"
    assert prov.get("memory_reference_ids") == []

    control = client.post(
        "/analyze",
        data={
            "mentor_mode": "Balanced Mentor",
            "cognition_run_purpose": "control",
            "image": (io.BytesIO(PNG), "ctrl.png"),
        },
    )
    assert control.status_code == 200
    control_prov = control.get_json().get("cognition_provenance") or {}
    assert control_prov.get("run_id")
    assert control_prov.get("run_mode") == "control"
    assert control_prov.get("run_purpose") == "control"

    with ledger._connect() as conn:
        rows = list(conn.execute("SELECT mode, run_purpose, retrieval_eligible FROM cognitive_runs ORDER BY started_at"))
        open_eps = conn.execute("SELECT COUNT(*) AS c FROM episodes WHERE status='open'").fetchone()["c"]
    assert [(row["mode"], row["run_purpose"], row["retrieval_eligible"]) for row in rows] == [
        ("baseline", "baseline", 0),
        ("control", "control", 0),
    ]
    assert open_eps == 0


def test_managed_server_restart_changes_pid(tmp_path):
    script = tmp_path / "tiny_server.py"
    script.write_text(TINY_SERVER, encoding="utf-8")
    cognition_dir = tmp_path / "cognition"
    cognition_dir.mkdir()
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TINY_PORT"] = str(port)
    env["FRAMED_COGNITION_DIR"] = str(cognition_dir)
    server = ManagedServer(
        command=[sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        url=f"http://127.0.0.1:{port}",
        log_path=tmp_path / "tiny.log",
        health_path="/",
    )
    first = server.start(timeout=15)
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            assert response.read().decode("utf-8") == str(cognition_dir)
        old_pid, new_pid = server.restart(timeout=15)
        assert old_pid == first
        assert new_pid != old_pid
        assert server.pid == new_pid
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            assert response.read().decode("utf-8") == str(cognition_dir)
    finally:
        server.stop()
    assert server.pid is None
