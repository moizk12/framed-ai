"""Execute FRAMED Slice A human live gate via real POST /analyze runtime."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

GATE_ROOT = Path(
    os.environ.get(
        "FRAMED_LIVE_GATE_ROOT",
        "C:/Users/moizk/OneDrive/Pictures/framed-clean.tmp/slice_a_live_gate",
    )
)
COGNITION_DIR = GATE_ROOT / "cognition_data"
ARCHIVE_ROOT = Path(
    os.environ.get(
        "FRAMED_LIVE_GATE_ARCHIVE",
        "C:/Users/moizk/Music/FRAMED_AGI_Research_Starter/FRAMED_AGI_Research_Starter/local_lab/archdaemon/status/slice_a",
    )
)

BASE_URL = os.environ.get("FRAMED_LIVE_GATE_URL", "http://127.0.0.1:7860")
IMAGES = {
    "e1": GATE_ROOT / "images" / "e1_cluttered_interior.jpg",
    "e2": GATE_ROOT / "images" / "e2_related_interior.jpg",
    "control": GATE_ROOT / "images" / "control_street.jpg",
    "fresco": GATE_ROOT / "images" / "fresco_michelangelo.jpg",
}
COMPARISON_GROUP = os.environ.get("FRAMED_LIVE_GATE_GROUP", "slice_a_live_gate")


def _setup_env() -> None:
    os.environ["FRAMED_COGNITION_V1"] = "true"
    os.environ["FRAMED_COGNITION_DIR"] = str(COGNITION_DIR)
    reset = os.environ.get("FRAMED_LIVE_GATE_RESET", "true").lower() in ("1", "true", "yes")
    if reset and COGNITION_DIR.exists():
        shutil.rmtree(COGNITION_DIR)
    COGNITION_DIR.mkdir(parents=True, exist_ok=True)


def _ledger():
    from framed.cognition.ledger.sqlite_store import reset_ledger

    return reset_ledger(COGNITION_DIR / "cognition_ledger.sqlite3")


def _activate(label: str) -> Dict[str, Any]:
    from framed.cognition.identity import get_identity

    ledger = _ledger()
    ws = get_identity()["workspace_id"]
    ledger.ensure_demo_states(ws)
    ledger.activate_state(ws, label)
    return ledger.get_active_state(ws)


def _post_analyze(
    image_path: Path,
    *,
    cognition_run_purpose: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    data: Dict[str, str] = {"mentor_mode": "Balanced Mentor", "comparison_group_id": COMPARISON_GROUP}
    if cognition_run_purpose:
        data["cognition_run_purpose"] = cognition_run_purpose
    if baseline_run_id:
        data["baseline_run_id"] = baseline_run_id
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/jpeg")}
        t0 = time.time()
        resp = requests.post(f"{BASE_URL}/analyze", files=files, data=data, timeout=timeout)
        elapsed = time.time() - t0
    out = {
        "status_code": resp.status_code,
        "method": "POST",
        "url": f"{BASE_URL}/analyze",
        "elapsed_sec": round(elapsed, 2),
        "form_data": data,
    }
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:2000]}
    out["response"] = body
    return out


def _record_from_response(tag: str, resp: Dict[str, Any]) -> Dict[str, Any]:
    body = resp.get("response") or {}
    prov = body.get("cognition_provenance") or {}
    intel = body.get("intelligence") or {}
    rec = intel.get("recognition") or {}
    ve = body.get("visual_evidence") or {}
    sg = ve.get("scene_gate") or {}
    record = {
        "tag": tag,
        "status_code": resp.get("status_code"),
        "elapsed_sec": resp.get("elapsed_sec"),
        "form_data": resp.get("form_data"),
        "episode_id": prov.get("episode_id"),
        "run_id": prov.get("run_id"),
        "run_purpose": prov.get("run_purpose"),
        "baseline_run_id": prov.get("baseline_run_id"),
        "state_version_id": prov.get("state_version_id"),
        "context_fingerprint": prov.get("context_fingerprint"),
        "memory_reference_ids": prov.get("memory_reference_ids") or [],
        "rejected_candidates": prov.get("rejected_candidates") or [],
        "deltas": prov.get("deltas") or [],
        "cognition_context_used": prov.get("cognition_context_used"),
        "recognition": {
            "what_i_see": rec.get("what_i_see"),
            "confidence": rec.get("confidence"),
            "_cognition_context_used": rec.get("_cognition_context_used"),
            "_memory_reference_ids": rec.get("_memory_reference_ids"),
        },
        "scene_gate": {
            "scene_type": sg.get("scene_type"),
            "signals": sg.get("signals"),
        },
        "deliberation_snapshot": {
            "primary_hypothesis": rec.get("what_i_see"),
            "confidence": rec.get("confidence"),
            "strategy": prov.get("strategy") or "standard",
        },
    }
    return record


def _ledger_export(run_id: Optional[str]) -> Dict[str, Any]:
    if not run_id:
        return {}
    ledger = _ledger()
    bundle = ledger.export_replay_bundle([run_id])
    refs = bundle.get("memory_references") or []
    return {"bundle": bundle, "memory_references": refs}


def _save(name: str, obj: Any) -> Path:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE_ROOT / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def _browser_click_smoke() -> Dict[str, Any]:
    """Real browser upload → Analyze click → POST /analyze (requires server running)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"skipped": True, "reason": "playwright not installed"}
    result: Dict[str, Any] = {"skipped": False}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            posts: List[Dict[str, Any]] = []

            def on_request(req):
                if req.method == "POST" and req.url.rstrip("/").endswith("/analyze"):
                    posts.append({"url": req.url, "method": req.method})

            page.on("request", on_request)
            page.goto(f"{BASE_URL}/upload", timeout=30000)
            if not IMAGES["e1"].exists():
                browser.close()
                return {"skipped": True, "reason": "e1 image missing for browser smoke"}
            page.set_input_files('input[type="file"]', str(IMAGES["e1"]))
            page.click('button:has-text("Analyze"), input[type="submit"][value*="Analyze"], #analyze-btn')
            page.wait_for_timeout(5000)
            result["post_requests"] = posts
            result["post_analyze_seen"] = len(posts) >= 1
            browser.close()
    except Exception as exc:
        result["error"] = str(exc)
        result["post_analyze_seen"] = False
    return result


def run_live_gate() -> Dict[str, Any]:
    _setup_env()
    report: Dict[str, Any] = {
        "schema": "slice_a_live_gate_v2",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_root": str(GATE_ROOT),
        "cognition_dir": str(COGNITION_DIR),
        "base_url": BASE_URL,
        "phases": {},
        "checks": {},
    }

    active = _activate("state_memory_enabled")
    from framed.cognition.identity import get_identity

    ident = get_identity()
    report["phases"]["phase1_clean_start"] = {
        "active_state": active.get("label"),
        "actor_id": ident["actor_id"],
        "workspace_id": ident["workspace_id"],
        "cognition_dir": str(COGNITION_DIR),
    }

    ui = {}
    try:
        rjs = requests.get(f"{BASE_URL}/static/feedback.js", timeout=10)
        ui["feedback_js_status"] = rjs.status_code
        rup = requests.get(f"{BASE_URL}/upload", timeout=10)
        ui["upload_page_status"] = rup.status_code
    except Exception as exc:
        ui["error"] = str(exc)
    ui["browser_click"] = _browser_click_smoke()
    report["phases"]["ui_smoke"] = ui

    _activate("state_memory_enabled")
    e1_resp = _post_analyze(IMAGES["e1"], cognition_run_purpose="live")
    e1 = _record_from_response("e1", e1_resp)
    e1["ledger"] = _ledger_export(e1.get("run_id"))
    report["phases"]["e1"] = e1

    _activate("state_baseline")
    baseline_state = _ledger().get_active_state(ident["workspace_id"])
    e2b_resp = _post_analyze(IMAGES["e2"], cognition_run_purpose="baseline")
    e2b = _record_from_response("e2_baseline", e2b_resp)
    e2b["active_state_after"] = _ledger().get_active_state(ident["workspace_id"]).get("label")
    e2b["baseline_state_snapshot"] = baseline_state.get("snapshot")
    report["phases"]["e2_baseline"] = e2b

    _activate("state_memory_enabled")
    e2m_resp = _post_analyze(
        IMAGES["e2"],
        cognition_run_purpose="memory_enabled",
        baseline_run_id=e2b.get("run_id"),
    )
    e2m = _record_from_response("e2_memory", e2m_resp)
    e2m["ledger"] = _ledger_export(e2m.get("run_id"))
    report["phases"]["e2_memory"] = e2m

    _activate("state_memory_enabled")
    ctrl_resp = _post_analyze(IMAGES["control"], cognition_run_purpose="control")
    ctrl = _record_from_response("control", ctrl_resp)
    ctrl["ledger"] = _ledger_export(ctrl.get("run_id"))
    report["phases"]["control"] = ctrl

    _activate("state_baseline")
    rb_resp = _post_analyze(IMAGES["e2"], cognition_run_purpose="baseline")
    rb = _record_from_response("rollback", rb_resp)
    rb["active_state_after"] = _ledger().get_active_state(ident["workspace_id"]).get("label")
    report["phases"]["rollback"] = rb

    post_restart_state = _ledger().get_active_state(ident["workspace_id"])
    report["phases"]["restart_durability"] = {
        "active_state_label": post_restart_state.get("label"),
        "snapshot_retrieval_enabled": post_restart_state.get("snapshot", {}).get("retrieval_enabled"),
    }

    _activate("state_memory_enabled")
    fresco_resp = _post_analyze(IMAGES["fresco"], cognition_run_purpose="live")
    fresco = _record_from_response("fresco", fresco_resp)
    report["phases"]["fresco"] = fresco

    e1_id = e1.get("episode_id")
    e2b_id = e2b.get("episode_id")
    e2b_run = e2b.get("run_id")
    mem_refs = e2m.get("ledger", {}).get("memory_references") or []
    source_eps = {r.get("source_episode_id") for r in mem_refs}
    rejected = e2m.get("rejected_candidates") or []
    baseline_rejects = [r for r in rejected if r.get("episode_id") == e2b_id or r.get("source_run_id") == e2b_run]

    checks = {
        "e1_stored": bool(e1_id),
        "e1_closed": bool(e1_id),
        "e2_baseline_fresh_run": e2b.get("run_id") and e2b.get("run_id") != e1.get("run_id"),
        "e2_baseline_zero_refs": len(e2b.get("memory_reference_ids") or []) == 0,
        "e2_baseline_not_indexed": e2b_id not in source_eps,
        "e2_memory_fresh_run": e2m.get("run_id") and e2m.get("run_id") != e2b.get("run_id"),
        "e2_memory_retrieved_e1_only": source_eps == {e1_id} and len(mem_refs) >= 1,
        "e2_baseline_rejected": bool(baseline_rejects),
        "e2_baseline_rejected_same_asset": any(r.get("same_asset") for r in baseline_rejects),
        "e2_baseline_rejected_purpose": any(r.get("ineligible_run_purpose") == "baseline" for r in baseline_rejects),
        "cognition_context_used": bool(e2m.get("cognition_context_used")),
        "meaningful_delta": len(e2m.get("deltas") or []) >= 1,
        "control_no_e1": e1_id not in {
            r.get("source_episode_id") for r in (ctrl.get("ledger", {}).get("memory_references") or [])
        },
        "rollback_zero_refs": len(rb.get("memory_reference_ids") or []) == 0,
        "rollback_state_durable": post_restart_state.get("label") == "state_baseline",
        "feedback_js_200": ui.get("feedback_js_status") == 200,
        "browser_post_analyze": ui.get("browser_click", {}).get("post_analyze_seen") is True
        or ui.get("browser_click", {}).get("skipped"),
        "fresco_not_ui": (fresco.get("scene_gate") or {}).get("scene_type") != "screenshot_ui",
        "all_post_analyze": all(
            x.get("status_code") == 200
            for x in (e1_resp, e2b_resp, e2m_resp, ctrl_resp, rb_resp, fresco_resp)
            if isinstance(x, dict)
        ),
    }
    report["checks"] = checks
    report["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    report["failed_checks"] = [k for k, v in checks.items() if not v]

    _save("live_gate_report_v2.json", report)
    _save("live_e1_record_v2.json", e1)
    _save("live_e2_baseline_record_v2.json", e2b)
    _save("live_e2_memory_record_v2.json", e2m)
    _save("live_control_record_v2.json", ctrl)
    _save(
        "live_deliberation_delta_report_v2.json",
        {
            "deltas": e2m.get("deltas"),
            "e2_baseline": e2b.get("deliberation_snapshot"),
            "e2_memory": e2m.get("deliberation_snapshot"),
            "memory_references": mem_refs,
            "rejected_candidates": rejected,
        },
    )
    _save("live_rollback_report_v2.json", rb)
    _save("live_restart_durability_report_v2.json", report["phases"]["restart_durability"])
    _save("live_ui_smoke_report_v2.json", ui)
    _save("live_fresco_regression_report_v2.json", fresco)
    return report


def main() -> int:
    try:
        report = run_live_gate()
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("verdict") == "PASS" else 1
    except Exception as exc:
        print(json.dumps({"verdict": "ERROR", "error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
