"""Measure process-cold and warm public inference on one immutable input.

Uses the configured providers. Outputs only public DTOs and safe failure types.
Model weights may already be cached; this does not simulate an empty HF volume.
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("test-results/inference-latency.json"))
    args = parser.parse_args()
    from framed.public_api import build_public_analysis_dto, run_public_analysis
    image_bytes = args.image.read_bytes()
    evidence = {"input_sha256": hashlib.sha256(image_bytes).hexdigest(), "cache_condition": "process cold; existing disk model caches retained", "runs": []}
    with tempfile.TemporaryDirectory(prefix="framed-benchmark-") as folder:
        image = Path(folder) / args.image.name
        image.write_bytes(image_bytes)
        for phase in ("cold", "warm"):
            started = time.perf_counter()
            try:
                result, duration = run_public_analysis(str(image), image.name)
                dto = build_public_analysis_dto(result, request_id="benchmark", analysis_id="benchmark", duration_ms=duration)
                row = {"phase": phase, "success": True, "public": dto}
            except Exception as exc:  # noqa: BLE001 - record safe failure types for both timing runs
                row = {"phase": phase, "success": False, "error_type": type(exc).__name__}
            row["seconds"] = round(time.perf_counter() - started, 2)
            evidence["runs"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in row.items() if k != "public"}), flush=True)
    return 0 if all(row["success"] for row in evidence["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
