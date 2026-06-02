#!/usr/bin/env python3
"""
Minimal A/B comparison: Qwen2.5-VL-7B vs Gemma (LOCAL_GEMMA4_E4B) on five FRAMED-style tasks.
Saves raw model output to eval_outputs/; never raises on parse failures.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from framed.analysis.llm_provider import (  # noqa: E402
    MODEL_CONFIGS,
    FRAMED_LOCAL_BASE_URL,
    FRAMED_LOCAL_API_KEY,
)


def _mime(path: Path) -> str:
    suf = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suf, "image/jpeg")


def _b64_image(path: Path) -> Tuple[str, str]:
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii"), _mime(path)


def vision_chat(
    model_id: str,
    system: str,
    user_text: str,
    image_path: Path,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from openai import OpenAI

    base = os.getenv("FRAMED_LOCAL_BASE_URL", FRAMED_LOCAL_BASE_URL).strip().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    api_key = os.getenv("FRAMED_LOCAL_API_KEY", FRAMED_LOCAL_API_KEY)
    b64, mime = _b64_image(image_path)
    data_url = f"data:{mime};base64,{b64}"
    client = OpenAI(base_url=base, api_key=api_key)
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        kwargs["response_format"] = response_format
    try:
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message if resp.choices else None
        text = (msg.content or "") if msg else ""
        return {"ok": True, "text": text, "error": None}
    except Exception as e:
        return {"ok": False, "text": "", "error": str(e)}


def task_prompts() -> List[Tuple[str, str, str, Optional[Dict[str, Any]]]]:
    """name, system, user, optional response_format for JSON reliability task."""
    schema_hint = '{"scene":"string","subjects":[],"mood":"string","confidence":0.0}'
    return [
        (
            "scene_interpretation",
            "You describe photographs precisely.",
            "Describe the scene: main subjects, composition, mood, lighting. Be specific.",
            None,
        ),
        (
            "critique_grounding",
            "You only use visible evidence.",
            "Write 2-3 sentences of photographic critique using ONLY what is visible in the image. No invented objects.",
            None,
        ),
        (
            "ocr_text",
            "You transcribe visible text.",
            "List all visible text in the image exactly as it appears. If none, say NONE.",
            None,
        ),
        (
            "object_localization",
            "You return structured detections when possible.",
            'List salient objects. Return JSON: {"objects":[{"label":"string","box":[x,y,w,h]}]} '
            "with box coordinates normalized 0-1 to image width/height if you can; else use empty box []. "
            "If JSON is not possible, return plain text listing objects.",
            None,
        ),
        (
            "json_reliability",
            "You output strict JSON only.",
            f"Return ONLY valid JSON matching this shape (fill fields): {schema_hint}",
            {"type": "json_object"},
        ),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="A/B local models (Qwen vs Gemma) on five tasks")
    p.add_argument("--image", required=True, help="Path to image")
    p.add_argument("--output-dir", default="eval_outputs", help="Directory for raw outputs")
    args = p.parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        print(f"Not a file: {image_path}", file=sys.stderr)
        return 1

    qwen = MODEL_CONFIGS["LOCAL_QWEN25_VL_7B"]
    gemma = MODEL_CONFIGS["LOCAL_GEMMA4_E4B"]
    models: List[Tuple[str, str, float]] = [
        ("LOCAL_QWEN25_VL_7B", qwen["model_name"], float(qwen.get("temperature", 0.3))),
        ("LOCAL_GEMMA4_E4B", gemma["model_name"], float(gemma.get("temperature", 0.5))),
    ]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = _project_root / args.output_dir / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary: List[str] = []
    for cfg_id, model_id, temp in models:
        sub = out_root / cfg_id
        sub.mkdir(exist_ok=True)
        summary.append(f"=== {cfg_id} ({model_id}) ===")
        for name, system, user, rf in task_prompts():
            r = vision_chat(
                model_id=model_id,
                system=system,
                user_text=user,
                image_path=image_path,
                max_tokens=2048,
                temperature=temp,
                response_format=rf,
            )
            payload = {
                "config_id": cfg_id,
                "model_id": model_id,
                "task": name,
                "ok": r["ok"],
                "error": r.get("error"),
                "text": r.get("text", ""),
            }
            fp = sub / f"{name}.json"
            fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            preview = (r.get("text") or "")[:120].replace("\n", " ")
            line = f"  {name}: {'OK' if r['ok'] else 'ERR'} len={len(r.get('text') or '')} {preview!r}"
            summary.append(line)
        summary.append("")

    report = out_root / "summary.txt"
    report.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nSaved under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
