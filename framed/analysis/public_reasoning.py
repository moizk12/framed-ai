"""Bounded, text-only public deliberation; no longitudinal state."""
import json

from .critique_finalization import CritiqueRuntimeError
from .intelligence_formatting import _safe_parse_layer_json
from .llm_provider import call_model_a

STANDALONE_CONTRACT = """This is a standalone public analysis.
You have no prior knowledge of this photographer unless information was explicitly
supplied in this request. Never imply that you remember the user, have seen their
earlier work, know their recurring habits, or know their creative trajectory.
Do not invent previous struggles, improvement over time, evolution, usual
tendencies, remembered preferences, or earlier conversations.
Judge only the supplied image evidence and explicit request context."""


def reason_standalone(recognition):
    result = call_model_a(
        prompt="Evaluate this recognition, its uncertainty, and concrete photographic choices. "
        "Return JSON with observations (list of strings), questions (list of strings), "
        "and why_i_believe_this (string).\n" + json.dumps(recognition),
        system_prompt=STANDALONE_CONTRACT,
        response_format={"type": "json_object"}, max_tokens=1000, temperature=0.3,
    )
    if result.get("error"):
        raise CritiqueRuntimeError("public_reasoning_failed")
    parsed = _safe_parse_layer_json(result.get("content") or "")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("why_i_believe_this"), str):
        raise CritiqueRuntimeError("public_reasoning_malformed")
    for key in ("observations", "questions"):
        if not isinstance(parsed.get(key), list) or not all(isinstance(v, str) for v in parsed[key]):
            raise CritiqueRuntimeError("public_reasoning_malformed")
    return {key: parsed[key] for key in ("observations", "questions", "why_i_believe_this")}
