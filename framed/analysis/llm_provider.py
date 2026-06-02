"""Model A/B LLM calls (local OpenAI-compat by default; OpenAI optional via OPENAI_API_KEY)."""

import os
import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from .providers.base import LLMProvider
from .providers.local_openai import LocalOpenAICompatProvider as ProvidersLocalOpenAICompatProvider
from .providers.openai_provider import OpenAIProvider as ProvidersOpenAIProvider

logger = logging.getLogger(__name__)


def _normalize_local_base(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


FRAMED_LOCAL_BASE_URL = _normalize_local_base(os.getenv("FRAMED_LOCAL_BASE_URL", "http://localhost:1234/v1"))
FRAMED_LOCAL_API_KEY = os.getenv("FRAMED_LOCAL_API_KEY", "lm-studio")
FRAMED_STRICT_LOCAL = os.getenv("FRAMED_STRICT_LOCAL", "false").lower() == "true"

MODEL_A_TYPE = os.getenv("FRAMED_MODEL_A", "LOCAL_QWEN25_VL_7B")
MODEL_B_TYPE = os.getenv("FRAMED_MODEL_B", "LOCAL_QWEN25_VL_7B")

MODEL_CONFIGS = {
    "PLACEHOLDER": {
        "provider": "placeholder",
        "model_name": "placeholder",
        "api_key_env": None,
        "max_tokens": 4000,
        "temperature": 0.7,
    },
    "LOCAL_QWEN25_VL_7B": {
        "provider": "local_openai",
        "model_name": "qwen2.5-vl-7b-instruct",
        "api_key_env": None,
        "max_tokens": 8192,
        "temperature": 0.3,
    },
    "LOCAL_GEMMA4_E4B": {
        "provider": "local_openai",
        "model_name": "google/gemma-3-4b-it",
        "api_key_env": None,
        "max_tokens": 8192,
        "temperature": 0.5,
    },
    "GPT_5_2": {
        "provider": "openai",
        "model_name": "gpt-5.2",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 8192,
        "temperature": None,
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "low"},
    },
    "GPT_5_MINI": {
        "provider": "openai",
        "model_name": "gpt-5-mini",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 2048,
        "temperature": 1,
        "text": {"verbosity": "medium"},
    },
}

MAX_RETRIES = 3
RETRY_DELAY = 1.0

FALLBACK_MODEL_A = os.getenv("FRAMED_FALLBACK_MODEL_A", MODEL_A_TYPE)
FALLBACK_MODEL_B = os.getenv("FRAMED_FALLBACK_MODEL_B", MODEL_B_TYPE)

RATE_LIMIT_CALLS_PER_MINUTE = {
    "PLACEHOLDER": 1000,
}

COST_PER_1M_TOKENS = {
    "PLACEHOLDER": {"input": 0.0, "output": 0.0},
    "LOCAL_QWEN25_VL_7B": {"input": 0.0, "output": 0.0},
    "LOCAL_GEMMA4_E4B": {"input": 0.0, "output": 0.0},
}


class PlaceholderProvider(LLMProvider):
    def __init__(self, model_name: str = "placeholder"):
        self.model_name = model_name
        self.is_available_flag = True

    def is_available(self) -> bool:
        return self.is_available_flag

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.warning("PLACEHOLDER LLM call model=%s", self.model_name)
        mock_content = {
            "recognition": {"what_i_see": "[PLACEHOLDER]", "evidence": [], "confidence": 0.85},
            "meta_cognition": {"why_i_believe_this": "[PLACEHOLDER]", "confidence": 0.85, "what_i_might_be_missing": "[PLACEHOLDER]"},
            "temporal": {"how_i_used_to_see_this": "[PLACEHOLDER]", "how_i_see_it_now": "[PLACEHOLDER]", "evolution_reason": "[PLACEHOLDER]"},
            "emotion": {"what_i_feel": "[PLACEHOLDER]", "why": "[PLACEHOLDER]", "evolution": "[PLACEHOLDER]"},
            "continuity": {"user_pattern": "[PLACEHOLDER]", "comparison": "[PLACEHOLDER]", "trajectory": "[PLACEHOLDER]"},
            "mentor": {"observations": ["[PLACEHOLDER]"], "questions": ["[PLACEHOLDER]"], "challenges": []},
            "self_critique": {"past_errors": ["[PLACEHOLDER]"], "evolution": "[PLACEHOLDER]"},
        }
        if response_format and response_format.get("type") == "json_object":
            content = json.dumps(mock_content, indent=2)
        else:
            content = "[PLACEHOLDER] mock critique"
        return {
            "content": content,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(prompt.split()) + len(content.split()),
            },
            "model": self.model_name,
            "error": None,
        }


def create_provider(model_type: str, role: Literal["reasoning", "expression"]) -> LLMProvider:
    config = MODEL_CONFIGS.get(model_type)

    if not config:
        logger.warning("Unknown model type %s; using LOCAL_QWEN25_VL_7B", model_type)
        return create_provider("LOCAL_QWEN25_VL_7B", role)

    provider_name = config["provider"]

    if provider_name == "placeholder":
        return PlaceholderProvider(model_name=f"placeholder-{role}")

    if provider_name == "local_openai":
        return ProvidersLocalOpenAICompatProvider(config, role)

    if provider_name == "openai":
        return ProvidersOpenAIProvider(config, role)

    logger.warning("Provider %s not implemented; using LOCAL_QWEN25_VL_7B", provider_name)
    return create_provider("LOCAL_QWEN25_VL_7B", role)


_model_a_provider: Optional[LLMProvider] = None
_model_b_provider: Optional[LLMProvider] = None


def get_model_a_provider() -> LLMProvider:
    global _model_a_provider
    if _model_a_provider is None:
        _model_a_provider = create_provider(MODEL_A_TYPE, "reasoning")
        logger.info("Model A provider: %s", MODEL_A_TYPE)
    return _model_a_provider


def get_model_b_provider() -> LLMProvider:
    global _model_b_provider
    if _model_b_provider is None:
        _model_b_provider = create_provider(MODEL_B_TYPE, "expression")
        logger.info("Model B provider: %s", MODEL_B_TYPE)
    return _model_b_provider


def _strict_local_fail_message(provider: LLMProvider) -> str:
    if isinstance(provider, ProvidersLocalOpenAICompatProvider):
        return ProvidersLocalOpenAICompatProvider.last_error or "Local LM Studio not available (see logs)."
    return "Provider not available"


def call_model_a(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[Dict[str, Any]] = None,
    use_fallback: bool = True,
) -> Dict[str, Any]:
    provider = get_model_a_provider()
    if FRAMED_STRICT_LOCAL and isinstance(provider, ProvidersLocalOpenAICompatProvider):
        if not provider.is_available():
            return {
                "content": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": MODEL_A_TYPE,
                "error": _strict_local_fail_message(provider),
            }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if not provider.is_available():
                raise ValueError(f"Model A not available: {MODEL_A_TYPE}")
            result = provider.call(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        except Exception as e:
            last_error = e
            logger.warning("Model A attempt %s/%s failed: %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    if use_fallback and FALLBACK_MODEL_A != MODEL_A_TYPE:
        fb = FALLBACK_MODEL_A
        if fb == "PLACEHOLDER" and MODEL_CONFIGS.get(MODEL_A_TYPE, {}).get("provider") == "local_openai":
            logger.error("Model A failed; skipping PLACEHOLDER fallback while using local_openai primary")
        else:
            logger.warning("Model A fallback: %s", fb)
            fallback_provider = create_provider(fb, "reasoning")
            try:
                return fallback_provider.call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            except Exception as e:
                logger.error("Fallback Model A failed: %s", e)

    return {
        "content": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "model": MODEL_A_TYPE,
        "error": str(last_error) if last_error else "Unknown error",
    }


def call_model_b(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    use_fallback: bool = True,
) -> Dict[str, Any]:
    provider = get_model_b_provider()
    if FRAMED_STRICT_LOCAL and isinstance(provider, ProvidersLocalOpenAICompatProvider):
        if not provider.is_available():
            return {
                "content": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": MODEL_B_TYPE,
                "error": _strict_local_fail_message(provider),
            }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if not provider.is_available():
                raise ValueError(f"Model B not available: {MODEL_B_TYPE}")
            result = provider.call(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=None,
            )
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        except Exception as e:
            last_error = e
            logger.warning("Model B attempt %s/%s failed: %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    if use_fallback and FALLBACK_MODEL_B != MODEL_B_TYPE:
        fb = FALLBACK_MODEL_B
        if fb == "PLACEHOLDER" and MODEL_CONFIGS.get(MODEL_B_TYPE, {}).get("provider") == "local_openai":
            logger.error("Model B failed; skipping PLACEHOLDER fallback while using local_openai primary")
        else:
            logger.warning("Model B fallback: %s", fb)
            fallback_provider = create_provider(fb, "expression")
            try:
                return fallback_provider.call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=None,
                )
            except Exception as e:
                logger.error("Fallback Model B failed: %s", e)

    return {
        "content": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "model": MODEL_B_TYPE,
        "error": str(last_error) if last_error else "Unknown error",
    }


def estimate_cost(model_type: str, input_tokens: int, output_tokens: int) -> float:
    costs = COST_PER_1M_TOKENS.get(model_type, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000) * costs["input"] + (output_tokens / 1_000_000) * costs["output"]
