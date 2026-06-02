import logging
import os
from typing import Any, Dict, Literal, Optional

from .base import LLMProvider

logger = logging.getLogger(__name__)


def _extract_text_from_responses_output(output: Any) -> str:
    if not output:
        return ""
    for item in output:
        if hasattr(item, "content") and item.content:
            for part in item.content:
                if hasattr(part, "text") and part.text:
                    return part.text
        if hasattr(item, "type") and item.type == "message":
            if hasattr(item, "content") and item.content:
                for part in item.content:
                    if getattr(part, "type", None) == "output_text" and hasattr(part, "text"):
                        return part.text
    return ""


class OpenAIProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any], role: Literal["reasoning", "expression"]):
        self.config = config
        self.role = role
        self.model_name = config["model_name"]
        self._api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        api_key = os.getenv(self._api_key_env, "").strip()
        if not api_key:
            try:
                from pathlib import Path
                from dotenv import load_dotenv

                for d in (Path(__file__).resolve().parents[3], Path.cwd()):
                    env_path = d / ".env"
                    if env_path.exists():
                        load_dotenv(env_path)
                        api_key = os.getenv(self._api_key_env, "").strip()
                        break
            except ImportError:
                pass
        if not api_key:
            return None
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except Exception as e:
                logger.error("OpenAI client init failed: %s", e)
                return None
        return self._client

    def is_available(self) -> bool:
        return self._get_client() is not None

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        client = self._get_client()
        if not client:
            return {"content": "", "usage": {}, "model": self.model_name, "error": "OpenAI client not available"}
        max_tokens = max_tokens or self.config.get("max_tokens", 4096)
        if self.role == "reasoning":
            return self._call_reasoning(client, prompt, system_prompt, max_tokens, response_format)
        return self._call_expression(
            client,
            prompt,
            system_prompt,
            max_tokens,
            temperature if temperature is not None else self.config.get("temperature"),
        )

    def _call_reasoning(self, client, prompt: str, system_prompt: Optional[str], max_tokens: int, response_format: Optional[Dict[str, Any]]):
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            try:
                kwargs: Dict[str, Any] = {"model": self.model_name, "input": [{"role": "user", "content": prompt}], "max_output_tokens": max_tokens}
                if system_prompt:
                    kwargs["input"] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                kwargs["reasoning"] = self.config.get("reasoning", {"effort": "medium"})
                kwargs["text"] = self.config.get("text", {"verbosity": "low"})
                resp = client.responses.create(**kwargs)
                text = getattr(resp, "output_text", None) or ""
                if not text and hasattr(resp, "output") and resp.output:
                    text = _extract_text_from_responses_output(resp.output)
                usage = {}
                if hasattr(resp, "usage") and resp.usage:
                    usage = {
                        "prompt_tokens": getattr(resp.usage, "input_tokens", 0),
                        "completion_tokens": getattr(resp.usage, "output_tokens", 0),
                        "total_tokens": getattr(resp.usage, "total_tokens", 0),
                    }
                return {"content": text, "usage": usage, "model": self.model_name, "error": None}
            except Exception as e:
                logger.warning("Responses API failed, falling back to Chat: %s", e)

        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        kwargs: Dict[str, Any] = {"model": self.model_name, "messages": messages, "max_completion_tokens": max_tokens, "temperature": 0.3}
        if response_format and response_format.get("type") == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message if resp.choices else None
        content = msg.content if msg else ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        }
        return {"content": content or "", "usage": usage, "model": self.model_name, "error": None}

    def _call_expression(self, client, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: Optional[float]):
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        kwargs: Dict[str, Any] = {"model": self.model_name, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message if resp.choices else None
        content = msg.content if msg else ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        }
        return {"content": content or "", "usage": usage, "model": self.model_name, "error": None}

