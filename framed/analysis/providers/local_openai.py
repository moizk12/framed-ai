import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests

from .base import LLMProvider

logger = logging.getLogger(__name__)


def _normalize_local_base(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def fetch_local_model_ids(base_url: str, api_key: str, timeout: float = 5.0) -> Tuple[Optional[List[str]], Optional[str]]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        if r.status_code != 200:
            return None, f"GET {url} -> HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        ids = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
        return ids, None
    except requests.RequestException as e:
        return None, f"Cannot reach LM Studio at {url}: {e}"


def model_id_registered(model_ids: List[str], want: str) -> bool:
    if not want:
        return False
    if want in model_ids:
        return True
    w = want.lower()
    for mid in model_ids:
        if w in mid.lower() or mid.lower() in w:
            return True
    return False


_IMAGE_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _image_mime_type(path: Path) -> Optional[str]:
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return _IMAGE_MIME_BY_SUFFIX.get(path.suffix.lower())


def build_user_content(prompt: str, image_path: Optional[str] = None) -> Any:
    """Build standard OpenAI-compatible text or text-plus-image content."""
    if not image_path:
        return prompt

    path = Path(image_path)
    mime_type = _image_mime_type(path)
    if not mime_type:
        raise ValueError(f"Cannot determine image MIME type for {path.name!r}")
    data_url = f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]


class LocalOpenAICompatProvider(LLMProvider):
    """OpenAI-compatible chat API (LM Studio default: localhost:1234/v1)."""

    last_error: str = ""

    def __init__(self, config: Dict[str, Any], role: Literal["reasoning", "expression"]):
        self.config = config
        self.role = role
        self.base_url = _normalize_local_base(os.getenv("FRAMED_LOCAL_BASE_URL", "http://localhost:1234/v1"))
        self.api_key = os.getenv("FRAMED_LOCAL_API_KEY", "lm-studio")
        override = (
            os.getenv("FRAMED_LOCAL_MODEL_A", "").strip()
            if role == "reasoning"
            else os.getenv("FRAMED_LOCAL_MODEL_B", "").strip()
        )
        self.model_name = override or config.get("model_name", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            except Exception as e:
                LocalOpenAICompatProvider.last_error = f"OpenAI client init failed: {e}"
                logger.error("%s", LocalOpenAICompatProvider.last_error)
                return None
        return self._client

    def is_available(self) -> bool:
        LocalOpenAICompatProvider.last_error = ""
        ids, err = fetch_local_model_ids(self.base_url, self.api_key)
        if err:
            LocalOpenAICompatProvider.last_error = err
            return False
        if not self.model_name:
            LocalOpenAICompatProvider.last_error = "Model id empty: set FRAMED_LOCAL_MODEL_A / FRAMED_LOCAL_MODEL_B"
            return False
        if not model_id_registered(ids, self.model_name):
            LocalOpenAICompatProvider.last_error = (
                f"Model {self.model_name!r} not in /v1/models. Loaded: {ids[:12]}{'...' if len(ids) > 12 else ''}"
            )
            return False
        return True

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._get_client()
        if not client:
            return {"content": "", "usage": {}, "model": self.model_name, "error": LocalOpenAICompatProvider.last_error or "no client"}

        max_tokens = max_tokens or self.config.get("max_tokens", 4096)
        temp = temperature if temperature is not None else self.config.get("temperature", 0.5)
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": build_user_content(prompt, image_path)})

            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
            }
            if response_format and response_format.get("type") not in (None, "json_object"):
                kwargs["response_format"] = response_format

            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message if resp.choices else None
            content = (msg.content or "") if msg else ""
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens or 0,
                    "completion_tokens": resp.usage.completion_tokens or 0,
                    "total_tokens": resp.usage.total_tokens or 0,
                }
            return {"content": content, "usage": usage, "model": self.model_name, "error": None}
        except Exception as e:
            err = str(e)
            logger.exception("LocalOpenAICompatProvider call failed: %s", err)
            return {"content": "", "usage": {}, "model": self.model_name, "error": err}

