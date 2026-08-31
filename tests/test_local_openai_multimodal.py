import base64
import inspect
from types import SimpleNamespace

from framed.analysis import llm_provider
from framed.analysis.intelligence_layers import reason_about_layers_2_7, reason_about_recognition
from framed.analysis.providers.local_openai import LocalOpenAICompatProvider, build_user_content


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _provider():
    provider = LocalOpenAICompatProvider(
        {"model_name": "qwen2.5-vl-7b-instruct", "max_tokens": 64, "temperature": 0.3},
        "reasoning",
    )
    completions = _FakeCompletions()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


def test_build_user_content_without_image_preserves_plain_text():
    assert build_user_content("structured evidence") == "structured evidence"


def test_build_user_content_adds_base64_image_url(tmp_path):
    image_bytes = b"\x89PNG\r\n\x1a\nframed-test"
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(image_bytes)

    content = build_user_content("structured evidence", str(image_path))

    assert content[0] == {"type": "text", "text": "structured evidence"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    )


def test_local_provider_sends_multimodal_content_only_when_image_is_supplied(tmp_path):
    provider, completions = _provider()
    image_path = tmp_path / "photo.webp"
    image_path.write_bytes(b"framed-webp-test")

    result = provider.call("evidence", system_prompt="recognize", image_path=str(image_path))

    assert result["error"] is None
    assert completions.kwargs["messages"][0] == {"role": "system", "content": "recognize"}
    assert completions.kwargs["messages"][1]["content"][0] == {"type": "text", "text": "evidence"}
    assert completions.kwargs["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/webp;base64,"
    )

    provider.call("expression only")
    assert completions.kwargs["messages"] == [{"role": "user", "content": "expression only"}]


def test_model_b_contract_remains_text_only():
    assert "image_path" not in inspect.signature(llm_provider.call_model_b).parameters
    assert "image_path" not in inspect.signature(reason_about_layers_2_7).parameters
    assert "image_path" in inspect.signature(reason_about_recognition).parameters


class _CapturingProvider:
    def __init__(self):
        self.kwargs = None

    def is_available(self):
        return True

    def call(self, **kwargs):
        self.kwargs = kwargs
        return {"content": "ok", "usage": {}, "model": "test", "error": None}


def test_model_a_facade_forwards_image_path(monkeypatch, tmp_path):
    provider = _CapturingProvider()
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"framed-jpeg-test")
    monkeypatch.setattr(llm_provider, "get_model_a_provider", lambda: provider)

    result = llm_provider.call_model_a("recognize", image_path=str(image_path), use_fallback=False)

    assert result["error"] is None
    assert provider.kwargs["image_path"] == str(image_path)


def test_model_b_facade_does_not_supply_image(monkeypatch):
    provider = _CapturingProvider()
    monkeypatch.setattr(llm_provider, "get_model_b_provider", lambda: provider)

    result = llm_provider.call_model_b("express", use_fallback=False)

    assert result["error"] is None
    assert "image_path" not in provider.kwargs
