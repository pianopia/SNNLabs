"""LLM backends for harness baselines (offline-scripted + optional HTTP).

Network access is **never** performed by the scripted backend used in tests.
The HTTP backend is opt-in for real API comparisons and is not imported by
default runners unless ``--llm-backend http`` is selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable
import json
import os
import re
import time
import urllib.error
import urllib.request


@runtime_checkable
class LlmBackend(Protocol):
    """Minimal completion interface shared by scripted and HTTP backends."""

    name: str

    def complete(self, prompt: str, *, system: str | None = None) -> "LlmCompletion":
        """Return a text completion for ``prompt``."""


@dataclass(frozen=True)
class LlmCompletion:
    text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    backend: str = ""
    raw: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Not a real tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class ScriptedLlmBackend:
    """Deterministic offline backend for tests and CI.

    ``responder`` maps a prompt to a reply string. If omitted, replies with
    class ``0`` (useful as a fixed weak baseline).
    """

    name: str = "scripted"
    responder: Callable[[str], str] | None = None
    fixed_reply: str = "0"
    artificial_latency_ms: float = 0.0

    def complete(self, prompt: str, *, system: str | None = None) -> LlmCompletion:
        del system  # scripted path ignores system prompt content
        start = time.perf_counter()
        if self.responder is not None:
            text = self.responder(prompt)
        else:
            text = self.fixed_reply
        elapsed = (time.perf_counter() - start) * 1000.0 + self.artificial_latency_ms
        return LlmCompletion(
            text=str(text),
            latency_ms=elapsed,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(str(text)),
            backend=self.name,
        )


@dataclass
class HttpChatLlmBackend:
    """OpenAI-compatible chat completions over HTTP (opt-in).

    Env (or constructor):
      - ``api_key`` / ``OPENAI_API_KEY``
      - ``base_url`` / ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``)
      - ``model`` / ``OPENAI_MODEL`` (default ``gpt-4o-mini``)
    """

    name: str = "http-chat"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_s: float = 60.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        env_base = os.environ.get("OPENAI_BASE_URL")
        if env_base:
            self.base_url = env_base.rstrip("/")
        env_model = os.environ.get("OPENAI_MODEL")
        if env_model:
            self.model = env_model

    def complete(self, prompt: str, *, system: str | None = None) -> LlmCompletion:
        if not self.api_key:
            raise RuntimeError(
                "HttpChatLlmBackend requires api_key or OPENAI_API_KEY. "
                "Use ScriptedLlmBackend for offline runs."
            )
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {payload!r}")
        text = choices[0].get("message", {}).get("content", "")
        usage = payload.get("usage") or {}
        return LlmCompletion(
            text=str(text),
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens") or estimate_tokens(prompt)),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(str(text))),
            backend=self.name,
            raw={"model": self.model, "usage": usage},
        )


def parse_class_id(text: str, *, num_classes: int) -> int | None:
    """Extract the first integer in ``[0, num_classes)`` from free-form text."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    for match in re.finditer(r"-?\d+", text):
        value = int(match.group(0))
        if 0 <= value < num_classes:
            return value
    return None


def make_llm_backend(kind: str, **kwargs) -> LlmBackend:
    """Factory: ``scripted`` (default) or ``http``."""
    kind = (kind or "scripted").lower()
    if kind in {"scripted", "mock", "offline"}:
        return ScriptedLlmBackend(**{k: v for k, v in kwargs.items() if k in {
            "name", "responder", "fixed_reply", "artificial_latency_ms"
        }})
    if kind in {"http", "openai", "chat"}:
        return HttpChatLlmBackend(**{k: v for k, v in kwargs.items() if k in {
            "name", "api_key", "base_url", "model", "timeout_s", "temperature"
        }})
    raise ValueError(f"unknown llm backend kind: {kind!r} (expected scripted|http)")
