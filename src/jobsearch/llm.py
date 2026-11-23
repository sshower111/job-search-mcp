"""Tiny OpenAI-compatible chat client (Ollama, LM Studio, vLLM, or a hosted API).

Uses only the standard library so there is no extra dependency. Only
``POST {base_url}/chat/completions`` is used, with optional ``tools``.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from .config import LLMSettings

log = logging.getLogger("jobsearch.llm")


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Return an assistant message dict: {"role": "assistant", "content": str|None, "tool_calls": [...]}."""
        ...


Transport = Callable[[str, dict[str, str], bytes, float], dict[str, Any]]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        raise LLMError("LLM base_url must be http(s)")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - scheme checked above
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError(f"Could not reach LLM endpoint {url}: {exc}") from exc


class OpenAICompatibleChat:
    def __init__(self, settings: LLMSettings, transport: Transport | None = None):
        self.settings = settings
        self.transport = transport or _urllib_transport

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        key = self.settings.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        log.debug("LLM request model=%s messages=%d tools=%d", self.settings.model, len(messages),
                  len(tools or []))
        data = self.transport(url, headers, json.dumps(payload).encode("utf-8"), self.settings.timeout_seconds)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response shape from LLM endpoint") from exc
        return {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
        }


def extract_json(text: str | None) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply (handles ```json fences and chatter)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start: i + 1])
                    break
        start = text.find("{", start + 1) if not candidates else -1
    for c in candidates:
        try:
            value = json.loads(c)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None
