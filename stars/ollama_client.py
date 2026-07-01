"""Ollama HTTP client for local Qwen3-8B inference."""

from __future__ import annotations

import os
from typing import Any

import httpx


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.host = (host or os.environ.get("STARS_OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.environ.get("STARS_OLLAMA_MODEL", "qwen3:8b")
        if timeout is None:
            timeout = float(os.environ.get("STARS_OLLAMA_TIMEOUT", "300"))
        self.timeout = timeout

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        # Qwen3 defaults to thinking mode, which is much slower over Ollama.
        if "qwen3" in self.model.lower():
            payload["think"] = False

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.post(f"{self.host}/api/chat", json=payload)
                    r.raise_for_status()
                    data = r.json()
                return str(data.get("message", {}).get("content", "")).strip()
            except httpx.ReadTimeout as exc:
                last_exc = exc
                if attempt >= 2:
                    break
        raise last_exc  # type: ignore[misc]

    def ping(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.host}/api/tags")
                return r.status_code == 200
        except Exception:
            return False
