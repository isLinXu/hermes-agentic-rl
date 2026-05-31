"""Production LLM judges for OPD and next-state PRM.

The implementations use the OpenAI-compatible ``/chat/completions`` HTTP
surface via stdlib ``urllib`` so the package does not need an SDK dependency.
They work with OpenAI, local vLLM servers exposing an OpenAI-compatible API,
and most hosted gateways.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.algos.opd import HINT_END, HINT_START, OPDJudge, extract_hint_text
from hermes_agentic_rl.rewards.next_state_prm import NextStateJudge


@dataclass(slots=True)
class OpenAICompatibleJudgeConfig:
    """Config for OpenAI-compatible chat-completion judges."""

    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = "OPENAI_API_KEY"
    api_key_envs: tuple[str, ...] = ()
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: float = 30.0
    retries: int = 2
    extra_headers: dict[str, str] = field(default_factory=dict, repr=False)
    extra_body: dict[str, Any] = field(default_factory=dict, repr=False)


class OpenAICompatibleJudgeClient:
    """Small synchronous HTTP client wrapped by async judge classes."""

    def __init__(self, cfg: OpenAICompatibleJudgeConfig | None = None) -> None:
        self.cfg = cfg or OpenAICompatibleJudgeConfig()

    async def complete(self, messages: list[dict[str, str]]) -> str:
        return await asyncio.to_thread(self.complete_sync, messages)

    def complete_sync(self, messages: list[dict[str, str]]) -> str:
        cfg = self.cfg
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": float(cfg.temperature),
            "max_tokens": int(cfg.max_tokens),
        }
        payload.update(dict(cfg.extra_body))
        data = self._post_json(payload)
        return _extract_chat_content(data)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg
        endpoint = cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            **dict(cfg.extra_headers),
        }
        api_key = _resolve_api_key(cfg)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        encoded = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        attempts = max(1, int(cfg.retries) + 1)
        for attempt in range(attempts):
            req = urllib.request.Request(
                endpoint,
                data=encoded,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=float(cfg.timeout)) as resp:
                    body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise RuntimeError("judge response was not a JSON object")
                return parsed
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(
                    f"judge HTTP {exc.code} from {endpoint}: {body}"
                )
            except Exception as exc:  # pragma: no cover - network timing dependent
                last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2.0, 0.25 * (2 ** attempt)))
        raise RuntimeError(f"judge request failed: {last_error}") from last_error


class OpenAICompatibleOPDJudge(OPDJudge):
    """OPD hint extractor backed by an OpenAI-compatible LLM judge."""

    def __init__(self, cfg: OpenAICompatibleJudgeConfig | None = None) -> None:
        self.client = OpenAICompatibleJudgeClient(cfg)
        super().__init__(self._judge)

    async def _judge(self, response: str, next_state: str) -> str | None:
        content = await self.client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You extract corrective training hints for on-policy "
                        "distillation. If the next-state signal contains useful "
                        "feedback, return exactly one concise hint wrapped as "
                        f"{HINT_START}...{HINT_END}. If no useful hint exists, "
                        "return exactly NO_HINT."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Assistant response:\n"
                        f"{response}\n\nNext-state signal:\n{next_state}"
                    ),
                },
            ]
        )
        if content.strip().upper() == "NO_HINT":
            return None
        return content

    async def extract_hint(self, response: str, next_state: str) -> str | None:
        raw = await self._judge(response, next_state)
        if raw is None:
            return None
        hint = extract_hint_text(raw) or raw.strip()
        if not hint or hint.upper() == "NO_HINT":
            return None
        return hint


class OpenAICompatibleNextStateJudge(NextStateJudge):
    """Next-state GOOD/BAD/NEUTRAL judge backed by an LLM."""

    def __init__(self, cfg: OpenAICompatibleJudgeConfig | None = None) -> None:
        self.client = OpenAICompatibleJudgeClient(cfg)
        super().__init__(self._judge)

    async def _judge(self, response: str, next_state: str) -> str:
        return await self.client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict process-reward judge. Return a vote "
                        "as GOOD, BAD, or NEUTRAL on the first line. If the "
                        "signal suggests a concrete correction, include one "
                        f"{HINT_START}...{HINT_END} block after the vote."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Assistant response:\n"
                        f"{response}\n\nNext-state signal:\n{next_state}"
                    ),
                },
            ]
        )


def _resolve_api_key(cfg: OpenAICompatibleJudgeConfig) -> str | None:
    if cfg.api_key:
        return cfg.api_key
    env_names: list[str] = []
    if cfg.api_key_env:
        env_names.append(cfg.api_key_env)
    env_names.extend(cfg.api_key_envs)
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("judge response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("judge response choice was not an object")
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise RuntimeError("judge response missing message.content")
