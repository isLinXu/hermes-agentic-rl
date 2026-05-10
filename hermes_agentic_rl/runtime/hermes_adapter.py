from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

from hermes_agentic_rl.runtime.base import BaseRuntimeAdapter
from hermes_agentic_rl.runtime.errors import (
    RuntimeConfigurationError,
    RuntimeUnavailableError,
)
from hermes_agentic_rl.runtime.hermes_entrypoints import (
    HermesEntrypointNotFoundError,
    find_hermes_entrypoint,
)
from hermes_agentic_rl.runtime.hermes_wrapper import HermesAIAgentWrapper, HermesLoopWrapper


def _default_importer() -> Any:
    # hermes-agent 的官方 Python Library 入口是顶层模块 run_agent.py
    # https://hermes-agent.nousresearch.com/docs/guides/python-library
    return importlib.import_module("run_agent")


class HermesRuntimeAdapter(BaseRuntimeAdapter):
    def __init__(self, importer: Callable[[], Any] | None = None) -> None:
        self._importer = importer or _default_importer
        self._cached_module: Any | None = None
        self._error_message = ""

    def _load(self) -> Any | None:
        if self._cached_module is not None:
            return self._cached_module

        try:
            self._cached_module = self._importer()
            self._error_message = ""
            return self._cached_module
        except ImportError as exc:
            self._error_message = str(exc)
            return None

    def is_available(self) -> bool:
        return self._load() is not None

    def describe_unavailable_reason(self) -> str:
        self._load()
        return self._error_message or "hermes-agent runtime is unavailable"

    def build_agent_loop(self, config: dict[str, Any]) -> Any:
        module = self._load()
        if module is None:
            raise RuntimeUnavailableError(self.describe_unavailable_reason())

        try:
            entrypoint = find_hermes_entrypoint()
        except HermesEntrypointNotFoundError as exc:
            raise RuntimeUnavailableError(str(exc)) from exc

        attr = entrypoint.load_attr()

        if entrypoint.name == "ai_agent":
            # AIAgent 作为最稳定、最面向用户的入口。它会在内部完成 tool-calling loop。
            try:
                # 通过 kwargs 保持前后兼容：不同版本 AIAgent.__init__ 参数较多
                runtime_cfg = (config or {}).get("runtime", {})
                model = runtime_cfg.get("model")
                provider = runtime_cfg.get("provider")
                base_url = runtime_cfg.get("base_url")
                api_key = runtime_cfg.get("api_key")
                api_key_env = runtime_cfg.get("api_key_env")
                if not api_key and isinstance(api_key_env, str) and api_key_env:
                    api_key = os.getenv(api_key_env)
                enabled_toolsets = runtime_cfg.get("enabled_toolsets")
                max_iterations = runtime_cfg.get("max_agent_turns")
                system_message = (config or {}).get("environment", {}).get("system_prompt")

                agent = attr(
                    base_url=base_url,
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    enabled_toolsets=enabled_toolsets,
                    quiet_mode=True,
                    skip_context_files=True,
                    skip_memory=True,
                    max_iterations=max_iterations,
                )
            except Exception as exc:
                raise RuntimeConfigurationError(f"failed to build AIAgent: {exc}") from exc

            return HermesAIAgentWrapper(agent=agent, entrypoint_name=entrypoint.name, system_message=system_message)

        # 其他入口：先以最小可用形态包装（后续如需要可再扩展 server/tool wiring）
        try:
            loop = attr() if callable(attr) else attr
        except Exception as exc:
            raise RuntimeConfigurationError(f"failed to build hermes loop: {exc}") from exc

        return HermesLoopWrapper(loop=loop, entrypoint_name=entrypoint.name)
