from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hermes_agentic_rl.collectors.sidecar import build_sidecar_from_runtime_config
from hermes_agentic_rl.integrations.hermes_repo import prepare_hermes_imports
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


def _clear_cached_hermes_modules(repo_path: Path | None) -> None:
    if repo_path is None:
        return

    module = sys.modules.get("run_agent")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return

    try:
        if Path(module_file).resolve().is_relative_to(repo_path):
            return
    except Exception:
        return

    sys.modules.pop("run_agent", None)
    for name in list(sys.modules):
        if name == "environments" or name.startswith("environments."):
            sys.modules.pop(name, None)


def _resolve_api_key(runtime_cfg: dict[str, Any]) -> str | None:
    api_key = runtime_cfg.get("api_key")
    if isinstance(api_key, str) and api_key:
        return api_key

    env_names: list[str] = []
    raw_envs = runtime_cfg.get("api_key_envs")
    if isinstance(raw_envs, list):
        env_names.extend(str(item).strip() for item in raw_envs if str(item).strip())
    api_key_env = runtime_cfg.get("api_key_env")
    if isinstance(api_key_env, str) and api_key_env.strip():
        env_names.append(api_key_env.strip())

    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value
    return None


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
        runtime_cfg = (config or {}).get("runtime", {})
        repo_resolution = prepare_hermes_imports(runtime_cfg)
        if repo_resolution.repo_path is not None:
            if not repo_resolution.repo_path.exists():
                raise RuntimeUnavailableError(
                    f"configured hermes repo path does not exist: {repo_resolution.repo_path}"
                )
            if not repo_resolution.repo_path.is_dir():
                raise RuntimeUnavailableError(
                    f"configured hermes repo path is not a directory: {repo_resolution.repo_path}"
                )
            if not repo_resolution.has_run_agent:
                raise RuntimeUnavailableError(
                    f"hermes repo path is missing run_agent.py: {repo_resolution.repo_path}"
                )
            _clear_cached_hermes_modules(repo_resolution.repo_path)

        module = self._load()
        if module is None:
            reason = self.describe_unavailable_reason()
            if repo_resolution.repo_path is not None:
                reason = f"{reason} (hermes repo: {repo_resolution.repo_path})"
            raise RuntimeUnavailableError(reason)

        try:
            entrypoint = find_hermes_entrypoint(module=module)
        except HermesEntrypointNotFoundError as exc:
            raise RuntimeUnavailableError(str(exc)) from exc

        attr = entrypoint.load_attr()

        if entrypoint.name == "ai_agent":
            # AIAgent 作为最稳定、最面向用户的入口。它会在内部完成 tool-calling loop。
            try:
                # 通过 kwargs 保持前后兼容：不同版本 AIAgent.__init__ 参数较多
                model = runtime_cfg.get("model")
                provider = runtime_cfg.get("provider")
                base_url = runtime_cfg.get("base_url")
                api_key = _resolve_api_key(runtime_cfg)
                enabled_toolsets = runtime_cfg.get("enabled_toolsets")
                max_iterations = runtime_cfg.get("max_agent_turns")
                session_log_path = runtime_cfg.get("session_log_path")
                session_sidecar = build_sidecar_from_runtime_config(runtime_cfg)
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

            return HermesAIAgentWrapper(
                agent=agent,
                entrypoint_name=entrypoint.name,
                system_message=system_message,
                session_log_path=session_log_path,
                session_sidecar=session_sidecar,
            )

        # 其他入口：先以最小可用形态包装（后续如需要可再扩展 server/tool wiring）
        try:
            session_log_path = runtime_cfg.get("session_log_path")
            session_sidecar = build_sidecar_from_runtime_config(runtime_cfg)
            loop = attr() if callable(attr) else attr
        except Exception as exc:
            raise RuntimeConfigurationError(f"failed to build hermes loop: {exc}") from exc

        return HermesLoopWrapper(
            loop=loop,
            entrypoint_name=entrypoint.name,
            session_log_path=session_log_path,
            session_sidecar=session_sidecar,
        )
