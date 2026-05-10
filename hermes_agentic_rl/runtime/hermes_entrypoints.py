from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


class HermesEntrypointNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HermesEntrypoint:
    name: str
    module_name: str
    attr_name: str

    def load_attr(self) -> Any:
        module = importlib.import_module(self.module_name)
        return getattr(module, self.attr_name)


_CANDIDATES: list[HermesEntrypoint] = [
    # hermes-agent 官方 Python Library 入口（优先）
    # docs: https://hermes-agent.nousresearch.com/docs/guides/python-library
    HermesEntrypoint(
        name="ai_agent",
        module_name="run_agent",
        attr_name="AIAgent",
    ),
    # hermes-agent environments 子系统的可复用多轮 loop（需要更复杂的 server/tool wiring）
    HermesEntrypoint(
        name="hermes_agent_loop",
        module_name="environments.agent_loop",
        attr_name="HermesAgentLoop",
    ),
]


def find_hermes_entrypoint() -> HermesEntrypoint:
    for candidate in _CANDIDATES:
        try:
            attr = candidate.load_attr()
            if attr is not None:
                return candidate
        except Exception:
            continue
    raise HermesEntrypointNotFoundError("no supported hermes-agent entrypoint found")
