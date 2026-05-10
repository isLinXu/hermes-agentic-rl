import types

import pytest

from hermes_agentic_rl.runtime.hermes_entrypoints import (
    HermesEntrypointNotFoundError,
    find_hermes_entrypoint,
)


def test_find_hermes_entrypoint_returns_first_available(monkeypatch):
    mod = types.ModuleType("run_agent")
    mod.AIAgent = object  # sentinel
    monkeypatch.setitem(__import__("sys").modules, "run_agent", mod)

    entrypoint = find_hermes_entrypoint()

    assert entrypoint.name == "ai_agent"
    assert entrypoint.module_name == "run_agent"
    assert entrypoint.attr_name == "AIAgent"


def test_find_hermes_entrypoint_raises_when_all_missing(monkeypatch):
    # 确保候选模块都不存在
    for name in [
        "run_agent",
        "environments.agent_loop",
    ]:
        monkeypatch.delitem(__import__("sys").modules, name, raising=False)

    with pytest.raises(HermesEntrypointNotFoundError):
        find_hermes_entrypoint()
