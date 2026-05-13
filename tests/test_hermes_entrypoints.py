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
    def _missing_import(name: str, *args, **kwargs):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(
        "hermes_agentic_rl.runtime.hermes_entrypoints.importlib.import_module",
        _missing_import,
    )

    with pytest.raises(HermesEntrypointNotFoundError):
        find_hermes_entrypoint()
