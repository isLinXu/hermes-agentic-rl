import asyncio

import pytest

from hermes_agentic_rl.runtime.base import BaseRuntimeAdapter
from hermes_agentic_rl.runtime.errors import RuntimeUnavailableError
from hermes_agentic_rl.runtime.fake_adapter import FakeRuntimeAdapter
from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter


def test_runtime_unavailable_error_string():
    error = RuntimeUnavailableError("hermes-agent is not installed")

    assert "hermes-agent is not installed" in str(error)


def test_base_runtime_adapter_subclass_contract():
    class DemoAdapter(BaseRuntimeAdapter):
        def is_available(self) -> bool:
            return True

        def describe_unavailable_reason(self) -> str:
            return ""

        def build_agent_loop(self, config: dict):
            return object()

    adapter = DemoAdapter()

    assert adapter.is_available() is True
    assert adapter.build_agent_loop({}) is not None


def test_fake_runtime_adapter_builds_protocol_compatible_loop():
    adapter = FakeRuntimeAdapter()
    loop = adapter.build_agent_loop({"runtime": {"model": "demo"}})

    payload = asyncio.run(loop.run("Create hello.txt"))

    assert payload["final_output"] == "done"
    assert payload["tool_calls"][0][0]["name"] == "write_file"
    assert payload["turns_used"] == 1


def test_hermes_runtime_adapter_reports_unavailable_when_dependency_missing():
    adapter = HermesRuntimeAdapter(
        importer=lambda: (_ for _ in ()).throw(ImportError("missing hermes"))
    )

    assert adapter.is_available() is False
    assert "missing hermes" in adapter.describe_unavailable_reason()

    with pytest.raises(RuntimeUnavailableError):
        adapter.build_agent_loop({"runtime": {"integration": "hermes"}})
