import asyncio
import importlib
import types
from dataclasses import dataclass

from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter
from hermes_agentic_rl.runtime.hermes_wrapper import HermesLoopWrapper


@dataclass
class FakeHermesResult:
    messages: list[dict]
    tool_calls: list[list[dict]]
    tool_results: list[list[dict]]
    final_output: str
    finished_naturally: bool
    turns_used: int


class FakeHermesLoop:
    async def run(self, prompt: str) -> FakeHermesResult:
        return FakeHermesResult(
            messages=[{"role": "assistant", "content": "done"}],
            tool_calls=[[{"name": "write_file", "arguments": {"path": "x.txt"}}]],
            tool_results=[[{"ok": True}]],
            final_output="done",
            finished_naturally=True,
            turns_used=1,
        )


def test_wrapper_normalizes_to_protocol():
    wrapper = HermesLoopWrapper(loop=FakeHermesLoop(), entrypoint_name="test")
    payload = asyncio.run(wrapper.run("Create x.txt"))

    assert payload["final_output"] == "done"
    assert payload["tool_calls"][0][0]["name"] == "write_file"
    assert payload["metadata"]["runtime"] == "hermes"
    assert payload["metadata"]["entrypoint"] == "test"


def test_hermes_adapter_builds_ai_agent_wrapper_when_run_agent_is_injected(monkeypatch):
    """
    通过注入式 run_agent.AIAgent 模拟“已安装 pip 包形态”的真实入口。
    不依赖实际安装 hermes-agent（因为 hermes-agent 要求 Python>=3.11）。
    """
    run_agent_mod = types.ModuleType("run_agent")

    class FakeAIAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_conversation(
            self,
            user_message: str,
            task_id: str | None = None,
            system_message: str | None = None,
        ):
            saved_task_id = task_id
            del system_message
            return {
                "final_response": "done",
                "messages": [
                    {"role": "user", "content": user_message},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "write_file", "arguments": {"path": "x.txt"}}],
                    },
                    {"role": "tool", "name": "write_file", "content": "{\"ok\": true}"},
                    {"role": "assistant", "content": "done"},
                ],
                "task_id": saved_task_id,
            }

    run_agent_mod.AIAgent = FakeAIAgent
    monkeypatch.setitem(importlib.sys.modules, "run_agent", run_agent_mod)

    adapter = HermesRuntimeAdapter(importer=lambda: run_agent_mod)
    loop = adapter.build_agent_loop(
        {
            "runtime": {
                "integration": "hermes",
                "model": "demo",
                "provider": "openrouter",
                "base_url": "http://example.invalid",
                "api_key": "k-demo",
            }
        }
    )
    payload = asyncio.run(loop.run("Create x.txt"))

    assert payload["metadata"]["runtime"] == "hermes"
    assert payload["final_output"] == "done"
    assert payload["tool_calls"][0][0]["name"] == "write_file"
    assert payload["tool_calls"][0][0]["name"] == "write_file"
