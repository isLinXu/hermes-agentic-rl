import asyncio
import importlib
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from hermes_agentic_rl.collectors.sidecar import close_all_sidecars
from hermes_agentic_rl.offline.replay_buffer import ReplayBuffer
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


def test_wrapper_can_append_session_logs(tmp_path: Path):
    session_log_path = tmp_path / "sessions.jsonl"
    wrapper = HermesLoopWrapper(
        loop=FakeHermesLoop(),
        entrypoint_name="test",
        session_log_path=str(session_log_path),
    )

    payload = asyncio.run(wrapper.run("Create x.txt"))

    assert session_log_path.exists()
    lines = [
        line for line in session_log_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["final_output"] == "done"
    assert payload["metadata"]["session_log_path"].endswith("sessions.jsonl")


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
                    {"role": "tool", "name": "write_file", "content": '{"ok": true}'},
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


def test_hermes_adapter_resolves_api_key_from_env_list(monkeypatch):
    run_agent_mod = types.ModuleType("run_agent")

    class FakeAIAgent:
        instances: ClassVar[list["FakeAIAgent"]] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.instances.append(self)

        def run_conversation(self, user_message: str, **_kwargs):
            return {
                "final_response": "done",
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "done"},
                ],
            }

    run_agent_mod.AIAgent = FakeAIAgent
    monkeypatch.setitem(importlib.sys.modules, "run_agent", run_agent_mod)
    monkeypatch.delenv("MISSING_HERMES_KEY", raising=False)
    monkeypatch.setenv("HERMES_TEST_API_KEY", "resolved-key")

    adapter = HermesRuntimeAdapter(importer=lambda: run_agent_mod)
    adapter.build_agent_loop(
        {
            "runtime": {
                "integration": "hermes",
                "model": "demo",
                "provider": "openai",
                "base_url": "http://example.invalid/v1",
                "api_key_envs": ["MISSING_HERMES_KEY", "HERMES_TEST_API_KEY"],
            }
        }
    )

    assert FakeAIAgent.instances[-1].kwargs["api_key"] == "resolved-key"


def test_hermes_adapter_passes_session_log_path_to_wrapper(tmp_path: Path, monkeypatch):
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
            del system_message
            return {
                "final_response": "done",
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "done"},
                ],
                "task_id": task_id,
            }

    run_agent_mod.AIAgent = FakeAIAgent
    monkeypatch.setitem(importlib.sys.modules, "run_agent", run_agent_mod)

    session_log_path = tmp_path / "adapter_sessions.jsonl"
    adapter = HermesRuntimeAdapter(importer=lambda: run_agent_mod)
    loop = adapter.build_agent_loop(
        {
            "runtime": {
                "integration": "hermes",
                "model": "demo",
                "provider": "openrouter",
                "base_url": "http://example.invalid",
                "api_key": "k-demo",
                "session_log_path": str(session_log_path),
            }
        }
    )

    payload = asyncio.run(loop.run("Create x.txt"))

    assert session_log_path.exists()
    record = json.loads(session_log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["metadata"]["task_id"] is not None
    assert payload["metadata"]["session_log_path"].endswith("adapter_sessions.jsonl")


def test_hermes_adapter_sidecar_writes_replay_output(tmp_path: Path, monkeypatch):
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
            del system_message
            return {
                "final_response": "done",
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "great, thanks"},
                ],
                "task_id": task_id,
            }

    run_agent_mod.AIAgent = FakeAIAgent
    monkeypatch.setitem(importlib.sys.modules, "run_agent", run_agent_mod)

    session_log_path = tmp_path / "sidecar_sessions.jsonl"
    replay_output_path = tmp_path / "sidecar_replay.jsonl"
    adapter = HermesRuntimeAdapter(importer=lambda: run_agent_mod)
    loop = adapter.build_agent_loop(
        {
            "runtime": {
                "integration": "hermes",
                "model": "demo",
                "provider": "openrouter",
                "base_url": "http://example.invalid",
                "api_key": "k-demo",
                "session_sidecar": {
                    "enabled": True,
                    "session_log_path": str(session_log_path),
                    "replay_output_path": str(replay_output_path),
                    "backend": {"name": "tiny"},
                    "flush_interval_sec": 0.01,
                    "max_batch_size": 1,
                },
            }
        }
    )

    try:
        payload = asyncio.run(loop.run("Create x.txt"))
    finally:
        close_all_sidecars()

    assert payload["metadata"]["session_sidecar"]["accepted"] is True
    assert payload["metadata"]["session_sidecar"]["stats"]["worker_alive"] is True
    assert "queue_size" in payload["metadata"]["session_sidecar"]["stats"]
    assert session_log_path.exists()
    assert replay_output_path.exists()
    buffer = ReplayBuffer.load_jsonl(replay_output_path)
    assert len(buffer.samples) == 1
    assert buffer.samples[0].reward > 0
