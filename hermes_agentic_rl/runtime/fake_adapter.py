from __future__ import annotations

from typing import Any

from hermes_agentic_rl.runtime.base import BaseRuntimeAdapter


class _FakeAgentLoop:
    async def run(self, prompt: str) -> dict[str, Any]:
        return {
            "messages": [{"role": "assistant", "content": "done"}],
            "tool_calls": [[{"name": "write_file", "arguments": {"path": "hello.txt"}}]],
            "tool_results": [[{"ok": True, "path": "hello.txt"}]],
            "final_output": "done",
            "finished_naturally": True,
            "turns_used": 1,
            "metadata": {"runtime": "fake", "prompt": prompt},
        }


class FakeRuntimeAdapter(BaseRuntimeAdapter):
    def is_available(self) -> bool:
        return True

    def describe_unavailable_reason(self) -> str:
        return ""

    def build_agent_loop(self, config: dict[str, Any]) -> _FakeAgentLoop:
        return _FakeAgentLoop()
