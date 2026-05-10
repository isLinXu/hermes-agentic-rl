from __future__ import annotations

import uuid
from typing import Any

from hermes_agentic_rl.runtime.errors import RuntimeExecutionError


class HermesLoopWrapper:
    def __init__(self, loop: Any, entrypoint_name: str) -> None:
        self._loop = loop
        self._entrypoint_name = entrypoint_name

    async def run(self, prompt: str) -> dict[str, Any]:
        try:
            result = await self._loop.run(prompt)
        except Exception as exc:
            raise RuntimeExecutionError(f"hermes loop execution failed: {exc}") from exc

        return {
            "messages": self._get_value(result, "messages", []),
            "tool_calls": self._get_value(result, "tool_calls", []),
            "tool_results": self._get_value(result, "tool_results", []),
            "final_output": self._get_value(result, "final_output"),
            "finished_naturally": bool(
                self._get_value(result, "finished_naturally", True)
            ),
            "turns_used": int(self._get_value(result, "turns_used", 0)),
            "metadata": {
                "runtime": "hermes",
                "entrypoint": self._entrypoint_name,
                "prompt": prompt,
            },
        }

    @staticmethod
    def _get_value(result: Any, key: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)


def _extract_tool_calls_and_results(messages: list[dict[str, Any]]) -> tuple[list[list[dict]], list[list[dict]]]:
    """
    从 OpenAI 风格 messages 中提取每一轮的 tool_calls 与对应 tool results。

    约定：
    - assistant 消息包含 "tool_calls" 时，视为一次“调用轮次”
    - 紧随其后的 role=tool 消息（直到下一个 assistant 消息）视为本轮结果
    """
    tool_calls_per_turn: list[list[dict]] = []
    tool_results_per_turn: list[list[dict]] = []

    current_results: list[dict] | None = None
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            tool_calls_per_turn.append(list(msg.get("tool_calls") or []))
            current_results = []
            tool_results_per_turn.append(current_results)
            continue

        if role == "tool" and current_results is not None:
            # tool 消息可以包含 content / name / tool_call_id 等字段
            current_results.append(dict(msg))

    return tool_calls_per_turn, tool_results_per_turn


class HermesAIAgentWrapper:
    """
    将 `run_agent.AIAgent` 包装为我们内部统一的 `run(prompt) -> protocol dict`。
    """

    def __init__(self, agent: Any, entrypoint_name: str, system_message: str | None = None) -> None:
        self._agent = agent
        self._entrypoint_name = entrypoint_name
        self._system_message = system_message

    async def run(self, prompt: str) -> dict[str, Any]:
        try:
            # AIAgent.run_conversation 是同步方法（内部用 asyncio），这里保持兼容：
            # 若未来变为 async，此处仍可正常 await（因为我们会在 except 分支捕获 TypeError）。
            task_id = f"hermes-agentic-rl-{uuid.uuid4().hex[:8]}"
            result = self._agent.run_conversation(
                user_message=prompt,
                task_id=task_id,
                system_message=self._system_message,
            )
        except TypeError:
            # 某些版本可能不接受 system_message/task_id 组合，降级为最小调用
            try:
                result = self._agent.run_conversation(user_message=prompt)
            except Exception as exc:
                raise RuntimeExecutionError(f"hermes AIAgent.run_conversation failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeExecutionError(f"hermes AIAgent.run_conversation failed: {exc}") from exc

        if not isinstance(result, dict):
            raise RuntimeExecutionError("hermes AIAgent returned non-dict result")

        messages = list(result.get("messages") or [])
        tool_calls, tool_results = _extract_tool_calls_and_results(messages)

        return {
            "messages": messages,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "final_output": result.get("final_response") or result.get("final_output"),
            "finished_naturally": True,
            "turns_used": int(result.get("turns_used", 0)) if result.get("turns_used") is not None else 0,
            "metadata": {
                "runtime": "hermes",
                "entrypoint": self._entrypoint_name,
                "prompt": prompt,
                "task_id": result.get("task_id"),
            },
        }
