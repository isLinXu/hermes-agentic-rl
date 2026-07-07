"""MCP tool environment adapter — bridges MCP tool protocol into the RL pipeline.

This module adapts the Model Context Protocol (MCP) tool-calling convention
to the existing ``ToolFn`` interface used by ``MultiTurnAgentLoop``. It
allows RL training with real external tools (filesystem, web search, code
execution, etc.) without changing the training loop.

Key components:

1. **MCPToolSpec** — declarative tool specification (name, description,
   JSON schema for parameters).

2. **MCPToolRegistry** — a collection of named tool specs + their Python
   callables. Acts as the ``tools`` dict for ``MultiTurnAgentLoop``.

3. **MCPToolEnv** — a ``BaseEnv`` that generates tasks requiring MCP tool
   usage, similar to ``SimToolEnv`` but with configurable external tools.

4. **MCPToolReward** — reward component that scores whether the correct
   MCP tool was called with correct arguments and produced the expected
   result.

Design rationale:
  - ART uses server-side MCP for tool execution. We keep the same tool
    protocol (⟦NAME(arg)⟧) but allow tools to be either local Python
    functions or remote MCP server calls.
  - The adapter handles argument serialization: the RL loop produces
    string args (from the text response); the MCP adapter parses them
    as JSON when the tool spec declares a schema, or passes them raw.
  - Tool execution failures are captured and returned as error strings
    (same pattern as ``calc_tool``), so the policy can learn to recover.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.agent_loop.multi_turn_loop import (
    ToolFn,
)
from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.rewards.base import BaseReward

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MCPToolSpec:
    """Declarative specification for a single MCP tool.

    Attributes
    ----------
    name : str
        Tool name as it appears in the tool-call protocol.
    description : str
        Human-readable description (shown to the policy via prompt).
    handler : Callable[[str], str]
        The actual function: takes the raw arg string, returns result string.
    param_schema : dict[str, Any] | None
        JSON schema for parameters. If None, the arg is passed as a raw string.
    """

    name: str
    description: str
    handler: Callable[[str], str]
    param_schema: dict[str, Any] | None = None

    def to_prompt_description(self) -> str:
        """Generate a prompt-friendly description for the policy."""
        schema_str = ""
        if self.param_schema:
            schema_str = f" (args: {json.dumps(self.param_schema)})"
        return f"- {self.name}: {self.description}{schema_str}"


class MCPToolRegistry:
    """Registry of MCP tools, compatible with ``MultiTurnAgentLoop``.

    Acts as both:
      - A ``Mapping[str, ToolFn]`` for the agent loop.
      - A source of tool specs for prompt generation.
    """

    def __init__(self) -> None:
        self._specs: dict[str, MCPToolSpec] = {}

    def register(self, spec: MCPToolSpec) -> None:
        """Register a tool specification."""
        if spec.name in self._specs:
            raise ValueError(f"Tool '{spec.name}' is already registered")
        self._specs[spec.name] = spec

    def register_simple(
        self,
        name: str,
        description: str,
        handler: Callable[[str], str],
        *,
        param_schema: dict[str, Any] | None = None,
    ) -> None:
        """Convenience: register without creating MCPToolSpec explicitly."""
        self.register(MCPToolSpec(
            name=name,
            description=description,
            param_schema=param_schema,
            handler=handler,
        ))

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> ToolFn:
        """Return a ToolFn compatible with MultiTurnAgentLoop."""
        if name not in self._specs:
            raise KeyError(f"Unknown tool: {name}")
        spec = self._specs[name]

        def _tool_fn(tool_name: str, arg: str) -> str:
            del tool_name  # already known from spec
            try:
                # If schema exists, try to parse arg as JSON.
                if spec.param_schema and arg.strip().startswith("{"):
                    parsed = json.loads(arg)
                    # For now, pass the parsed dict as JSON string to handler.
                    # Handlers can decide how to interpret.
                    return spec.handler(json.dumps(parsed))
                return spec.handler(arg)
            except Exception as exc:
                return f"error:{type(exc).__name__}:{exc}"

        return _tool_fn

    def keys(self) -> list[str]:
        return list(self._specs.keys())

    def to_tool_mapping(self) -> dict[str, ToolFn]:
        """Convert to a plain dict for MultiTurnAgentLoop."""
        return {name: self[name] for name in self._specs}

    def to_prompt_descriptions(self) -> str:
        """Generate a multi-line description of all available tools."""
        return "\n".join(spec.to_prompt_description() for spec in self._specs.values())

    @property
    def num_tools(self) -> int:
        return len(self._specs)

    def get_spec(self, name: str) -> MCPToolSpec | None:
        return self._specs.get(name)


@dataclass(slots=True)
class MCPToolTask:
    """A single MCP tool-use task for training."""

    task_id: str
    instruction: str
    expected_tool: str
    expected_arg: str | None
    expected_result: str | None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


class MCPToolEnv(BaseEnv):
    """Environment that trains the policy to use MCP tools correctly.

    Parameters
    ----------
    registry : MCPToolRegistry
        The available tools.
    tasks : list[MCPToolTask]
        Task instances for training.
    max_turns : int
        Maximum agent turns per episode.
    """

    def __init__(
        self,
        registry: MCPToolRegistry,
        tasks: list[MCPToolTask],
        *,
        max_turns: int = 3,
    ) -> None:
        self.registry = registry
        self.tasks = tasks
        self.max_turns = max_turns
        self._task_index = 0

    async def setup(self) -> None:
        pass

    async def get_next_item(self) -> dict[str, Any]:
        task = self.tasks[self._task_index % len(self.tasks)]
        self._task_index += 1
        return {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "expected_tool": task.expected_tool,
            "expected_arg": task.expected_arg,
            "expected_result": task.expected_result,
            "tools_description": self.registry.to_prompt_descriptions(),
        }

    def format_prompt(self, item: dict[str, Any]) -> str:
        tools_desc = item.get("tools_description", "")
        instruction = item["instruction"]
        return (
            f"Available tools:\n{tools_desc}\n\n"
            f"Task: {instruction}\n"
            f"Call the appropriate tool to complete the task."
        )

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        expected_tool = item.get("expected_tool", "")
        expected_arg = item.get("expected_arg")
        expected_result = item.get("expected_result")

        # Extract tool calls from trajectory.
        tool_calls = []
        tool_results = []
        for step in trajectory.steps:
            tool_calls.extend(step.tool_calls)
            tool_results.extend(step.tool_results)

        # Score components.
        results: list[RewardResult] = []

        # 1. Did the policy call the expected tool?
        called_tool = any(tc.get("name") == expected_tool for tc in tool_calls)
        results.append(RewardResult(
            name="tool_called",
            score=1.0 if called_tool else 0.0,
            reason=(
                f"Called '{expected_tool}'"
                if called_tool
                else f"Did not call '{expected_tool}'"
            ),
            weight=0.4,
        ))

        # 2. Was the argument correct?
        if expected_arg is not None and called_tool:
            arg_correct = any(
                tc.get("arg") == expected_arg
                for tc in tool_calls
                if tc.get("name") == expected_tool
            )
            results.append(RewardResult(
                name="arg_correct",
                score=1.0 if arg_correct else 0.0,
                reason="Correct arg" if arg_correct else "Wrong arg",
                weight=0.2,
            ))
        elif called_tool:
            results.append(RewardResult(
                name="arg_correct",
                score=0.5,  # partial credit if no expected arg defined
                reason="No expected arg to check",
                weight=0.2,
            ))
        else:
            results.append(RewardResult(
                name="arg_correct",
                score=0.0,
                reason="Tool not called",
                weight=0.2,
            ))

        # 3. Was the result correct?
        if expected_result is not None:
            result_correct = any(
                expected_result in str(tr.get("result", ""))
                for tr in tool_results
            )
            results.append(RewardResult(
                name="result_correct",
                score=1.0 if result_correct else 0.0,
                reason="Correct result" if result_correct else "Wrong result",
                weight=0.4,
            ))
        else:
            results.append(RewardResult(
                name="result_correct",
                score=0.0,
                reason="No expected result",
                weight=0.4,
            ))

        return results

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        """Optional SFT sample showing the correct tool call."""
        expected_tool = item.get("expected_tool", "")
        expected_arg = item.get("expected_arg", "")
        if expected_arg:
            response = f"<tool_call>{expected_tool}({expected_arg})⟩"
        else:
            response = f"<tool_call>{expected_tool}()⟩"
        return [SupervisedSample(
            instruction=self.format_prompt(item),
            response=response,
        )]


class MCPToolReward(BaseReward):
    """Reward component for MCP tool usage.

    Wraps the reward logic from ``MCPToolEnv.compute_reward`` into a
    standalone ``BaseReward`` for use with the reward manager.
    """

    name = "mcp_tool"

    def __init__(self, weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any = None,
    ) -> RewardResult:
        expected_tool = item.get("expected_tool", "")
        expected_arg = item.get("expected_arg")
        expected_result = item.get("expected_result")

        tool_calls = []
        tool_results = []
        for step in trajectory.steps:
            tool_calls.extend(step.tool_calls)
            tool_results.extend(step.tool_results)

        called_tool = any(tc.get("name") == expected_tool for tc in tool_calls)

        score = 0.0
        reason_parts: list[str] = []

        # Tool called (0.4)
        if called_tool:
            score += 0.4
            reason_parts.append("tool_called")
        else:
            reason_parts.append("tool_not_called")

        # Arg correct (0.2)
        if called_tool and expected_arg is not None:
            arg_correct = any(
                tc.get("arg") == expected_arg
                for tc in tool_calls
                if tc.get("name") == expected_tool
            )
            if arg_correct:
                score += 0.2
                reason_parts.append("arg_correct")
            else:
                reason_parts.append("arg_wrong")

        # Result correct (0.4)
        if called_tool and expected_result is not None:
            result_correct = any(
                expected_result in str(tr.get("result", ""))
                for tr in tool_results
            )
            if result_correct:
                score += 0.4
                reason_parts.append("result_correct")
            else:
                reason_parts.append("result_wrong")

        return RewardResult(
            name="mcp_tool",
            score=score,
            reason=", ".join(reason_parts),
            weight=self._weight,
        )


def build_mcp_tool_dataset(
    registry: MCPToolRegistry,
    n: int = 16,
    *,
    seed: int = 0,
) -> list[MCPToolTask]:
    """Generate a simple dataset of tool-use tasks from the registry.

    For each tool, creates tasks that ask the policy to call it with
    known arguments. This is a basic template — real datasets should
    provide domain-specific tasks.
    """
    import random

    rng = random.Random(seed)
    tasks: list[MCPToolTask] = []

    tool_names = registry.keys()
    if not tool_names:
        return tasks

    for i in range(n):
        tool_name = rng.choice(tool_names)
        spec = registry.get_spec(tool_name)
        assert spec is not None

        # Generate a simple arg.
        arg = f"input_{i}"
        # Compute expected result by calling the handler directly.
        try:
            expected = spec.handler(arg)
        except Exception:
            expected = None

        tasks.append(MCPToolTask(
            task_id=f"mcp-{tool_name}-{i}",
            instruction=f"Use the '{tool_name}' tool with argument '{arg}'.",
            expected_tool=tool_name,
            expected_arg=arg,
            expected_result=expected,
        ))

    return tasks
