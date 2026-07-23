"""Simulated tool-use environment with zero external dependencies.

Goal: provide a learnable multi-turn task that stresses the tool-calling
loop without requiring any real runtime (terminals, filesystems, MCPs…).

Task template:
    "What is <A> <op> <B>? Answer as: answer=<number>"

Expected policy behavior:
    Turn 1:  emit  <tool_call>calc(A op B)</tool_call>
    Turn 2:  observe <tool_result>VALUE</tool_result>, then emit
             "answer=VALUE"

Reward (∈ [0, 1]):
    0.4 · used_calc_tool
    0.3 · tool_result_correct
    0.3 · final_answer_correct

This reward is dense enough to be learnable by a random tiny policy within
a couple hundred iterations on CPU, and it cleanly separates the two
sub-skills (call the tool + integrate its result).
"""

from __future__ import annotations

import ast
import operator as op
from typing import Any

from hermes_agentic_rl.agent_loop.multi_turn_loop import DEFAULT_TOOL_RESULT_TEMPLATE
from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.rewards.base import BaseReward

SAFE_OPS: dict[type[ast.AST], Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_eval(expr: str) -> str:
    """Safely evaluate a small arithmetic expression. Returns string result.

    Supports: +, -, *, //, unary ±, integer literals, parentheses. Any
    unsupported construct raises ValueError (converted to error:... by the
    caller).
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"bad_syntax:{e}") from e

    def _walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float):
                return node.value
            raise ValueError(f"bad_literal:{type(node.value).__name__}")
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](_walk(node.operand))
        raise ValueError(f"bad_node:{type(node).__name__}")

    val = _walk(tree)
    return str(val)


def calc_tool(name: str, arg: str) -> str:
    """The `calc` tool — used by MultiTurnAgentLoop."""
    del name
    try:
        return safe_eval(arg)
    except Exception as exc:
        return f"error:{type(exc).__name__}:{exc}"


DEFAULT_TOOLS = {"calc": calc_tool}


def build_sim_tool_dataset(n: int = 16, *, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic dataset of tiny arithmetic tasks."""
    import random

    rng = random.Random(seed)
    ops = [("+", op.add), ("-", op.sub), ("*", op.mul)]
    items: list[dict[str, Any]] = []
    for i in range(n):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        sym, func = rng.choice(ops)
        target = str(func(a, b))
        items.append(
            {
                "task_id": f"calc-{i}",
                "instruction": (f"What is {a} {sym} {b}? Use calc tool, then answer as: answer=N"),
                "expr": f"{a} {sym} {b}",
                "target": target,
            }
        )
    return items


class SimToolRewardComponent(BaseReward):
    name = "sim_tool_reward"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del tool_context
        target = str(item.get("target", ""))
        expr = str(item.get("expr", ""))
        # runtime rl metadata is inspected elsewhere; we only use
        # Trajectory.steps below for tool-call accounting.
        _ = trajectory.metadata.get("runtime")
        # walk Trajectory.steps.tool_calls
        used_calc = 0.0
        correct_tool_result = 0.0
        for step in trajectory.steps:
            for call in step.tool_calls:
                if call.get("name") == "calc":
                    used_calc = 1.0
                    arg = str(call.get("arg", "")).strip()
                    if arg.replace(" ", "") == expr.replace(" ", ""):
                        # the arg was literally the task expression → result
                        # will be correct if the tool executed.
                        correct_tool_result = 1.0
            for res in step.tool_results:
                if res.get("name") == "calc" and str(res.get("result", "")).strip() == target:
                    correct_tool_result = 1.0

        final = str(trajectory.final_output or "")
        wanted = f"answer={target}"
        final_correct = 1.0 if wanted in final else 0.0

        score = 0.4 * used_calc + 0.3 * correct_tool_result + 0.3 * final_correct
        reason = (
            f"used_calc={used_calc} tool_ok={correct_tool_result} "
            f"final_ok={final_correct} target={target}"
        )
        return RewardResult(
            name=self.name,
            score=float(score),
            reason=reason,
            weight=self.weight,
            metadata={
                "used_calc": used_calc,
                "correct_tool_result": correct_tool_result,
                "final_correct": final_correct,
                "target": target,
            },
        )


class SimToolEnv(BaseEnv):
    """Round-robin sim-tool env."""

    def __init__(self, dataset: list[dict[str, Any]]) -> None:
        self.dataset = dataset
        self._index = 0
        self._reward = SimToolRewardComponent(weight=1.0)

    async def setup(self) -> None:
        self._index = 0

    async def get_next_item(self) -> dict[str, Any]:
        item = self.dataset[self._index % len(self.dataset)]
        self._index += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        return item["instruction"]

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        return [await self._reward.evaluate(item, trajectory, tool_context)]

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        instruction = str(item.get("instruction", "")).strip()
        expr = str(item.get("expr", "")).strip()
        target = str(item.get("target", "")).strip()
        if not instruction or not expr or not target:
            return []
        tool_call = f"<tool_call>calc({expr})</tool_call>"
        tool_result = DEFAULT_TOOL_RESULT_TEMPLATE.format(result=target)
        return [
            SupervisedSample(
                instruction=instruction,
                response=tool_call,
                metadata={"turn_index": 0, "kind": "tool_call"},
            ),
            SupervisedSample(
                instruction=instruction,
                prompt_suffix=tool_call + tool_result,
                response=f"answer={target}",
                metadata={"turn_index": 1, "kind": "final_answer"},
            ),
        ]
