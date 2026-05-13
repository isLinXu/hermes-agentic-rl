# Hermes-Agentic-RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 `hermes-agentic-rl` 的第一阶段最小可运行框架，打通 `Trajectory → Reward → Trainer Export` 的训练闭环。

**Architecture:** 采用 `core` 作为统一编排层，`envs` 负责任务抽象，`rewards` 负责结构化评分，`trainers` 负责导出 Atropos/GRPO 可消费样本。第一阶段只实现 `TerminalTaskEnv + OutcomeReward + ToolcallReward + JSONL/Atropos 导出`，其余组件仅留出清晰扩展边界。

**Tech Stack:** Python 3.11+、`dataclasses`、`pathlib`、`typing`、`yaml`、`pytest`

---

## 文件结构

本计划会创建以下文件：

- `pyproject.toml`
- `README.md`
- `hermes_agentic_rl/__init__.py`
- `hermes_agentic_rl/config.py`
- `hermes_agentic_rl/core/types.py`
- `hermes_agentic_rl/core/trajectory.py`
- `hermes_agentic_rl/core/registry.py`
- `hermes_agentic_rl/core/reward_manager.py`
- `hermes_agentic_rl/core/trainer_bridge.py`
- `hermes_agentic_rl/core/rollout_manager.py`
- `hermes_agentic_rl/envs/base_env.py`
- `hermes_agentic_rl/envs/terminal_task_env.py`
- `hermes_agentic_rl/rewards/base.py`
- `hermes_agentic_rl/rewards/outcome_reward.py`
- `hermes_agentic_rl/rewards/toolcall_reward.py`
- `hermes_agentic_rl/rewards/aggregate.py`
- `hermes_agentic_rl/trainers/base.py`
- `hermes_agentic_rl/trainers/atropos_grpo.py`
- `hermes_agentic_rl/cli/main.py`
- `configs/terminal_grpo.yaml`
- `examples/minimal_terminal_task.py`
- `tests/test_types_and_trajectory.py`
- `tests/test_registry.py`
- `tests/test_reward_aggregate.py`
- `tests/test_toolcall_reward.py`
- `tests/test_reward_manager.py`
- `tests/test_trainer_export.py`
- `tests/test_cli_smoke.py`

本计划会修改以下文件：

- `README.md`

---

### Task 1: 初始化项目打包与目录骨架

**Files:**
- Create: `pyproject.toml`
- Create: `hermes_agentic_rl/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: 写一个最小项目元数据测试草案**

在 `tests/test_cli_smoke.py` 里先写一个会失败的测试，验证包版本与 CLI 入口模块可导入：

```python
from hermes_agentic_rl import __version__
from hermes_agentic_rl.cli.main import build_parser


def test_package_version_and_cli_parser():
    assert __version__ == "0.1.0"
    parser = build_parser()
    assert parser.prog == "hermes-agentic-rl"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cli_smoke.py::test_package_version_and_cli_parser -v`
Expected: FAIL，提示 `ModuleNotFoundError: No module named 'hermes_agentic_rl'`

- [ ] **Step 3: 创建最小包骨架**

创建 `pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hermes-agentic-rl"
version = "0.1.0"
description = "A minimal agentic RL framework for hermes-agent"
readme = "README.md"
requires-python = ">=3.11"
dependencies = ["PyYAML>=6.0"]

[project.scripts]
hermes-agentic-rl = "hermes_agentic_rl.cli.main:main"

[tool.setuptools.packages.find]
include = ["hermes_agentic_rl*"]
```

创建 `hermes_agentic_rl/__init__.py`：

```python
__all__ = ["__version__"]

__version__ = "0.1.0"
```

将 `README.md` 改为：

```md
# hermes-agentic-rl

最小可运行的 Hermes Agent 原生 agentic RL 框架。
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cli_smoke.py::test_package_version_and_cli_parser -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml README.md hermes_agentic_rl/__init__.py tests/test_cli_smoke.py
git commit -m "chore: bootstrap hermes-agentic-rl package"
```

### Task 2: 实现核心类型与轨迹序列化

**Files:**
- Create: `hermes_agentic_rl/core/types.py`
- Create: `hermes_agentic_rl/core/trajectory.py`
- Create: `tests/test_types_and_trajectory.py`

- [ ] **Step 1: 写失败测试，覆盖类型构造与序列化**

在 `tests/test_types_and_trajectory.py` 写：

```python
from hermes_agentic_rl.core.trajectory import trajectory_from_dict, trajectory_to_dict
from hermes_agentic_rl.core.types import RewardResult, RolloutStep, Trajectory


def test_trajectory_roundtrip():
    trajectory = Trajectory(
        task_id="task-1",
        prompt="create file",
        steps=[
            RolloutStep(
                turn_index=0,
                assistant_message="I will create the file",
                tool_calls=[{"name": "write_file", "arguments": {"path": "a.txt"}}],
                tool_results=[{"ok": True}],
            )
        ],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
        metadata={"source": "test"},
    )

    payload = trajectory_to_dict(trajectory)
    restored = trajectory_from_dict(payload)

    assert restored.task_id == "task-1"
    assert restored.steps[0].tool_calls[0]["name"] == "write_file"
    assert restored.metadata["source"] == "test"


def test_reward_result_defaults():
    result = RewardResult(name="outcome_reward", score=1.0, reason="ok")
    assert result.metadata == {}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_types_and_trajectory.py -v`
Expected: FAIL，提示缺少 `hermes_agentic_rl.core`

- [ ] **Step 3: 实现类型与轨迹工具**

创建 `hermes_agentic_rl/core/types.py`：

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RolloutStep:
    turn_index: int
    assistant_message: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    task_id: str
    prompt: str
    steps: list[RolloutStep]
    final_output: str | None
    finished_naturally: bool
    turns_used: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RewardResult:
    name: str
    score: float
    reason: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RewardSummary:
    final_score: float
    components: list[RewardResult]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainSample:
    task_id: str
    prompt: str
    final_output: str | None
    reward: float
    trajectory: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
```

创建 `hermes_agentic_rl/core/trajectory.py`：

```python
from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RolloutStep, Trajectory


def trajectory_to_dict(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "task_id": trajectory.task_id,
        "prompt": trajectory.prompt,
        "steps": [
            {
                "turn_index": step.turn_index,
                "assistant_message": step.assistant_message,
                "tool_calls": step.tool_calls,
                "tool_results": step.tool_results,
                "reasoning": step.reasoning,
                "metadata": step.metadata,
            }
            for step in trajectory.steps
        ],
        "final_output": trajectory.final_output,
        "finished_naturally": trajectory.finished_naturally,
        "turns_used": trajectory.turns_used,
        "metadata": trajectory.metadata,
    }


def trajectory_from_dict(payload: dict[str, Any]) -> Trajectory:
    return Trajectory(
        task_id=payload["task_id"],
        prompt=payload["prompt"],
        steps=[
            RolloutStep(
                turn_index=step["turn_index"],
                assistant_message=step.get("assistant_message"),
                tool_calls=step.get("tool_calls", []),
                tool_results=step.get("tool_results", []),
                reasoning=step.get("reasoning"),
                metadata=step.get("metadata", {}),
            )
            for step in payload.get("steps", [])
        ],
        final_output=payload.get("final_output"),
        finished_naturally=payload["finished_naturally"],
        turns_used=payload["turns_used"],
        metadata=payload.get("metadata", {}),
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_types_and_trajectory.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/core/types.py hermes_agentic_rl/core/trajectory.py tests/test_types_and_trajectory.py
git commit -m "feat: add trajectory core types"
```

### Task 3: 实现注册表

**Files:**
- Create: `hermes_agentic_rl/core/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_registry.py` 写：

```python
from hermes_agentic_rl.core.registry import Registry


def test_registry_register_and_get():
    registry = Registry()

    class Demo:
        pass

    registry.register("reward", "demo", Demo)
    resolved = registry.get("reward", "demo")

    assert resolved is Demo


def test_registry_duplicate_registration_raises():
    registry = Registry()

    class Demo:
        pass

    registry.register("reward", "demo", Demo)

    try:
        registry.register("reward", "demo", Demo)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL，提示缺少 `Registry`

- [ ] **Step 3: 实现注册表**

创建 `hermes_agentic_rl/core/registry.py`：

```python
from __future__ import annotations

from collections import defaultdict
from typing import Any


class Registry:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = defaultdict(dict)

    def register(self, namespace: str, name: str, value: Any) -> None:
        if name in self._store[namespace]:
            raise ValueError(f"{namespace}:{name} already registered")
        self._store[namespace][name] = value

    def get(self, namespace: str, name: str) -> Any:
        try:
            return self._store[namespace][name]
        except KeyError as exc:
            raise KeyError(f"{namespace}:{name} is not registered") from exc

    def list_names(self, namespace: str) -> list[str]:
        return sorted(self._store[namespace].keys())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/core/registry.py tests/test_registry.py
git commit -m "feat: add pluggable component registry"
```

### Task 4: 实现 reward 基类与聚合器

**Files:**
- Create: `hermes_agentic_rl/rewards/base.py`
- Create: `hermes_agentic_rl/rewards/aggregate.py`
- Create: `tests/test_reward_aggregate.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_reward_aggregate.py` 写：

```python
from hermes_agentic_rl.core.types import RewardResult
from hermes_agentic_rl.rewards.aggregate import weighted_sum


def test_weighted_sum_uses_component_weights():
    summary = weighted_sum(
        [
            RewardResult(name="a", score=1.0, reason="ok", weight=0.75),
            RewardResult(name="b", score=0.5, reason="partial", weight=0.25),
        ]
    )

    assert round(summary.final_score, 4) == 0.875
    assert len(summary.components) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_reward_aggregate.py -v`
Expected: FAIL

- [ ] **Step 3: 实现基类与聚合器**

创建 `hermes_agentic_rl/rewards/base.py`：

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory


class BaseReward(ABC):
    name: str

    @abstractmethod
    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        raise NotImplementedError
```

创建 `hermes_agentic_rl/rewards/aggregate.py`：

```python
from __future__ import annotations

from hermes_agentic_rl.core.types import RewardResult, RewardSummary


def weighted_sum(results: list[RewardResult]) -> RewardSummary:
    if not results:
        return RewardSummary(final_score=0.0, components=[], metadata={"aggregator": "weighted_sum"})

    total_weight = sum(result.weight for result in results)
    if total_weight <= 0:
        raise ValueError("total reward weight must be positive")

    final_score = sum(result.score * result.weight for result in results) / total_weight
    return RewardSummary(
        final_score=final_score,
        components=results,
        metadata={"aggregator": "weighted_sum", "total_weight": total_weight},
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_reward_aggregate.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/rewards/base.py hermes_agentic_rl/rewards/aggregate.py tests/test_reward_aggregate.py
git commit -m "feat: add reward base and weighted aggregation"
```

### Task 5: 实现 OutcomeReward 与 ToolcallReward

**Files:**
- Create: `hermes_agentic_rl/rewards/outcome_reward.py`
- Create: `hermes_agentic_rl/rewards/toolcall_reward.py`
- Create: `tests/test_toolcall_reward.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_toolcall_reward.py` 写：

```python
import asyncio

from hermes_agentic_rl.core.types import RolloutStep, Trajectory
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward


def test_outcome_reward_from_expected_output():
    reward = OutcomeReward(weight=0.7)
    trajectory = Trajectory(
        task_id="task-1",
        prompt="do x",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )

    result = asyncio.run(
        reward.evaluate({"expected_output": "done"}, trajectory, tool_context=None)
    )

    assert result.score == 1.0
    assert result.weight == 0.7


def test_toolcall_reward_penalizes_empty_calls():
    reward = ToolcallReward(weight=0.3)
    trajectory = Trajectory(
        task_id="task-1",
        prompt="do x",
        steps=[RolloutStep(turn_index=0, tool_calls=[])],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )

    result = asyncio.run(reward.evaluate({}, trajectory, tool_context=None))

    assert result.score == 0.0
    assert "no tool calls" in result.reason
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_toolcall_reward.py -v`
Expected: FAIL

- [ ] **Step 3: 实现两个 reward**

创建 `hermes_agentic_rl/rewards/outcome_reward.py`：

```python
from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward


class OutcomeReward(BaseReward):
    name = "outcome_reward"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        expected_output = item.get("expected_output")
        if expected_output is None:
            return RewardResult(
                name=self.name,
                score=1.0 if trajectory.final_output else 0.0,
                reason="no expected_output provided; used final_output presence",
                weight=self.weight,
            )

        score = 1.0 if trajectory.final_output == expected_output else 0.0
        reason = "final_output matched expected_output" if score == 1.0 else "final_output mismatched expected_output"
        return RewardResult(name=self.name, score=score, reason=reason, weight=self.weight)
```

创建 `hermes_agentic_rl/rewards/toolcall_reward.py`：

```python
from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward


class ToolcallReward(BaseReward):
    name = "toolcall_reward"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        call_count = sum(len(step.tool_calls) for step in trajectory.steps)
        if call_count == 0:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no tool calls found in trajectory",
                weight=self.weight,
                metadata={"tool_call_count": 0},
            )

        invalid_calls = 0
        for step in trajectory.steps:
            for call in step.tool_calls:
                if "name" not in call:
                    invalid_calls += 1

        score = max(0.0, 1.0 - (invalid_calls / call_count))
        reason = "all tool calls contain name" if invalid_calls == 0 else "some tool calls are missing name"
        return RewardResult(
            name=self.name,
            score=score,
            reason=reason,
            weight=self.weight,
            metadata={"tool_call_count": call_count, "invalid_calls": invalid_calls},
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_toolcall_reward.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/rewards/outcome_reward.py hermes_agentic_rl/rewards/toolcall_reward.py tests/test_toolcall_reward.py
git commit -m "feat: add outcome and toolcall rewards"
```

### Task 6: 实现 RewardManager

**Files:**
- Create: `hermes_agentic_rl/core/reward_manager.py`
- Create: `tests/test_reward_manager.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_reward_manager.py` 写：

```python
import asyncio

from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward


def test_reward_manager_runs_components_and_aggregates():
    trajectory = Trajectory(
        task_id="task-1",
        prompt="create file",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )

    manager = RewardManager(
        rewards=[OutcomeReward(weight=0.7), ToolcallReward(weight=0.3)]
    )

    summary = asyncio.run(
        manager.evaluate({"expected_output": "done"}, trajectory, tool_context=None)
    )

    assert round(summary.final_score, 4) == 0.7
    assert len(summary.components) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_reward_manager.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 RewardManager**

创建 `hermes_agentic_rl/core/reward_manager.py`：

```python
from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardSummary, Trajectory
from hermes_agentic_rl.rewards.aggregate import weighted_sum
from hermes_agentic_rl.rewards.base import BaseReward


class RewardManager:
    def __init__(self, rewards: list[BaseReward]) -> None:
        self.rewards = rewards

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardSummary:
        results = []
        for reward in self.rewards:
            results.append(await reward.evaluate(item, trajectory, tool_context))
        return weighted_sum(results)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_reward_manager.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/core/reward_manager.py tests/test_reward_manager.py
git commit -m "feat: add reward manager"
```

### Task 7: 实现环境抽象与最小 TerminalTaskEnv

**Files:**
- Create: `hermes_agentic_rl/envs/base_env.py`
- Create: `hermes_agentic_rl/envs/terminal_task_env.py`
- Create: `examples/minimal_terminal_task.py`

- [ ] **Step 1: 写最小环境使用样例**

在 `examples/minimal_terminal_task.py` 写一个最小调用样例，先允许它失败：

```python
from hermes_agentic_rl.envs.terminal_task_env import TerminalTaskEnv


def build_env() -> TerminalTaskEnv:
    data = [
        {
            "task_id": "write-hello",
            "instruction": "Create hello.txt and write hello",
            "expected_output": "done",
        }
    ]
    return TerminalTaskEnv(dataset=data)


if __name__ == "__main__":
    env = build_env()
    print(env.format_prompt(env.dataset[0]))
```

- [ ] **Step 2: 运行样例确认失败**

Run: `python examples/minimal_terminal_task.py`
Expected: FAIL，提示缺少 `TerminalTaskEnv`

- [ ] **Step 3: 实现环境基类与终端环境**

创建 `hermes_agentic_rl/envs/base_env.py`：

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory


class BaseEnv(ABC):
    @abstractmethod
    async def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_next_item(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def format_prompt(self, item: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        raise NotImplementedError
```

创建 `hermes_agentic_rl/envs/terminal_task_env.py`：

```python
from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward


class TerminalTaskEnv(BaseEnv):
    def __init__(self, dataset: list[dict[str, Any]]) -> None:
        self.dataset = dataset
        self._index = 0
        self._outcome_reward = OutcomeReward(weight=0.7)
        self._toolcall_reward = ToolcallReward(weight=0.3)

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
        return [
            await self._outcome_reward.evaluate(item, trajectory, tool_context),
            await self._toolcall_reward.evaluate(item, trajectory, tool_context),
        ]
```

- [ ] **Step 4: 运行样例确认通过**

Run: `python examples/minimal_terminal_task.py`
Expected: 输出 `Create hello.txt and write hello`

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/envs/base_env.py hermes_agentic_rl/envs/terminal_task_env.py examples/minimal_terminal_task.py
git commit -m "feat: add terminal task environment"
```

### Task 8: 实现 RolloutManager

**Files:**
- Create: `hermes_agentic_rl/core/rollout_manager.py`

- [ ] **Step 1: 写一个失败的集成测试**

在 `tests/test_cli_smoke.py` 追加：

```python
import asyncio

from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.types import Trajectory


class FakeAgentLoop:
    async def run(self, prompt: str) -> dict:
        return {
            "messages": [{"role": "assistant", "content": "done"}],
            "tool_calls": [[{"name": "write_file", "arguments": {"path": "a.txt"}}]],
            "tool_results": [[{"ok": True}]],
            "final_output": "done",
            "finished_naturally": True,
            "turns_used": 1,
        }


def test_rollout_manager_collects_trajectory():
    manager = RolloutManager(agent_loop=FakeAgentLoop())
    trajectory = asyncio.run(manager.collect({"task_id": "t1"}, "create a file"))
    assert isinstance(trajectory, Trajectory)
    assert trajectory.final_output == "done"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cli_smoke.py::test_rollout_manager_collects_trajectory -v`
Expected: FAIL

- [ ] **Step 3: 实现 RolloutManager**

创建 `hermes_agentic_rl/core/rollout_manager.py`：

```python
from __future__ import annotations

from hermes_agentic_rl.core.types import RolloutStep, Trajectory


class RolloutManager:
    def __init__(self, agent_loop) -> None:
        self.agent_loop = agent_loop

    async def collect(self, item: dict, prompt: str) -> Trajectory:
        raw = await self.agent_loop.run(prompt)
        tool_calls_per_turn = raw.get("tool_calls", [])
        tool_results_per_turn = raw.get("tool_results", [])

        steps = []
        turn_count = max(len(tool_calls_per_turn), len(tool_results_per_turn), raw.get("turns_used", 0))
        for index in range(turn_count):
            steps.append(
                RolloutStep(
                    turn_index=index,
                    assistant_message=None,
                    tool_calls=tool_calls_per_turn[index] if index < len(tool_calls_per_turn) else [],
                    tool_results=tool_results_per_turn[index] if index < len(tool_results_per_turn) else [],
                )
            )

        return Trajectory(
            task_id=item["task_id"],
            prompt=prompt,
            steps=steps,
            final_output=raw.get("final_output"),
            finished_naturally=raw.get("finished_naturally", False),
            turns_used=raw.get("turns_used", 0),
            metadata={"messages": raw.get("messages", [])},
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cli_smoke.py::test_rollout_manager_collects_trajectory -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/core/rollout_manager.py tests/test_cli_smoke.py
git commit -m "feat: add rollout manager"
```

### Task 9: 实现 Trainer 基类与 Atropos/JSONL 导出

**Files:**
- Create: `hermes_agentic_rl/trainers/base.py`
- Create: `hermes_agentic_rl/trainers/atropos_grpo.py`
- Create: `hermes_agentic_rl/core/trainer_bridge.py`
- Create: `tests/test_trainer_export.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_trainer_export.py` 写：

```python
import json
from pathlib import Path

from hermes_agentic_rl.core.trajectory import trajectory_to_dict
from hermes_agentic_rl.core.types import RewardResult, RewardSummary, Trajectory
from hermes_agentic_rl.trainers.atropos_grpo import AtroposGrpoTrainer


def test_atropos_grpo_trainer_writes_jsonl(tmp_path: Path):
    trajectory = Trajectory(
        task_id="task-1",
        prompt="create a file",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )
    summary = RewardSummary(
        final_score=0.8,
        components=[RewardResult(name="outcome_reward", score=0.8, reason="ok")],
    )
    output_path = tmp_path / "train.jsonl"
    trainer = AtroposGrpoTrainer(output_path=output_path)

    payload = trainer.submit_sync({"task_id": "task-1"}, trajectory, summary)

    assert payload["reward"] == 0.8
    written = json.loads(output_path.read_text().strip())
    assert written["task_id"] == "task-1"
    assert written["trajectory"]["final_output"] == "done"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_trainer_export.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 trainer 与 bridge**

创建 `hermes_agentic_rl/trainers/base.py`：

```python
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    async def submit(self, item: dict, trajectory, reward_summary):
        raise NotImplementedError
```

创建 `hermes_agentic_rl/trainers/atropos_grpo.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

from hermes_agentic_rl.core.trajectory import trajectory_to_dict
from hermes_agentic_rl.trainers.base import BaseTrainer


class AtroposGrpoTrainer(BaseTrainer):
    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_payload(self, item: dict, trajectory, reward_summary) -> dict:
        return {
            "task_id": item["task_id"],
            "prompt": trajectory.prompt,
            "final_output": trajectory.final_output,
            "reward": reward_summary.final_score,
            "reward_components": [
                {
                    "name": component.name,
                    "score": component.score,
                    "reason": component.reason,
                    "weight": component.weight,
                    "metadata": component.metadata,
                }
                for component in reward_summary.components
            ],
            "trajectory": trajectory_to_dict(trajectory),
            "metadata": reward_summary.metadata,
        }

    async def submit(self, item: dict, trajectory, reward_summary):
        payload = self._build_payload(item, trajectory, reward_summary)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def submit_sync(self, item: dict, trajectory, reward_summary):
        payload = self._build_payload(item, trajectory, reward_summary)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload
```

创建 `hermes_agentic_rl/core/trainer_bridge.py`：

```python
from __future__ import annotations


class TrainerBridge:
    def __init__(self, trainer) -> None:
        self.trainer = trainer

    async def submit(self, item: dict, trajectory, reward_summary):
        return await self.trainer.submit(item, trajectory, reward_summary)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_trainer_export.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/trainers/base.py hermes_agentic_rl/trainers/atropos_grpo.py hermes_agentic_rl/core/trainer_bridge.py tests/test_trainer_export.py
git commit -m "feat: add atropos grpo export trainer"
```

### Task 10: 实现配置加载与 CLI

**Files:**
- Create: `hermes_agentic_rl/config.py`
- Create: `hermes_agentic_rl/cli/main.py`
- Create: `configs/terminal_grpo.yaml`

- [ ] **Step 1: 写失败测试**

在 `tests/test_cli_smoke.py` 追加：

```python
from pathlib import Path

from hermes_agentic_rl.config import load_config


def test_load_config_reads_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "runtime:\n  model: demo\nenvironment:\n  type: terminal_task_env\nreward:\n  aggregator: weighted_sum\ntrainer:\n  type: atropos_grpo\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config["runtime"]["model"] == "demo"
    assert config["trainer"]["type"] == "atropos_grpo"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cli_smoke.py::test_load_config_reads_yaml -v`
Expected: FAIL

- [ ] **Step 3: 实现配置与 CLI**

创建 `hermes_agentic_rl/config.py`：

```python
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
```

创建 `hermes_agentic_rl/cli/main.py`：

```python
from __future__ import annotations

import argparse

from hermes_agentic_rl import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-agentic-rl")
    parser.add_argument("--version", action="store_true", help="show version")
    parser.add_argument("command", nargs="?", default="help", choices=["help", "rollout", "train"])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        print(__version__)
        return 0
    if args.command == "help":
        parser.print_help()
        return 0
    print(f"command {args.command} is wired but not yet implemented")
    return 0
```

创建 `configs/terminal_grpo.yaml`：

```yaml
runtime:
  provider: openai
  model: demo-model
  temperature: 0.7
  max_agent_turns: 20
  terminal_backend: local
  terminal_timeout: 120

environment:
  type: terminal_task_env
  dataset_path: data/minimal_terminal_tasks.jsonl
  system_prompt: null

reward:
  aggregator: weighted_sum
  components:
    - name: outcome_reward
      weight: 0.7
    - name: toolcall_reward
      weight: 0.3

trainer:
  type: atropos_grpo
  export_training_path: outputs/train_samples.jsonl
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cli_smoke.py::test_load_config_reads_yaml -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl/config.py hermes_agentic_rl/cli/main.py configs/terminal_grpo.yaml tests/test_cli_smoke.py
git commit -m "feat: add config loader and cli shell"
```

### Task 11: 跑通最小闭环烟测

**Files:**
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: 写失败烟测**

在 `tests/test_cli_smoke.py` 追加：

```python
import asyncio
from pathlib import Path

from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.trainer_bridge import TrainerBridge
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward
from hermes_agentic_rl.trainers.atropos_grpo import AtroposGrpoTrainer


class FakeEndToEndLoop:
    async def run(self, prompt: str) -> dict:
        return {
            "messages": [{"role": "assistant", "content": "done"}],
            "tool_calls": [[{"name": "write_file", "arguments": {"path": "hello.txt"}}]],
            "tool_results": [[{"ok": True, "path": "hello.txt"}]],
            "final_output": "done",
            "finished_naturally": True,
            "turns_used": 1,
        }


def test_end_to_end_minimal_pipeline(tmp_path: Path):
    item = {"task_id": "task-1", "expected_output": "done"}
    rollout_manager = RolloutManager(agent_loop=FakeEndToEndLoop())
    trajectory = asyncio.run(rollout_manager.collect(item, "create hello.txt"))

    reward_manager = RewardManager(
        rewards=[OutcomeReward(weight=0.7), ToolcallReward(weight=0.3)]
    )
    summary = asyncio.run(reward_manager.evaluate(item, trajectory, tool_context=None))

    bridge = TrainerBridge(AtroposGrpoTrainer(output_path=tmp_path / "train.jsonl"))
    payload = asyncio.run(bridge.submit(item, trajectory, summary))

    assert summary.final_score == 1.0
    assert payload["trajectory"]["final_output"] == "done"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cli_smoke.py::test_end_to_end_minimal_pipeline -v`
Expected: FAIL

- [ ] **Step 3: 修正已有实现直到烟测通过**

如测试失败，优先修正以下代码，不新增新模块：

`hermes_agentic_rl/core/rollout_manager.py`

```python
from __future__ import annotations

from hermes_agentic_rl.core.types import RolloutStep, Trajectory


class RolloutManager:
    def __init__(self, agent_loop) -> None:
        self.agent_loop = agent_loop

    async def collect(self, item: dict, prompt: str) -> Trajectory:
        raw = await self.agent_loop.run(prompt)
        tool_calls_per_turn = raw.get("tool_calls", [])
        tool_results_per_turn = raw.get("tool_results", [])
        messages = raw.get("messages", [])

        steps = []
        turn_count = max(len(tool_calls_per_turn), len(tool_results_per_turn), raw.get("turns_used", 0))
        for index in range(turn_count):
            assistant_message = None
            if index < len(messages) and isinstance(messages[index], dict):
                assistant_message = messages[index].get("content")

            steps.append(
                RolloutStep(
                    turn_index=index,
                    assistant_message=assistant_message,
                    tool_calls=tool_calls_per_turn[index] if index < len(tool_calls_per_turn) else [],
                    tool_results=tool_results_per_turn[index] if index < len(tool_results_per_turn) else [],
                )
            )

        return Trajectory(
            task_id=item["task_id"],
            prompt=prompt,
            steps=steps,
            final_output=raw.get("final_output"),
            finished_naturally=raw.get("finished_naturally", False),
            turns_used=raw.get("turns_used", 0),
            metadata={"messages": messages},
        )
```

- [ ] **Step 4: 运行完整测试套件**

Run: `pytest tests -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add hermes_agentic_rl tests configs examples
git commit -m "feat: complete minimal hermes-agentic-rl pipeline"
```

## 自检结果

### Spec 覆盖
- `Trajectory` 数据结构：Task 2
- 注册表：Task 3
- Reward 组件与聚合：Task 4、Task 5、Task 6
- `TerminalTaskEnv`：Task 7
- Rollout：Task 8
- Trainer export：Task 9
- 配置与 CLI：Task 10
- 端到端闭环：Task 11

### 占位符检查
- 没有使用 `TODO`、`TBD`、`implement later`
- 每个代码步骤都给出具体代码
- 每个测试步骤都给出精确命令

### 类型一致性检查
- `RewardSummary.final_score`、`RewardResult.weight`、`Trajectory` 字段在各任务中名称一致
- `RolloutManager.collect()`、`RewardManager.evaluate()`、`BaseTrainer.submit()` 的参数命名保持一致

## 执行交接

Plan complete and saved to `docs/superpowers/plans/2026-05-07-hermes-agentic-rl.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
