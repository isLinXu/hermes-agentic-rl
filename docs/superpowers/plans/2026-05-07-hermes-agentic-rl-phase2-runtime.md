# Hermes-Agentic-RL Phase 2 Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `hermes-agentic-rl` 增加可选的真实 `hermes-agent` 运行时集成，并让 `rollout` / `train` CLI 命令具备最小可用行为。

**Architecture:** 在现有最小框架之上新增 `runtime` 和 `datasets` 两层：`runtime` 负责 fake / hermes 两种运行时的统一协议，`datasets` 负责 JSONL 任务加载；CLI 只做配置读取、任务装配和流程编排，不直接承担 reward、trajectory 或 trainer 逻辑。真实 `hermes-agent` 通过运行时适配器按需导入，未安装时返回结构化错误而不破坏默认测试。

**Tech Stack:** Python 3.11+、`argparse`、`json`、`pathlib`、`typing`、`PyYAML`、`pytest`

---

## 文件结构

本计划会创建以下文件：

- `hermes_agentic_rl/runtime/__init__.py`
- `hermes_agentic_rl/runtime/errors.py`
- `hermes_agentic_rl/runtime/base.py`
- `hermes_agentic_rl/runtime/hermes_adapter.py`
- `hermes_agentic_rl/runtime/fake_adapter.py`
- `hermes_agentic_rl/datasets/__init__.py`
- `hermes_agentic_rl/datasets/jsonl_loader.py`
- `tests/test_dataset_loader.py`
- `tests/test_runtime_adapter.py`
- `tests/test_cli_runtime_commands.py`

本计划会修改以下文件：

- `hermes_agentic_rl/config.py`
- `hermes_agentic_rl/core/rollout_manager.py`
- `hermes_agentic_rl/cli/main.py`
- `configs/terminal_grpo.yaml`

---

### Task 1: 添加 JSONL 数据集加载器

**Files:**
- Create: `hermes_agentic_rl/datasets/__init__.py`
- Create: `hermes_agentic_rl/datasets/jsonl_loader.py`
- Test: `tests/test_dataset_loader.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_dataset_loader.py`：

```python
import json
from pathlib import Path

from hermes_agentic_rl.datasets.jsonl_loader import load_jsonl_dataset


def test_load_jsonl_dataset_reads_items_and_preserves_task_id(tmp_path: Path):
    dataset_path = tmp_path / "tasks.jsonl"
    rows = [
        {"task_id": "task-1", "instruction": "Create a.txt", "expected_output": "done"},
        {"task_id": "task-2", "instruction": "Create b.txt", "expected_output": "done"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    items = load_jsonl_dataset(dataset_path)

    assert len(items) == 2
    assert items[0]["task_id"] == "task-1"
    assert items[1]["instruction"] == "Create b.txt"


def test_load_jsonl_dataset_generates_task_id_when_missing(tmp_path: Path):
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps({"instruction": "Create c.txt", "expected_output": "done"}),
        encoding="utf-8",
    )

    items = load_jsonl_dataset(dataset_path)

    assert len(items) == 1
    assert items[0]["task_id"] == "generated-0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes_agentic_rl.datasets'`

- [ ] **Step 3: Write minimal implementation**

创建 `hermes_agentic_rl/datasets/__init__.py`：

```python
from hermes_agentic_rl.datasets.jsonl_loader import load_jsonl_dataset

__all__ = ["load_jsonl_dataset"]
```

创建 `hermes_agentic_rl/datasets/jsonl_loader.py`：

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    items: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            item.setdefault("task_id", f"generated-{index}")
            items.append(item)
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/datasets tests/test_dataset_loader.py
git commit -m "feat: add jsonl dataset loader"
```

### Task 2: 定义运行时错误与基础协议

**Files:**
- Create: `hermes_agentic_rl/runtime/__init__.py`
- Create: `hermes_agentic_rl/runtime/errors.py`
- Create: `hermes_agentic_rl/runtime/base.py`
- Test: `tests/test_runtime_adapter.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_runtime_adapter.py`：

```python
from hermes_agentic_rl.runtime.errors import RuntimeUnavailableError
from hermes_agentic_rl.runtime.base import BaseRuntimeAdapter


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runtime_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes_agentic_rl.runtime'`

- [ ] **Step 3: Write minimal implementation**

创建 `hermes_agentic_rl/runtime/errors.py`：

```python
class RuntimeUnavailableError(RuntimeError):
    pass


class RuntimeConfigurationError(RuntimeError):
    pass


class RuntimeExecutionError(RuntimeError):
    pass
```

创建 `hermes_agentic_rl/runtime/base.py`：

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRuntimeAdapter(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def describe_unavailable_reason(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_agent_loop(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError
```

创建 `hermes_agentic_rl/runtime/__init__.py`：

```python
from hermes_agentic_rl.runtime.base import BaseRuntimeAdapter
from hermes_agentic_rl.runtime.errors import (
    RuntimeConfigurationError,
    RuntimeExecutionError,
    RuntimeUnavailableError,
)

__all__ = [
    "BaseRuntimeAdapter",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "RuntimeUnavailableError",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runtime_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/runtime tests/test_runtime_adapter.py
git commit -m "feat: add runtime base protocol and errors"
```

### Task 3: 实现 fake runtime adapter

**Files:**
- Create: `hermes_agentic_rl/runtime/fake_adapter.py`
- Modify: `tests/test_runtime_adapter.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_runtime_adapter.py` 追加：

```python
import asyncio

from hermes_agentic_rl.runtime.fake_adapter import FakeRuntimeAdapter


def test_fake_runtime_adapter_builds_protocol_compatible_loop():
    adapter = FakeRuntimeAdapter()
    loop = adapter.build_agent_loop({"runtime": {"model": "demo"}})

    payload = asyncio.run(loop.run("Create hello.txt"))

    assert payload["final_output"] == "done"
    assert payload["tool_calls"][0][0]["name"] == "write_file"
    assert payload["turns_used"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runtime_adapter.py::test_fake_runtime_adapter_builds_protocol_compatible_loop -v`
Expected: FAIL with `ModuleNotFoundError` for `fake_adapter`

- [ ] **Step 3: Write minimal implementation**

创建 `hermes_agentic_rl/runtime/fake_adapter.py`：

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runtime_adapter.py::test_fake_runtime_adapter_builds_protocol_compatible_loop -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/runtime/fake_adapter.py tests/test_runtime_adapter.py
git commit -m "feat: add fake runtime adapter"
```

### Task 4: 实现 hermes runtime adapter 的不可用路径

**Files:**
- Create: `hermes_agentic_rl/runtime/hermes_adapter.py`
- Modify: `tests/test_runtime_adapter.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_runtime_adapter.py` 追加：

```python
import pytest

from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter
from hermes_agentic_rl.runtime.errors import RuntimeUnavailableError


def test_hermes_runtime_adapter_reports_unavailable_when_dependency_missing():
    adapter = HermesRuntimeAdapter(importer=lambda: (_ for _ in ()).throw(ImportError("missing hermes")))

    assert adapter.is_available() is False
    assert "missing hermes" in adapter.describe_unavailable_reason()

    with pytest.raises(RuntimeUnavailableError):
        adapter.build_agent_loop({"runtime": {"integration": "hermes"}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runtime_adapter.py::test_hermes_runtime_adapter_reports_unavailable_when_dependency_missing -v`
Expected: FAIL with `ModuleNotFoundError` for `hermes_adapter`

- [ ] **Step 3: Write minimal implementation**

创建 `hermes_agentic_rl/runtime/hermes_adapter.py`：

```python
from __future__ import annotations

from typing import Any, Callable

from hermes_agentic_rl.runtime.base import BaseRuntimeAdapter
from hermes_agentic_rl.runtime.errors import RuntimeUnavailableError


def _default_importer() -> Any:
    import hermes_agent  # type: ignore

    return hermes_agent


class HermesRuntimeAdapter(BaseRuntimeAdapter):
    def __init__(self, importer: Callable[[], Any] | None = None) -> None:
        self._importer = importer or _default_importer
        self._cached_module: Any | None = None
        self._error_message = ""

    def _load(self) -> Any | None:
        if self._cached_module is not None:
            return self._cached_module
        try:
            self._cached_module = self._importer()
            self._error_message = ""
            return self._cached_module
        except ImportError as exc:
            self._error_message = str(exc)
            return None

    def is_available(self) -> bool:
        return self._load() is not None

    def describe_unavailable_reason(self) -> str:
        self._load()
        return self._error_message or "hermes-agent runtime is unavailable"

    def build_agent_loop(self, config: dict[str, Any]) -> Any:
        module = self._load()
        if module is None:
            raise RuntimeUnavailableError(self.describe_unavailable_reason())
        raise RuntimeUnavailableError(
            "real hermes-agent loop wiring is not implemented yet; adapter availability is confirmed"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runtime_adapter.py::test_hermes_runtime_adapter_reports_unavailable_when_dependency_missing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/runtime/hermes_adapter.py tests/test_runtime_adapter.py
git commit -m "feat: add hermes runtime unavailable path"
```

### Task 5: 扩展配置解析

**Files:**
- Modify: `hermes_agentic_rl/config.py`
- Modify: `configs/terminal_grpo.yaml`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cli_smoke.py` 追加：

```python
def test_load_config_reads_runtime_integration_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  model: demo\n"
            "environment:\n"
            "  dataset_path: data/tasks.jsonl\n"
            "trainer:\n"
            "  export_training_path: outputs/train.jsonl\n"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["runtime"]["integration"] == "fake"
    assert config["environment"]["dataset_path"] == "data/tasks.jsonl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_smoke.py::test_load_config_reads_runtime_integration_defaults -v`
Expected: FAIL because `integration` key is missing

- [ ] **Step 3: Write minimal implementation**

将 `hermes_agentic_rl/config.py` 修改为：

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    runtime = config.setdefault("runtime", {})
    runtime.setdefault("integration", "fake")
    runtime.setdefault("max_agent_turns", 20)
    runtime.setdefault("enabled_toolsets", ["terminal", "file"])

    environment = config.setdefault("environment", {})
    trainer = config.setdefault("trainer", {})
    reward = config.setdefault("reward", {})
    reward.setdefault("aggregator", "weighted_sum")

    return {
        "runtime": runtime,
        "environment": environment,
        "reward": reward,
        "trainer": trainer,
    }
```

将 `configs/terminal_grpo.yaml` 修改为：

```yaml
runtime:
  integration: fake
  provider: openai
  model: demo-model
  temperature: 0.7
  max_agent_turns: 20
  terminal_backend: local
  terminal_timeout: 120
  enabled_toolsets:
    - terminal
    - file

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_smoke.py::test_load_config_reads_runtime_integration_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/config.py configs/terminal_grpo.yaml tests/test_cli_smoke.py
git commit -m "feat: extend runtime config defaults"
```

### Task 6: 增强 RolloutManager 以保留运行时 metadata

**Files:**
- Modify: `hermes_agentic_rl/core/rollout_manager.py`
- Modify: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cli_smoke.py` 追加：

```python
class FakeLoopWithMetadata:
    async def run(self, prompt: str) -> dict:
        return {
            "messages": [{"role": "assistant", "content": "done"}],
            "tool_calls": [[{"name": "write_file", "arguments": {"path": "meta.txt"}}]],
            "tool_results": [[{"ok": True}]],
            "final_output": "done",
            "finished_naturally": True,
            "turns_used": 1,
            "metadata": {"runtime": "fake", "session_id": "demo-session"},
        }


def test_rollout_manager_preserves_runtime_metadata():
    manager = RolloutManager(agent_loop=FakeLoopWithMetadata())

    trajectory = asyncio.run(manager.collect({"task_id": "t-meta"}, "create meta.txt"))

    assert trajectory.metadata["runtime"]["runtime"] == "fake"
    assert trajectory.metadata["runtime"]["session_id"] == "demo-session"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_smoke.py::test_rollout_manager_preserves_runtime_metadata -v`
Expected: FAIL because `runtime` metadata key is absent

- [ ] **Step 3: Write minimal implementation**

将 `hermes_agentic_rl/core/rollout_manager.py` 修改为：

```python
from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RolloutStep, Trajectory


class RolloutManager:
    def __init__(self, agent_loop: Any) -> None:
        self.agent_loop = agent_loop

    async def collect(self, item: dict[str, Any], prompt: str) -> Trajectory:
        raw = await self.agent_loop.run(prompt)
        tool_calls_per_turn = raw.get("tool_calls", [])
        tool_results_per_turn = raw.get("tool_results", [])
        messages = raw.get("messages", [])
        runtime_metadata = raw.get("metadata", {})

        steps: list[RolloutStep] = []
        turn_count = max(
            len(tool_calls_per_turn),
            len(tool_results_per_turn),
            raw.get("turns_used", 0),
        )
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
            metadata={"messages": messages, "runtime": runtime_metadata},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_smoke.py::test_rollout_manager_preserves_runtime_metadata -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/core/rollout_manager.py tests/test_cli_smoke.py
git commit -m "feat: preserve runtime metadata in rollout manager"
```

### Task 7: 让 CLI `rollout` 命令真正可执行

**Files:**
- Modify: `hermes_agentic_rl/cli/main.py`
- Create: `tests/test_cli_runtime_commands.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_cli_runtime_commands.py`：

```python
import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_cli_rollout_writes_trajectory_json(tmp_path: Path, monkeypatch, capsys):
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps({"task_id": "task-1", "instruction": "Create hello.txt", "expected_output": "done"}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "  model: demo\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "trainer:\n"
            f"  export_training_path: {tmp_path / 'train.jsonl'}\n"
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "trajectory.json"

    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "rollout", "--config", str(config_path), "--output", str(output_path)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "trajectory saved" in captured.out
    assert output_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_runtime_commands.py::test_cli_rollout_writes_trajectory_json -v`
Expected: FAIL because current CLI does not accept `--config` / `--output`

- [ ] **Step 3: Write minimal implementation**

将 `hermes_agentic_rl/cli/main.py` 修改为：

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes_agentic_rl import __version__
from hermes_agentic_rl.config import load_config
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.trajectory import trajectory_to_dict
from hermes_agentic_rl.datasets.jsonl_loader import load_jsonl_dataset
from hermes_agentic_rl.runtime.fake_adapter import FakeRuntimeAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-agentic-rl")
    parser.add_argument("--version", action="store_true", help="show version")
    parser.add_argument("command", nargs="?", default="help", choices=["help", "rollout", "train"])
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--output", dest="output_path")
    return parser


def _build_runtime_adapter(config: dict):
    integration = config["runtime"]["integration"]
    if integration == "fake":
        return FakeRuntimeAdapter()
    raise RuntimeError(f"unsupported integration: {integration}")


def _run_rollout(config_path: str, output_path: str | None) -> int:
    config = load_config(config_path)
    dataset = load_jsonl_dataset(config["environment"]["dataset_path"])
    item = dataset[0]
    adapter = _build_runtime_adapter(config)
    loop = adapter.build_agent_loop(config)
    trajectory = __import__("asyncio").run(RolloutManager(loop).collect(item, item["instruction"]))

    if output_path:
        Path(output_path).write_text(
            json.dumps(trajectory_to_dict(trajectory), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"trajectory saved to {output_path}")
    else:
        print(json.dumps(trajectory_to_dict(trajectory), ensure_ascii=False))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        print(__version__)
        return 0
    if args.command == "help":
        parser.print_help()
        return 0
    if args.command == "rollout":
        return _run_rollout(args.config_path, args.output_path)
    print(f"command {args.command} is wired but not yet implemented")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_runtime_commands.py::test_cli_rollout_writes_trajectory_json -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/cli/main.py tests/test_cli_runtime_commands.py
git commit -m "feat: add executable rollout cli command"
```

### Task 8: 让 CLI `train` 命令真正可执行

**Files:**
- Modify: `hermes_agentic_rl/cli/main.py`
- Modify: `tests/test_cli_runtime_commands.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cli_runtime_commands.py` 追加：

```python
def test_cli_train_exports_training_jsonl(tmp_path: Path, monkeypatch, capsys):
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps({"task_id": "task-1", "instruction": "Create hello.txt", "expected_output": "done"}),
        encoding="utf-8",
    )
    output_path = tmp_path / "train.jsonl"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "  model: demo\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "reward:\n"
            "  aggregator: weighted_sum\n"
            "trainer:\n"
            f"  export_training_path: {output_path}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["hermes-agentic-rl", "train", "--config", str(config_path)])

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "training sample saved" in captured.out
    assert output_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_runtime_commands.py::test_cli_train_exports_training_jsonl -v`
Expected: FAIL because `train` still prints placeholder

- [ ] **Step 3: Write minimal implementation**

继续修改 `hermes_agentic_rl/cli/main.py`：

```python
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.trainer_bridge import TrainerBridge
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward
from hermes_agentic_rl.trainers.atropos_grpo import AtroposGrpoTrainer
```

增加：

```python
def _build_reward_manager(config: dict) -> RewardManager:
    return RewardManager(
        rewards=[
            OutcomeReward(weight=0.7),
            ToolcallReward(weight=0.3),
        ]
    )


def _run_train(config_path: str) -> int:
    config = load_config(config_path)
    dataset = load_jsonl_dataset(config["environment"]["dataset_path"])
    item = dataset[0]
    adapter = _build_runtime_adapter(config)
    loop = adapter.build_agent_loop(config)
    trajectory = __import__("asyncio").run(RolloutManager(loop).collect(item, item["instruction"]))
    summary = __import__("asyncio").run(
        _build_reward_manager(config).evaluate(item, trajectory, tool_context=None)
    )
    bridge = TrainerBridge(
        AtroposGrpoTrainer(output_path=Path(config["trainer"]["export_training_path"]))
    )
    __import__("asyncio").run(bridge.submit(item, trajectory, summary))
    print(f"training sample saved to {config['trainer']['export_training_path']}")
    return 0
```

将 `main()` 中的命令分支改为：

```python
    if args.command == "rollout":
        return _run_rollout(args.config_path, args.output_path)
    if args.command == "train":
        return _run_train(args.config_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_runtime_commands.py::test_cli_train_exports_training_jsonl -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/cli/main.py tests/test_cli_runtime_commands.py
git commit -m "feat: add executable train cli command"
```

### Task 9: hermes 集成不可用时返回明确错误

**Files:**
- Modify: `hermes_agentic_rl/cli/main.py`
- Modify: `tests/test_cli_runtime_commands.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cli_runtime_commands.py` 追加：

```python
def test_cli_rollout_reports_clear_error_when_hermes_runtime_is_unavailable(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps({"task_id": "task-1", "instruction": "Create hello.txt", "expected_output": "done"}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: hermes\n"
            "  model: demo\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "trainer:\n"
            f"  export_training_path: {tmp_path / 'train.jsonl'}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["hermes-agentic-rl", "rollout", "--config", str(config_path)])

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "runtime unavailable" in captured.out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_runtime_commands.py::test_cli_rollout_reports_clear_error_when_hermes_runtime_is_unavailable -v`
Expected: FAIL because CLI currently raises generic runtime error

- [ ] **Step 3: Write minimal implementation**

继续修改 `hermes_agentic_rl/cli/main.py`：

```python
from hermes_agentic_rl.runtime.errors import RuntimeUnavailableError
from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter
```

替换 `_build_runtime_adapter`：

```python
def _build_runtime_adapter(config: dict):
    integration = config["runtime"]["integration"]
    if integration == "fake":
        return FakeRuntimeAdapter()
    if integration == "hermes":
        return HermesRuntimeAdapter()
    raise RuntimeError(f"unsupported integration: {integration}")
```

为 `_run_rollout` 和 `_run_train` 加上错误处理，示例：

```python
def _create_loop_or_raise(config: dict):
    adapter = _build_runtime_adapter(config)
    if not adapter.is_available():
        raise RuntimeUnavailableError(adapter.describe_unavailable_reason())
    return adapter.build_agent_loop(config)
```

在 `main()` 中用：

```python
    try:
        if args.command == "rollout":
            return _run_rollout(args.config_path, args.output_path)
        if args.command == "train":
            return _run_train(args.config_path)
    except RuntimeUnavailableError as exc:
        print(f"runtime unavailable: {exc}")
        return 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_runtime_commands.py::test_cli_rollout_reports_clear_error_when_hermes_runtime_is_unavailable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/cli/main.py tests/test_cli_runtime_commands.py
git commit -m "feat: add clear hermes runtime unavailable error"
```

### Task 10: 运行第二阶段完整测试集

**Files:**
- Modify: `tests/test_cli_runtime_commands.py`

- [ ] **Step 1: Write one final smoke assertion**

在 `tests/test_cli_runtime_commands.py` 最后追加：

```python
def test_cli_help_still_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["hermes-agentic-rl", "help"])
    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out.lower()
```

- [ ] **Step 2: Run test to verify it fails if needed**

Run: `pytest tests/test_cli_runtime_commands.py::test_cli_help_still_returns_zero -v`
Expected: PASS or minimal adjustment needed. If it already passes, keep it and continue.

- [ ] **Step 3: Run full phase-2 focused suite**

Run: `pytest tests/test_dataset_loader.py tests/test_runtime_adapter.py tests/test_cli_runtime_commands.py tests/test_cli_smoke.py -v`
Expected: PASS

- [ ] **Step 4: Run full repository tests**

Run: `pytest tests -v`
Expected: PASS with all previous stage tests still green

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl tests configs
git commit -m "feat: complete phase2 runtime integration and cli"
```

## 自检结果

### Spec coverage
- JSONL 数据加载：Task 1
- 运行时协议与错误：Task 2、Task 3、Task 4
- 配置增强：Task 5
- `RolloutManager` 运行时 metadata：Task 6
- 可执行 `rollout` CLI：Task 7
- 可执行 `train` CLI：Task 8
- hermes 不可用路径：Task 9
- 完整测试与回归：Task 10

### Placeholder scan
- 未使用 `TODO`、`TBD`、`implement later`
- 每个步骤都给出了具体代码或具体命令
- 每项测试都有明确的失败和通过条件

### Type consistency
- `BaseRuntimeAdapter.build_agent_loop(config)` 在所有任务中签名一致
- `RolloutManager.collect(item, prompt)` 的调用方式保持一致
- `load_config()` 始终返回包含 `runtime`、`environment`、`reward`、`trainer` 的字典

## 执行交接

Plan complete and saved to `docs/superpowers/plans/2026-05-07-hermes-agentic-rl-phase2-runtime.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
