# Hermes-Agentic-RL Phase 3 Hermes Runtime Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `HermesRuntimeAdapter` 从“依赖存在性检测器”升级为可在已安装 pip 包场景下真实接线 `hermes-agent` 的运行时桥接器，并让 `integration=hermes` 的 `rollout` 在可注入的测试环境中跑通。

**Architecture:** 新增 `runtime/hermes_entrypoints.py` 管理有限候选入口探测；新增 `runtime/hermes_wrapper.py` 负责将真实 Hermes 对象包装为统一 `run(prompt)` 协议并规范化输出；增强 `runtime/hermes_adapter.py` 调用入口探测与 wrapper 构建；CLI 继续复用 `_build_agent_loop()`，只要 adapter 能返回真实 loop 即可。测试通过 monkeypatch 注入“假的 hermes_agent 包模块”来验证接线，不依赖真实安装。

**Tech Stack:** Python 3.11+、`importlib`、`types.ModuleType`、`dataclasses`、`pytest`

---

## 文件结构

本计划会创建以下文件：

- `hermes_agentic_rl/runtime/hermes_entrypoints.py`
- `hermes_agentic_rl/runtime/hermes_wrapper.py`
- `tests/test_hermes_entrypoints.py`
- `tests/test_hermes_runtime_integration.py`

本计划会修改以下文件：

- `hermes_agentic_rl/runtime/hermes_adapter.py`
- `hermes_agentic_rl/cli/main.py`（预期只需极小改动或无需改动）

---

### Task 1: 实现有限入口探测（hermes_entrypoints）

**Files:**
- Create: `hermes_agentic_rl/runtime/hermes_entrypoints.py`
- Test: `tests/test_hermes_entrypoints.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_hermes_entrypoints.py`：

```python
import importlib
import types

import pytest

from hermes_agentic_rl.runtime.hermes_entrypoints import (
    HermesEntrypointNotFoundError,
    find_hermes_entrypoint,
)


def _inject_module(monkeypatch, name: str, module: types.ModuleType) -> None:
    # 让 importlib.import_module(name) 返回我们注入的 module
    monkeypatch.setitem(importlib.sys.modules, name, module)


def test_find_hermes_entrypoint_returns_first_available(monkeypatch):
    mod = types.ModuleType("hermes_agent.environments.agent_loop")
    mod.HermesAgentLoop = object  # sentinel
    _inject_module(monkeypatch, "hermes_agent.environments.agent_loop", mod)

    entrypoint = find_hermes_entrypoint()

    assert entrypoint.name == "hermes_agent_loop"
    assert entrypoint.module_name == "hermes_agent.environments.agent_loop"
    assert entrypoint.attr_name == "HermesAgentLoop"


def test_find_hermes_entrypoint_raises_when_all_missing(monkeypatch):
    # 确保候选模块都不存在
    for name in [
        "hermes_agent.environments.agent_loop",
        "hermes_agent.run_agent",
    ]:
        monkeypatch.delitem(importlib.sys.modules, name, raising=False)

    with pytest.raises(HermesEntrypointNotFoundError):
        find_hermes_entrypoint()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_hermes_entrypoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes_agentic_rl.runtime.hermes_entrypoints'`

- [ ] **Step 3: Write minimal implementation**

创建 `hermes_agentic_rl/runtime/hermes_entrypoints.py`：

```python
from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


class HermesEntrypointNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HermesEntrypoint:
    name: str
    module_name: str
    attr_name: str

    def load_attr(self) -> Any:
        module = importlib.import_module(self.module_name)
        return getattr(module, self.attr_name)


_CANDIDATES: list[HermesEntrypoint] = [
    HermesEntrypoint(
        name="hermes_agent_loop",
        module_name="hermes_agent.environments.agent_loop",
        attr_name="HermesAgentLoop",
    ),
    HermesEntrypoint(
        name="ai_agent",
        module_name="hermes_agent.run_agent",
        attr_name="AIAgent",
    ),
]


def find_hermes_entrypoint() -> HermesEntrypoint:
    for candidate in _CANDIDATES:
        try:
            attr = candidate.load_attr()
            if attr is not None:
                return candidate
        except Exception:
            continue
    raise HermesEntrypointNotFoundError("no supported hermes-agent entrypoint found")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_hermes_entrypoints.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/runtime/hermes_entrypoints.py tests/test_hermes_entrypoints.py
git commit -m "feat: add hermes entrypoint discovery"
```

---

### Task 2: 实现 Hermes wrapper（统一 run 协议与最小映射）

**Files:**
- Create: `hermes_agentic_rl/runtime/hermes_wrapper.py`
- Test: `tests/test_hermes_runtime_integration.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_hermes_runtime_integration.py`：

```python
import asyncio
from dataclasses import dataclass

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_hermes_runtime_integration.py::test_wrapper_normalizes_to_protocol -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hermes_agentic_rl.runtime.hermes_wrapper'`

- [ ] **Step 3: Write minimal implementation**

创建 `hermes_agentic_rl/runtime/hermes_wrapper.py`：

```python
from __future__ import annotations

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

        # 允许 result 是对象或 dict
        def _get(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        return {
            "messages": _get(result, "messages", []),
            "tool_calls": _get(result, "tool_calls", []),
            "tool_results": _get(result, "tool_results", []),
            "final_output": _get(result, "final_output"),
            "finished_naturally": bool(_get(result, "finished_naturally", True)),
            "turns_used": int(_get(result, "turns_used", 0)),
            "metadata": {
                "runtime": "hermes",
                "entrypoint": self._entrypoint_name,
                "prompt": prompt,
            },
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_hermes_runtime_integration.py::test_wrapper_normalizes_to_protocol -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/runtime/hermes_wrapper.py tests/test_hermes_runtime_integration.py
git commit -m "feat: add hermes loop wrapper and normalizer"
```

---

### Task 3: 让 HermesRuntimeAdapter 真正 build loop（通过入口探测 + wrapper）

**Files:**
- Modify: `hermes_agentic_rl/runtime/hermes_adapter.py`
- Modify: `tests/test_hermes_runtime_integration.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_hermes_runtime_integration.py` 追加：

```python
import importlib
import types

import pytest

from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter


def test_hermes_adapter_builds_wrapper_when_entrypoint_is_injected(monkeypatch):
    # 伪造 hermes_agent.environments.agent_loop.HermesAgentLoop
    module = types.ModuleType("hermes_agent.environments.agent_loop")

    class InjectedHermesLoop:
        async def run(self, prompt: str):
            return {
                "messages": [{"role": "assistant", "content": "done"}],
                "tool_calls": [[{"name": "write_file", "arguments": {"path": "x.txt"}}]],
                "tool_results": [[{"ok": True}]],
                "final_output": "done",
                "finished_naturally": True,
                "turns_used": 1,
            }

    # 入口对象本身充当“工厂”，直接返回 loop 实例
    module.HermesAgentLoop = lambda **kwargs: InjectedHermesLoop()
    monkeypatch.setitem(importlib.sys.modules, "hermes_agent.environments.agent_loop", module)

    adapter = HermesRuntimeAdapter(importer=lambda: object())  # importer 只用于判定“包存在”
    loop = adapter.build_agent_loop({"runtime": {"integration": "hermes"}})

    payload = __import__("asyncio").run(loop.run("Create x.txt"))

    assert payload["metadata"]["runtime"] == "hermes"
    assert payload["final_output"] == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_hermes_runtime_integration.py::test_hermes_adapter_builds_wrapper_when_entrypoint_is_injected -v`
Expected: FAIL because adapter currently always raises “not implemented yet”

- [ ] **Step 3: Write minimal implementation**

修改 `hermes_agentic_rl/runtime/hermes_adapter.py`：

```python
from hermes_agentic_rl.runtime.errors import RuntimeConfigurationError, RuntimeUnavailableError
from hermes_agentic_rl.runtime.hermes_entrypoints import HermesEntrypointNotFoundError, find_hermes_entrypoint
from hermes_agentic_rl.runtime.hermes_wrapper import HermesLoopWrapper
```

把 `build_agent_loop` 改为：

```python
    def build_agent_loop(self, config: dict[str, Any]) -> Any:
        module = self._load()
        if module is None:
            raise RuntimeUnavailableError(self.describe_unavailable_reason())

        try:
            entrypoint = find_hermes_entrypoint()
        except HermesEntrypointNotFoundError as exc:
            raise RuntimeUnavailableError(str(exc)) from exc

        attr = entrypoint.load_attr()

        # 极简构造策略：如果是可调用对象则当作工厂，否则当作实例
        try:
            if callable(attr):
                loop = attr()
            else:
                loop = attr
        except Exception as exc:
            raise RuntimeConfigurationError(f"failed to build hermes loop: {exc}") from exc

        return HermesLoopWrapper(loop=loop, entrypoint_name=entrypoint.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_hermes_runtime_integration.py::test_hermes_adapter_builds_wrapper_when_entrypoint_is_injected -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/runtime/hermes_adapter.py tests/test_hermes_runtime_integration.py
git commit -m "feat: wire hermes runtime adapter via entrypoint discovery"
```

---

### Task 4: 让 CLI hermes rollout 在注入环境下跑通（无需真实安装）

**Files:**
- Modify: `tests/test_cli_runtime_commands.py`
- (Optional) Modify: `hermes_agentic_rl/cli/main.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_cli_runtime_commands.py` 追加：

```python
import importlib
import types


def test_cli_rollout_works_with_hermes_when_entrypoint_is_injected(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    module = types.ModuleType("hermes_agent.environments.agent_loop")

    class InjectedHermesLoop:
        async def run(self, prompt: str):
            return {
                "messages": [{"role": "assistant", "content": "done"}],
                "tool_calls": [[{"name": "write_file", "arguments": {"path": "x.txt"}}]],
                "tool_results": [[{"ok": True}]],
                "final_output": "done",
                "finished_naturally": True,
                "turns_used": 1,
            }

    module.HermesAgentLoop = lambda **kwargs: InjectedHermesLoop()
    monkeypatch.setitem(importlib.sys.modules, "hermes_agent.environments.agent_loop", module)

    # 伪造 hermes_agent 顶层模块，避免 adapter 判定“未安装”
    monkeypatch.setitem(importlib.sys.modules, "hermes_agent", types.ModuleType("hermes_agent"))

    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps({"task_id": "task-1", "instruction": "Create x.txt", "expected_output": "done"}),
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
    output_path = tmp_path / "trajectory.json"
    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "rollout", "--config", str(config_path), "--output", str(output_path)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "trajectory saved" in captured.out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["runtime"]["runtime"] == "hermes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli_runtime_commands.py::test_cli_rollout_works_with_hermes_when_entrypoint_is_injected -v`
Expected: FAIL before Task 3完成（现在应该会在 Task 3 后变为绿）

- [ ] **Step 3: Make minimal adjustments if needed**

如果失败原因是 CLI 在 hermes integration 下仍返回 runtime unavailable：

- 确认 `HermesRuntimeAdapter._default_importer()` 使用的是 `import hermes_agent`
- 测试里已经注入了 `hermes_agent` 顶层模块，应当可通过
- 若仍失败，调整 `HermesRuntimeAdapter` 的 importer 使用 `importlib.import_module("hermes_agent")`（允许模块注入）

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cli_runtime_commands.py::test_cli_rollout_works_with_hermes_when_entrypoint_is_injected -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli_runtime_commands.py hermes_agentic_rl/runtime/hermes_adapter.py hermes_agentic_rl/runtime/hermes_entrypoints.py hermes_agentic_rl/runtime/hermes_wrapper.py
git commit -m "test: add injected hermes cli rollout integration"
```

---

### Task 5: 全量回归

**Files:**
- None (test-only)

- [ ] **Step 1: Run focused suite**

Run:
```bash
python3 -m pytest \
  tests/test_hermes_entrypoints.py \
  tests/test_hermes_runtime_integration.py \
  tests/test_cli_runtime_commands.py \
  tests/test_runtime_adapter.py \
  -v
```
Expected: PASS

- [ ] **Step 2: Run full repo suite**

Run: `python3 -m pytest tests -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests
git commit -m "test: add phase3 hermes runtime wiring coverage"
```

---

## 自检结果

### Spec coverage
- 入口探测：Task 1
- wrapper 统一协议：Task 2
- adapter 真正 build loop：Task 3
- CLI hermes rollout（注入式集成）：Task 4
- 全量回归：Task 5

### Placeholder scan
- 无 TBD/TODO
- 每一步给出具体测试、具体实现代码、具体命令

### Type consistency
- wrapper 输出严格符合当前 runtime 统一协议字段集合
- CLI 继续复用 `HermesRuntimeAdapter`，不引入新的特殊路径

## 执行交接

Plan complete and saved to `docs/superpowers/plans/2026-05-07-hermes-agentic-rl-phase3-hermes-runtime.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
