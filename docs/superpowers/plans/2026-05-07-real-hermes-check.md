# Real Hermes Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个可执行的真实 Hermes 环境自检脚本和配套文档，用于在 Python 3.11+ 环境中检查 `hermes-agent` 安装、公开入口、adapter 构建条件，以及样例配置/数据是否齐备。

**Architecture:** 将自检逻辑分成可测试的纯函数和一个轻量命令行脚本两层。测试主要覆盖检查函数和结果汇总，不依赖真实联网或真实 `hermes-agent` 安装；脚本只负责组织检查项、打印人类可读摘要，并输出 JSON 结果。文档单独说明如何在 Python 3.11+ 环境中使用该脚本以及如何在检查通过后执行真实 rollout。

**Tech Stack:** Python 3.10+/3.11+、`argparse`、`json`、`pathlib`、`importlib`、`pytest`

---

## 文件结构

本计划会创建以下文件：

- `scripts/check_real_hermes.py`
- `docs/real-hermes-check.md`
- `tests/test_real_hermes_check.py`

本计划可能修改以下文件：

- `README.md`

---

### Task 1: 提取并测试自检核心函数

**Files:**
- Create: `tests/test_real_hermes_check.py`
- Create: `scripts/check_real_hermes.py`

- [ ] **Step 1: Write the failing test**

先创建 `tests/test_real_hermes_check.py`，只测试纯函数，不测试终端打印：

```python
from pathlib import Path

from scripts.check_real_hermes import (
    build_overall_status,
    check_python_version,
    check_path_exists,
)


def test_check_python_version_flags_python310_as_error():
    result = check_python_version((3, 10, 12))

    assert result["status"] == "error"
    assert "3.11+" in result["details"]


def test_check_python_version_flags_python311_as_ok():
    result = check_python_version((3, 11, 9))

    assert result["status"] == "ok"


def test_check_path_exists_reports_missing_file(tmp_path: Path):
    result = check_path_exists(tmp_path / "missing.txt", label="sample_config")

    assert result["status"] == "warning"
    assert "missing" in result["details"].lower()


def test_build_overall_status_uses_error_over_warning():
    checks = {
        "python_version": {"status": "ok", "details": "ok"},
        "adapter_build": {"status": "error", "details": "broken"},
        "sample_dataset": {"status": "warning", "details": "missing"},
    }

    assert build_overall_status(checks) == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_real_hermes_check.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.check_real_hermes'`

- [ ] **Step 3: Write minimal implementation**

创建 `scripts/check_real_hermes.py`：

```python
from __future__ import annotations

from pathlib import Path


def check_python_version(version_info: tuple[int, int, int]) -> dict[str, str]:
    major, minor, patch = version_info
    if (major, minor) >= (3, 11):
        return {
            "status": "ok",
            "details": f"Python version is {major}.{minor}.{patch}",
        }
    return {
        "status": "error",
        "details": f"Python {major}.{minor}.{patch} is too old; hermes-agent requires Python 3.11+",
    }


def check_path_exists(path: Path, label: str) -> dict[str, str]:
    if path.exists():
        return {"status": "ok", "details": f"{label} exists: {path}"}
    return {"status": "warning", "details": f"{label} missing: {path}"}


def build_overall_status(checks: dict[str, dict[str, str]]) -> str:
    statuses = [check["status"] for check in checks.values()]
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_real_hermes_check.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/check_real_hermes.py tests/test_real_hermes_check.py
git commit -m "feat: add real hermes check core functions"
```

### Task 2: 检查 run_agent 导入、AIAgent 符号、入口探测

**Files:**
- Modify: `tests/test_real_hermes_check.py`
- Modify: `scripts/check_real_hermes.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_real_hermes_check.py` 追加：

```python
import types

from scripts.check_real_hermes import (
    check_ai_agent_symbol,
    check_run_agent_import,
)


def test_check_run_agent_import_reports_missing_module(monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "run_agent", raising=False)

    result = check_run_agent_import(importer=lambda: (_ for _ in ()).throw(ImportError("missing run_agent")))

    assert result["status"] == "error"
    assert "missing run_agent" in result["details"]


def test_check_ai_agent_symbol_reports_ok_when_present():
    module = types.SimpleNamespace(AIAgent=object)
    result = check_ai_agent_symbol(module)

    assert result["status"] == "ok"
    assert "AIAgent" in result["details"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_real_hermes_check.py::test_check_run_agent_import_reports_missing_module -v`
Expected: FAIL because functions do not exist

- [ ] **Step 3: Write minimal implementation**

扩展 `scripts/check_real_hermes.py`：

```python
import importlib
from typing import Any, Callable

from hermes_agentic_rl.runtime.hermes_entrypoints import (
    HermesEntrypointNotFoundError,
    find_hermes_entrypoint,
)
```

```python
def default_run_agent_importer() -> Any:
    return importlib.import_module("run_agent")


def check_run_agent_import(importer: Callable[[], Any] | None = None) -> dict[str, str | Any]:
    importer = importer or default_run_agent_importer
    try:
        module = importer()
        return {"status": "ok", "details": "run_agent importable", "module": module}
    except ImportError as exc:
        return {"status": "error", "details": str(exc), "module": None}


def check_ai_agent_symbol(module: Any) -> dict[str, str]:
    if module is None:
        return {"status": "error", "details": "run_agent module unavailable"}
    if hasattr(module, "AIAgent"):
        return {"status": "ok", "details": "AIAgent symbol found"}
    return {"status": "error", "details": "AIAgent symbol missing"}


def check_entrypoint_detection() -> dict[str, str]:
    try:
        entrypoint = find_hermes_entrypoint()
        return {
            "status": "ok",
            "details": f"detected entrypoint: {entrypoint.module_name}:{entrypoint.attr_name}",
        }
    except HermesEntrypointNotFoundError as exc:
        return {"status": "error", "details": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python3 -m pytest tests/test_real_hermes_check.py::test_check_run_agent_import_reports_missing_module -v
python3 -m pytest tests/test_real_hermes_check.py::test_check_ai_agent_symbol_reports_ok_when_present -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/check_real_hermes.py tests/test_real_hermes_check.py
git commit -m "feat: add run_agent and AIAgent checks"
```

### Task 3: 增加 adapter build 检查与结果汇总

**Files:**
- Modify: `tests/test_real_hermes_check.py`
- Modify: `scripts/check_real_hermes.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_real_hermes_check.py` 追加：

```python
from scripts.check_real_hermes import check_adapter_build


def test_check_adapter_build_reports_error_when_python_too_old():
    result = check_adapter_build(
        version_info=(3, 10, 12),
        adapter_builder=lambda: None,
    )

    assert result["status"] == "error"
    assert "Python 3.11+" in result["details"]


def test_check_adapter_build_reports_ok_when_builder_succeeds():
    result = check_adapter_build(
        version_info=(3, 11, 9),
        adapter_builder=lambda: object(),
    )

    assert result["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_real_hermes_check.py::test_check_adapter_build_reports_error_when_python_too_old -v`
Expected: FAIL because function does not exist

- [ ] **Step 3: Write minimal implementation**

扩展 `scripts/check_real_hermes.py`：

```python
from hermes_agentic_rl.runtime.hermes_adapter import HermesRuntimeAdapter
```

```python
def check_adapter_build(
    version_info: tuple[int, int, int],
    adapter_builder=None,
) -> dict[str, str]:
    major, minor, patch = version_info
    if (major, minor) < (3, 11):
        return {
            "status": "error",
            "details": f"adapter build skipped: Python {major}.{minor}.{patch} does not satisfy Python 3.11+",
        }

    adapter_builder = adapter_builder or (lambda: HermesRuntimeAdapter().build_agent_loop({"runtime": {"integration": "hermes"}}))
    try:
        adapter_builder()
        return {"status": "ok", "details": "Hermes runtime adapter build succeeded"}
    except Exception as exc:
        return {"status": "warning", "details": f"adapter build failed: {exc}"}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python3 -m pytest tests/test_real_hermes_check.py::test_check_adapter_build_reports_error_when_python_too_old -v
python3 -m pytest tests/test_real_hermes_check.py::test_check_adapter_build_reports_ok_when_builder_succeeds -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/check_real_hermes.py tests/test_real_hermes_check.py
git commit -m "feat: add adapter build check"
```

### Task 4: 让脚本输出人类可读摘要与 JSON 结果

**Files:**
- Modify: `tests/test_real_hermes_check.py`
- Modify: `scripts/check_real_hermes.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_real_hermes_check.py` 追加：

```python
from scripts.check_real_hermes import format_summary_lines


def test_format_summary_lines_contains_status_prefixes():
    checks = {
        "python_version": {"status": "ok", "details": "Python version is 3.11.9"},
        "sample_dataset": {"status": "warning", "details": "sample dataset missing"},
    }

    lines = format_summary_lines(checks)

    assert any(line.startswith("[OK]") for line in lines)
    assert any(line.startswith("[WARN]") for line in lines)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_real_hermes_check.py::test_format_summary_lines_contains_status_prefixes -v`
Expected: FAIL because function does not exist

- [ ] **Step 3: Write minimal implementation**

继续扩展 `scripts/check_real_hermes.py`：

```python
import json
import sys
```

```python
def format_summary_lines(checks: dict[str, dict[str, str]]) -> list[str]:
    mapping = {"ok": "[OK]", "warning": "[WARN]", "error": "[ERROR]"}
    lines: list[str] = []
    for name, result in checks.items():
        prefix = mapping.get(result["status"], "[INFO]")
        lines.append(f"{prefix} {name}: {result['details']}")
    return lines


def run_checks() -> dict:
    workspace_root = Path(__file__).resolve().parent.parent
    python_result = check_python_version(sys.version_info[:3])
    run_agent_result = check_run_agent_import()
    ai_agent_result = check_ai_agent_symbol(run_agent_result.get("module"))
    entrypoint_result = check_entrypoint_detection()
    adapter_result = check_adapter_build(sys.version_info[:3])
    config_result = check_path_exists(workspace_root / "configs" / "terminal_grpo_hermes.yaml", "sample_config")
    dataset_result = check_path_exists(workspace_root / "data" / "minimal_terminal_tasks.jsonl", "sample_dataset")

    checks = {
        "python_version": python_result,
        "run_agent_import": {k: v for k, v in run_agent_result.items() if k != "module"},
        "ai_agent_symbol": ai_agent_result,
        "entrypoint_detection": entrypoint_result,
        "adapter_build": adapter_result,
        "sample_config": config_result,
        "sample_dataset": dataset_result,
    }
    return {
        "overall_status": build_overall_status(checks),
        "checks": checks,
    }


def main() -> int:
    report = run_checks()
    for line in format_summary_lines(report["checks"]):
        print(line)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_real_hermes_check.py::test_format_summary_lines_contains_status_prefixes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/check_real_hermes.py tests/test_real_hermes_check.py
git commit -m "feat: add real hermes check reporting output"
```

### Task 5: 编写用户文档

**Files:**
- Create: `docs/real-hermes-check.md`
- Modify: `README.md`

- [ ] **Step 1: Write the failing documentation-oriented test**

在 `tests/test_real_hermes_check.py` 追加一个最小存在性测试：

```python
def test_real_hermes_docs_exist():
    workspace_root = Path(__file__).resolve().parent.parent

    assert (workspace_root / "docs" / "real-hermes-check.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_real_hermes_check.py::test_real_hermes_docs_exist -v`
Expected: FAIL because the doc file does not exist

- [ ] **Step 3: Write minimal implementation**

创建 `docs/real-hermes-check.md`：

```md
# Real Hermes Check

## 目的

用于在 Python 3.11+ 环境中检查：

- `hermes-agent` 是否已安装
- `run_agent` 是否可导入
- `AIAgent` 是否存在
- 当前仓库的 `HermesRuntimeAdapter` 是否具备真实入口探测条件
- hermes 配置与示例数据是否存在

## 前置条件

- Python 3.11+
- 已安装本仓库
- 如需真实 Hermes 检查，已安装：

```bash
python -m pip install "git+https://github.com/NousResearch/hermes-agent.git"
```

## 运行方式

```bash
python scripts/check_real_hermes.py
```

## 输出解释

- `[OK]`：该项通过
- `[WARN]`：该项未完全通过，但不一定阻止你继续手动检查
- `[ERROR]`：该项是硬阻塞

## 常见失败

### Python 版本不足

如果输出提示 Python 版本低于 3.11，说明当前环境不能真实安装 hermes-agent。

### run_agent 无法导入

说明 hermes-agent 尚未安装，或当前虚拟环境不正确。

### adapter build failed

说明入口存在，但当前环境可能缺少 provider 配置、依赖或其他运行条件。

## 自检通过后执行真实 rollout

```bash
python -m hermes_agentic_rl.cli.main rollout --config configs/terminal_grpo_hermes.yaml --output outputs/hermes_trajectory.json
```
```

在 `README.md` 增加一节短链接：

```md
## 真实 Hermes 自检

详见 `docs/real-hermes-check.md`。
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_real_hermes_check.py::test_real_hermes_docs_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/real-hermes-check.md README.md tests/test_real_hermes_check.py
git commit -m "docs: add real hermes check guide"
```

### Task 6: 全量回归

**Files:**
- None

- [ ] **Step 1: Run focused tests**

Run:
```bash
python3 -m pytest tests/test_real_hermes_check.py tests/test_samples.py -v
```
Expected: PASS

- [ ] **Step 2: Run full suite**

Run:
```bash
python3 -m pytest tests -v
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add scripts docs tests README.md
git commit -m "feat: add real hermes environment self-check deliverable"
```

## 自检结果

### Spec coverage
- 自检脚本：Task 1、Task 2、Task 3、Task 4
- 配套文档：Task 5
- 全量回归：Task 6

### Placeholder scan
- 未使用 `TODO`、`TBD`、`implement later`
- 所有任务都有具体代码与具体命令

### Type consistency
- 检查结果始终使用 `{"status": "...", "details": "..."}` 结构
- `run_checks()` 最终统一输出 `{"overall_status": ..., "checks": ...}`

## 执行交接

Plan complete and saved to `docs/superpowers/plans/2026-05-07-real-hermes-check.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
