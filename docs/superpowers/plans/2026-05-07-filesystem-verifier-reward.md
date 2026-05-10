# FileSystem Verifier Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 引入 `expected_files` 数据字段与 `FileSystemVerifierReward`，用文件系统校验强化训练数据质量；同时让 `train` 输出 verifier 统计并支持质量门槛（min verifier pass ratio），最后在真实 hermes 配置下验证产出。

**Architecture:** 新增 reward 组件文件 `filesystem_verifier_reward.py`（仅依赖 pathlib/标准库），按 item.expected_files 执行存在性与内容校验；在 CLI `train` 统计中增加 `verifier_pass_ratio` 并支持 `--min-verifier-pass-ratio`（同时支持 config 覆盖）；更新样例数据 `data/minimal_terminal_tasks.jsonl` 以包含 expected_files；新增单元测试覆盖 verifier 行为与 CLI 统计字段；最后用 hermes 真实 train 回归确认输出。

**Tech Stack:** Python、pytest、pathlib、json

---

## 文件结构

新增：
- `hermes_agentic_rl/rewards/filesystem_verifier_reward.py`
- `tests/test_filesystem_verifier_reward.py`

修改：
- `hermes_agentic_rl/cli/main.py`
- `data/minimal_terminal_tasks.jsonl`
- `configs/terminal_grpo.yaml`（可选：加入 reward 配置说明）
- `configs/terminal_grpo_hermes.yaml`（加入 reward/门槛配置）
- `README.md`（可选：补充 expected_files 格式说明）

---

### Task 1: 新增 FileSystemVerifierReward（先红后绿）

**Files:**
- Create: `tests/test_filesystem_verifier_reward.py`
- Create: `hermes_agentic_rl/rewards/filesystem_verifier_reward.py`

- [ ] **Step 1: Write the failing tests**

创建 `tests/test_filesystem_verifier_reward.py`：

```python
import asyncio
from pathlib import Path

from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.filesystem_verifier_reward import FileSystemVerifierReward


def _empty_traj() -> Trajectory:
    return Trajectory(
        task_id="t",
        prompt="p",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )


def test_verifier_skips_when_no_expected_files(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reward = FileSystemVerifierReward(weight=0.6)
    result = asyncio.run(reward.evaluate(item={}, trajectory=_empty_traj(), tool_context=None))

    assert result.weight == 0.0
    assert result.score == 0.0


def test_verifier_fails_when_file_missing(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "hello.txt", "equals": "hello"}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 0.0
    assert result.metadata["checked_files"] == 1
    assert result.metadata["passed_files"] == 0


def test_verifier_passes_on_equals_match(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")

    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "hello.txt", "equals": "hello"}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 1.0
    assert result.metadata["passed_files"] == 1


def test_verifier_passes_on_contains_match(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "todo.txt").write_text("buy milk\nand eggs", encoding="utf-8")

    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "todo.txt", "contains": "buy milk"}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 1.0
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python3 -m pytest tests/test_filesystem_verifier_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: hermes_agentic_rl.rewards.filesystem_verifier_reward`

- [ ] **Step 3: Minimal implementation**

创建 `hermes_agentic_rl/rewards/filesystem_verifier_reward.py`：

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward


class FileSystemVerifierReward(BaseReward):
    name = "filesystem_verifier_reward"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del trajectory, tool_context

        expected_files = item.get("expected_files")
        if not expected_files:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no expected_files provided; skipped",
                weight=0.0,
                metadata={"checked_files": 0, "passed_files": 0, "failures": []},
            )

        checked = 0
        passed = 0
        failures: list[dict[str, str]] = []
        root = Path.cwd()

        for spec in expected_files:
            checked += 1
            path = str(spec.get("path", "")).strip()
            if not path:
                failures.append({"path": "", "reason": "missing path"})
                continue
            full_path = root / path
            if not full_path.exists():
                failures.append({"path": path, "reason": "file missing"})
                continue
            try:
                content = full_path.read_text(encoding="utf-8")
            except Exception as exc:
                failures.append({"path": path, "reason": f"read failed: {exc}"})
                continue

            equals = spec.get("equals")
            contains = spec.get("contains")
            if equals is not None:
                if content.strip() == str(equals).strip():
                    passed += 1
                else:
                    failures.append({"path": path, "reason": "content mismatch (equals)"})
                continue
            if contains is not None:
                if str(contains) in content:
                    passed += 1
                else:
                    failures.append({"path": path, "reason": "content mismatch (contains)"})
                continue

            failures.append({"path": path, "reason": "no equals/contains rule provided"})

        score = 1.0 if checked > 0 and passed == checked else 0.0
        reason = "all expected_files verified" if score == 1.0 else "file verification failed"
        return RewardResult(
            name=self.name,
            score=score,
            reason=reason,
            weight=self.weight,
            metadata={"checked_files": checked, "passed_files": passed, "failures": failures},
        )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python3 -m pytest tests/test_filesystem_verifier_reward.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hermes_agentic_rl/rewards/filesystem_verifier_reward.py tests/test_filesystem_verifier_reward.py
git commit -m "feat: add filesystem verifier reward"
```

---

### Task 2: 更新样例数据集 expected_files

**Files:**
- Modify: `data/minimal_terminal_tasks.jsonl`

- [ ] **Step 1: Update dataset**

将两条样例补齐 expected_files：

- task-1: `hello.txt == hello`
- task-2: `notes/todo.txt == buy milk`

- [ ] **Step 2: Quick sanity**

Run: `python3 -m pytest tests/test_samples.py -v`
Expected: PASS（不强制校验 expected_files 内容，但保证格式仍可读）

---

### Task 3: 将 verifier 接入 reward manager（配置驱动）

**Files:**
- Modify: `hermes_agentic_rl/cli/main.py`
- Modify: `configs/terminal_grpo_hermes.yaml`

- [ ] **Step 1: Write failing test**

扩展 `tests/test_train_multisample.py` 或新增 `tests/test_train_quality_stats.py`：
验证 `train` 输出包含 `verifier_pass_ratio` 字段（在 expected_files 存在且通过的情况下）。

- [ ] **Step 2: Implement**

1) `_build_reward_manager(config)` 改为：
- 读取 `config.get("reward", {}).get("components", ...)`
- 默认组件列表包含 `OutcomeReward` / `ToolcallReward` / `FileSystemVerifierReward`

2) `train summary` 增加 verifier 统计：
- 仅统计 `item.expected_files` 存在的样本
- `verifier_pass_ratio = passed / total`

3) 新增 CLI 参数 `--min-verifier-pass-ratio` 并支持 config `trainer.min_verifier_pass_ratio`

- [ ] **Step 3: Run tests**

Run: `python3 -m pytest tests/test_train_multisample.py -v`
Expected: PASS

---

### Task 4: 真实 hermes 回归验证

**Files:**
- None

- [ ] **Step 1: Run full suite**

Run: `python3 -m pytest tests -v`
Expected: PASS

- [ ] **Step 2: Run hermes train**

Run（示例）：
```bash
mkdir -p outputs
LKEAP_API_KEY=... .venv311/bin/hermes-agentic-rl train \
  --config configs/terminal_grpo_hermes.yaml \
  --limit 2 --overwrite \
  --min-verifier-pass-ratio 1.0
```

Expected:
- 退出码 0
- summary 中 `verifier_pass_ratio=1.0000`
- 导出样本中包含 `filesystem_verifier_reward` component

---

## 自检结果
- 只新增一个 verifier reward，不引入外部依赖
- 用 expected_files 明确校验目标，避免解析 instruction 的不稳定性
- 训练输出增加统计与质量门槛，便于规模化数据采集
