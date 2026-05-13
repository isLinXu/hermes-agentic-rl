# CLI Overrides + Workdir Clean Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `train` 增加 `--dataset/--export/--workdir-clean`，实现安全清理 workdir_base 并支持 CLI 覆盖配置，提升可操作性与实验复现能力。

**Architecture:** 在 `hermes_agentic_rl/cli/main.py` 的 argparse 增加新参数；在 `_run_train` 中将 CLI 覆盖注入到 config 的 dataset/export/workdir_base；实现一个安全的 `_clean_workdir_base()`，仅允许清理 `base_cwd/outputs/` 之下的 workdir_base；新增 2-3 个 pytest 用例覆盖覆盖优先级与清理安全。

**Tech Stack:** Python、pytest、pathlib、shutil

---

## 文件结构

修改：
- `hermes_agentic_rl/cli/main.py`

新增：
- `tests/test_train_cli_overrides.py`
- `tests/test_train_workdir_clean.py`

---

### Task 1: TDD — CLI dataset/export 覆盖（先红后绿）

**Files:**
- Create: `tests/test_train_cli_overrides.py`
- Modify: `hermes_agentic_rl/cli/main.py`

- [ ] **Step 1: Write failing test**

创建 `tests/test_train_cli_overrides.py`：

```python
import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_train_cli_overrides_dataset_and_export(tmp_path: Path, monkeypatch):
    # dataset A (config default) has 1 sample
    dataset_a = tmp_path / "a.jsonl"
    dataset_a.write_text(
        json.dumps(
            {
                "task_id": "a",
                "instruction": "Create x.txt and write hello",
                "expected_output": "done",
                "expected_files": [{"path": "x.txt", "equals": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # dataset B (CLI override) has 2 samples
    dataset_b = tmp_path / "b.jsonl"
    dataset_b.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": f"b{i}",
                        "instruction": "Create x.txt and write hello",
                        "expected_output": "done",
                        "expected_files": [{"path": "x.txt", "equals": "hello"}],
                    }
                )
                for i in range(1, 3)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Prepare per-item workdir with files so verifier passes (fake runtime)
    workdir_base = tmp_path / "outputs" / "workdirs"
    for tid in ["b1", "b2"]:
        d = workdir_base / tid
        d.mkdir(parents=True, exist_ok=True)
        (d / "x.txt").write_text("hello", encoding="utf-8")

    export_override = tmp_path / "outputs" / "export.jsonl"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset_a}\n"
            "reward:\n"
            "  aggregator: weighted_sum\n"
            "  components:\n"
            "    - name: filesystem_verifier_reward\n"
            "      weight: 1.0\n"
            "trainer:\n"
            f"  export_training_path: {tmp_path/'outputs'/'default.jsonl'}\n"
            f"  workdir_base: {workdir_base}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "train",
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_b),
            "--export",
            str(export_override),
            "--limit",
            "2",
            "--overwrite",
            "--min-verifier-pass-ratio",
            "1.0",
        ],
    )

    assert main() == 0
    lines = [l for l in export_override.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
```

- [ ] **Step 2: Run test (RED)**

Run: `python3 -m pytest tests/test_train_cli_overrides.py -v`
Expected: FAIL because CLI args not supported (`unrecognized arguments: --dataset --export`)

- [ ] **Step 3: Implement minimal CLI support**

在 `build_parser()` 增加：
- `--dataset`
- `--export`
- `--workdir-clean`（先留空实现也可，但 parser 先支持）

在 `main()` 调用 `_run_train()` 传入 dataset/export/workdir_clean。

在 `_run_train()` 里：
- dataset_path 使用 CLI dataset（若提供）否则用 config
- export_path 使用 CLI export（若提供）否则用 config

- [ ] **Step 4: Run test (GREEN)**

Run: `python3 -m pytest tests/test_train_cli_overrides.py -v`
Expected: PASS

---

### Task 2: TDD — workdir-clean 安全清理（先红后绿）

**Files:**
- Create: `tests/test_train_workdir_clean.py`
- Modify: `hermes_agentic_rl/cli/main.py`

- [ ] **Step 1: Write failing tests**

创建 `tests/test_train_workdir_clean.py`：

```python
import json
from pathlib import Path

import pytest

from hermes_agentic_rl.cli.main import main


def test_workdir_clean_removes_task_dirs_under_outputs(tmp_path: Path, monkeypatch):
    workdir_base = tmp_path / "outputs" / "workdirs" / "hermes"
    (workdir_base / "task-1").mkdir(parents=True, exist_ok=True)
    (workdir_base / "task-1" / "x.txt").write_text("x", encoding="utf-8")

    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create x.txt and write hello",
                "expected_output": "done",
                "expected_files": [{"path": "x.txt", "equals": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    export_path = tmp_path / "outputs" / "out.jsonl"
    config = tmp_path / "config.yaml"
    config.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset}\n"
            "reward:\n"
            "  aggregator: weighted_sum\n"
            "  components:\n"
            "    - name: filesystem_verifier_reward\n"
            "      weight: 1.0\n"
            "trainer:\n"
            f"  export_training_path: {export_path}\n"
            f"  workdir_base: {workdir_base}\n"
        ),
        encoding="utf-8",
    )

    # also prepare fresh verifier file after clean
    (workdir_base / "task-1").mkdir(parents=True, exist_ok=True)
    (workdir_base / "task-1" / "x.txt").write_text("hello", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "train",
            "--config",
            str(config),
            "--workdir-clean",
            "--limit",
            "1",
            "--overwrite",
            "--min-verifier-pass-ratio",
            "1.0",
        ],
    )

    assert main() == 0
    assert workdir_base.exists()


def test_workdir_clean_refuses_to_delete_outside_outputs(tmp_path: Path, monkeypatch):
    workdir_base = tmp_path / "evil"
    workdir_base.mkdir(parents=True, exist_ok=True)
    (workdir_base / "task-1").mkdir(parents=True, exist_ok=True)

    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("{}", encoding="utf-8")

    export_path = tmp_path / "outputs" / "out.jsonl"
    config = tmp_path / "config.yaml"
    config.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset}\n"
            "trainer:\n"
            f"  export_training_path: {export_path}\n"
            f"  workdir_base: {workdir_base}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "train",
            "--config",
            str(config),
            "--workdir-clean",
        ],
    )

    assert main() != 0
```

- [ ] **Step 2: Run tests (RED)**

Run: `python3 -m pytest tests/test_train_workdir_clean.py -v`
Expected: FAIL (workdir-clean 尚未实现)

- [ ] **Step 3: Implement `_clean_workdir_base(base_cwd, workdir_base)`**

实现逻辑：
- 若 workdir_base 为空：报错 return non-zero
- 解析为绝对路径：若非 absolute 则 join base_cwd
- 仅允许 `workdir_base` 位于 `(base_cwd / "outputs")` 下
- 遍历 workdir_base 下的子目录，逐个 `shutil.rmtree`

- [ ] **Step 4: Run tests (GREEN)**

Run: `python3 -m pytest tests/test_train_workdir_clean.py -v`
Expected: PASS

---

### Task 3: 回归验证

**Files:**
- None

- [ ] **Step 1: Run full suite**

Run: `python3 -m pytest tests -q`
Expected: PASS

- [ ] **Step 2: Hermes smoke**

Run（示例）：
```bash
LKEAP_API_KEY=... .venv311/bin/hermes-agentic-rl train \
  --config configs/terminal_grpo_hermes_mixed.yaml \
  --dataset data/terminal_tasks_50_mixed.jsonl \
  --export outputs/train_samples_hermes_mixed_cli.jsonl \
  --limit 10 --overwrite --min-verifier-pass-ratio 0.6 \
  --workdir-clean
```

Expected:
- 退出码 0
- `outputs/workdirs/...` 被清空后重新生成
- export JSONL 生成 10 行

---

## 自检
- 所有新功能都有单测
- 清理功能有严格的 outputs/ 目录约束，避免误删
- CLI 覆盖优先于 config，便于快速试验不同数据集与输出文件
