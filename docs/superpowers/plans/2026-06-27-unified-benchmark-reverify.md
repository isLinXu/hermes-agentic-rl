# Unified Benchmark Reverify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `scripts/reverify_real_benchmark.py` 在本地 parquet 缺失时自动切换到 Hugging Face 数据源，并保持现有 parquet 工作流不变。

**Architecture:** 在脚本层引入“数据源解析 -> 运行时配置物化 -> 命令构建”三段式流程。HF 模式通过临时 YAML 副本覆盖 `environment` 的数据字段，并重写 `benchmark_suite` 的 `config_path`，不改正式配置文件与评估逻辑。

**Tech Stack:** Python 3.11+, `argparse`, `pathlib`, `tempfile`, `yaml.safe_load` / `yaml.safe_dump`, pytest

---

## File map

- Modify: `scripts/reverify_real_benchmark.py`
  - 新增 HF CLI 参数、数据源解析、运行时配置物化与命令构建重构。
- Modify: `tests/test_reverify_real_benchmark_script.py`
  - 增加 HF fallback、配置重写与命令构建回归测试。
- Read-only reference: `configs/hermes_reasoning_traces_eval_rl.yaml`
  - 作为 HF 模式下生成临时 eval 配置的模板来源。
- Read-only reference: `configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml`
  - 作为 HF 模式下生成临时 stage2 eval 配置的模板来源。
- Read-only reference: `configs/benchmark_suite.yaml`
  - 作为 HF 模式下生成临时 benchmark-suite 配置的模板来源。

---

### Task 1: 先固定 HF fallback 的外部行为

**Files:**
- Modify: `tests/test_reverify_real_benchmark_script.py`
- Test: `tests/test_reverify_real_benchmark_script.py`

- [ ] **Step 1: 写第一个失败测试，固定“默认 parquet 缺失时 dry-run 自动切到 HF”**

在 `tests/test_reverify_real_benchmark_script.py` 追加以下测试：

```python
def test_main_dry_run_falls_back_to_hf_when_default_parquet_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import scripts.reverify_real_benchmark as module

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reverify_real_benchmark.py",
            "--dry-run",
            "--hf-repo-id",
            "lambda/hermes-agent-reasoning-traces",
            "--hf-config-name",
            "kimi",
            "--hf-split",
            "train",
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "数据源模式: hf" in captured.out
    assert "lambda/hermes-agent-reasoning-traces" in captured.out
```

- [ ] **Step 2: 运行单测，确认它先红掉**

Run:

```bash
pytest tests/test_reverify_real_benchmark_script.py::test_main_dry_run_falls_back_to_hf_when_default_parquet_missing -v
```

Expected:

- FAIL
- 当前失败原因应是脚本仍然把“默认 parquet 缺失”当作错误并返回 `2`

- [ ] **Step 3: 再写一个失败测试，固定“显式传 --dataset 时 parquet 优先级更高”**

在同一文件追加：

```python
def test_explicit_dataset_keeps_parquet_priority(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import scripts.reverify_real_benchmark as module

    dataset = tmp_path / "external" / "train.parquet"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"parquet")

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reverify_real_benchmark.py",
            "--dry-run",
            "--dataset",
            str(dataset),
            "--hf-repo-id",
            "lambda/hermes-agent-reasoning-traces",
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "数据源模式: parquet" in captured.out
    assert str(dataset) in captured.out
```

- [ ] **Step 4: 运行这两个测试，确认第二个也先失败或输出不符合预期**

Run:

```bash
pytest tests/test_reverify_real_benchmark_script.py::test_main_dry_run_falls_back_to_hf_when_default_parquet_missing tests/test_reverify_real_benchmark_script.py::test_explicit_dataset_keeps_parquet_priority -v
```

Expected:

- 至少有 1 个 FAIL
- 失败点应集中在 dry-run 输出与数据源选择逻辑，而不是语法错误

- [ ] **Step 5: 提交测试骨架**

```bash
git add tests/test_reverify_real_benchmark_script.py
git commit -m "test: lock data source selection for benchmark reverify"
```

---

### Task 2: 固定运行时配置物化与命令构建行为

**Files:**
- Modify: `tests/test_reverify_real_benchmark_script.py`
- Test: `tests/test_reverify_real_benchmark_script.py`

- [ ] **Step 1: 写失败测试，固定 HF 模式下临时 eval 配置不再保留 `dataset_path`**

在 `tests/test_reverify_real_benchmark_script.py` 追加：

```python
def test_materialize_runtime_configs_builds_hf_eval_configs_without_dataset_path(
    tmp_path: Path,
) -> None:
    import scripts.reverify_real_benchmark as module

    plan = module.RuntimePlan(
        mode="hf",
        description="HF fallback",
        dataset_path=None,
        dataset_override=None,
        hf_repo_id="lambda/hermes-agent-reasoning-traces",
        hf_config_name="kimi",
        hf_split="train",
        hf_streaming=True,
        hf_rows_api_only=True,
    )

    runtime = module.materialize_runtime_configs(tmp_path, plan)
    payload = module._load_yaml(runtime.eval_config_path)

    assert "dataset_path" not in payload["environment"]
    assert payload["environment"]["repo_id"] == "lambda/hermes-agent-reasoning-traces"
    assert payload["environment"]["config_name"] == "kimi"
    assert payload["environment"]["split"] == "train"
    assert payload["environment"]["streaming"] is True
    assert payload["environment"]["rows_api_only"] is True
```

- [ ] **Step 2: 写失败测试，固定临时 benchmark-suite 配置会重写 `config_path`**

继续追加：

```python
def test_materialize_runtime_configs_rewrites_benchmark_suite_config_paths(
    tmp_path: Path,
) -> None:
    import scripts.reverify_real_benchmark as module

    plan = module.RuntimePlan(
        mode="hf",
        description="HF fallback",
        dataset_path=None,
        dataset_override=None,
        hf_repo_id="lambda/hermes-agent-reasoning-traces",
        hf_config_name="kimi",
        hf_split="train",
        hf_streaming=True,
        hf_rows_api_only=True,
    )

    runtime = module.materialize_runtime_configs(tmp_path, plan)
    payload = module._load_yaml(runtime.benchmark_suite_config_path)
    config_paths = [item["config_path"] for item in payload["benchmark_suite"]["benchmarks"]]

    assert str(runtime.eval_config_path) in config_paths or str(runtime.stage2_eval_config_path) in config_paths
    assert "configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml" not in config_paths
```

- [ ] **Step 3: 写失败测试，固定 `build_commands()` 使用运行时配置路径**

替换现有 `test_build_commands_covers_preflight_eval_and_suite`，改为：

```python
def test_build_commands_uses_runtime_config_paths() -> None:
    from pathlib import Path

    from scripts.reverify_real_benchmark import build_commands

    commands = build_commands(
        eval_config_path=Path("/tmp/eval.yaml"),
        stage2_eval_config_path=Path("/tmp/stage2.yaml"),
        benchmark_suite_config_path=Path("/tmp/suite.yaml"),
        include_preflight=True,
    )

    assert len(commands) == 4
    assert commands[0][-1] == "hermes-preflight"
    assert commands[1][-2:] == ["--config", "/tmp/eval.yaml"]
    assert commands[2][-2:] == ["--config", "/tmp/stage2.yaml"]
    assert commands[3][-2:] == ["--config", "/tmp/suite.yaml"]
```

- [ ] **Step 4: 运行新增的 3 个测试，确认先失败**

Run:

```bash
pytest tests/test_reverify_real_benchmark_script.py::test_materialize_runtime_configs_builds_hf_eval_configs_without_dataset_path tests/test_reverify_real_benchmark_script.py::test_materialize_runtime_configs_rewrites_benchmark_suite_config_paths tests/test_reverify_real_benchmark_script.py::test_build_commands_uses_runtime_config_paths -v
```

Expected:

- FAIL
- 失败原因应为 `RuntimePlan`、`materialize_runtime_configs()`、新 `build_commands()` 尚不存在

- [ ] **Step 5: 提交第二批测试**

```bash
git add tests/test_reverify_real_benchmark_script.py
git commit -m "test: define runtime config rewrite behavior"
```

---

### Task 3: 用最小实现让测试转绿

**Files:**
- Modify: `scripts/reverify_real_benchmark.py`
- Test: `tests/test_reverify_real_benchmark_script.py`

- [ ] **Step 1: 先补最小数据结构和 YAML 辅助函数**

在 `scripts/reverify_real_benchmark.py` 顶部引入：

```python
from dataclasses import dataclass
import tempfile

import yaml
```

并添加两个数据类与 YAML 辅助函数：

```python
@dataclass(slots=True)
class RuntimePlan:
    mode: str
    description: str
    dataset_path: Path | None
    dataset_override: Path | None
    hf_repo_id: str
    hf_config_name: str
    hf_split: str
    hf_streaming: bool
    hf_rows_api_only: bool


@dataclass(slots=True)
class RuntimeConfigs:
    eval_config_path: Path
    stage2_eval_config_path: Path
    benchmark_suite_config_path: Path
    temp_dir: Path | None = None


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
```

- [ ] **Step 2: 运行配置物化相关测试，确认失败形态往前推进**

Run:

```bash
pytest tests/test_reverify_real_benchmark_script.py::test_materialize_runtime_configs_builds_hf_eval_configs_without_dataset_path tests/test_reverify_real_benchmark_script.py::test_materialize_runtime_configs_rewrites_benchmark_suite_config_paths -v
```

Expected:

- 仍 FAIL
- 但失败应转为 `materialize_runtime_configs` 未定义或行为不完整

- [ ] **Step 3: 实现数据源解析与运行时配置物化**

在脚本中加入 CLI 参数：

```python
parser.add_argument("--hf-repo-id", default="lambda/hermes-agent-reasoning-traces")
parser.add_argument("--hf-config-name", default="kimi")
parser.add_argument("--hf-split", default="train")
parser.add_argument("--hf-streaming", action="store_true", default=True)
parser.add_argument("--hf-rows-api-only", action="store_true", default=True)
```

然后加入最小实现：

```python
def resolve_data_source(args: argparse.Namespace, repo_root: Path) -> RuntimePlan:
    default_dataset = repo_root / DEFAULT_DATASET_RELATIVE_PATH
    if args.dataset is not None:
        source = args.dataset.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"指定的数据文件不存在：{source}")
        return RuntimePlan(
            mode="parquet",
            description=f"显式数据文件：{source}",
            dataset_path=default_dataset,
            dataset_override=source,
            hf_repo_id=args.hf_repo_id,
            hf_config_name=args.hf_config_name,
            hf_split=args.hf_split,
            hf_streaming=args.hf_streaming,
            hf_rows_api_only=args.hf_rows_api_only,
        )
    if default_dataset.exists():
        return RuntimePlan(
            mode="parquet",
            description=f"默认数据文件：{default_dataset}",
            dataset_path=default_dataset,
            dataset_override=None,
            hf_repo_id=args.hf_repo_id,
            hf_config_name=args.hf_config_name,
            hf_split=args.hf_split,
            hf_streaming=args.hf_streaming,
            hf_rows_api_only=args.hf_rows_api_only,
        )
    return RuntimePlan(
        mode="hf",
        description=f"HF 数据源：{args.hf_repo_id}/{args.hf_config_name}/{args.hf_split}",
        dataset_path=None,
        dataset_override=None,
        hf_repo_id=args.hf_repo_id,
        hf_config_name=args.hf_config_name,
        hf_split=args.hf_split,
        hf_streaming=args.hf_streaming,
        hf_rows_api_only=args.hf_rows_api_only,
    )
```

实现运行时配置物化：

```python
def materialize_runtime_configs(repo_root: Path, plan: RuntimePlan) -> RuntimeConfigs:
    if plan.mode == "parquet":
        return RuntimeConfigs(
            eval_config_path=repo_root / EVAL_CONFIG_PATH,
            stage2_eval_config_path=repo_root / STAGE2_EVAL_CONFIG_PATH,
            benchmark_suite_config_path=repo_root / BENCHMARK_SUITE_CONFIG_PATH,
            temp_dir=None,
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="reverify_real_benchmark_", dir=repo_root / ".tmp"))
    eval_payload = _load_yaml(repo_root / EVAL_CONFIG_PATH)
    stage2_payload = _load_yaml(repo_root / STAGE2_EVAL_CONFIG_PATH)
    suite_payload = _load_yaml(repo_root / BENCHMARK_SUITE_CONFIG_PATH)

    for payload in (eval_payload, stage2_payload):
        environment = payload["environment"]
        environment.pop("dataset_path", None)
        environment["repo_id"] = plan.hf_repo_id
        environment["config_name"] = plan.hf_config_name
        environment["split"] = plan.hf_split
        environment["streaming"] = plan.hf_streaming
        environment["rows_api_only"] = plan.hf_rows_api_only

    eval_path = temp_dir / "hermes_reasoning_traces_eval_rl.hf.yaml"
    stage2_path = temp_dir / "hermes_reasoning_traces_eval_rl_terminal_command_stage2.hf.yaml"
    _dump_yaml(eval_path, eval_payload)
    _dump_yaml(stage2_path, stage2_payload)

    for benchmark in suite_payload["benchmark_suite"]["benchmarks"]:
        config_path = benchmark["config_path"]
        if config_path == EVAL_CONFIG_PATH:
            benchmark["config_path"] = str(eval_path)
        elif config_path == STAGE2_EVAL_CONFIG_PATH:
            benchmark["config_path"] = str(stage2_path)

    suite_path = temp_dir / "benchmark_suite.hf.yaml"
    _dump_yaml(suite_path, suite_payload)
    return RuntimeConfigs(eval_path, stage2_path, suite_path, temp_dir=temp_dir)
```

- [ ] **Step 4: 改造 `build_commands()` 和 `main()`，让行为跑通**

把 `build_commands()` 改成：

```python
def build_commands(
    *,
    eval_config_path: Path,
    stage2_eval_config_path: Path,
    benchmark_suite_config_path: Path,
    include_preflight: bool = True,
) -> list[list[str]]:
    base = ["uv", "run", "python", "-m", "hermes_agentic_rl.cli.main"]
    commands: list[list[str]] = []
    if include_preflight:
        commands.append([*base, "hermes-preflight"])
    commands.append([*base, "eval-rl", "--config", str(eval_config_path)])
    commands.append([*base, "eval-rl", "--config", str(stage2_eval_config_path)])
    commands.append([*base, "benchmark-suite", "--config", str(benchmark_suite_config_path)])
    return commands
```

把 `main()` 改成：

```python
def main() -> int:
    args = build_parser().parse_args()
    include_preflight = not args.skip_preflight
    try:
        plan = resolve_data_source(args, PROJECT_ROOT)
        runtime = materialize_runtime_configs(PROJECT_ROOT, plan)
        commands = build_commands(
            eval_config_path=runtime.eval_config_path,
            stage2_eval_config_path=runtime.stage2_eval_config_path,
            benchmark_suite_config_path=runtime.benchmark_suite_config_path,
            include_preflight=include_preflight,
        )
        if args.dry_run:
            print(f"数据源模式: {plan.mode}")
            print(f"数据源说明: {plan.description}")
            if runtime.temp_dir is not None:
                print(f"临时配置目录: {runtime.temp_dir}")
            for command in commands:
                print("  " + " ".join(shlex.quote(part) for part in command))
            return 0
        if plan.mode == "parquet":
            resolved_dataset = ensure_dataset_link(PROJECT_ROOT, plan.dataset_override)
            print(f"[DATASET] 已就绪：{resolved_dataset}")
        else:
            print(f"[DATASET] 已切换到 {plan.description}")
        return run_commands(PROJECT_ROOT, commands)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败，退出码={exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1
```

- [ ] **Step 5: 跑完整测试文件，确认全部转绿**

Run:

```bash
pytest tests/test_reverify_real_benchmark_script.py -v
```

Expected:

- 全部 PASS
- 旧测试若与新接口冲突，按新设计同步更新断言，但不要放宽行为约束

- [ ] **Step 6: 提交脚本实现**

```bash
git add scripts/reverify_real_benchmark.py tests/test_reverify_real_benchmark_script.py
git commit -m "feat: add hf fallback to benchmark reverify"
```

---

### Task 4: 做一次命令级验收

**Files:**
- Modify: `scripts/reverify_real_benchmark.py`
- Test: `tests/test_reverify_real_benchmark_script.py`

- [ ] **Step 1: 用 dry-run 验证 HF fallback 输出**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH uv run python scripts/reverify_real_benchmark.py --dry-run
```

Expected:

- 本地 parquet 缺失时返回 `0`
- 输出包含：
  - `数据源模式: hf`
  - `lambda/hermes-agent-reasoning-traces`
  - 临时配置目录
  - 3 条评估相关命令

- [ ] **Step 2: 用显式 parquet 验证优先级没有回归**

先准备一个临时 parquet 占位文件，然后运行：

```bash
mkdir -p /tmp/hermes_reasoning_traces && touch /tmp/hermes_reasoning_traces/train.parquet
env -u PYTHONHOME -u PYTHONPATH uv run python scripts/reverify_real_benchmark.py --dry-run --dataset /tmp/hermes_reasoning_traces/train.parquet
```

Expected:

- 返回 `0`
- 输出包含 `数据源模式: parquet`
- 不应打印 HF fallback 说明

- [ ] **Step 3: 只跑关键单测，作为最终回归确认**

Run:

```bash
pytest tests/test_reverify_real_benchmark_script.py -q
```

Expected:

- 全绿

- [ ] **Step 4: 如果 dry-run 输出不够清晰，做最后一次最小整理**

允许的最后整理范围：

```python
print(f"数据源模式: {plan.mode}")
print(f"数据源说明: {plan.description}")
```

不要在这一步继续扩大功能范围，不要引入新的 CLI 语义。

- [ ] **Step 5: 提交验收后的最终整理**

```bash
git add scripts/reverify_real_benchmark.py tests/test_reverify_real_benchmark_script.py
git commit -m "chore: polish benchmark reverify dry-run output"
```

---

## Self-review

- Spec coverage:
  - HF fallback：由 Task 1 + Task 3 覆盖
  - parquet 优先级：由 Task 1 + Task 4 覆盖
  - 临时 eval 配置生成：由 Task 2 + Task 3 覆盖
  - benchmark-suite `config_path` 重写：由 Task 2 + Task 3 覆盖
  - dry-run 语义变化：由 Task 1 + Task 4 覆盖
- Placeholder scan:
  - 计划中没有 `TODO`、`TBD` 或“自行处理”的占位描述
- Type consistency:
  - 统一使用 `RuntimePlan`、`RuntimeConfigs`、`Path`、`build_commands()`、`materialize_runtime_configs()` 这些名称，前后保持一致
