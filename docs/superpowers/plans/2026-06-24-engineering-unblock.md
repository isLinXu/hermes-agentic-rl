# Engineering Unblock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复当前阻塞真实闭环的工程问题，让 `hermes-preflight` 不再被 Hermes 子模块语法错误直接打断，并让 `atropos-preflight` 对缺失子模块/依赖给出清晰结果。

**Architecture:** 先从最小可确定修复入手，直接修 `subprojects/hermes-agent/hermes_cli/config.py` 中的语法损坏；然后通过现有 preflight CLI 与单元测试把错误从“导入崩溃”收敛成“结构化缺项”。本批次不改训练主干，不扩展 `eval-rl`，只动 preflight 相关路径与必要测试。

**Tech Stack:** Python 3.11+, pytest, 本地 git submodule, `hermes_agentic_rl.integrations.*`, `subprojects/hermes-agent`

---

## File map

- Modify: `subprojects/hermes-agent/hermes_cli/config.py`
  - 修复当前确定性的语法损坏，目标是恢复 Hermes 子模块基本可导入性。
- Modify: `tests/test_hermes_local_repo.py`
  - 增加或调整 preflight 回归测试，确保不会再因明显坏掉的 Hermes 代码路径产生不可控崩溃。
- Modify: `tests/test_atropos_preflight_cli.py`
  - 强化 CLI 输出断言，让“子模块缺失/依赖缺失”是结构化结果。
- Modify: `hermes_agentic_rl/integrations/hermes_preflight.py`
  - 仅在必要时增强导入失败信息，避免非结构化异常直出。
- Modify: `hermes_agentic_rl/integrations/atropos_preflight.py`
  - 仅在必要时增强缺失子模块与缺失依赖的区分。

---

### Task 1: 修复 Hermes 子模块的确定性语法损坏

**Files:**
- Modify: `subprojects/hermes-agent/hermes_cli/config.py`
- Test: `tests/test_hermes_local_repo.py`

- [ ] **Step 1: 写一个最小回归测试，覆盖“预检不应因 Hermes 子模块语法错误直接崩溃”**

在 `tests/test_hermes_local_repo.py` 追加一个聚焦 preflight 可调用性的测试，先固定“返回结构化结果”这个行为。

```python
def test_run_hermes_preflight_returns_structured_result_for_repo_root():
    from pathlib import Path

    from hermes_agentic_rl.integrations.hermes_preflight import run_hermes_preflight

    workspace_root = Path(__file__).resolve().parent.parent
    result = run_hermes_preflight(workspace_root)

    assert isinstance(result.python_ok, dict)
    assert isinstance(result.missing, list)
    assert result.repo_source in {None, "config", "env", "subproject"}
```

- [ ] **Step 2: 运行该测试，确认当前行为失败或不稳定**

Run:

```bash
pytest tests/test_hermes_local_repo.py::test_run_hermes_preflight_returns_structured_result_for_repo_root -v
```

Expected:

- 可能直接失败，或者在 import Hermes 相关路径时暴露异常；
- 失败本身是预期的，因为当前 `subprojects/hermes-agent/hermes_cli/config.py` 已确认有语法损坏。

- [ ] **Step 3: 修复 `subprojects/hermes-agent/hermes_cli/config.py` 中的损坏 lambda**

将当前损坏代码：

```python
return re.sub(
    r"\${([^}]+)}",
    def def def lambda m: os.environ.get(m.group(1), m.group(0)),
    obj,
)
```

改为最小正确实现：

```python
return re.sub(
    r"\${([^}]+)}",
    lambda m: os.environ.get(m.group(1), m.group(0)),
    obj,
)
```

- [ ] **Step 4: 重新运行回归测试，验证该路径恢复**

Run:

```bash
pytest tests/test_hermes_local_repo.py::test_run_hermes_preflight_returns_structured_result_for_repo_root -v
```

Expected:

- PASS
- 说明 preflight 至少能返回结构化结果对象，而不是被 Hermes 子模块语法错误拦截。

- [ ] **Step 5: 提交这个最小修复**

```bash
git add subprojects/hermes-agent/hermes_cli/config.py tests/test_hermes_local_repo.py
git commit -m "fix: repair hermes config syntax for preflight"
```

---

### Task 2: 收敛 Hermes preflight 的失败形态

**Files:**
- Modify: `hermes_agentic_rl/integrations/hermes_preflight.py`
- Modify: `tests/test_hermes_local_repo.py`

- [ ] **Step 1: 写一个失败测试，固定“当前仓库根目录运行 preflight 时，不应抛出未处理异常”**

在 `tests/test_hermes_local_repo.py` 增加一个更贴近 CLI 行为的测试。

```python
def test_hermes_preflight_repo_root_does_not_raise():
    from pathlib import Path

    from hermes_agentic_rl.integrations.hermes_preflight import run_hermes_preflight

    workspace_root = Path(__file__).resolve().parent.parent
    result = run_hermes_preflight(workspace_root)

    assert hasattr(result, "missing")
    assert hasattr(result, "python_ok")
    assert isinstance(result.missing, list)
```

- [ ] **Step 2: 运行与 Hermes preflight 相关测试，查看新的真实失败点**

Run:

```bash
pytest tests/test_hermes_local_repo.py -v
```

Expected:

- 如果仍失败，新的失败点应该来自 Hermes 依赖缺失或导入异常，而不是语法错误。

- [ ] **Step 3: 仅在必要时增强 `run_hermes_preflight()` 的容错信息**

如果当前实现会因为 `find_spec()` 或导入副作用抛异常，按最小范围补充保护。示例目标形态如下：

```python
def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False
```

如果需要额外记录导入失败原因，保留 `python_ok` 结构，同时只把缺项追加到 `missing`，不要在这里引入复杂错误对象。

- [ ] **Step 4: 跑 Hermes 相关测试与 CLI 命令**

Run:

```bash
pytest tests/test_hermes_local_repo.py -v
python -m hermes_agentic_rl.cli.main hermes-preflight
```

Expected:

- 单测 PASS
- CLI 不再因未处理异常直接终止
- 若仍失败，应输出 JSON，并将问题收敛为缺失模块或缺失目录

- [ ] **Step 5: 提交 preflight 收敛改动**

```bash
git add hermes_agentic_rl/integrations/hermes_preflight.py tests/test_hermes_local_repo.py
git commit -m "fix: harden hermes preflight failure reporting"
```

---

### Task 3: 收敛 Atropos preflight 的缺项表达

**Files:**
- Modify: `hermes_agentic_rl/integrations/atropos_preflight.py`
- Modify: `tests/test_atropos_preflight_cli.py`

- [ ] **Step 1: 写一个失败测试，固定“缺失 tinker-atropos 时仍返回结构化 JSON”**

在 `tests/test_atropos_preflight_cli.py` 增加一个专门覆盖“目录缺失”的测试。

```python
def test_atropos_preflight_cli_reports_missing_submodule_as_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    atropos_dir = tmp_path / "subprojects" / "atropos"
    atropos_dir.mkdir(parents=True, exist_ok=True)
    (atropos_dir / "atroposlib").mkdir(parents=True, exist_ok=True)
    (atropos_dir / "atroposlib" / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["hermes-agentic-rl", "atropos-preflight"])

    exit_code = main()
    payload = json.loads(capsys.readouterr().out.strip())

    assert exit_code == 1
    assert "local_dir:subprojects/tinker-atropos" in payload["missing"]
```

- [ ] **Step 2: 运行 Atropos preflight 测试**

Run:

```bash
pytest tests/test_atropos_preflight_cli.py -v
```

Expected:

- 如果失败，失败点应集中在 CLI 输出结构或缺项表达，而不是杂乱的 import 栈。

- [ ] **Step 3: 仅在必要时增强 `run_atropos_preflight()` 的缺项表达**

如需修改，目标是保留当前轻量实现，只增强区分度，例如确保：

```python
if atropos_dir is None:
    missing.append("local_dir:subprojects/atropos")
if tinker_atropos_dir is None:
    missing.append("local_dir:subprojects/tinker-atropos")
if not python_ok["atroposlib"]:
    missing.append("python:atroposlib")
if not python_ok["tinker_atropos.config"]:
    missing.append("python:tinker_atropos")
```

不要把这里改造成复杂诊断框架；只确保“目录缺失”和“Python 模块缺失”可区分。

- [ ] **Step 4: 运行测试与真实 CLI 验证**

Run:

```bash
pytest tests/test_atropos_preflight_cli.py -v
python -m hermes_agentic_rl.cli.main atropos-preflight
```

Expected:

- 单测 PASS
- 真实 CLI 输出 JSON
- 当前工作区若子模块未初始化，应清晰出现 `local_dir:subprojects/tinker-atropos` 或同类缺项

- [ ] **Step 5: 提交 Atropos preflight 收敛改动**

```bash
git add hermes_agentic_rl/integrations/atropos_preflight.py tests/test_atropos_preflight_cli.py
git commit -m "fix: clarify atropos preflight missing states"
```

---

### Task 4: 最终验证与结果整理

**Files:**
- Modify: `docs/superpowers/specs/2026-06-24-engineering-unblock-design.md`（仅在实现偏离 spec 时）
- Modify: `EXECUTION_REPORT.md`（仅在项目已有相同记录习惯时）

- [ ] **Step 1: 运行最终目标命令**

Run:

```bash
pytest tests/test_hermes_local_repo.py tests/test_atropos_preflight_cli.py -v
python -m hermes_agentic_rl.cli.main hermes-preflight
python -m hermes_agentic_rl.cli.main atropos-preflight
```

Expected:

- 核心 preflight 测试全部 PASS
- 两个 CLI 都不应再因为语法损坏直接崩溃

- [ ] **Step 2: 核对实现是否与 spec 一致**

人工检查以下结论是否成立：

```text
1. Hermes 子模块的确定性语法损坏已修复
2. hermes-preflight 的失败已收敛为结构化缺项
3. atropos-preflight 能区分子模块缺失与 Python 导入缺失
4. 没有修改 eval-rl、训练器或环境主干
```

- [ ] **Step 3: 如有必要，补充执行结果记录**

若项目当前确实用 `EXECUTION_REPORT.md` 记录这类修复，则追加最小摘要：

```markdown
## 2026-06-24 engineering unblock

- repaired deterministic syntax corruption in `subprojects/hermes-agent/hermes_cli/config.py`
- hardened preflight result shape for Hermes / Atropos integration checks
- verified `hermes-preflight` and `atropos-preflight` no longer fail at the same point as before
```

如果该文件并非当前任务的既有记录入口，则跳过，不新增无关文档。

- [ ] **Step 4: 提交最终验证结果**

```bash
git add tests/test_hermes_local_repo.py tests/test_atropos_preflight_cli.py hermes_agentic_rl/integrations/hermes_preflight.py hermes_agentic_rl/integrations/atropos_preflight.py subprojects/hermes-agent/hermes_cli/config.py EXECUTION_REPORT.md
git commit -m "fix: unblock hermes and atropos preflight checks"
```

---

## Self-review

### Spec coverage

- 修 Hermes 子模块确定性损坏：Task 1 覆盖
- 收敛 `hermes-preflight`：Task 2 覆盖
- 收敛 `atropos-preflight`：Task 3 覆盖
- 最终验证与结果输出：Task 4 覆盖

### Placeholder scan

- 未使用 `TODO` / `TBD`
- 每个任务都给了明确文件、命令与期望结果

### Type consistency

- `run_hermes_preflight()` / `run_atropos_preflight()` 命名与现有代码一致
- 测试文件路径与仓库现状一致
- 未引入新的未定义接口

