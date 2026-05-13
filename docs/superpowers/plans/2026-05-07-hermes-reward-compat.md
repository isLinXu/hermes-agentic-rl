# Hermes Reward Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `OutcomeReward` 与 `ToolcallReward` 对真实 Hermes/OpenAI 风格输出的兼容问题，使真实 `train` 导出的样本 reward 合理（非 0），同时保持 fake 路径与既有测试不回归。

**Architecture:** 只改 reward 层：`OutcomeReward` 在 `expected_output` 为短成功标记时启用保守的成功语义匹配；`ToolcallReward` 增加统一的 tool-name 提取逻辑，兼容 `call["name"]` 与 `call["function"]["name"]`。新增一个专门的 reward 兼容测试文件覆盖新行为；最后用真实 Hermes 配置再跑一次 `train` 作为回归验证。

**Tech Stack:** Python、pytest、regex（标准库 `re`）

---

## 文件结构

本计划会修改：
- `hermes_agentic_rl/rewards/outcome_reward.py`
- `hermes_agentic_rl/rewards/toolcall_reward.py`

本计划会新增：
- `tests/test_reward_compat.py`

（可选）会读取输出文件验证：
- `outputs/train_samples_hermes.jsonl`

---

### Task 1: 为 ToolcallReward 增加 OpenAI/Hermes tool_call 结构测试

**Files:**
- Create: `tests/test_reward_compat.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_reward_compat.py`（先只放 ToolcallReward 相关测试）：

```python
import asyncio

from hermes_agentic_rl.core.types import RolloutStep, Trajectory
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward


def _trajectory_with_calls(calls: list[dict]) -> Trajectory:
    return Trajectory(
        task_id="t",
        prompt="p",
        steps=[
            RolloutStep(
                turn_index=0,
                assistant_message=None,
                tool_calls=calls,
                tool_results=[],
            )
        ],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )


def test_toolcall_reward_accepts_openai_function_name_format():
    reward = ToolcallReward(weight=1.0)
    traj = _trajectory_with_calls(
        [
            {
                "type": "function",
                "function": {"name": "write_file", "arguments": "{}"},
            }
        ]
    )
    result = asyncio.run(reward.evaluate(item={}, trajectory=traj, tool_context=None))
    assert result.score == 1.0


def test_toolcall_reward_rejects_call_with_no_name_anywhere():
    reward = ToolcallReward(weight=1.0)
    traj = _trajectory_with_calls([{"type": "function", "function": {"arguments": "{}"}}])
    result = asyncio.run(reward.evaluate(item={}, trajectory=traj, tool_context=None))
    assert result.score == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reward_compat.py::test_toolcall_reward_accepts_openai_function_name_format -v`
Expected: FAIL with `AssertionError`（当前实现会把 call 判为 missing name）

- [ ] **Step 3: Commit the red test (optional)**

如需严格分步提交，可先提交红测试；否则跳过。

---

### Task 2: 修改 ToolcallReward 兼容 function.name

**Files:**
- Modify: `hermes_agentic_rl/rewards/toolcall_reward.py`
- Test: `tests/test_reward_compat.py`

- [ ] **Step 1: Implement minimal code**

在 `ToolcallReward` 内新增一个私有 helper（或内联）：

```python
def _extract_tool_name(call: dict) -> str | None:
    if isinstance(call.get("name"), str) and call["name"].strip():
        return call["name"].strip()
    func = call.get("function")
    if isinstance(func, dict) and isinstance(func.get("name"), str) and func["name"].strip():
        return func["name"].strip()
    return None
```

把判 invalid 的逻辑从：

```python
if "name" not in call:
    invalid_calls += 1
```

改为：

```python
if _extract_tool_name(call) is None:
    invalid_calls += 1
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/test_reward_compat.py::test_toolcall_reward_accepts_openai_function_name_format -v`
Expected: PASS

- [ ] **Step 3: Run full compat file**

Run: `python3 -m pytest tests/test_reward_compat.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add hermes_agentic_rl/rewards/toolcall_reward.py tests/test_reward_compat.py
git commit -m "fix: support OpenAI/Hermes tool_calls in ToolcallReward"
```

---

### Task 3: 为 OutcomeReward 增加“done + 自然语言成功输出”测试

**Files:**
- Modify: `tests/test_reward_compat.py`

- [ ] **Step 1: Write the failing test**

在 `tests/test_reward_compat.py` 追加：

```python
import asyncio

from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward


def test_outcome_reward_accepts_successful_natural_language_when_expected_done():
    reward = OutcomeReward(weight=1.0)
    traj = Trajectory(
        task_id="t",
        prompt="p",
        steps=[],
        final_output="I've successfully created hello.txt with content hello.",
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )
    item = {"expected_output": "done", "instruction": "Create hello.txt and write hello"}
    result = asyncio.run(reward.evaluate(item=item, trajectory=traj, tool_context=None))
    assert result.score == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reward_compat.py::test_outcome_reward_accepts_successful_natural_language_when_expected_done -v`
Expected: FAIL（当前 OutcomeReward 只做精确匹配）

---

### Task 4: 修改 OutcomeReward：短成功标记 + 保守语义匹配

**Files:**
- Modify: `hermes_agentic_rl/rewards/outcome_reward.py`
- Test: `tests/test_reward_compat.py`

- [ ] **Step 1: Implement minimal code**

实现策略（保守）：

1. 只在 `expected_output` 属于短成功标记集合时启用启发式：
   - `{"done","ok","okay","success","successful","completed","complete"}`
2. `final_output` 必须存在且满足以下任一条件：
   - 包含成功关键词（`success`, `successfully`, `created`, `completed`, `done`）
   - 或同时包含 instruction 中的一个文件名 token（例如 `hello.txt`）

建议实现两个 helper：

```python
import re

_SUCCESS_MARKERS = {"done", "ok", "okay", "success", "successful", "completed", "complete"}
_SUCCESS_KEYWORDS = ("success", "successfully", "created", "completed", "done", "written")

def _extract_filenames(text: str) -> list[str]:
    # 非严格：抓取形如 hello.txt / notes/todo.md 的 token
    return re.findall(r"[A-Za-z0-9_./-]+\\.[A-Za-z0-9]{1,8}", text)

def _looks_successful(final_output: str, instruction: str | None) -> bool:
    low = final_output.lower()
    if any(k in low for k in _SUCCESS_KEYWORDS):
        return True
    if instruction:
        for token in _extract_filenames(instruction):
            if token and token in final_output:
                return True
    return False
```

然后在 `evaluate()` 中插入：

```python
elif isinstance(expected_output, str) and expected_output.strip().lower() in _SUCCESS_MARKERS:
    if trajectory.final_output and _looks_successful(trajectory.final_output, item.get("instruction")):
        score = 1.0
        reason = "expected_output is a success marker; inferred success from final_output"
    else:
        score = 0.0
        reason = "expected_output is a success marker; final_output did not look successful"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/test_reward_compat.py::test_outcome_reward_accepts_successful_natural_language_when_expected_done -v`
Expected: PASS

- [ ] **Step 3: Run full suite**

Run: `python3 -m pytest tests/test_reward_compat.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add hermes_agentic_rl/rewards/outcome_reward.py tests/test_reward_compat.py
git commit -m "fix: make OutcomeReward accept success-marker outputs"
```

---

### Task 5: 全量回归 + 真实 Hermes train 再跑一次

**Files:**
- None (commands only)

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests -v`
Expected: PASS

- [ ] **Step 2: Re-run hermes train**

使用当前 hermes 配置重新生成训练样本：

Run:
```bash
mkdir -p outputs
LKEAP_API_KEY=... .venv311/bin/hermes-agentic-rl train --config configs/terminal_grpo_hermes.yaml
```

- [ ] **Step 3: Verify reward improved**

打开 `outputs/train_samples_hermes.jsonl`，确认：
- `reward` > 0
- `reward_components` 中 Outcome/Toolcall 至少有一个为 1.0（或接近 1.0）

- [ ] **Step 4: Commit (optional)**

如果需要最后一个汇总提交：

```bash
git add hermes_agentic_rl/rewards tests
git commit -m "fix: reward compatibility for Hermes/OpenAI outputs"
```

---

## 自检结果

- 覆盖 spec 中的两个问题：OutcomeReward + ToolcallReward
- 不引入新依赖、不增加 judge、不改训练框架
- 新增测试文件专门保护兼容逻辑，避免回归
