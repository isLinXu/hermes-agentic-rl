# Tool-call Stability Reward Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收紧 Hermes 原生 tool-call 的奖励梯度，让结构失败、半结构和合法但语义偏差的预测在 reward 上稳定分层，同时保持 `terminal_command_tool_call` 适配路径不回归。

**Architecture:** 第一批只改 `hermes_agentic_rl/envs/hermes_reasoning_traces.py` 内部 reward 逻辑，不改配置 schema 与训练入口。实现通过新增一个专门的 reward 测试文件，先锁定结构层级和 `hybrid` 聚合行为，再做最小代码调整使这些测试转绿。

**Tech Stack:** Python 3.11, pytest, dataclasses, `SequenceMatcher`, Hermes Agentic RL reward/env code

---

## File map

- Modify: `hermes_agentic_rl/envs/hermes_reasoning_traces.py`
  - 重构 `_partial_tool_call_structure()`、`_structured_tool_call_score()` 与 `HermesReasoningTraceReward.evaluate()` 的聚合规则。
- Create: `tests/test_hermes_reasoning_traces_reward.py`
  - 为结构分层、`hybrid` 收紧和 `terminal_command_tool_call` 适配路径补专门的回归测试。
- Read-only reference: `docs/superpowers/specs/2026-06-28-tool-call-stability-design.md`
  - 对齐成功标准与第一批范围。

---

### Task 1: 锁定结构分层的外部行为

**Files:**
- Create: `tests/test_hermes_reasoning_traces_reward.py`
- Test: `tests/test_hermes_reasoning_traces_reward.py`

- [ ] **Step 1: 创建测试文件并写测试 helper**

在 `tests/test_hermes_reasoning_traces_reward.py` 写入：

```python
from __future__ import annotations

import pytest

from hermes_agentic_rl.envs.hermes_reasoning_traces import (
    _structured_tool_call_score,
)


TARGET_TOOL_CALL = """<think>
</think>
<tool_call>
{"name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>"""


def assert_descending(*scores: float) -> None:
    for left, right in zip(scores, scores[1:]):
        assert left > right, (scores, left, right)
```

- [ ] **Step 2: 写第一个失败测试，锁定“完整合法 > 合法但 value 偏差 > 半结构 > 无结构”**

继续写入：

```python
def test_structured_tool_call_score_orders_quality_levels() -> None:
    valid_exact = """<think>
</think>
<tool_call>
{"name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>"""
    valid_wrong_value = """<think>
</think>
<tool_call>
{"name": "terminal", "arguments": {"command": "pwd"}}
</tool_call>"""
    partial = """<think>
</think>
<tool_call>
{"name": "terminal", "arguments":
</tool_call>"""
    no_structure = "run ls -la in the current directory"

    valid_score, _ = _structured_tool_call_score(valid_exact, TARGET_TOOL_CALL)
    wrong_value_score, _ = _structured_tool_call_score(valid_wrong_value, TARGET_TOOL_CALL)
    partial_score, _ = _structured_tool_call_score(partial, TARGET_TOOL_CALL)
    no_structure_score, _ = _structured_tool_call_score(no_structure, TARGET_TOOL_CALL)

    assert_descending(valid_score, wrong_value_score, partial_score, no_structure_score)
```

- [ ] **Step 3: 运行单测，确认它先红掉**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_hermes_reasoning_traces_reward.py::test_structured_tool_call_score_orders_quality_levels -v
```

Expected:

- FAIL
- 失败应表现为当前 `partial_score` 与 `wrong_value_score`、或 `no_structure_score` 的层级不符合新设计预期

- [ ] **Step 4: 写第二个失败测试，锁定“合法但 tool name 错误低于完全合法，高于半结构”**

继续写入：

```python
def test_structured_tool_call_score_penalizes_tool_name_mismatch() -> None:
    valid_exact = """<think>
</think>
<tool_call>
{"name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>"""
    wrong_name = """<think>
</think>
<tool_call>
{"name": "patch", "arguments": {"command": "ls -la"}}
</tool_call>"""
    partial = """<tool_call>{"name": "terminal"</tool_call>"""

    valid_score, _ = _structured_tool_call_score(valid_exact, TARGET_TOOL_CALL)
    wrong_name_score, _ = _structured_tool_call_score(wrong_name, TARGET_TOOL_CALL)
    partial_score, _ = _structured_tool_call_score(partial, TARGET_TOOL_CALL)

    assert valid_score > wrong_name_score > partial_score
```

- [ ] **Step 5: 再次运行两个测试，确认都红掉**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_hermes_reasoning_traces_reward.py::test_structured_tool_call_score_orders_quality_levels tests/test_hermes_reasoning_traces_reward.py::test_structured_tool_call_score_penalizes_tool_name_mismatch -v
```

Expected:

- 至少 1 个 FAIL
- 失败原因应是当前结构分层不够锋利，而不是导入错误

- [ ] **Step 6: 提交第一批测试骨架**

```bash
git add tests/test_hermes_reasoning_traces_reward.py
git commit -m "test: lock tool-call structure quality ordering"
```

---

### Task 2: 锁定 hybrid 聚合与 adapter 不回归

**Files:**
- Modify: `tests/test_hermes_reasoning_traces_reward.py`
- Test: `tests/test_hermes_reasoning_traces_reward.py`

- [ ] **Step 1: 写第三个失败测试，锁定“结构失败时文本相似不能主导 hybrid 得分”**

在同一文件追加：

```python
@pytest.mark.asyncio
async def test_hybrid_reward_does_not_let_text_similarity_mask_structure_failure() -> None:
    from hermes_agentic_rl.core.types import Trajectory
    from hermes_agentic_rl.envs.hermes_reasoning_traces import HermesReasoningTraceReward

    reward = HermesReasoningTraceReward(
        reward_mode="hybrid",
        tool_call_reward_weight=0.85,
        text_reward_weight=0.15,
    )
    item = {"target_response_full": TARGET_TOOL_CALL, "target_response": TARGET_TOOL_CALL}

    semantically_similar_but_invalid = Trajectory(final_output="""<tool_call>{"name":"terminal","arguments":""")
    valid_but_imperfect = Trajectory(
        final_output="""<think>
</think>
<tool_call>
{"name": "terminal", "arguments": {"command": "pwd"}}
</tool_call>"""
    )

    invalid_result = await reward.evaluate(item, semantically_similar_but_invalid, None)
    imperfect_result = await reward.evaluate(item, valid_but_imperfect, None)

    assert imperfect_result.score > invalid_result.score
```

- [ ] **Step 2: 写第四个失败测试，锁定 `terminal_command_tool_call` 适配路径不回归**

继续追加：

```python
@pytest.mark.asyncio
async def test_terminal_command_adapter_still_rewards_valid_command_content() -> None:
    from hermes_agentic_rl.core.types import Trajectory
    from hermes_agentic_rl.envs.hermes_reasoning_traces import HermesReasoningTraceReward

    reward = HermesReasoningTraceReward(
        reward_mode="hybrid",
        tool_call_reward_weight=0.45,
        text_reward_weight=0.55,
    )
    item = {
        "target_response": "ls -la\n",
        "target_response_full": TARGET_TOOL_CALL,
        "response_adapter": "terminal_command_tool_call",
        "response_suffix": "\n",
    }

    good = await reward.evaluate(item, Trajectory(final_output="ls -la"), None)
    bad = await reward.evaluate(item, Trajectory(final_output="pwd"), None)

    assert good.score > bad.score
    assert good.metadata["response_adapter"] == "terminal_command_tool_call"
```

- [ ] **Step 3: 运行新增测试，确认先红掉**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_hermes_reasoning_traces_reward.py::test_hybrid_reward_does_not_let_text_similarity_mask_structure_failure tests/test_hermes_reasoning_traces_reward.py::test_terminal_command_adapter_still_rewards_valid_command_content -v
```

Expected:

- 至少 1 个 FAIL
- 失败点应集中在 `hybrid` 聚合和结构惩罚不足

- [ ] **Step 4: 跑整个新测试文件，确认形成稳定红灯**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_hermes_reasoning_traces_reward.py -v
```

Expected:

- 多个 FAIL
- 红灯覆盖结构排序和 hybrid 聚合两个维度

- [ ] **Step 5: 提交第二批测试**

```bash
git add tests/test_hermes_reasoning_traces_reward.py
git commit -m "test: define hybrid tool-call reward behavior"
```

---

### Task 3: 最小实现结构失败分层

**Files:**
- Modify: `hermes_agentic_rl/envs/hermes_reasoning_traces.py`
- Test: `tests/test_hermes_reasoning_traces_reward.py`

- [ ] **Step 1: 先收紧 `_partial_tool_call_structure()` 的分布**

把 `hermes_agentic_rl/envs/hermes_reasoning_traces.py` 中的 `_partial_tool_call_structure()` 替换为：

```python
def _partial_tool_call_structure(prediction: str) -> tuple[float, dict[str, float]]:
    lowered = prediction.lower()
    open_tag = 1.0 if "<tool_call" in lowered else 0.0
    close_tag = 1.0 if "</tool_call>" in lowered else 0.0
    json_braces = 1.0 if "{" in prediction and "}" in prediction else 0.0
    name_key = 1.0 if '"name"' in lowered or "'name'" in lowered else 0.0
    arguments_key = 1.0 if '"arguments"' in lowered or "'arguments'" in lowered else 0.0
    components = {
        "partial_open_tag": open_tag,
        "partial_close_tag": close_tag,
        "partial_json_braces": json_braces,
        "partial_name_key": name_key,
        "partial_arguments_key": arguments_key,
    }
    weak_signal = max(open_tag, close_tag, name_key, arguments_key, json_braces)
    rich_structure = open_tag + close_tag + json_braces + name_key + arguments_key
    if rich_structure >= 4:
        score = 0.22
    elif rich_structure >= 2:
        score = 0.08
    elif weak_signal > 0:
        score = 0.02
    else:
        score = 0.0
    return score, components
```

- [ ] **Step 2: 运行结构排序测试，确认失败形态推进**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_hermes_reasoning_traces_reward.py::test_structured_tool_call_score_orders_quality_levels tests/test_hermes_reasoning_traces_reward.py::test_structured_tool_call_score_penalizes_tool_name_mismatch -v
```

Expected:

- 仍可能 FAIL
- 但应把问题推进到合法结构和 hybrid 聚合逻辑，而不再是 partial 分布明显过宽

- [ ] **Step 3: 在 `_structured_tool_call_score()` 里明确 5 档失败层级**

在 `if not prediction_calls:` 分支下保留 partial 梯度，同时在有 `prediction_calls` 时加入结构下限逻辑。将相关段落改为：

```python
    mean_score = sum(best_scores) / len(best_scores)
    for key in best_components[0]:
        metadata[key] = sum(component[key] for component in best_components) / len(best_components)
    metadata["tool_call_score"] = mean_score

    parse_ok = float(metadata.get("tool_call_parse_ok", 0.0))
    name_match = float(metadata.get("tool_name_match", 0.0))
    key_overlap = float(metadata.get("argument_key_overlap", 0.0))
    value_similarity = float(metadata.get("argument_value_similarity", 0.0))

    if parse_ok >= 1.0:
        if name_match >= 1.0 and key_overlap >= 0.95 and value_similarity >= 0.95:
            final_score = max(mean_score, 0.85)
        elif name_match >= 1.0:
            final_score = max(mean_score, 0.45)
        else:
            final_score = max(min(mean_score, 0.60), 0.30)
    else:
        partial_score, partial_metadata = _partial_tool_call_structure(prediction)
        metadata["partial_tool_call_score"] = partial_score
        metadata.update(partial_metadata)
        final_score = max(partial_score, min(mean_score, 0.24))

    return max(0.0, min(1.0, final_score)), metadata
```

- [ ] **Step 4: 跑整个新测试文件，确认只剩 hybrid 聚合相关问题或已基本转绿**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_hermes_reasoning_traces_reward.py -v
```

Expected:

- 结构排序相关测试应 PASS
- 如果仍 FAIL，应主要集中在 `HermesReasoningTraceReward.evaluate()` 的 `hybrid` 聚合逻辑

- [ ] **Step 5: 提交结构分层实现**

```bash
git add hermes_agentic_rl/envs/hermes_reasoning_traces.py tests/test_hermes_reasoning_traces_reward.py
git commit -m "feat: tighten tool-call structure reward tiers"
```

---

### Task 4: 最小实现 hybrid 收紧并做最终回归

**Files:**
- Modify: `hermes_agentic_rl/envs/hermes_reasoning_traces.py`
- Test: `tests/test_hermes_reasoning_traces_reward.py`

- [ ] **Step 1: 收紧 `HermesReasoningTraceReward.evaluate()` 中的 hybrid 聚合**

把 `evaluate()` 中的这段：

```python
        elif self.reward_mode == "tool_call":
            score = tool_call_score
        else:
            denom = self.tool_call_reward_weight + self.text_reward_weight
            score = (
                self.tool_call_reward_weight * tool_call_score
                + self.text_reward_weight * text_score
            ) / denom
```

替换为：

```python
        elif self.reward_mode == "tool_call":
            score = tool_call_score
        else:
            denom = self.tool_call_reward_weight + self.text_reward_weight
            parse_ok = float(tool_metadata.get("tool_call_parse_ok", 0.0))
            if has_target_tool_call and parse_ok < 1.0:
                effective_text_weight = min(self.text_reward_weight, 0.05)
                effective_tool_weight = self.tool_call_reward_weight
            else:
                effective_text_weight = self.text_reward_weight
                effective_tool_weight = self.tool_call_reward_weight
            effective_denom = effective_tool_weight + effective_text_weight
            score = (
                effective_tool_weight * tool_call_score
                + effective_text_weight * text_score
            ) / effective_denom
```

- [ ] **Step 2: 跑两个 hybrid / adapter 测试，确认转绿**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_hermes_reasoning_traces_reward.py::test_hybrid_reward_does_not_let_text_similarity_mask_structure_failure tests/test_hermes_reasoning_traces_reward.py::test_terminal_command_adapter_still_rewards_valid_command_content -v
```

Expected:

- PASS

- [ ] **Step 3: 跑整个新测试文件**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_hermes_reasoning_traces_reward.py -v
```

Expected:

- 全绿

- [ ] **Step 4: 跑一次已有统一复验入口相关测试，确认未误伤周边**

Run:

```bash
env -u PYTHONHOME -u PYTHONPATH pytest tests/test_reverify_real_benchmark_script.py -q
```

Expected:

- 全绿

- [ ] **Step 5: 如果 reward reason / metadata 需要最小整理，最后做一次局部清理**

允许的最小整理范围：

```python
reason=(
    f"mode={self.reward_mode} score={score:.4f} "
    f"text_similarity={text_score:.4f} tool_call={tool_call_score:.4f}"
)
```

不要在这一步引入新 reward mode、新配置字段或新训练逻辑。

- [ ] **Step 6: 提交最终实现**

```bash
git add hermes_agentic_rl/envs/hermes_reasoning_traces.py tests/test_hermes_reasoning_traces_reward.py
git commit -m "feat: tighten hermes tool-call reward shaping"
```

---

## Self-review

- Spec coverage:
  - 结构失败分层：Task 1 + Task 3
  - `partial_tool_call_score` 收紧：Task 3
  - `hybrid` 不再让文本相似掩盖结构失败：Task 2 + Task 4
  - `terminal_command_tool_call` 适配路径不回归：Task 2 + Task 4
  - 课程层仅设计不实现：本计划未包含训练配置或 curriculum 代码改动
- Placeholder scan:
  - 计划没有 `TODO`、`TBD` 或“自行决定”的占位描述
- Type consistency:
  - 统一使用 `_structured_tool_call_score()`、`_partial_tool_call_structure()`、`HermesReasoningTraceReward.evaluate()`、`terminal_command_tool_call` 这些符号名，前后一致
