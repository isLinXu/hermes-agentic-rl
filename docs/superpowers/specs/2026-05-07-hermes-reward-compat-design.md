# hermes-reward-compat 设计文档

## 概述
本设计定义一个小范围兼容修复：让 `hermes-agentic-rl` 的 reward 逻辑能够正确处理真实 Hermes / OpenAI 风格输出，避免真实 `train` 导出的样本长期被错误判为 `0.0`。

本次修复只覆盖两个问题：

- `OutcomeReward` 对 `expected_output="done"` 这类短成功标记过于严格，无法接受自然语言完成描述
- `ToolcallReward` 只识别 `call["name"]`，不能识别 OpenAI/Hermes 常见的 `call["function"]["name"]`

## 设计目标
- 保持当前 reward 框架不变
- 让真实 Hermes 轨迹在最小示例数据上获得合理的非零 reward
- 不引入新的外部依赖
- 不引入 LLM judge 或复杂 verifier

## 非目标
本设计不包含：

- 通用任务级 judge
- 文件系统状态验证器
- 基于 tool result 的复杂任务完成度评估
- 新的 reward 类型
- 新的 dataset schema

## 背景
当前真实 Hermes 运行已经跑通：

- `rollout` 能生成真实 trajectory
- `train` 能导出真实训练样本

但导出的样本 reward 仍然是 `0.0`，原因有两个：

1. `OutcomeReward`
   - 当前逻辑要求 `trajectory.final_output == expected_output`
   - 数据集中的 `expected_output` 目前是 `"done"`
   - 真实 Hermes `final_output` 是自然语言，例如：
     - `"I've successfully created hello.txt ..."`

2. `ToolcallReward`
   - 当前逻辑只看 `call["name"]`
   - 真实 Hermes tool call 结构是：
     - `call["function"]["name"]`

这两个问题都属于“格式兼容”而不是“训练方法错误”。

## 方案选择
### 方案 A：最小兼容修复，推荐
做法：

- `OutcomeReward`
  - 保留精确匹配
  - 但在 `expected_output` 是短成功标记时，允许通过自然语言成功描述判定为成功
- `ToolcallReward`
  - 同时兼容 `name` 和 `function.name`

优点：

- 范围最小
- 能直接改善真实 Hermes 样本 reward
- 不改变整体架构

缺点：

- 仍然是启发式规则

### 方案 B：只修 ToolcallReward
优点：

- 风险更低

缺点：

- Outcome 仍会被 `"done"` 精确比较卡死

### 方案 C：引入通用 judge
优点：

- 长期最强

缺点：

- 范围过大
- 超出当前问题本身

### 结论
本次采用方案 A：`最小兼容修复`。

## 核心设计
### OutcomeReward
当前逻辑：

- 无 `expected_output`：看 `final_output` 是否存在
- 有 `expected_output`：只做精确字符串匹配

新逻辑：

1. 如果 `expected_output is None`
   - 保持现状

2. 如果 `trajectory.final_output == expected_output`
   - 仍然直接给 `1.0`

3. 如果 `expected_output` 是短成功标记
   - 如：`done`、`ok`、`success`、`completed`
   - 则允许对 `final_output` 做最小成功语义匹配

建议匹配信号：

- 包含成功动词：
  - `done`
  - `success`
  - `successfully`
  - `created`
  - `completed`
  - `written`
- 或包含数据项中的关键目标，例如：
  - `instruction` 中出现的文件名 `hello.txt`

为了控制范围，建议采用：

- 成功关键词匹配
- 文件名/路径片段匹配（只取最简单的 `.txt`/带路径 token）

### ToolcallReward
当前逻辑：

- 遍历 `step.tool_calls`
- 仅检查顶层是否有 `name`

新逻辑：

- 增加一个统一 helper：
  - 先读 `call["name"]`
  - 再读 `call["function"]["name"]`
- 只要能提取出非空名称，就视为有效

## 模块职责
### `hermes_agentic_rl/rewards/outcome_reward.py`
负责：

- 识别短成功标记
- 做精确匹配与启发式成功匹配
- 给出更具体的 `reason`

### `hermes_agentic_rl/rewards/toolcall_reward.py`
负责：

- 提供统一 tool call 名称提取逻辑
- 对 Hermes/OpenAI 风格结构做兼容

### `tests/test_rewards.py`
建议新增或扩展现有 reward 测试，覆盖：

- 精确匹配
- `"done"` + 自然语言成功输出
- `name`
- `function.name`

## 判分规则
### OutcomeReward
建议判分顺序：

1. `expected_output is None`
   - 有 `final_output` 则 `1.0`

2. 精确匹配
   - `1.0`

3. `expected_output` 为短成功标记，且 `final_output` 呈现成功语义
   - `1.0`

4. 其他情况
   - `0.0`

### ToolcallReward
建议逻辑：

- `call_count == 0` → `0.0`
- 能提取到名称的 call 视为有效
- 分数仍然保持：
  - `1.0 - invalid_calls / call_count`

## 风险控制
### 风险一：OutcomeReward 过于宽松
缓解：

- 只对短成功标记启用启发式匹配
- 不对任意 `expected_output` 启用模糊匹配

### 风险二：误判自然语言中的成功词
缓解：

- 关键词集合保持很小
- 尽量结合 instruction 里的文件名信号

### 风险三：ToolcallReward 未来再出现新结构
缓解：

- 名称提取单独封装，后续只改一个 helper

## 测试策略
至少覆盖：

1. `OutcomeReward`
   - `expected_output="done"`，`final_output="I've successfully created hello.txt"` → `1.0`
   - 精确匹配仍为 `1.0`
   - 明显失败描述仍为 `0.0`

2. `ToolcallReward`
   - `{"name": "write_file"}` → 有效
   - `{"function": {"name": "write_file"}}` → 有效
   - 缺失两者 → 无效

3. 真实回归
   - 重新跑一次 Hermes `train`
   - 检查 `outputs/train_samples_hermes.jsonl` 中 reward 是否改善

## 验收标准
本次完成后，必须满足：

1. 真实 Hermes 样本不再因为 `function.name` 被误判 tool call 无效
2. `expected_output="done"` 时，合理的自然语言成功输出能拿到 outcome 分
3. 全量测试继续通过
4. 重新生成的 `outputs/train_samples_hermes.jsonl` 中 reward 高于 `0.0`

## 结论
这是一次格式兼容修复，而不是训练范式变化。通过限制在 `OutcomeReward` 与 `ToolcallReward` 两处最小修复，就能把真实 Hermes 轨迹从“已经跑通但 reward 错判”为“可用于下一步训练调优”的状态。
