# Tool-call 稳定性优化设计

- 日期：2026-06-28
- 主题：提升 Hermes 原生 tool-call 协议的结构稳定性，同时设计配套的数据课程分层；第一批只实现评估/奖励层优化
- 范围：
  - `hermes_agentic_rl/envs/hermes_reasoning_traces.py`
  - 与 `HermesReasoningTraceReward` 直接相关的测试
  - 训练/课程层的设计说明与后续实施边界
- 第一批实现范围：
  - 仅落地评估/奖励层优化
  - 不在本批次修改训练配置、采样配比与 checkpoint 选择逻辑
- 非范围：
  - 长上下文 `prompt-context-retention` 优化
  - 新模型结构或更大 backend
  - `reverify_real_benchmark.py` 统一入口逻辑
  - W&B / TensorBoard 指标系统重构

## 背景

真实 HF 复验已经证明：

1. 统一 benchmark 入口已被打通，真实 HF 数据源可以贯穿：
   - `eval-rl`
   - `stage2 eval-rl`
   - `benchmark-suite`
2. `tool-call heldout` 在当前链路下可以跑出有效 promotion 结果。
3. 但在原生 Hermes tool-call 路径上，`metadata/tool_call_parse_ok`、`tool_name_match`、`argument_key_overlap`、`argument_value_similarity` 仍不稳定。
4. 同时，stage2 `terminal_command_tool_call` 适配路径明显更容易得到更好的 held-out 指标。

这说明当前问题不是“项目不会训练”，而是：

- 奖励信号对“半结构正确”的区分还不够锋利
- 数据课程把“原生协议学习”和“受限动作空间学习”揉在一起，导致模型更容易学会命令内容，而不是完整 Hermes 原生协议

## 现状判断

### 一、评估/奖励层已经有基础，但还不够聚焦

当前 `HermesReasoningTraceReward` 已经支持：

- `similarity`
- `tool_call`
- `hybrid`

并且元数据中已经记录了：

- `tool_call_parse_ok`
- `tool_call_json_valid`
- `argument_json_valid`
- `argument_schema_ok`
- `tool_name_match`
- `argument_key_overlap`
- `argument_value_similarity`
- `argument_value_quality`
- `partial_tool_call_score`

这意味着当前系统并不缺少结构化信号，而是缺少更清晰的奖励梯度设计。

### 二、训练/课程层已经出现两条混杂路径

当前仓库里，至少存在两条风格不同的训练/评估路径：

1. **原生 Hermes tool-call 路径**
   - 模型直接生成 `<tool_call> ... </tool_call>` 结构
   - 更接近真实协议
   - 更难学稳

2. **`terminal_command_tool_call` 适配路径**
   - 模型先生成 command string
   - 评估期再包一层固定的 tool-call wrapper
   - 更容易把 `argument_value_similarity` 做高
   - 但容易掩盖原生协议结构学习不足

这两条路径都有效，但解决的是不同问题。当前最需要的是把它们从“混合使用”变成“分阶段使用”。

## 目标

本次设计同时覆盖两层，但第一批只落地第一层。

### 总体目标

1. 让 reward 更准确地区分三类 tool-call 失败：
   - 有外壳但 JSON / schema 不合法
   - schema 合法但 tool name 不匹配
   - key 基本对齐但 argument value 偏差大
2. 避免“文本很像，但 Hermes 原生协议没学稳”时仍获得过高 reward。
3. 为后续课程分层明确训练阶段边界，避免 stage2 适配路径掩盖原生协议短板。

### 第一批实现目标

1. 在不改训练配置的前提下，重新定义 `tool_call` / `hybrid` 奖励聚合逻辑。
2. 将 `partial_tool_call_score` 从“宽松补偿项”改为“有限失败梯度”。
3. 为结构性错误建立更清晰的 reward profile 和测试。
4. 不改变 `eval-rl` / `benchmark-suite` 的接口与产物格式。

## 成功标准

第一批完成后，应满足：

1. 对同一批预测样本，以下情况的 reward 排序应稳定成立：
   - 完全合法 + name 匹配 + arguments 对齐
   - 有合法 wrapper + name 对 + value 偏差
   - 有 wrapper 但 JSON / schema 非法
   - 只有零散 `<tool_call>` 痕迹
   - 完全无 tool-call 结构
2. 当 target 本身包含 tool-call 时，单纯文本相似不应再轻易掩盖结构失败。
3. `partial_tool_call_score` 不得让“明显非法结构”得到接近合法 tool-call 的分数。
4. 现有 `terminal_command_tool_call` 适配路径仍能工作，不被本次改动破坏。

## 方案对比

### 方案 A：先做评估/奖励层实现，课程层先设计，推荐

做法：

1. 重新定义 `HermesReasoningTraceReward` 的结构化打分逻辑。
2. 保留现有训练配置，先验证 reward 变化对 held-out 指标的影响。
3. 同时把课程分层设计清楚，但不在本批次改配置。

优点：

- 改动最小
- 最容易验证“是不是 reward 信号不够锋利”
- 不会同时引入 reward 与 curriculum 两个变量

缺点：

- 第一轮只能提升判别质量，不能立刻保证模型学得更稳

### 方案 B：先做课程层实现，奖励层只设计

做法：

1. 增加训练阶段和采样配比
2. 先让模型更频繁看到协议样本
3. reward 层暂不调整

优点：

- 直接改“怎么学”

缺点：

- 如果 reward 本身不够准，训练仍可能学偏
- 第一轮试验归因困难

### 方案 C：两层同时落地

优点：

- 理论上更完整

缺点：

- 最难判断收益来源
- 第一批风险最大

### 结论

采用 **方案 A**：

- 两层一起设计
- 第一批只实现评估/奖励层

## 总体设计

### 一、评估/奖励层设计

当前 `_score_tool_call_pair()` 的优势是指标丰富，但存在两个问题：

1. `text_score` 在 `hybrid` 模式下仍可能过度补偿结构失败
2. `partial_tool_call_score` 的存在形式更像“软补贴”，而不是“失败梯度”

因此，第一批设计分三步：

#### 1. 建立结构失败分层

将预测结果分为 5 档：

1. **合法完整**
   - `parse_ok = 1`
   - `tool_name_match` 高
   - `argument_key_overlap` / `argument_value_similarity` 可比较

2. **合法但语义偏差**
   - JSON / schema 合法
   - tool name 或 arguments 偏差明显

3. **半结构化**
   - 有 `<tool_call>` 外壳
   - 有 name / arguments key
   - 但 JSON 或 schema 不合法

4. **弱结构痕迹**
   - 只有部分标签、括号、字段名
   - 无法进入 object-like 解析

5. **完全无结构**
   - 没有可识别 tool-call 痕迹

目标不是直接暴露 5 个 public API，而是在 reward 内部用这套层次调整分数。

#### 2. 重定义 `partial_tool_call_score`

新的原则：

- `partial_tool_call_score` 只负责给“接近合法结构”的失败样本保留梯度
- 绝不能让它与“合法结构但语义偏差”的分数重叠太多

约束：

- `partial_tool_call_score` 上限必须低于“合法但 name 错 / args 错”的最低稳定得分带
- 对仅有 tag 痕迹、没有 JSON/schema 可解析性的输出，得分应更低

建议的行为区间：

- 合法完整：`0.75 ~ 1.0`
- 合法但语义偏差：`0.35 ~ 0.74`
- 半结构化：`0.12 ~ 0.34`
- 弱结构痕迹：`0.02 ~ 0.11`
- 完全无结构：`0.0 ~ 0.01`

这里不是硬编码绝对分值，而是定义 reward 带宽约束。

#### 3. 收紧 `hybrid` 聚合逻辑

当前 `hybrid` 的问题是，当 target 是 tool-call 时，文本相似度有时会让 reward 看起来不低。

新的聚合原则：

- 当 `target` 含 tool-call 时：
  - 如果 `parse_ok = 0`，则文本相似度只能作为弱补充项
  - 不能与结构分数平权
- 当 `parse_ok = 1` 后：
  - 文本相似度才作为 arguments value 细粒度差异补充

也就是说：

1. 先过结构门槛
2. 再谈文本相似

第一批不改变 `reward_mode` 的 public 枚举，只重写内部聚合。

### 二、训练/课程层设计

这一层先只设计，不立即实现。

#### 课程分层

定义四个阶段：

1. **Stage 0：结构模板阶段**
   - 目标：稳定学习 `<tool_call>` 基本骨架
   - 样本：短上下文、固定工具名、短 arguments

2. **Stage 1：tool name 对齐阶段**
   - 目标：学会从工具集合中选择正确 `name`
   - 样本：多个工具名、低歧义 arguments

3. **Stage 2：arguments key/value 对齐阶段**
   - 目标：提升 `argument_key_overlap` 与 `argument_value_similarity`
   - 样本：真实 trace 中结构清晰、字段稳定的子集

4. **Stage 3：开放式真实轨迹混合阶段**
   - 目标：把结构能力迁移到更长、更真实的多轮上下文中
   - 样本：原始 Hermes reasoning traces + 部分困难样本

#### 路径边界

- `terminal_command_tool_call` 路径应继续保留
  - 用于 command-content 学习
  - 不再被当成“原生协议稳定性”的主要代理指标

- 原生 Hermes tool-call 路径应单独评估
  - 用于观察 `tool_call_parse_ok`
  - 用于观察 `tool_name_match`
  - 用于观察 `argument_*`

#### 后续实施原则

后续进入课程层实现时，优先改：

1. 数据过滤规则
2. 训练 stage 配置拆分
3. checkpoint / eval 配对方式

不优先改：

1. backend 规模
2. optimizer 结构
3. 复杂外部 reward model

## 第一批实现设计

第一批只实现奖励层，预计只触及 `hermes_agentic_rl/envs/hermes_reasoning_traces.py` 和对应测试。

### 计划内改动

1. 重写 `_partial_tool_call_structure()` 的打分分布
2. 为 `_structured_tool_call_score()` 增加更清晰的失败层级处理
3. 调整 `HermesReasoningTraceReward.evaluate()` 中 `hybrid` 聚合规则
4. 补充回归测试，锁定：
   - 合法结构 > 半结构 > 弱结构 > 无结构
   - `parse_ok = 0` 时文本相似不能主导最终分数
   - `terminal_command_tool_call` 适配路径不回归

### 不改内容

1. `HermesReasoningTracesConfig` 的对外字段
2. 现有 `reward_mode` 枚举名
3. `eval-rl` / `benchmark-suite` 的配置 schema
4. `reverify_real_benchmark.py`

## 测试设计

第一批至少需要以下测试：

1. **完整合法 tool-call 得分最高**
   - target 为合法 tool-call
   - prediction 完全合法且 name / args 对齐

2. **合法但 name 错误低于完全合法**
   - prediction schema 合法
   - 但 `tool_name_match = 0`

3. **合法但 value 偏差低于完全合法且高于半结构**
   - prediction schema 合法
   - `argument_value_similarity` 明显低于 1

4. **半结构输出只保留有限梯度**
   - prediction 有 `<tool_call>`、`name`、`arguments`
   - 但 JSON / object schema 非法

5. **无结构输出最低**
   - prediction 不包含可识别 tool-call 痕迹

6. **hybrid 不再让文本相似掩盖结构失败**
   - 构造一个文本非常像 target，但结构非法的 prediction
   - 其 reward 必须低于结构合法但 value 不完美的 prediction

7. **terminal command adapter 不回归**
   - `assistant_response_adapter = terminal_command_tool_call`
   - command 包装逻辑与 reward 路径保持可用

## 风险与回退

### 风险

1. reward 收紧后，现有 checkpoint 的 `mean_reward` 可能整体下降
2. 某些先前“看起来还不错”的模型会被重新判定为结构不稳定
3. 如果第一批只改奖励，不改课程，训练阶段短期可能更难收敛

### 回退策略

1. 改动只集中在 reward 逻辑与测试，回退成本低
2. 不改配置 schema，可快速恢复
3. 若验证发现 reward 过于苛刻，可只回调聚合系数，不必回滚整体设计

## 预期输出

完成第一批后，应产出：

1. 更稳定的 tool-call 结构奖励行为
2. 一组覆盖结构失败层级的测试
3. 一份已准备好的课程层设计，可作为第二批实现输入

## 自检结论

1. 文档已明确分离“整体设计”和“第一批实现范围”。
2. 第一批只实现评估/奖励层，未把课程层实现混入当前范围。
3. 成功标准、测试标准、风险与回退路径均已写清。
4. 设计目标紧扣当前真实 HF 复验暴露出的结构稳定性问题，没有扩展到长上下文优化。
