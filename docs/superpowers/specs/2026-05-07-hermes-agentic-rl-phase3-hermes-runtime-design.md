# hermes-agentic-rl 第三阶段设计文档

## 概述
本设计定义 `hermes-agentic-rl` 的第三阶段目标：在第二阶段已经具备 fake runtime、可执行 CLI 和 hermes 不可用错误处理的基础上，实现对“已安装 pip 包形式的 `hermes-agent`”的真实运行时接线。

本阶段聚焦于：

- 发现并调用已安装 `hermes-agent` 的可用入口
- 构建真实的 runtime loop 包装器
- 把真实 Hermes 结果规范化为框架内部统一的运行时协议
- 优先让 `integration=hermes` 的 `rollout` 路径真实跑通

本阶段不解决源码仓开发模式，也不同时兼容多种不稳定入口策略。

## 设计目标
- 支持从已安装的 `hermes-agent` pip 包中发现真实运行时入口
- 让 `HermesRuntimeAdapter` 在 runtime 可用时返回真实 agent loop 包装器
- 将真实 Hermes 执行结果映射为框架统一协议
- 让 `rollout` 命令在 `integration=hermes` 时可执行
- 保持 `integration=fake` 路径完全兼容

## 非目标
本阶段不包含：

- 源码仓模式接入
- 多版本 Hermes 全兼容层
- `train` 的真实多样本训练编排增强
- 在线训练、异步 judge、异步 sidecar
- benchmark、session replay、memory 回灌

## 背景与当前状态
截至第二阶段，当前仓库已经具备：

- `Trajectory` / `RewardSummary` 等核心数据结构
- `RolloutManager` / `RewardManager` / `TrainerBridge`
- fake runtime adapter
- hermes runtime 不可用路径
- 可执行的 `rollout` 与 `train` CLI

但 `HermesRuntimeAdapter` 仍然只有“检测依赖是否安装”和“明确报未实现”的能力，还不能真正跑通已安装 Hermes 的真实 loop。

因此第三阶段的核心问题是：

> 如何在不绑定 `hermes-agent` 主仓源码的前提下，可靠地探测并调用已安装包中的真实 agent 入口？

## 方案选择
### 方案 A：单入口探测
做法：

- 假定 `hermes-agent` 存在一个单一稳定入口
- 直接写死导入路径

优点：

- 实现最简单

缺点：

- 一旦入口变动，适配器立即失效
- 风险较高

### 方案 B：有限入口探测，推荐
做法：

- 在一个很小的候选入口集合中按顺序探测
- 找到可用入口后构建真实 loop
- 找不到则给出“当前版本不支持”的明确错误

优点：

- 比单入口更稳
- 范围仍然可控

缺点：

- 需要维护一个小型探测矩阵

### 方案 C：通用反射式探测
做法：

- 对整个安装包进行模块扫描和反射
- 动态猜测 agent loop 入口

优点：

- 理论上最灵活

缺点：

- 实现不稳定
- 容易误判
- 不适合本阶段

### 结论
本阶段采用方案 B：`有限入口探测`。

## 总体架构
建议新增或修改以下模块：

```text
hermes_agentic_rl/
├── runtime/
│   ├── hermes_adapter.py
│   ├── hermes_entrypoints.py
│   └── hermes_wrapper.py
└── cli/
    └── main.py
```

同时会补充新的测试文件：

```text
tests/
├── test_hermes_entrypoints.py
└── test_hermes_runtime_integration.py
```

## 核心设计原则
### 入口发现显式化
不要把 Hermes 探测逻辑写散在 adapter 内部。

应将“候选入口列表、探测顺序、可用性检查”集中到独立模块中，例如 `hermes_entrypoints.py`。

### 真实运行时只暴露统一协议
一旦找到 Hermes 的真实入口，框架其他模块不再直接感知 Hermes 内部对象。

对外只暴露：

- 一个能执行 `run(prompt)` 的 loop wrapper
- 一个符合统一协议的返回值

### 失败要可诊断
必须区分：

- 包未安装
- 安装了但找不到支持的入口
- 找到了入口但构造失败
- 运行成功但返回结构无法映射

每种情况都要给出不同的错误信息。

### 优先 rollout，再扩 train
本阶段优先保证：

- `integration=hermes` 下 `rollout` 真能跑

`train` 只要能复用真实 rollout 结果即可，不额外扩功能。

## 模块职责
### `runtime/hermes_entrypoints.py`
负责定义有限入口探测策略。

建议职责：

- 维护候选入口描述列表
- 顺序尝试导入候选模块
- 判断模块中是否存在可用构造器或 loop 类
- 返回第一个成功的入口描述

入口描述建议至少包含：

- `name`
- `module_name`
- `factory_name`
- `build_loop(config) -> object`

### `runtime/hermes_wrapper.py`
负责把真实 Hermes loop 包装成统一协议。

建议职责：

- 接收由入口工厂创建出的真实对象
- 提供统一 `run(prompt)` 方法
- 捕获运行异常并转为 `RuntimeExecutionError`
- 将结果交给规范化函数映射为标准字典

### `runtime/hermes_adapter.py`
第三阶段的主入口。

职责：

- 调用 `hermes_entrypoints` 做入口探测
- 构建 `HermesRuntimeLoopWrapper`
- 在失败时返回结构化错误

不再只是“可用 / 不可用”判断，而是真正进入 runtime build 流程。

### `cli/main.py`
本阶段只需最小修改：

- 让 `integration=hermes` 时真正调用新的 `HermesRuntimeAdapter`
- 在 `rollout` 命令中使用真实 loop wrapper
- `train` 继续复用 `_build_agent_loop(config)`，无需新增专属逻辑

## 入口探测策略
### 候选入口范围
本阶段只允许一个很小的候选集合，避免无限扩张。

建议探测顺序基于公开文档中稳定度较高的运行时路径：

1. `hermes_agent.environments.agent_loop`
2. `hermes_agent.run_agent`
3. 未来如果需要，再补一个兼容入口

注意：

- 这里的候选模块名只是设计位，具体实现前需要基于已安装包实际结构校验
- 如果运行时发现候选集合全部不可用，应报“当前安装版本不支持”

### 入口探测结果
建议用结构化对象表示：

```python
{
    "name": "hermes_agent_loop",
    "module_name": "hermes_agent.environments.agent_loop",
    "factory_name": "HermesAgentLoop",
}
```

或更完整的 dataclass。

## 统一运行时协议
真实 Hermes loop 的输出最终仍要映射为：

```python
{
    "messages": list[dict],
    "tool_calls": list[list[dict]],
    "tool_results": list[list[dict]],
    "final_output": str | None,
    "finished_naturally": bool,
    "turns_used": int,
    "metadata": dict,
}
```

### 映射原则
- `messages`
  - 尽量保留原始多轮消息
- `tool_calls`
  - 若真实 Hermes 有结构化工具调用，直接映射
  - 若只有消息事件，需从事件中提取
- `tool_results`
  - 映射真实工具返回
- `final_output`
  - 优先取最终 assistant 文本
- `finished_naturally`
  - 若真实 API 提供状态，按状态映射
  - 否则按“无异常完成”推断
- `metadata`
  - 保留入口名、运行时版本、原始返回摘要

## 配置要求
本阶段配置层原则上不需要大改，但建议允许补充：

```yaml
runtime:
  integration: hermes
  provider: openai
  model: some-model
  max_agent_turns: 20
  enabled_toolsets:
    - terminal
    - file
```

### 运行要求
当 `integration=hermes` 时：

- 必须存在可用的 Hermes 入口
- `provider` / `model` 至少要能被 wrapper 传递给真实运行时

如果某些参数当前版本无法传递，应在错误信息中说明，而不是静默忽略。

## 错误处理
### 包未安装
表现：

- `import hermes_agent` 失败

处理：

- 抛 `RuntimeUnavailableError`
- 文案应提示“未安装 hermes-agent”

### 安装了但入口不支持
表现：

- 包存在
- 候选入口全部探测失败

处理：

- 抛 `RuntimeUnavailableError`
- 文案应提示“当前已安装版本不支持”

### 入口可用但构造失败
表现：

- 入口模块可导入
- 但构造 loop 时缺少必要参数或对象

处理：

- 抛 `RuntimeConfigurationError`
- 文案应说明是构造失败而不是未安装

### 运行失败
表现：

- loop 执行时抛异常

处理：

- 抛 `RuntimeExecutionError`
- 附带入口名和简要上下文

### 返回结构无法映射
表现：

- 真实结果缺少必要字段

处理：

- 抛 `RuntimeExecutionError`
- 明确说明哪一类数据无法映射

## 测试策略
### 单元测试
新增覆盖：

- 候选入口探测成功
- 候选入口全部失败
- 构造 wrapper 失败
- wrapper 将 fake 风格“真实对象”映射到统一协议

### 集成测试
新增覆盖：

- `integration=hermes` 时 CLI rollout 成功路径
  - 用 monkeypatch/假的 Hermes 模块注入，不依赖真实包安装
- `integration=hermes` 时 CLI rollout 不支持版本路径

### 真实环境冒烟测试
如果宿主环境实际安装了 `hermes-agent`，可额外做：

- 单次 `rollout --config ...` 冒烟

但这不进入默认测试集。

## 对现有代码的影响
### 不变部分
- `Trajectory`
- `RewardManager`
- `TrainerBridge`
- `AtroposGrpoTrainer`
- fake runtime

### 增强部分
- `HermesRuntimeAdapter`
- CLI 的 hermes 路径
- runtime 错误细分

## 验收标准
本阶段完成后，必须满足：

1. `integration=hermes` 时，框架会尝试真实入口探测
2. 探测成功时，`rollout` 可以运行真实 wrapper
3. 探测失败时，错误能区分“未安装”和“当前版本不支持”
4. `integration=fake` 路径保持全部兼容
5. 默认测试仍然可在无真实 Hermes 环境下全部通过

## 风险与缓解
### 风险一：pip 包暴露入口不稳定
缓解：

- 使用有限候选入口集合
- 所有探测逻辑集中在 `hermes_entrypoints.py`

### 风险二：真实返回格式太复杂，难以一次性完整映射
缓解：

- 第三阶段只要求映射最小可训练字段
- 复杂调试信息先放进 `metadata`

### 风险三：真实接线污染 fake 路径
缓解：

- fake 与 hermes adapter 分离
- CLI 只在 `integration=hermes` 时进入真实接线

## 实现顺序建议
推荐顺序：

1. `hermes_entrypoints.py`
2. `hermes_wrapper.py`
3. `hermes_adapter.py` 真正 build loop
4. `tests/test_hermes_entrypoints.py`
5. `tests/test_hermes_runtime_integration.py`
6. CLI hermes rollout 接线
7. 全量测试回归

## 结论
第三阶段的核心，不是再扩一层框架抽象，而是把 `HermesRuntimeAdapter` 从“存在性检测器”提升为“真实运行时桥接器”。

通过限定在“已安装 pip 包 + 有限入口探测 + rollout 优先”的范围内，本阶段可以以较可控的复杂度，把 `hermes-agentic-rl` 真正推进到可接真实 Hermes 的状态。
