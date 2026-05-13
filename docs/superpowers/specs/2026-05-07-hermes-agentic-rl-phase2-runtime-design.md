# hermes-agentic-rl 第二阶段设计文档

## 概述
本设计定义 `hermes-agentic-rl` 的第二阶段目标：在第一阶段最小闭环框架的基础上，引入对真实 `hermes-agent` 运行时的可选集成，并将当前仅具备壳层能力的 CLI 提升为可执行的 `rollout` / `train` 命令。

本阶段采用“可选运行时集成”路线，而非直接绑定 `hermes-agent` 主仓源码。也就是说，当前仓库继续保持独立可测试；如果宿主环境中安装了 `hermes-agent`，则通过运行时适配器连接真实 `HermesAgentLoop`，否则给出明确错误或回退到测试路径。

## 设计目标
- 在不破坏当前独立仓库结构的前提下，支持运行时接入真实 `hermes-agent`
- 为 `RolloutManager` 增加对真实 Hermes rollout 输出的适配能力
- 让 CLI 的 `rollout` 与 `train` 命令具备最小可用行为
- 通过配置控制是否启用 Hermes 运行时集成
- 保持测试套件在未安装 `hermes-agent` 的环境中仍可运行

## 非目标
本阶段不包含以下内容：

- 真正对接 Atropos 在线训练服务
- 多任务并发 rollout 调度
- 从真实 Hermes session 自动抽样训练数据
- LLM judge、PRM、多数投票判分
- benchmark runner
- 异步 judge 服务或 Hermes-native 四段侧车架构

## 背景与问题定义
第一阶段已经完成：

- `Trajectory`、`RewardSummary` 等统一类型
- `RewardManager`
- `BaseEnv` 与 `TerminalTaskEnv`
- `RolloutManager`
- `AtroposGrpoTrainer` 导出器
- `config` / `CLI` 最小壳层

但当前系统仍存在两个明显缺口：

1. `RolloutManager` 只支持 fake loop / 测试 loop，无法运行真实 `hermes-agent`
2. CLI 虽有命令名，但还不能真正执行 rollout 或 train

因此第二阶段的核心不是继续扩 reward 或 env，而是补齐“框架与真实 agent 运行时之间的桥”。

## 方案选择
### 方案 A：可选运行时集成
做法：

- 新增运行时适配器层
- 在运行时尝试导入 `hermes-agent`
- 成功时连接真实 `HermesAgentLoop`
- 失败时保留现有测试路径，并对 CLI 给出明确错误

优点：

- 当前独立仓库仍可自洽
- 兼容未来并入 `hermes-agent` 主仓
- 测试不被真实依赖阻塞

缺点：

- 需要做一层协议映射
- 对真实 `hermes-agent` 的版本差异要更小心

### 方案 B：直接绑定主仓源码
做法：

- 当前代码直接写死导入 `hermes-agent` 源码路径

优点：

- 集成路径最短

缺点：

- 当前仓库不再独立
- 不适合现阶段

### 方案 C：只做协议层，不做真实导入
做法：

- 只定义接口与配置
- 不真正接入运行时

优点：

- 风险最低

缺点：

- 无法满足“真实接入”的目标

### 结论
本阶段采用方案 A：`可选运行时集成`。

## 总体架构
建议新增以下目录与模块：

```text
hermes_agentic_rl/
├── runtime/
│   ├── __init__.py
│   ├── base.py
│   ├── hermes_adapter.py
│   └── errors.py
├── datasets/
│   ├── __init__.py
│   └── jsonl_loader.py
└── cli/
    └── main.py
```

同时会修改：

- `hermes_agentic_rl/core/rollout_manager.py`
- `hermes_agentic_rl/config.py`
- `configs/terminal_grpo.yaml`
- `examples/minimal_terminal_task.py`

## 核心设计原则
### 运行时依赖可选
`hermes-agent` 必须是可选依赖，而不是安装时硬依赖。

要求：

- `pip install hermes-agentic-rl` 不要求必须安装 `hermes-agent`
- 只有在用户显式使用 `integration: hermes` 时才尝试导入真实运行时
- 导入失败时必须给出清晰错误，说明如何安装或切换模式

### 内部协议稳定
无论真实 Hermes 返回什么格式，框架内部都统一落为 `Trajectory`。

因此：

- `runtime adapter` 负责处理外部格式差异
- `RolloutManager` 只消费统一协议
- reward / trainer 不直接感知 `hermes-agent` 实现细节

### CLI 只做编排，不做业务
CLI 负责：

- 读取配置
- 载入任务
- 选择 runtime adapter
- 驱动 rollout / reward / trainer

CLI 不负责：

- reward 逻辑
- trajectory 转换
- 训练样本格式拼装

### 保持无真实依赖的测试能力
所有核心测试仍可在没有安装 `hermes-agent` 的环境中运行。

因此：

- 单元测试使用 fake runtime
- 真实 Hermes 集成测试可以作为可选测试组，不进入默认 CI

## 模块职责
### `runtime/base.py`
定义运行时适配接口，例如：

- `build_agent_loop(config)`
- `is_available()`
- `describe_unavailable_reason()`

这个接口对上层隐藏真实 `hermes-agent` 的模块结构。

### `runtime/errors.py`
定义明确的运行时错误：

- `RuntimeUnavailableError`
- `RuntimeConfigurationError`
- `RuntimeExecutionError`

用于避免 CLI 里出现模糊的 `ImportError` 或堆栈直出。

### `runtime/hermes_adapter.py`
这是第二阶段的核心模块。

职责：

- 尝试导入真实 `hermes-agent`
- 组装真实 `HermesAgentLoop` 所需对象
- 将真实 loop 暴露为统一的 `run(prompt)` 协议
- 将原始返回内容映射为框架内部可理解的结构

这个适配器不直接输出 `Trajectory`，而是输出一个标准化的原始结果对象，交给 `RolloutManager` 转换。

### `datasets/jsonl_loader.py`
提供最小数据集加载能力：

- 读取 JSONL
- 返回 `list[dict]`
- 对缺失的 `task_id` 做最小生成或报错

第二阶段只需要 JSONL，不引入更复杂的数据源。

### `core/rollout_manager.py`
需要从“只接受 fake loop”升级为“接受任意满足协议的 runtime loop”。

新增能力：

- 允许从 runtime adapter 注入 loop
- 处理真实 Hermes 返回中的 `messages` / `tool_calls` / `tool_results`
- 尽可能保留原始消息到 `trajectory.metadata`

### `config.py`
需要从简单 YAML 读取升级为最小结构化配置解释器，至少支持：

- `runtime.integration`
- `runtime.provider`
- `runtime.model`
- `runtime.max_agent_turns`
- `runtime.system_prompt`
- `runtime.enabled_toolsets`
- `environment.dataset_path`
- `trainer.export_training_path`

本阶段仍可以返回 `dict`，不强制上 dataclass 全量重构。

### `cli/main.py`
要从“命令壳层”变成最小执行入口。

#### `rollout`
行为：

1. 读配置
2. 读数据集
3. 取一个任务
4. 创建 runtime adapter
5. 执行一次 rollout
6. 将 trajectory 输出到终端或 JSON 文件

#### `train`
行为：

1. 读配置
2. 取一个任务样本
3. 执行 rollout
4. 计算 reward
5. 通过 trainer 导出训练样本

本阶段 `train` 是“单样本最小闭环”，不是批量训练 orchestration。

## 数据流
### `rollout` 命令
```text
CLI
  ↓
load_config
  ↓
load_jsonl_dataset
  ↓
HermesRuntimeAdapter
  ↓
RolloutManager.collect()
  ↓
Trajectory
  ↓
stdout / file
```

### `train` 命令
```text
CLI
  ↓
load_config
  ↓
load_jsonl_dataset
  ↓
HermesRuntimeAdapter
  ↓
RolloutManager.collect()
  ↓
RewardManager.evaluate()
  ↓
TrainerBridge.submit()
  ↓
JSONL export
```

## 配置扩展
建议新增如下配置结构：

```yaml
runtime:
  integration: hermes
  provider: openai
  model: demo-model
  max_agent_turns: 20
  system_prompt: null
  enabled_toolsets:
    - terminal
    - file

environment:
  type: terminal_task_env
  dataset_path: data/minimal_terminal_tasks.jsonl

reward:
  aggregator: weighted_sum
  components:
    - name: outcome_reward
      weight: 0.7
    - name: toolcall_reward
      weight: 0.3

trainer:
  type: atropos_grpo
  export_training_path: outputs/train_samples.jsonl
```

### 字段约束
- `runtime.integration`
  - 允许值：`fake`、`hermes`
  - 默认值建议为 `fake`
- `environment.dataset_path`
  - 必填
- `trainer.export_training_path`
  - `train` 命令时必填

## 运行时协议设计
建议统一 runtime loop 的最小返回协议：

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

说明：

- 真实 Hermes adapter 负责把原始输出规整成这个协议
- `FakeAgentLoop` 也继续遵守这个协议
- `RolloutManager` 不应该知道自己面对的是真实 loop 还是 fake loop

## 错误处理
### 未安装 `hermes-agent`
当 `integration: hermes` 且运行时导入失败时：

- CLI 返回非零退出码
- 输出简明错误，例如：
  - 未检测到 `hermes-agent`
  - 请先安装，或将 `runtime.integration` 改为 `fake`

### 配置缺失
例如：

- 缺少 `dataset_path`
- 缺少 `export_training_path`

要求：

- 抛出结构化配置错误
- 错误信息指出缺失字段名

### rollout 执行失败
例如：

- Hermes loop 抛异常
- 返回协议字段缺失

要求：

- 包装为 `RuntimeExecutionError`
- CLI 只展示必要上下文，不直接倾倒复杂堆栈

## 测试策略
### 单元测试
新增测试应覆盖：

- `load_jsonl_dataset`
- `runtime.integration` 配置解析
- `HermesRuntimeAdapter` 在未安装 `hermes-agent` 时的错误行为
- `RolloutManager` 对标准化 runtime payload 的处理

### CLI 集成测试
至少覆盖：

- `rollout` 命令在 `fake` 模式下成功执行
- `train` 命令在 `fake` 模式下成功导出 JSONL
- `rollout` 命令在 `hermes` 模式但运行时不可用时给出明确错误

### 可选真实集成测试
如果宿主环境安装了 `hermes-agent`，可额外执行：

- 单次真实 rollout 冒烟测试

但这不进入默认测试集。

## 第一阶段代码的影响范围
本阶段不会推翻第一阶段实现，而是做增强：

- `Trajectory` 结构保持不变
- `RewardManager` 保持不变
- `AtroposGrpoTrainer` 保持主逻辑不变
- `TerminalTaskEnv` 保持主结构不变

主要增量是：

- runtime adapter
- dataset loader
- CLI 编排增强
- 配置增强

## 验收标准
本阶段完成后，必须满足：

1. `rollout` 命令可以在 `fake` 模式下跑通并输出 trajectory
2. `train` 命令可以在 `fake` 模式下跑通并写出训练 JSONL
3. 当配置 `integration: hermes` 时，框架会尝试接入真实运行时
4. 未安装 `hermes-agent` 时，错误提示明确且可操作
5. 默认测试环境下仍能跑通全部单元测试与 CLI 集成测试

## 风险与缓解
### 风险一：真实 `hermes-agent` 模块结构不稳定
缓解：

- 把真实导入逻辑集中在 `runtime/hermes_adapter.py`
- 不在其他模块散落直接导入

### 风险二：CLI 过早承担过多逻辑
缓解：

- CLI 只做 orchestration
- 数据加载、runtime 构建、reward、trainer 都走独立模块

### 风险三：真实集成破坏现有测试稳定性
缓解：

- 默认测试使用 fake integration
- 真实 Hermes 集成测试设为可选

## 实现顺序建议
推荐实现顺序：

1. `datasets/jsonl_loader.py`
2. `runtime/errors.py` 与 `runtime/base.py`
3. `runtime/hermes_adapter.py`
4. `config.py` 扩展
5. `core/rollout_manager.py` 增强
6. `cli/main.py` 的 `rollout`
7. `cli/main.py` 的 `train`
8. fake 模式 CLI 集成测试
9. hermes 不可用错误测试

## 结论
第二阶段的重点不是继续扩展框架抽象，而是建立框架与真实 `hermes-agent` 运行时之间的稳定连接，同时让 CLI 真正可执行。

采用“可选运行时集成”方案，可以在保持当前仓库独立性的同时，向真实 Hermes 使用路径迈进一步，并为后续更深层的在线 agentic RL 集成打下基础。
