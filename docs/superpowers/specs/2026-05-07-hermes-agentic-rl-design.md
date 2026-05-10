# hermes-agentic-rl 设计文档

## 概述
`hermes-agentic-rl` 的目标是参考 `OpenClaw-RL` 对 `OpenClaw` 的强化学习思路，为 `hermes-agent` 提供一套原生、可扩展、可训练的 agentic RL 框架。

第一阶段不重做一套完整的异步在线训练基础设施，而是优先复用 `hermes-agent` 已有的 `environments/`、`HermesAgentLoop`、`ToolContext`、trajectory 与 Atropos 对接能力，构建一个最小可运行、边界清晰、后续可演化为异步侧车架构的训练框架。

## 设计目标
- 基于 `hermes-agent` 现有环境框架实现原生 agentic RL 框架
- 支持多轮 tool-calling rollout 的轨迹采集与统一表示
- 支持基于真实 sandbox 状态的 reward 计算，而不只依赖最终文本
- 支持将轨迹与 reward 转换为 Atropos/GRPO 可消费的训练样本
- 通过配置与注册表实现 `env`、`reward`、`judge`、`trainer` 的可插拔
- 为未来扩展到 `OpenClaw-RL` 风格的异步采集、异步判分、异步训练预留接口

## 非目标
以下能力明确不属于第一阶段交付范围：

- 在线持续学习服务
- 异步 judge 服务与多数投票
- 从真实 Hermes 会话自动回灌训练
- OPD / hindsight distillation
- 多节点训练调度与资源编排
- UI、dashboard、可视化控制台
- 覆盖所有 terminal backend 的大规模兼容矩阵

## 外部参考与设计依据
### OpenClaw-RL 的关键启发
参考 `OpenClaw-RL` 的思路，核心启发有三点：

1. 将 `agent serving`、`rollout collection`、`judging`、`policy training` 解耦
2. 将多轮会话组织为可训练的 session-aware trajectory
3. 将来自用户、环境、工具反馈的后继状态视为天然训练信号

### Hermes Agent 的现有基础
`hermes-agent` 已具备第一阶段所需的核心构件：

- `environments/` 与 Atropos 集成
- `HermesAgentLoop` 多轮工具调用执行引擎
- `ToolContext`，可在同一 sandbox 内做 reward 复验
- trajectory 与 benchmark / data generation 通路

因此，第一阶段最合理的路径不是复制 `OpenClaw-RL` 的完整异步训练系统，而是建立一个面向 `hermes-agent` 的框架层，使 Hermes 现有环境能力可以被统一组织成可训练流水线。

## 总体方案
### 方案选择
本设计采用“`Hermes 原生框架 + 异步化预留`”路线：

- 第一阶段：以 `hermes-agent/environments` 为底座，构建原生 agentic RL 框架
- 第二阶段：在 collector / judge / trainer 边界稳定后，扩展到异步侧车架构

### 总体架构
推荐目录结构如下：

```text
hermes-agentic-rl/
├── README.md
├── pyproject.toml
├── hermes_agentic_rl/
│   ├── core/
│   │   ├── types.py
│   │   ├── registry.py
│   │   ├── trajectory.py
│   │   ├── rollout_manager.py
│   │   ├── reward_manager.py
│   │   └── trainer_bridge.py
│   ├── envs/
│   │   ├── base_env.py
│   │   ├── terminal_task_env.py
│   │   ├── tooluse_task_env.py
│   │   └── swe_task_env.py
│   ├── rewards/
│   │   ├── base.py
│   │   ├── outcome_reward.py
│   │   ├── process_reward.py
│   │   ├── toolcall_reward.py
│   │   └── aggregate.py
│   ├── judges/
│   │   ├── base.py
│   │   ├── heuristic_judge.py
│   │   └── llm_judge.py
│   ├── trainers/
│   │   ├── base.py
│   │   ├── atropos_grpo.py
│   │   └── offline_sft_exporter.py
│   ├── collectors/
│   │   ├── conversation_collector.py
│   │   └── trajectory_adapter.py
│   └── cli/
│       └── main.py
├── configs/
│   ├── terminal_grpo.yaml
│   ├── tooluse_grpo.yaml
│   └── swe_grpo.yaml
├── examples/
│   └── minimal_terminal_task.py
└── tests/
```

## 核心设计原则
### 环境优先
第一阶段以 `Env` 为训练主入口，而不是直接从 session 日志训练。

原因：

- Hermes 已经具备环境抽象与 Atropos 对接能力
- reward 可以利用 `ToolContext` 复验 agent 实际行为
- 更容易快速得到可运行的最小闭环

### 轨迹优先
`Trajectory` 是跨模块的唯一事实来源。

所有与训练相关的组件都围绕统一轨迹结构工作：

- rollout 只负责生成轨迹
- reward 只负责从轨迹与 sandbox 状态中评估信号
- trainer 只负责消费轨迹与 reward 摘要

### 奖励可组合
reward 不写死为单一分数，而是拆成多个结构化组件，再统一聚合：

- `OutcomeReward`
- `ToolcallReward`
- `ProcessReward`
- 未来可加 `SafetyReward`
- 未来可加 `FormatReward`

### 训练后端可插拔
第一阶段主训练后端是 Atropos / GRPO，但接口不能绑定单一实现。

因此通过 `TrainerBridge` 与 `BaseTrainer` 抽象训练消费接口，未来可加：

- 异步在线 trainer
- offline SFT exporter
- DPO / preference style exporter

### 面向异步演进
虽然第一阶段不实现异步四段式架构，但 collector、judge、trainer 的接口设计必须允许后续拆分为独立进程或独立服务。

## 模块职责
### `core/`
#### `types.py`
定义核心数据结构：

- `TaskSpec`
- `RolloutStep`
- `Trajectory`
- `RewardResult`
- `RewardSummary`
- `TrainBatch`

#### `registry.py`
负责注册与查找：

- `env`
- `reward`
- `judge`
- `trainer`

通过字符串名称完成配置驱动装配。

#### `trajectory.py`
统一 trajectory 的构造、序列化、反序列化与元数据扩展，兼容：

- `messages`
- `tool_calls`
- `tool_results`
- `reasoning`
- `final_output`
- `metadata`

#### `rollout_manager.py`
负责：

- 调用 `HermesAgentLoop`
- 控制最大 turn、超时与异常收敛
- 采集完整 rollout 轨迹
- 返回标准化 `Trajectory`

不负责 reward 聚合，不负责训练提交。

#### `reward_manager.py`
负责：

- 执行多个 reward 组件
- 聚合为最终分数
- 输出 breakdown 与解释信息

#### `trainer_bridge.py`
负责：

- 将 `Trajectory + RewardSummary` 映射到训练后端所需格式
- 第一阶段输出 Atropos/GRPO 训练样本
- 同时支持 JSONL 导出用于 debug 与离线分析

### `envs/`
#### `base_env.py`
定义环境统一接口：

- `setup()`
- `get_next_item()`
- `format_prompt(item)`
- `compute_reward(...)`
- `evaluate()`

#### `terminal_task_env.py`
第一阶段主环境，也是第一阶段唯一必须实现的具体环境。

面向终端任务，例如：

- 创建文件
- 修改文件
- 执行脚本
- 修复简单命令或 repo 状态

#### `tooluse_task_env.py`
第二阶段预留，关注：

- 工具选择是否正确
- 参数是否正确
- 调用顺序是否合理

#### `swe_task_env.py`
第二阶段预留，关注：

- 测试结果
- repo diff
- 修复质量

### `rewards/`
#### `base.py`
定义统一 reward 接口。

#### `outcome_reward.py`
根据任务是否真正完成打分，例如：

- 文件是否存在
- 内容是否匹配
- 命令是否成功
- 测试是否通过

#### `toolcall_reward.py`
根据工具调用质量打分，例如：

- 是否使用了允许的工具
- 参数是否合理
- 是否出现明显无效调用
- 是否存在浪费型调用序列

#### `process_reward.py`
第一阶段可仅预留接口；如果实现，则用于评估：

- 是否陷入无效循环
- 是否过早停止
- 是否频繁产生错误调用

#### `aggregate.py`
第一阶段采用 `weighted_sum`。

要求：

- 输出 `final_score`
- 保留各子项 `score / weight / reason`
- 可供训练日志直接消费

### `judges/`
judge 不单独挂在主流程上，而作为某些 reward 的内部依赖。

#### `heuristic_judge.py`
第一阶段默认实现，特点：

- 稳定
- 成本低
- 可测试
- 可解释

#### `llm_judge.py`
第二阶段扩展，用于弱可验证质量信号。

### `trainers/`
#### `base.py`
定义统一训练接口：

- `submit(...)`
- `flush()`
- `close()`

#### `atropos_grpo.py`
第一阶段主实现：

- 接收标准化 trajectory
- 转为 Atropos/GRPO 所需样本结构
- 支持写入训练数据与调试数据

#### `offline_sft_exporter.py`
第二阶段附近可启用，用于导出高质量轨迹至 SFT 数据集。

### `collectors/`
#### `conversation_collector.py`
第一阶段仅预留接口，未来用于从真实 Hermes 会话采样轨迹。

#### `trajectory_adapter.py`
用于把不同来源的轨迹统一成框架内部的 `Trajectory`。

### `cli/`
建议提供以下入口：

- `train`
- `rollout`
- `evaluate`
- `export`

第一阶段可先实现最小命令集，优先保证可用性而非完整性。

## 数据流
### 主数据流
```text
Task Dataset / Task Generator
    ↓
Env.format_prompt()
    ↓
RolloutManager → HermesAgentLoop
    ↓
Trajectory Capture
    ↓
RewardManager
    ├── OutcomeReward
    ├── ToolcallReward
    └── optional ProcessReward
    ↓
TrainerBridge
    ↓
Atropos GRPO / JSONL Export
```

### 单次 rollout 流程
1. `Env` 获取一个任务样本
2. `format_prompt()` 生成发给 Hermes 的用户任务
3. `RolloutManager` 调用 `HermesAgentLoop` 执行多轮 agent 流程
4. 采集完整 `Trajectory`
5. `RewardManager` 基于轨迹、最终输出、工具结果与 sandbox 状态计算 reward
6. `TrainerBridge` 将样本转换为训练后端可消费格式
7. 输出训练样本、调试 JSONL 与评估日志

## 配置方案
### 配置形态
第一阶段使用 `YAML + dataclass` 双层配置：

- YAML 负责实验切换
- dataclass 负责默认值与类型约束

### 配置分层
#### `runtime`
包括：

- provider
- model
- temperature
- `max_agent_turns`
- terminal backend
- timeout
- sandbox lifetime

#### `environment`
包括：

- environment 类型
- dataset 路径
- prompt 模板
- system prompt
- 评估方式

#### `reward`
包括：

- 聚合器类型
- reward 组件列表
- 权重
- 是否输出 breakdown

#### `trainer`
包括：

- trainer 类型
- export 路径
- batch size
- 并发
- checkpoint / logging

### 示例配置
```yaml
runtime:
  provider: openai
  model: gpt-4.1
  temperature: 0.7
  max_agent_turns: 20
  terminal_backend: local
  terminal_timeout: 120

environment:
  type: terminal_task_env
  dataset_path: data/minimal_terminal_tasks.jsonl
  system_prompt: null

reward:
  aggregator: weighted_sum
  components:
    - name: outcome_reward
      weight: 0.7
    - name: toolcall_reward
      weight: 0.3

trainer:
  type: atropos_grpo
  export_trajectory_path: outputs/trajectories.jsonl
  export_training_path: outputs/train_samples.jsonl
  save_reward_breakdown: true
```

## 接口草图
### 核心数据结构
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RolloutStep:
    turn_index: int
    assistant_message: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    prompt: str
    steps: List[RolloutStep]
    final_output: Optional[str]
    finished_naturally: bool
    turns_used: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardResult:
    name: str
    score: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `BaseEnv`
```python
class BaseEnv:
    async def setup(self) -> None:
        ...

    async def get_next_item(self) -> Dict[str, Any]:
        ...

    def format_prompt(self, item: Dict[str, Any]) -> str:
        ...

    async def compute_reward(
        self,
        item: Dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> List[RewardResult]:
        ...
```

约束：

- `Env.compute_reward()` 产出的是环境侧 reward signal，而不是最终聚合分数
- 环境侧 reward signal 可以直接返回多个 `RewardResult`
- 最终 reward 聚合始终属于 `RewardManager`
- `Env` 不负责训练提交

### `BaseReward`
```python
class BaseReward:
    name: str

    async def evaluate(
        self,
        item: Dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        ...
```

### `RolloutManager`
```python
class RolloutManager:
    async def collect(self, item: Dict[str, Any], prompt: str) -> Trajectory:
        ...
```

约束：

- 只负责驱动 Hermes rollout
- 只负责收集标准化轨迹
- 不参与 reward 与 trainer 逻辑

### `RewardManager`
```python
class RewardManager:
    async def evaluate(
        self,
        item: Dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> Dict[str, Any]:
        ...
```

建议返回：

```python
{
    "final_score": 0.82,
    "components": [...],
    "metadata": {...},
}
```

### `BaseTrainer`
```python
class BaseTrainer:
    async def submit(
        self,
        item: Dict[str, Any],
        trajectory: Trajectory,
        reward_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...
```

第一阶段可先实现：

- 写标准化 JSONL
- 转换为 Atropos/GRPO 样本

## 第一阶段最小实现范围
### 必做
- 统一 trajectory 数据结构
- `BaseEnv`
- `TerminalTaskEnv`
- `OutcomeReward`
- `ToolcallReward`
- `RewardManager`
- `RolloutManager`
- `AtroposGrpoTrainer`
- JSONL trajectory exporter
- 1 个最小示例环境
- 基础测试

### 第一阶段仅预留
- `TooluseTaskEnv`
- `SweTaskEnv`
- `ProcessReward`
- `heuristic_judge` 之外的 judge 实现

### 暂缓
- 异步 judge
- 会话回灌训练
- OPD / hindsight distillation
- 多节点训练编排
- dashboard

## 验收标准
第一阶段完成后，必须满足以下条件：

1. 可以跑通一个最小 terminal agent 任务
2. 可以保存完整 trajectory
3. 可以在同一 sandbox 中复验结果并生成 reward
4. 可以导出 Atropos/GRPO 可消费训练样本
5. 可以通过配置切换 env 与 reward 组合

## 测试策略
### 单元测试
必须覆盖：

- `trajectory.py` 的序列化 / 反序列化
- `registry.py` 的注册与查找
- `aggregate.py` 的加权聚合逻辑
- `toolcall_reward.py` 的合法 / 非法 / 空调用路径

### 集成测试
至少包括：

- mock `HermesAgentLoop` 的最小 rollout 集成测试
- 带工具结果的 reward 集成测试

### 端到端烟测
建议最小任务：

- 创建指定文件
- 写入固定内容

验证项：

- trajectory 落盘成功
- reward 正确
- training export 成功

## 约束与边界
为避免后续演进时耦合失控，本设计固定以下约束：

- `core/` 不直接依赖具体 `env` 实现
- `reward` 不直接写文件
- `trainer` 不直接调用 Hermes rollout
- `Trajectory` 是跨模块唯一事实来源
- 所有可替换组件都通过 `registry` 装配

## 后续演进路线
### 第二阶段
- 引入 `conversation_collector`
- 支持从真实 Hermes 会话中提取训练轨迹
- 将 judge 从 reward 内部依赖升级为可异步执行的组件

### 第三阶段
- 引入 `OpenClaw-RL` 风格的异步四段式架构
- 支持异步 rollout、异步 judging、异步训练提交
- 支持 OPD / hindsight / richer process reward

## 风险与缓解
### 风险一：过早做成通用平台导致实现失焦
缓解：

- 第一阶段仅围绕 `TerminalTaskEnv` 打通闭环
- 明确其余 env 为预留，不强行同时实现

### 风险二：reward 过度依赖文本判断，无法真实反映 agent 行为
缓解：

- 尽量优先使用 `ToolContext` 复验文件、命令、测试与环境状态
- LLM judge 只作为第二阶段补充

### 风险三：训练后端接口被 Atropos 细节污染
缓解：

- 所有训练转换逻辑集中于 `trainer_bridge.py` 与 `trainers/`
- `core` 中的数据结构保持训练后端无关

## 实现顺序建议
建议按以下顺序推进实现：

1. `core/types.py` 与 `trajectory.py`
2. `BaseEnv` 与 `TerminalTaskEnv`
3. `OutcomeReward`、`ToolcallReward`、`RewardManager`
4. `RolloutManager`
5. `AtroposGrpoTrainer` 与 JSONL exporter
6. `example`
7. `unit + integration + smoke tests`

## 结论
`hermes-agentic-rl` 第一阶段应被定义为一个建立在 `hermes-agent` 原生环境体系之上的 agentic RL 框架层，而不是 `OpenClaw-RL` 异步训练系统的直接复制品。

该方案既能快速获得最小可运行闭环，也能在接口层面为未来演进到异步在线训练架构预留足够空间。
