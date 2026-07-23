# hermes-agentic-rl 项目深度分析

> 分析日期：2026-07-07  
> 分析锚点：当前分支 `feat/v0.12-engineering-hardening`，版本 `0.11.0`  
> 项目路径：`/Users/gatilin/PycharmProjects/hermes-agentic-rl`  
> 代码规模：`hermes_agentic_rl/` 下约 **38,782 行 Python 代码**，**181+ 模块**，**53+ 测试文件**

---

## 1. 整体架构

### 1.1 项目定位

`hermes-agentic-rl` 是围绕 **Hermes-agent**（执行层）构建的 **RL 学习层**，将 Hermes 真实运行轨迹转化为 _可训练 / 可评估 / 可促晋_ 的反馈闭环。它**不是**通用 LLM 微调框架，而是专门面向 agentic RL（工具调用、终端命令、多轮回复）的垂直框架。

### 1.2 模块结构图

```
hermes_agentic_rl/                      ~38,782 LOC, 181+ modules
├── cli/              11   训练/评估/在线循环 CLI 入口
├── core/              7   Trajectory / RolloutStep / RewardResult /
│                          RolloutManager / RewardManager / TrainerBridge / Registry
├── algos/            18   GRPO / PPO / RLOO / OPD / Hybrid / Factored /
│                          SimPO / GSPO / Best-of-N + common/ 工具
├── trainers/         33   OnPolicyTrainer (skeleton ~1960 行) →
│                          GRPO/PPO/HybridTrainer 薄包装
├── backends/          6   LLMBackend protocol + Tiny / HF / vLLM / batch_generate
├── runtime/           7   fake / hermes adapter + wrapper + 多入口探测
├── envs/             16   echo / sim_tool / curriculum / letter_counting /
│                          hermes_reasoning_traces / multi_stream / code_fix / sandbox
├── rewards/          26   outcome / toolcall / fs_verifier / next_turn_feedback /
│                          PRM / NextStatePRM / RewardModel / lagrangian / shaping /
│                          opd_hint_extractor / judge_cache / dynamic_balancer / composer
├── eval/              6   rl_eval / capability_axes / harness / ab_test / benchmark_suite
├── offline/           5   BC / DPO / replay_buffer / per_buffer
├── collectors/        8   sidecar / replay_export / replay_quality / session_judge /
│                          preference_mining / conversation_collector / trajectory_adapter
├── exporters/         3   self-evolution / atropos jsonl
├── monitor/           4   JSONL / TensorBoard / W&B / live dashboard (stdlib HTTP)
├── distributed/       5   MP / Ray / ExperienceQueue / fault_tolerant_pool
├── peft/              2   LoRA injection (target_patterns + merge_into_base)
├── mdp/               4   PromptStateEncoder / observation / action_space
├── integrations/      6   atropos / hermes preflight + repo 路径解析
├── agent_loop/        4   PolicyAgentLoop / MultiTurnAgentLoop（<tool_call> 协议）
├── datasets/          3   JSONL / HF parquet 加载器
├── framework/         2   EnvTrainingPipeline / SessionTrainingPipeline
└── utils/             1   coerce 工具
```

### 1.3 核心目录职责

| 目录 | 职责 | 关键文件 |
|---|---|---|
| `core/` | 数据契约与核心编排 | `types.py`, `rollout_manager.py`, `reward_manager.py`, `trainer_bridge.py` |
| `algos/` | 纯函数算法实现 | `grpo.py`, `ppo.py`, `opd.py`, `hybrid.py`, `common/loss.py` |
| `trainers/` | 训练循环骨架与工程特性 | `on_policy.py`, `grpo_trainer.py`, `async_loop.py`, `checkpoint.py` |
| `backends/` | LLM 推理与训练后端协议 | `base.py`, `tiny.py`, `hf.py`, `vllm_backend.py` |
| `envs/` | 环境接口与具体实现 | `base_env.py`, `sim_tool_env.py`, `curriculum.py` |
| `rewards/` | 可组合奖励组件 | `base.py`, `composer.py`, `toolcall_reward.py`, `outcome_reward.py` |
| `runtime/` | 外部运行时适配 | `hermes_adapter.py`, `fake_adapter.py` |
| `framework/` | 高层流水线编排 | `pipeline.py` |
| `collectors/` | 数据采集与落盘 | `sidecar.py`, `replay_export.py` |
| `distributed/` | 分布式 rollout | `mp_pool.py`, `ray_runner.py`, `experience_queue.py` |

### 1.4 分层视图

```
┌─────────────────────────────────────────────────────────────┐
│ CLI 层  (cli/)                                               │
│  main.py + train_rl.py + online_cycle_cli.py + ...          │
└─────────────┬───────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ Trainer 层  (trainers/)                                      │
│  OnPolicyTrainer  ◀── 共享 rollout→loss→step skeleton      │
│  GRPOTrainer / PPOTrainer / HybridTrainer  ◀── 薄包装       │
│  AsyncLoop ◀── 异步训练（ExperienceQueue 消费）               │
│  CheckpointManager  ◀── {model, optim, rng, stats, kl, rms} │
└───────┬──────────────────────────────────────────────┬──────┘
        ↓                                              ↓
┌───────────────┐  ┌───────────────────┐  ┌──────────────────┐
│ Algo (algos/) │  │ Rewards (rewards/)│  │ Backend          │
│  GRPO         │  │  RewardManager    │  │  LLMBackend 协议 │
│  PPO + GAE    │  │  多组件加权       │  │  tiny / hf       │
│  RLOO         │  │  PRM/NextStatePRM │  │  value head      │
│  OPD          │  │  RewardModel head │  │  LoRA inject     │
│  Hybrid       │  │  Lagrangian       │  │  vLLM rollout    │
│  Factored     │  │  JudgeCache       │  └──────────────────┘
│  SimPO/GSPO   │  │  ProcessReward    │
└───────────────┘  └───────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ Env + MDP + Agent Loop                                       │
│  BaseEnv 子类（含 CurriculumEnv / MultiStreamEnv）          │
│  PolicyAgentLoop / MultiTurnAgentLoop（<tool_call> 协议）    │
│  PromptStateEncoder（tokenizer-agnostic）                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.5 四个核心架构原则

1. **算法是纯函数**：`BaseAlgo.compute_loss(policy, ref, batch) → (loss, AlgoUpdateStats)`，不持有 optimizer，可单元测试，可热替换。
2. **Backend 用 protocol，不用继承**：任何实现 `generate / score / score_batch / trainable_parameters` 的对象都是合法 policy。
3. **Trajectory 仅元数据耦合**：RL loop 只读 `trajectory.metadata["runtime"]["rl"]`，不依赖任何 Hermes 类型 → fake / hermes / atropos runtime 共用同一 trainer。
4. **Opt-in 一切**：LoRA / Lagrangian / value head / reference policy / per-token advantage / 分布式 rollout 全部默认关闭。

---

## 2. 核心组件分析

### 2.1 Agent 定义与生命周期

**环境接口** (`envs/base_env.py`)：

```python
class BaseEnv(ABC):
    @abstractmethod
    async def setup(self) -> None: ...
    @abstractmethod
    async def get_next_item(self) -> dict[str, Any]: ...
    @abstractmethod
    def format_prompt(self, item: dict[str, Any]) -> str: ...
    @abstractmethod
    async def compute_reward(self, item, trajectory, tool_context) -> list[RewardResult]: ...
```

**轨迹数据结构** (`core/types.py`)：
- `RolloutStep`：单轮数据（assistant_message, tool_calls, tool_results, reasoning, metadata）
- `Trajectory`：完整轨迹（task_id, prompt, steps[], final_output, finished_naturally, turns_used, metadata）
- `RewardResult`：单组件奖励（name, score, reason, weight, metadata）
- `RewardSummary`：聚合奖励（final_score, components[], metadata）
- `TrainSample`：训练样本（task_id, prompt, final_output, reward, trajectory, metadata）

**Rollout 收集** (`core/rollout_manager.py`)：
- `RolloutManager` 封装 `agent_loop`，将原始运行输出转换为结构化 `Trajectory`
- 支持按 `turn_index` 显式路由或按位置顺序回退
- 所有运行时的原始数据保留在 `metadata` 中

**Agent Loop** (`agent_loop/` 下)：
- `PolicyAgentLoop`：单轮推理
- `MultiTurnAgentLoop`：多轮交互，含 `<tool_call>` 协议解析

### 2.2 训练循环（GRPO / MTGRPO）

#### 2.2.1 GRPO 算法核心 (`algos/grpo.py`)

**Group Relative Policy Optimization**（DeepSeek-Math / DeepSeek-R1 风格）：

```
A_i = (R_i − μ_g) / (σ_g + ε)   # 组内归一化优势
```

关键设计：
- **无 critic head**：优势纯由组内 reward 统计计算
- **5 种 advantage 归一化**：`group`（默认）/ `dapo`（全等丢弃整组）/ `batch` / `whiten` / `none`
- **非对称裁剪**：`clip_eps_high=0.28 > clip_eps=0.2`（OpenClaw-RL §3.1，缓解策略坍塌）
- **per-token advantage (REINFORCE++)**：`<answer>` 标签后 token 拿到全 reward，之前 gamma 衰减
- **3 种 KL 估计器**：`k1`（直接差）、`k2`（½χ²）、`k3`（TRL/DeepSeek/verl 推荐，无偏非负低方差）
- **3 种 loss 聚合**：`mean_token` / `sum_token` / `dr_grpo`（长度无偏）
- **Off-policy 校正（TIS）**：`tis_rho_clip > 0` 时启用 Truncated Importance Sampling

**批量前向优化**（v0.8 关键性能改进）：
- `policy.score_batch(prompt_ids_list, response_ids_list)` 一次得到 `[B, T_max]` logp + mask
- 在 HF GPT-2 上实现 **5–10× 加速**
- `stack_old_logprobs` + `build_advantage_tensor` 向量化组装

#### 2.2.2 训练器骨架 (`trainers/on_policy.py`)

`OnPolicyTrainer` 是所有 on-policy 训练器的共享骨架（~1960 行），工程特性矩阵：

| 类别 | 特性 | 配置项 |
|---|---|---|
| 学习率 | LR scheduler | `lr_schedule` (constant / linear / cosine / warmup_cosine) |
| 优化 | Update epochs + minibatch | `update_epochs / minibatch_size` |
| trust region | per-minibatch 早停 | `target_kl` |
| trust region | Adaptive KL | `adaptive_kl` (InstructGPT A.2 / PID 控制器) |
| 奖励 | RunningMeanStd 白化 | `normalize_reward` |
| 检查点 | 周期 + 最佳并存 | `checkpoint_every / save_best_checkpoint` |
| 检查点 | Auto resume | `auto_resume / resume_from` |
| 检查点 | Async checkpoint | `async_checkpoint` |
| 早停 | 容忍轮数 | `early_stop_patience / early_stop_min_delta` |
| 抗遗忘 | Interleaved / Bootstrap SFT | `interleave_sft_every / bootstrap_sft_rounds` |
| 精度 | 混合精度 | `amp_dtype` (fp16 / bf16 / fp32 / auto) |
| 吞吐 | 梯度累积 | `grad_accum_steps` |
| 分布式 | FSDP / DDP | `distributed_strategy` |
| 分布式 | MP / Ray rollout pool | `rollout_pool` |
| 加速 | vLLM rollout | `vllm_rollout_model` |
| 多轮 | Multi-turn credit | `multi_turn_credit` (5 模式) |
| 课程 | 自动晋级 | `CurriculumEnv.observe(reward)` |
| 约束 | Lagrangian | `lagrangian.penalty_term(loss)` |
| 异步 | Async training | `AsyncLoop + ExperienceQueue` |
| 流水线 | Pipelined rollout | `pipeline_rollouts` |
| EMA | EMA rollout | `use_ema_rollout` |
| PRM | Process Reward Model | `prm_pipeline` |

#### 2.2.3 `train_mtgrpo.py` 独立训练入口

`train_mtgrpo.py` 是 MT-GRPO（Multi-Turn GRPO）的**独立训练脚本**，直接面向用户场景：

```python
# 核心流程
1. 创建 CurriculumScheduler（三阶段课程：结构 → 内容 → 总结）
2. 创建 Backend（tiny 或 hf）
3. 创建 SimToolEnv（零依赖模拟工具环境）
4. 创建 RewardComposer（ToolcallReward + OutcomeReward）
5. 创建 GRPOTrainer
6. 运行 training loop（含 curriculum 观察和自动晋级）
```

关键参数：
- `--credit-mode`: shared / terminal / discounted / judge / hybrid（多轮 credit 分配）
- `--curriculum-auto-advance`: 自动晋级课程阶段
- `--per-token-advantage`: REINFORCE++ token 级优势

#### 2.2.4 Multi-turn Credit 5 种模式

| 模式 | 每轮 reward 公式 |
|---|---|
| `shared` | 所有轮共享 `final_reward` |
| `terminal` | 仅最后一轮拿 `final_reward`，其余 0 |
| `discounted` | 按 `γ^(N−t−1)` 衰减分配 |
| `judge` | 仅用本地 judge 局部分（`final_weight=0`，`local_weight=1`）|
| `hybrid` | `w_final·final + w_local·local` |

### 2.3 工具 / 环境集成

#### 2.3.1 环境体系

| 环境 | 用途 | 特点 |
|---|---|---|
| `EchoTaskEnv` | 最小验证 | 回显输入，用于 smoke test |
| `SimToolEnv` | 模拟工具调用 | 零外部依赖，算数任务，CPU 可学 |
| `LetterCountingEnv` | 简单计数 | OPD hint 提取的可验证基准 |
| `CurriculumEnv` | 自动晋级 | 移动均值阈值控制难度 |
| `MultiStreamEnv` | 异构任务统一 | 加权自适应重采样 |
| `HermesReasoningTracesEnv` | 真实数据 | HF dataset / parquet 加载 |
| `TerminalTaskEnv` | 终端命令 | 真实 shell 执行 |
| `CodeSandboxEnv` | 代码执行 | SWE 风格 Atropos 集成 |

#### 2.3.2 模拟工具环境 (`envs/sim_tool_env.py`)

零外部依赖，专门测试多轮工具调用循环：

```
Task: "What is 3 + 5? Use calc tool, then answer as: answer=N"
Expected:
  Turn 1: <tool_call>calc(3 + 5)</tool_call>
  Turn 2: answer=8
Reward: 0.4·used_calc + 0.3·tool_result_correct + 0.3·final_answer_correct
```

内置 `safe_eval` 使用 AST 解析（仅支持 `+ - * //`），安全可控。

#### 2.3.3 工具调用奖励 (`rewards/toolcall_reward.py`)

三轴评分系统：
- **name** (weight 0.4): 工具名是否有效
- **schema** (weight 0.3): 参数结构是否正确
- **value** (weight 0.3): 参数值是否合理

```python
score = name_weight * name_score + schema_weight * schema_score + value_weight * value_score
```

支持 `function.arguments` 和 `args` 两种参数格式，自动降级处理。

### 2.4 配置系统

#### 2.4.1 YAML 结构

```yaml
runtime:
  integration: fake | hermes | hf | vllm | sglang | openai | anthropic
  max_agent_turns: 20
  enabled_toolsets: [terminal, file]

environment:
  type: echo | sim_tool | curriculum | letter_counting | hermes_reasoning_traces | multi_stream
  dataset_path: ...

trainer:
  algo: grpo | ppo | hybrid | opd | rloo | factored | simpo | gspo
  n_iters: 20
  group_size: 4
  lr: 1e-3
  # 50+ 可选参数...

reward:
  aggregator: weighted_sum
  components: [...]

backend:
  name: tiny | hf
  model_name: ...
  # LoRA / value_head / flash_attention
```

#### 2.4.2 配置验证 (`config.py`)

- 双重验证路径：Pydantic v2（`pip install '[config]'`）或轻量级 dict 回退
- 验证字段：`runtime.integration` 已知集合、`max_agent_turns` 正整数、`trainer.n_iters/lr` 正数
- `load_config()` 自动填充默认值并运行验证

#### 2.4.3 配置工厂

- `build_shared_on_policy_config()` (v0.9.1): 从任何 trainer-config dataclass 拷贝重叠字段到 `OnPolicyTrainerConfig`，消除 ~180 行重复映射

### 2.5 数据流和状态管理

#### 2.5.1 On-Policy 训练数据流

```
Env.get_next_item() ──→ format_prompt() ──→ Backend.generate() ──→ Trajectory
                                                         │
                                                         ↓
                                              RewardManager.evaluate()
                                                         │
                                                         ↓
                                              RolloutRecord (prompt_ids, response_ids,
                                              old_logprobs, reward, metadata)
                                                         │
                                                         ↓
                                              Algo.compute_loss(policy, ref, batch)
                                                         │
                                                         ↓
                                              optimizer.step()
```

#### 2.5.2 Online Cycle 闭环数据流

```
1. Hermes-agent 真实运行 ──→ session traces
2. Sidecar 异步落盘 ──→ replay JSONL
3. Replay 质量门控 ──→ 过滤低质量记录
4. Worker 训练（BC / DPO / RM）──→ 本地模型更新
5. Self-Evolution 导出 ──→ JSONL + 评估报告
```

#### 2.5.3 AsyncLoop 架构（v0.10+）

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│ Rollout     │────→│ ExperienceQueue │────→│ Learner     │
│ Workers     │     │ (bounded, drop) │     │ (AsyncLoop) │
└─────────────┘     └─────────────────┘     └─────────────┘
       ↑                                          ↓
       └──────────── 权重广播 ←────────────────────┘
```

- **staleness accounting**: `learner_version − behavior_version`
- **staleness-gated TIS**: staleness > threshold 时启用 off-policy 校正
- **queue 可观测性**: `queue_depth`, `queue_lag`, `queue_drops`

#### 2.5.4 奖励系统数据流

```
RewardComposer (有状态，运行统计)
  ├── Per-component running normalization (Welford 在线均值/方差)
  ├── Conditional activation (按课程级别或元数据条件触发)
  └── Distance-based discount (γ^turns_used)
  
→ 输出 RewardSummary (final_score, components[], metadata)
```

**RewardComposer** 是 `RewardManager` 的 drop-in 替代，额外提供：
- 并行/串行评估模式
- 条件失败追踪（连续 5 次失败后自动禁用组件）
- 运行统计 snapshot（用于日志和调试）

#### 2.5.5 检查点状态 (`trainers/checkpoint.py`)

完整状态保存：
```python
{
    "model": model_state_dict,
    "optimizer": optim_state_dict,
    "rng": rng_state,
    "stats": train_stats,
    "kl_controller": kl_state,
    "rms": running_mean_std_state,
}
```

支持 async checkpoint（后台线程写盘）、best checkpoint 永不修剪、auto resume。

---

## 3. 当前痛点与可优化点

### 3.1 `train_mtgrpo.py` 训练循环设计缺陷

**问题**：`run_training()` 中存在一个**逻辑断裂**的循环：

```python
for iter_idx in range(args.n_iters):
    batch_stats = {}  # 始终为空字典！
    curriculum.observe_batch_stats(iter_idx, batch_stats)  # 无实际数据输入
    if curriculum.should_advance():
        curriculum.advance()  # 几乎永远不会触发，因为 batch_stats 为空
# ... 然后才调用 trainer.train()
```

`trainer.train()` 内部的迭代与外部的 curriculum 观察循环**完全分离**，导致：
- Curriculum 的 `batch_stats` 始终为空，无法正确判断阶段晋级
- 外部循环的空跑消耗 `n_iters` 次迭代但不做任何实际训练
- `trainer.train()` 内部有自己的独立循环，两个循环各自为政

**建议**：将 curriculum 观察逻辑集成到 `OnPolicyTrainer` 的 `_one_iter()` 回调中，通过 `metrics_sink` 或 `_after_iter_hook` 传递真实 batch stats。

### 3.2 RewardManager vs RewardComposer 接口不一致

**问题**：
- `RewardManager` 的 `__init__` 签名是 `(self, rewards: list[BaseReward])`
- `RewardComposer` 的 `__init__` 签名是 `(self, components: list[BaseReward], config: ...)`
- 两者都暴露 `async evaluate(item, trajectory, tool_context) -> RewardSummary`，但 `RewardComposer` 内部使用 `_shape_one` + `_aggregate`，而 `RewardManager` 直接调用 `weighted_sum`
- `train_mtgrpo.py` 中创建了一个空的 `RewardManager`（`components=[]`），却传入 `GRPOTrainer`，这意味着训练时奖励始终为 0

**建议**：统一接口为 `BaseRewardEvaluator` Protocol，确保 `RewardComposer` 和 `RewardManager` 完全互换；修复 `train_mtgrpo.py` 中的空 `RewardManager` 问题。

### 3.3 配置验证深度不足

**问题**：
- 当前 `config.py` 的 dict fallback 仅验证 4 个字段（integration, max_turns, n_iters, lr）
- 即使启用 Pydantic，也仅覆盖顶层结构，对 `trainer` 内 50+ 参数的交叉约束（如 `group_size >= 2` 否则 GRPO group norm 失效）缺乏验证
- `train_mtgrpo.py` 的 argparse 和 YAML 配置两套系统并存，易产生不一致

**建议**：
- 在 Pydantic model 中添加字段级 validator（如 `group_size` 必须 `>= 2` 当 `algo=grpo` 时）
- 统一 argparse 和 YAML 配置源，优先级明确

### 3.4 `BaseReward.__init__` 设计模式问题

**问题**：
- `BaseReward.__init__` 虽然修复了 `weight` 参数传递问题（v3.0 修复），但采用 `**kwargs` 模式导致子类签名不透明
- `BaseReward` 的 `name` 是类属性（`name = "..."`），但部分子类在 `__init__` 中覆盖，部分不覆盖，一致性差
- `evaluate()` 的 `tool_context` 参数在许多实现中被忽略（`del tool_context`），但接口保留它

**建议**：
- 将 `name` 设为 `@abstractproperty` 或 `@classmethod` 强制子类声明
- 考虑将 `tool_context` 移入 `Trajectory.metadata` 以减少参数冗余

### 3.5 课程调度器与训练器的耦合松散

**问题**：
- `CurriculumScheduler` 是独立状态机，但 `OnPolicyTrainer` 对其无原生感知
- `CurriculumEnv` 的 `observe(reward)` 需要环境自己维护状态，但训练器通过 `reward_manager` 间接控制奖励
- `MultiStreamEnv` 的 per-stream 权重调整逻辑分散在 `on_policy.py`、环境类和配置中

**建议**：
- 在 `OnPolicyTrainer` 中增加 `_curriculum_hook(iter_idx, batch_stats)` 回调点
- 将 `CurriculumScheduler` 作为 trainer 的一等公民（类似 `kl_controller`）

### 3.6 Tiny Backend 能力天花板

**问题**：
- `TinyCausalLMBackend` 默认 dim=32, n_layers=2, vocab_size≈100，参数量 <100K
- 字符级 tokenizer 无法处理真实语言任务，仅用于 smoke test
- `train_mtgrpo.py` 的默认 backend 是 `tiny`，用户可能误以为这是 production 配置

**建议**：
- 在 `--backend tiny` 时打印醒目的 capacity warning
- 添加 `--backend hf` 作为默认推荐的快速启动路径

### 3.7 测试对外部模型的依赖

**问题**：
- 部分测试依赖 HF GPT-2 下载，CI 慢且不稳定
- 缺少纯 mock backend 的端到端测试覆盖
- `test_mtgrpo.py` 不存在，`train_mtgrpo.py` 无直接测试

**建议**：
- 为 `train_mtgrpo.py` 添加基于 `FakeBackend` 的集成测试
- 增加更多 `TinyBackend` 端到端测试替代 HF 依赖

### 3.8 mypy 深层类型问题（剩余 11 个）

**问题**：
- `stack_cached_logprobs` 的 `None` 处理涉及复杂设计模式
- `_DataclassT` 的 `replace` 类型变量问题
- `SFTMixin` 通过 `TYPE_CHECKING` 声明属性，长期脆弱

**建议**：
- 逐步收紧 mypy 配置（`disallow_untyped_defs = true`）
- 将 `SFTMixin` 重构为 Protocol 或显式接口

### 3.9 版本迭代快导致的 API 不稳定

**问题**：
- 版本迭代密度极高（~5 天一个 minor），v0.9→v0.11 间 OPD 大改、Hybrid 重构
- `GRPOTrainerConfig` 已从 v0.2 的 10 个字段膨胀到 50+ 个字段
- 测试文件按版本命名（`test_v08_*.py`, `test_v09_*.py`）暗示 API 不稳定性

**建议**：
- 发布 v1.0 路线图，冻结核心 API（`BaseAlgo`, `LLMBackend`, `BaseEnv`）
- 新增特性走 `@experimental` 标记，至少保留一个 minor 周期的兼容性

### 3.10 性能基准缺失

**问题**：
- 缺少系统级的 throughput / memory / scaling 基准测试套件
- `benchmarks/` 目录仅存在 `.benchmarks/`（pytest cache），无实际 benchmark 代码
- 无法量化 `score_batch` 相比 `score` 的 5-10× 加速在真实场景中的表现

**建议**：
- 添加 `pytest-benchmark` 套件，覆盖不同 backend（Tiny/HF/vLLM）和 batch size
- 添加 memory profiling 基准，监控峰值显存使用

---

## 4. 技术栈概览

### 4.1 主要依赖库

| 类别 | 库 | 用途 |
|---|---|---|
| 深度学习 | `torch>=2.1` | 训练后端、张量运算 |
| 模型推理 | `transformers>=4.40` | HF AutoModelForCausalLM |
| 数据处理 | `datasets>=2.19`, `pyarrow>=14` | HF dataset / parquet 加载 |
| 配置解析 | `PyYAML>=6.0` | YAML 配置 |
| 可选验证 | `pydantic>=2.7` | 类型安全配置验证 |
| 可观测性 | `tensorboard>=2.14`, `wandb>=0.17` | 训练曲线、实验管理 |
| HTTP | `requests>=2.33`, `httpx>=0.28` | Hermes/Atropos 集成 |
| WebSocket | `websockets>=16` | 实时通信 |
| 可选推理 | `vllm` | 分布式 rollout 后端 |

### 4.2 训练和推理后端

| 后端 | 定位 | 训练 | 推理 | 适用场景 |
|---|---|---|---|---|
| **Tiny** | 自包含小模型 | ✅ AdamW | ✅ greedy/multinomial | Smoke test、单元测试 |
| **HF** | HuggingFace CausalLM | ✅ full/LoRA | ✅ generate | 本地训练主路径 |
| **vLLM** | 高性能推理引擎 | ❌（仅推理） | ✅ paged attention | 分布式 rollout |
| **OpenAI/Anthropic** | 远端 API | ❌ | ✅ 通过 adapter | Online cycle 数据采集 |

**后端协议** (`backends/base.py`):
- `generate(prompt_ids, max_new_tokens, temperature, seed, stop_strings) -> GenerationOutput`
- `score(prompt_ids, response_ids, temperature) -> torch.Tensor`
- `score_batch(...) -> (logprobs [B,T], mask [B,T])`
- `trainable_parameters() -> Iterable[nn.Parameter]`

### 4.3 开发工具链

| 工具 | 配置 | 状态 |
|---|---|---|
| **ruff** | `target-version = "py311"`, `line-length = 100` | ✅ 0 错误（v3.0 修复）|
| **mypy** | `python_version = "3.11"`, soft start | ⚠️ 剩余 11 个深层错误 |
| **pytest** | `minversion = "7.4"`, markers: integration/slow/benchmark | ✅ 53+ 测试文件 |
| **coverage.py** | `fail_under = 45` | ✅ 配置完整 |
| **pre-commit** | ruff + mypy | ✅ 存在 |
| **pip-audit** | 依赖安全审计 | ✅ 存在 |
| **Docker** | 基础 Dockerfile + docker-compose | ⚠️ 较简单 |
| **Makefile** | 常用命令封装 | ✅ 存在 |
| **Sphinx** | 文档构建 | ✅ 配置存在 |

### 4.4 算法矩阵

| 算法 | 核心公式 | 适用场景 | 状态 |
|---|---|---|---|
| **GRPO** | `A_i = (R_i − μ_g)/(σ_g + ε)` | 默认主力，无 critic | ⭐⭐⭐⭐⭐ |
| **PPO** | GAE + clipped value loss | 有稳定 critic 场景 | ⭐⭐⭐⭐⭐ |
| **RLOO** | `b_i = Σ_{j≠i} R_j/(G−1)` | G≥3 方差更低 | ⭐⭐⭐⭐ |
| **OPD** | `A_t^OPD = log π_T(a_t\|s+hint) − log π_θ(a_t\|s)` | 有 next-state hint | ⭐⭐⭐⭐⭐ (v0.11 闭环) |
| **Hybrid** | `L = w_RL·L_GRPO + w_OPD·L_OPD` | 评估+指令融合 | ⭐⭐⭐⭐⭐ |
| **Factored** | `L = Σ_h w_h·L_clip(π_h_new, π_h_old, A)` | 5 头分解动作空间 | ⭐⭐⭐⭐ |
| **SimPO** | `L = −log σ(β·(R̃_w − R̃_l − γ))` | 偏好优化 | ⭐⭐⭐⭐ |
| **GSPO** | Group-based SimPO | 组内偏好优化 | ⭐⭐⭐⭐ |

### 4.5 奖励组件矩阵

| 组件 | 信号类型 | 密度 | 关键文件 |
|---|---|---|---|
| `OutcomeReward` | 最终输出正确性 | 稀疏 | `rewards/outcome_reward.py` |
| `ToolcallReward` | JSON 格式/参数匹配 | 半稠密 | `rewards/toolcall_reward.py` |
| `FilesystemVerifierReward` | 文件系统状态验证 | 稀疏 | `rewards/fs_verifier_reward.py` |
| `NextTurnFeedbackReward` | 下一回合反馈 | 半稠密 | `rewards/next_turn_feedback.py` |
| `PRM` | 过程奖励模型 | 稠密（步骤级）| `rewards/prm.py` |
| `NextStatePRM` | 基于 next-state 的 PRM | 稠密 | `rewards/next_state_prm.py` |
| `RewardModel` | 学习式奖励模型 | 稠密 | `rewards/reward_model.py` |
| `LagrangianReward` | 约束惩罚（RCPO）| 稠密 | `rewards/lagrangian.py` |
| `LengthPenaltyReward` | 长度惩罚 | 稠密 | `rewards/length_penalty.py` |
| `DynamicRewardBalancer` | 自适应权重平衡 | 元 | `rewards/dynamic_reward_balancer.py` |
| `MemoryRewardShaper` | 历史轨迹塑形 | 元 | `rewards/memory_reward_shaper.py` |
| `RewardComposer` | 组合+归一化+条件激活 | 元 | `rewards/composer.py` |

---

## 5. 附录

### 5.1 阅读源码推荐路径

| 顺序 | 文件 | 理由 |
|---|---|---|
| 1 | `core/types.py` | 所有数据契约的源头 |
| 2 | `backends/base.py` | LLMBackend protocol，理解 generate/score 双路径设计 |
| 3 | `envs/base_env.py` | 环境接口，理解 Agent 生命周期 |
| 4 | `core/rollout_manager.py` | Trajectory 收集逻辑 |
| 5 | `rewards/base.py` + `composer.py` | 奖励系统扩展点 |
| 6 | `algos/grpo.py` | 主力算法实现 |
| 7 | `trainers/grpo_trainer.py` | Trainer 薄包装，理解 config 映射 |
| 8 | `framework/pipeline.py` | 高层流水线编排 |
| 9 | `train_mtgrpo.py` | 端到端使用示例 |
| 10 | `curriculum.py` | 课程学习调度 |

### 5.2 关键环境变量

```bash
export HERMES_AGENT_REPO=/path/to/hermes-agent
export ATROPOS_REPO=/path/to/atropos
export NEWAPI_API_KEY=...
export WANDB_API_KEY=...
export PYTORCH_ENABLE_MPS_FALLBACK=1  # macOS MPS 训练
```

### 5.3 常用 CLI 命令

```bash
# 预检
python -m hermes_agentic_rl.cli.main hermes-preflight
python -m hermes_agentic_rl.cli.main atropos-preflight

# 训练
python -m hermes_agentic_rl.cli.main train-rl --config <yaml> --output <dir>
python train_mtgrpo.py --backend tiny --n-iters 5

# 评估
python -m hermes_agentic_rl.cli.main eval-rl --config <yaml>
python -m hermes_agentic_rl.cli.main eval-gate --config <yaml>  # exit 3 = 阻断

# 在线循环
python -m hermes_agentic_rl.cli.main online-cycle --config <yaml> --once --limit 1
```

---

*本报告基于源码逐层穿读（181+ 模块）、现有深度分析报告（v3.0）和 README/CHANGELOG 交叉验证生成。*
