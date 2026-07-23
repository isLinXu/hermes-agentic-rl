# hermes-agentic-rl scorecard 2026-06-26

## 结论

当前仓库已经具备本地训练、最小评估和关键测试的可执行能力，但“真实 Hermes held-out tool-use benchmark 的重新现跑”仍受数据入口约束。当前最可靠的综合判断是：

- 本地工程环境：已就绪
- 最小离线闭环：已跑通
- prompt/context benchmark：已跑通
- 真实 tool-use held-out benchmark：已有历史强证据，可复用现有产物；本次重新现跑被数据访问问题阻塞

## 本次实际执行

### 1. 环境与测试

| 项目 | 结果 | 说明 |
|---|---|---|
| `uv sync --extra integration --extra rl --extra data --extra metrics --extra test` | 通过 | 已补齐 `torch`、`datasets`、`tensorboard`、`wandb`、`pytest` 等依赖 |
| `hermes-preflight` | 通过 | `missing=[]` |
| `atropos-preflight` | 通过 | `missing=[]` |
| 关键测试 | 通过 | `15 passed in 19.97s`，覆盖 `eval-rl CLI`、`atropos-preflight CLI`、`MVP end-to-end` |

### 2. 最小离线闭环

| 评估 | baseline | best checkpoint | 结论 |
|---|---:|---:|---|
| `output/trae_echo_eval_run_20260625/eval_summary.json` | `mean_reward=0.0365` | `candidate:iter_00030 mean_reward=0.0837` | `reward_delta=+0.0472`，promotion=`promote` |

### 3. Prompt / Context benchmark

| 指标 | 结果 |
|---|---:|
| `mean_reward` | `0.20` |
| `metadata/context_required_fact_recall` | `0.00` |
| `metadata/context_constraint_satisfaction` | `0.00` |
| `metadata/context_tool_summary_retention` | `0.00` |
| `metadata/context_distractor_avoidance` | `1.00` |
| `metadata/context_compression_ok` | `1.00` |

说明：当前 baseline 在“避免干扰信息”和“压缩长度合规”上稳定，但关键信息召回与约束满足仍为零，说明长上下文能力不是当前 tiny baseline 的强项。

## 真实 tool-use held-out 结果

本次未能重新现跑真实 `hermes_reasoning_traces` held-out benchmark，原因有两层：

1. 标准路径缺少本地 parquet：
   - 期望路径：`data/hermes_reasoning_traces/train.parquet`
2. 尝试改用 Hugging Face 数据源时，访问 `huggingface.co` 触发 SSL 错误：
   - `SSL: WRONG_VERSION_NUMBER`

因此，这一部分暂时采用现有本地历史产物作为当前最强证据：

| 产物 | baseline | best checkpoint | 结论 |
|---|---:|---:|---|
| `outputs/hermes_reasoning_traces_parquet_mps_terminal_command_stage2_best21_eval_v2/eval_summary.json` | `mean_reward=0.3813` / `success_rate=0.0000` | `mean_reward=0.4437` / `success_rate=0.3438` | `reward_delta=+0.0623`，工具结构可靠，命令内容有实质提升 |
| `outputs/hermes_reasoning_traces_parquet_mps_eval_rl_sweep/eval_summary.json` | `mean_reward=0.2621` / `success_rate=0.6250` | `mean_reward=0.2714` / `success_rate=0.8750` | 有增益，但原生 tool-call 结构仍未完全学稳 |

## 当前判断

| 维度 | 评级 | 说明 |
|---|---|---|
| 框架完整性 | 高 | 训练、评估、A/B、gate、online-cycle、self-evolution 链路都在 |
| 本地可执行性 | 高 | 依赖、预检、关键测试、最小评估均已通过 |
| 训练有效性证据 | 中高 | `echo`、stage2、历史 held-out 都显示正向增益 |
| 真实 benchmark 现跑能力 | 中 | 代码没问题，但当前缺少 parquet / HF 数据源被 SSL 阻塞 |

## 下一步

1. 把真实 `train.parquet` 放回 `data/hermes_reasoning_traces/train.parquet`
2. 或修复当前到 Hugging Face 的 SSL/代理链路
3. 然后重跑：
   - `configs/hermes_reasoning_traces_eval_rl.yaml`
   - `configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml`
   - `configs/benchmark_suite.yaml`
