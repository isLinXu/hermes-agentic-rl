# v1.0_release_prep 执行计划

## 目标
为 v1.0.0 稳定版发布做准备，冻结核心 API 并编写迁移指南。

## 阶段

### Stage 1 — API 稳定性标记
- 修改 `hermes_agentic_rl/core/types.py`：为 Trajectory, RolloutStep, RewardResult, RewardSummary, TrainSample, dataclass_to_dict 添加 @stable
- 修改 `hermes_agentic_rl/algos/base.py`：为 BaseAlgo, RolloutRecord, RolloutBatch, AlgoUpdateStats, stack_cached_logprobs, old_logprobs_tensor 添加 @stable
- 修改 `hermes_agentic_rl/backends/base.py`：为 LLMBackend, TokenizerProtocol, GenerationOutput, BackendUnavailableError 添加 @stable
- 修改 `hermes_agentic_rl/trainers/on_policy.py`：为 OnPolicyTrainer 公共接口添加 @stable（train, __init__ 等）
- 修改 `hermes_agentic_rl/rewards/base.py`：为 BaseReward 添加 @stable

### Stage 2 — 迁移指南
- 创建 `docs/migration/v0.12-to-v1.0.md`
- 创建 `docs/migration/v1.0-release-checklist.md`

### Stage 3 — 版本号冻结
- 修改 `hermes_agentic_rl/__init__.py`：`__version__ = "1.0.0-rc1"`

### Stage 4 — 验证
- ruff check 所有修改的 .py 文件
- mypy --follow-imports=skip 所有修改的 .py 文件
- python3 -m hermes_agentic_rl.cli.main --version 应输出 1.0.0-rc1

### Stage 5 — model_parallel_tp_impl（后续，如果时间允许）
