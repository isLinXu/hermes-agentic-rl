# hermes-agentic-rl: Training Code Comprehensive Documentation

> Generated from codebase exploration of `isLinXu/hermes-agentic-rl` @ `feat/v0.12-engineering-hardening`

---

## 1. Project Structure Overview

```
hermes-agentic-rl/
├── hermes_agentic_rl/           # Core library
│   ├── algos/                   # RL algorithms (GRPO, PPO, GSPO, OPD, Hybrid)
│   ├── backends/                # LLM backends (tiny, HF, vLLM, quantized)
│   ├── cli/                     # CLI entry points
│   ├── core/                    # Core types, dataset utils, reward manager
│   ├── datasets/                # HuggingFace dataset loader
│   ├── envs/                    # Training environments
│   ├── rewards/                 # Reward components & composers
│   ├── trainers/                # Trainer implementations
│   └── distributed/             # Distributed training (DDP, FSDP)
├── configs/                     # YAML training configurations (49 configs)
├── examples/                    # Minimal runnable examples
├── scripts/                     # Training scripts & smoke tests
├── train_mtgrpo.py              # Root-level MT-GRPO training script
└── tests/                       # Test suite
```

---

## 2. Training-Related Files by Category

### 2.1 Main Training Entry Points

| File | Role | How to Invoke |
|------|------|---------------|
| `train_mtgrpo.py` | Standalone MT-GRPO script with curriculum | `python train_mtgrpo.py --backend tiny --n-iters 20` |
| `hermes_agentic_rl/cli/train_rl.py` | Primary CLI for RL training (YAML-driven) | `python -m hermes_agentic_rl.cli.main train-rl --config configs/echo_grpo_mvp.yaml` |
| `hermes_agentic_rl/cli/main.py` | Main CLI router (dispatches to subcommands) | `python -m hermes_agentic_rl.cli.main <subcommand>` |
| `examples/minimal_grpo.py` | Minimal 5-minute GRPO quickstart | `python examples/minimal_grpo.py` |
| `examples/sft_plus_rl.py` | SFT + RL hybrid demo | `python examples/sft_plus_rl.py [--real]` |
| `scripts/train_v05.py` | v0.5 feature validation script | `python scripts/train_v05.py` |
| `scripts/train_mps.py` | MPS backend training script | `python scripts/train_mps.py` |
| `scripts/train_mps_grpo.py` | GRPO on MPS | `python scripts/train_mps_grpo.py` |
| `scripts/sft_warmup.py` | SFT warmup script | `python scripts/sft_warmup.py` |
| `scripts/smoke_letter_counting.py` | Letter counting smoke test | `python scripts/smoke_letter_counting.py` |

### 2.2 Core Trainer Classes

| File | Class | Description |
|------|-------|-------------|
| `hermes_agentic_rl/trainers/on_policy.py` | `OnPolicyTrainer` | **Base trainer** — rollout → reward → update loop. All trainers inherit from this. |
| `hermes_agentic_rl/trainers/grpo_trainer.py` | `GRPOTrainer` | Group Relative Policy Optimization (DeepSeek-style). No value head. |
| `hermes_agentic_rl/trainers/ppo_trainer.py` | `PPOTrainer` | Proximal Policy Optimization. Requires value-head backend. |
| `hermes_agentic_rl/trainers/hybrid_trainer.py` | `HybridTrainer` | GRPO + OPD (OpenClaw-RL §3.3) hybrid objective. |
| `hermes_agentic_rl/trainers/gspo_trainer.py` | `GSPOTrainer` | Group Self-Play Optimization. |
| `hermes_agentic_rl/trainers/base.py` | `BaseTrainer` | Legacy ABC (exporter-style, deprecated). |

### 2.3 Algorithm Implementations

| File | Class | Key Method |
|------|-------|------------|
| `hermes_agentic_rl/algos/grpo.py` | `GRPO` | `compute_loss(policy, ref_policy, batch) → (loss, stats)` |
| `hermes_agentic_rl/algos/ppo.py` | `PPO` | `compute_loss(policy, ref_policy, batch) → (loss, stats)` |
| `hermes_agentic_rl/algos/base.py` | `BaseAlgo` (ABC) | `compute_loss(...)` abstract method |
| `hermes_agentic_rl/algos/hybrid.py` | `HybridAlgo` | Combines GRPO + OPD losses with weights |
| `hermes_agentic_rl/algos/gspo.py` | `GSPO` | Group Self-Play variant |
| `hermes_agentic_rl/algos/opd.py` | `OPD` | OpenClaw Directive (teacher-distillation) |
| `hermes_agentic_rl/algos/common/loss.py` | `clipped_surrogate_loss_batched` | Shared PPO/GRPO clipped surrogate |
| `hermes_agentic_rl/algos/common/gae.py` | `compute_gae_batched` | GAE(γ, λ) for PPO |
| `hermes_agentic_rl/algos/common/advantage.py` | `group_normalize_advantage` | Group-relative advantage normalization |
| `hermes_agentic_rl/algos/common/reinforce_pp.py` | `reinforce_plusplus_advantage` | Per-token advantage (REINFORCE++) |
| `hermes_agentic_rl/algos/common/kl.py` | `compute_kl_penalty` | KL divergence penalty (k1/k2/k3 estimators) |

### 2.4 Data Loading & Environments

| File | Role |
|------|------|
| `hermes_agentic_rl/core/dataset.py` | Dataset splitting utilities (`split_dataset`, `select_items_by_group`) |
| `hermes_agentic_rl/datasets/hf_loader.py` | HuggingFace dataset loader (`load_hf_dataset`) |
| `hermes_agentic_rl/envs/base_env.py` | `BaseEnv` ABC — all envs implement this |
| `hermes_agentic_rl/envs/sim_tool_env.py` | `SimToolEnv` — arithmetic tool-use (simulated, zero deps) |
| `hermes_agentic_rl/envs/echo_task_env.py` | `EchoTaskEnv` — simplest env (echo task) |
| `hermes_agentic_rl/envs/letter_counting.py` | `LetterCountingEnv` — structured reasoning task |
| `hermes_agentic_rl/envs/curriculum.py` | `CurriculumEnv`, `MixedCurriculumEnv` — progressive difficulty |
| `hermes_agentic_rl/envs/hermes_reasoning_traces.py` | `HermesReasoningTraceEnv` — reasoning trace env |
| `hermes_agentic_rl/core/rollout_manager.py` | `RolloutManager` — collects trajectories from agent loops |

### 2.5 Reward System

| File | Class | Role |
|------|-------|------|
| `hermes_agentic_rl/core/reward_manager.py` | `RewardManager` | Flat-list reward aggregator |
| `hermes_agentic_rl/rewards/base.py` | `BaseReward` (ABC) | All reward components inherit from this |
| `hermes_agentic_rl/rewards/composer.py` | `RewardComposer` | **Advanced aggregator** with normalization, conditional activation, turn discount |
| `hermes_agentic_rl/rewards/outcome_reward.py` | `OutcomeReward` | Checks final output correctness |
| `hermes_agentic_rl/rewards/toolcall_reward.py` | `ToolcallReward` | Rewards proper tool call structure |
| `hermes_agentic_rl/rewards/length_penalty.py` | `LengthPenaltyReward` | Penalizes overly long responses |
| `hermes_agentic_rl/rewards/llm_judge.py` | `LLMJudgeReward` | LLM-as-judge reward |
| `hermes_agentic_rl/rewards/memory_reward_shaper.py` | `MemoryAwareRewardShaper` | Cross-session improvement bonus |
| `hermes_agentic_rl/rewards/dynamic_reward_balancer.py` | `DynamicRewardBalancer` | Adaptive component weight scheduling |

### 2.6 Training Infrastructure

| File | Role |
|------|------|
| `hermes_agentic_rl/trainers/on_policy_config.py` | `OnPolicyTrainerConfig` — shared config dataclass |
| `hermes_agentic_rl/trainers/minibatch_builder.py` | Minibatch splitting & aggregation |
| `hermes_agentic_rl/trainers/_train_loop_ops.py` | Post-iteration hooks (EMA, PRM, curriculum, checkpoint) |
| `hermes_agentic_rl/trainers/_checkpoint_ops.py` | Checkpoint save/resume utilities |
| `hermes_agentic_rl/trainers/_rollout_helpers.py` | Trajectory metadata extraction helpers |
| `hermes_agentic_rl/trainers/checkpoint.py` | `CheckpointManager`, `AsyncCheckpointSaver` |
| `hermes_agentic_rl/trainers/kl_controller.py` | Adaptive KL controller (P / PID) |
| `hermes_agentic_rl/trainers/ema.py` | `EMAModel` — EMA shadow for stable rollouts |
| `hermes_agentic_rl/trainers/lr_schedule.py` | LR schedulers (constant, linear, cosine, warmup) |
| `hermes_agentic_rl/trainers/mixed_precision.py` | AMP context, gradient accumulator |
| `hermes_agentic_rl/trainers/sft_mixin.py` | `SFTMixin` — bootstrap & interleaved SFT |
| `hermes_agentic_rl/trainers/replay_buffer.py` | Off-policy replay with TIS correction |
| `hermes_agentic_rl/trainers/multi_turn_credit.py` | Turn-level credit assignment |

---

## 3. Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE (OnPolicyTrainer)                  │
└─────────────────────────────────────────────────────────────────────────────┘

for iter in range(n_iters):
    │
    ├── 1. ROLLOUT COLLECTION (_collect_for_iter)
    │   ├── Get next item(s) from env → prompt
    │   ├── For each prompt, sample G=group_size responses
    │   │   └── AgentLoop generates trajectory (MultiTurnAgentLoop for tool-use)
    │   ├── RewardManager.evaluate(item, trajectory) → RewardSummary
    │   ├── Extract RL metadata: prompt_ids, response_ids, old_logprobs
    │   └── Build RolloutRecord per response
    │
    ├── 2. REWARD PROCESSING (_update_on_records)
    │   ├── Optional: normalize rewards (running mean/std)
    │   ├── Optional: mix in replay buffer records (off-policy)
    │   ├── Optional: memory-aware reward shaping
    │   ├── Optional: OPD hint extraction + teacher logprob fill
    │   └── Build RolloutBatch from records
    │
    ├── 3. UPDATE (_update_on_records, continued)
    │   ├── Build minibatches (update_epochs × minibatch_size)
    │   ├── For each minibatch:
    │   │   ├── algo.compute_loss(policy, ref_policy, batch)
    │   │   │   ├── GRPO: group-normalize advantages → clipped surrogate
    │   │   │   ├── PPO: GAE → clipped surrogate + value loss
    │   │   │   └── Add KL penalty + entropy bonus
    │   │   ├── loss.backward()
    │   │   ├── Gradient clipping
    │   │   └── optimizer.step() + LR scheduler step
    │   └── Aggregate stats across minibatches
    │
    └── 4. POST-ITERATION (run_all_post_iter)
        ├── EMA update (if enabled)
        ├── PRM co-training (if enabled)
        ├── Curriculum observe/advance (if enabled)
        ├── Dynamic reward balancer update
        ├── Checkpoint save (if configured)
        ├── Eval hook (if configured)
        └── Early stopping check
```

### Key Data Flow

```
Dataset Item ──► Env.format_prompt() ──► AgentLoop.generate()
                                              │
                                              ▼
                                        Trajectory (steps, tool_calls, final_output)
                                              │
                                              ▼
                                        RewardManager.evaluate()
                                              │
                                              ▼
                                        RewardSummary (final_score, components)
                                              │
                                              ▼
                                        RolloutRecord (prompt_ids, response_ids, old_logprobs, reward)
                                              │
                                              ▼
                                        RolloutBatch ──► Algo.compute_loss() ──► backward() ──► step()
```

---

## 4. How to Invoke Training

### 4.1 Method 1: YAML Config + CLI (Recommended)

```bash
# GRPO on echo task (CPU, tiny model)
python -m hermes_agentic_rl.cli.main train-rl --config configs/echo_grpo_mvp.yaml

# PPO on letter counting (MPS, real model)
python -m hermes_agentic_rl.cli.main train-rl --config configs/letter_counting_grpo_baseline.yaml

# Hybrid GRPO+OPD
python -m hermes_agentic_rl.cli.main train-rl --config configs/letter_counting_hybrid_opd.yaml

# MT-GRPO with curriculum
python -m hermes_agentic_rl.cli.main train-rl --config configs/mtgrpo_tiny.yaml

# SFT + RL hybrid
python -m hermes_agentic_rl.cli.main train-rl --config configs/sft_plus_rl.yaml
```

### 4.2 Method 2: Standalone Script

```bash
# MT-GRPO standalone (no YAML)
python train_mtgrpo.py --backend tiny --n-iters 20 --group-size 4

# With HF backend
python train_mtgrpo.py --backend hf --model-name Qwen/Qwen2.5-1.5B-Instruct --n-iters 100
```

### 4.3 Method 3: Programmatic (Python)

```python
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv, build_sim_tool_dataset
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

# 1. Backend
backend = TinyCausalLMBackend(TinyBackendConfig(dim=32, n_heads=4, n_layers=2))

# 2. Environment
dataset = build_sim_tool_dataset(n=50, seed=42)
env = SimToolEnv(dataset=dataset)

# 3. Reward
reward_manager = RewardManager([OutcomeReward(weight=1.0)])

# 4. Config
cfg = GRPOTrainerConfig(
    n_iters=20,
    group_size=4,
    prompts_per_iter=2,
    lr=1e-3,
    max_new_tokens=32,
    log_every=1,
)

# 5. Train
trainer = GRPOTrainer(policy=backend, env=env, reward_manager=reward_manager, cfg=cfg)
stats = trainer.train()
```

---

## 5. Key Configuration Fields

### GRPOTrainerConfig / OnPolicyTrainerConfig

| Field | Default | Description |
|-------|---------|-------------|
| `n_iters` | 20 | Total training iterations |
| `group_size` | 4 | GRPO group size (≥2 for group norm) |
| `prompts_per_iter` | 2 | Prompts sampled per iteration |
| `lr` | 1e-3 | Learning rate |
| `max_new_tokens` | 16 | Max tokens per response |
| `temperature` | 1.0 | Sampling temperature |
| `clip_eps` | 0.2 | PPO/GRPO clip epsilon |
| `kl_coef` | 0.0 | KL penalty coefficient |
| `entropy_coef` | 0.0 | Entropy bonus coefficient |
| `update_epochs` | 1 | SGD epochs per iteration |
| `minibatch_size` | 0 | 0 = full batch |
| `grad_clip` | 1.0 | Gradient clipping max norm |
| `use_reference` | False | Enable reference policy for KL |
| `per_token_advantage` | False | REINFORCE++ per-token advantages |
| `advantage_norm` | "group" | "group" / "batch" / "whiten" / "dapo" |
| `multi_turn` | False | Enable multi-turn credit assignment |
| `multi_turn_credit` | None | Dict: `{mode: "discounted", gamma: 0.9}` |
| `interleave_sft_every` | 0 | SFT every N iters (0 = disabled) |
| `bootstrap_sft_rounds` | 0 | SFT rounds before RL |
| `checkpoint_every` | 0 | Save checkpoint every N iters |
| `save_best_checkpoint` | False | Save best checkpoint by reward |
| `early_stop_patience` | 0 | Early stopping patience |
| `vllm_rollout_model` | None | Use vLLM for fast rollouts |
| `distributed_strategy` | "none" | "none" / "ddp" / "fsdp" |
| `amp_dtype` | "fp32" | "fp16" / "bf16" / "fp32" |
| `grad_accum_steps` | 1 | Gradient accumulation |
| `curriculum` | None | Dict for CurriculumScheduler |
| `replay_buffer` | None | Dict for off-policy replay |

---

## 6. Example Configurations

### 6.1 Minimal GRPO (echo task, CPU)
```yaml
backend:
  name: tiny
  dim: 32
  n_layers: 2

environment:
  type: echo

train_rl:
  n_iters: 60
  group_size: 8
  lr: 0.005
  clip_eps: 0.2
```

### 6.2 Hybrid GRPO + OPD (real model, MPS)
```yaml
algo: hybrid
backend:
  name: hf
  model_name_or_path: gpt2
  device: mps

environment:
  type: letter_counting

reward:
  components:
    - type: letter_counting_next_state
      weight: 1.0

train_rl:
  n_iters: 100
  group_size: 6
  lr: 5e-5
  use_reference: true
  kl_coef: 0.04
  w_rl: 1.0
  w_opd: 1.0

opd:
  teacher_fill: true
  kl_coef: 0.02
```

### 6.3 MT-GRPO with Curriculum
```yaml
backend: tiny
n_iters: 20
group_size: 4

multi_turn: true
multi_turn_credit:
  mode: discounted
  gamma: 0.9

per_token_advantage: true
advantage_norm: group

curriculum:
  auto_advance: true
  min_iters_per_stage: 10

rewards:
  components:
    - type: ToolcallReward
      weight: 1.0
    - type: OutcomeReward
      weight: 1.0
```

---

## 7. Key Classes & Methods Reference

### OnPolicyTrainer (hermes_agentic_rl/trainers/on_policy.py:206)

**Constructor:**
```python
OnPolicyTrainer(
    policy: LLMBackend,
    env: BaseEnv,
    reward_manager: RewardManager,
    algo: BaseAlgo,
    cfg: OnPolicyTrainerConfig,
    agent_loop_factory: AgentLoopFactory | None = None,
    rollout_pool: Any = None,
)
```

**Key Methods:**
- `train() → TrainStats` — Main training loop
- `_collect_group(item) → list[RolloutRecord]` — Collect G rollouts for one prompt
- `_update_on_records(records, iter_idx) → AlgoUpdateStats` — Run SGD on rollout batch
- `_optimizer_step() → float` — Unscale → clip → step → LR update

### GRPO (hermes_agentic_rl/algos/grpo.py:90)

```python
GRPO(cfg: GRPOConfig)
grpo.compute_loss(policy, ref_policy, batch) → (loss_tensor, AlgoUpdateStats)
```

**Internals:**
1. Group-normalize advantages: `A_i = (R_i - mean_g(R)) / (std_g(R) + eps)`
2. Optional per-token advantage via REINFORCE++
3. Batched forward: `policy.score_batch(prompts, responses)`
4. Clipped surrogate loss + KL penalty + entropy bonus

### PPO (hermes_agentic_rl/algos/ppo.py:67)

```python
PPO(cfg: PPOConfig)
ppo.compute_loss(policy, ref_policy, batch) → (loss_tensor, AlgoUpdateStats)
```

**Internals:**
1. Batched forward: `policy.score_with_value_batch()` → logp, entropy, values
2. Batched GAE: `compute_gae_batched(token_rewards, values, mask)`
3. Advantage normalization + optional whitening
4. Clipped surrogate + clipped value loss + entropy + KL

### RewardComposer (hermes_agentic_rl/rewards/composer.py:155)

```python
RewardComposer(
    components: list[BaseReward],
    config: RewardComposerConfig,
)
await composer.evaluate(item, trajectory, tool_context) → RewardSummary
```

**Features:**
- Per-component running normalization (Welford's online algorithm)
- Conditional activation via lambda conditions
- Turn discount: `score × γ^turns_used`
- Parallel component evaluation via `asyncio.gather`

---

## 8. Available Environments

| Environment | Type | Description | Best For |
|-------------|------|-------------|----------|
| `echo` | `EchoTaskEnv` | Echo back input | Sanity checks |
| `sim_tool` | `SimToolEnv` | Arithmetic tool-use | Multi-turn GRPO |
| `letter_counting` | `LetterCountingEnv` | Count letters in words | Structured reasoning |
| `curriculum` | `CurriculumEnv` | Progressive difficulty | Curriculum learning |
| `multi_stream` | `MixedCurriculumEnv` | Multiple task streams | Multi-task training |
| `hermes_reasoning_traces` | `HermesReasoningTraceEnv` | Reasoning traces | Real reasoning |
| `context_benchmark` | `ContextBenchmarkEnv` | Context window tests | Long-context eval |

---

## 9. Backends

| Backend | Class | Use Case |
|---------|-------|----------|
| `tiny` | `TinyCausalLMBackend` | CPU testing, <1MB model |
| `hf` | `HFCausalLMBackend` | Real HuggingFace models |
| `vllm` | `VLLMRolloutBackend` | Fast distributed rollouts |
| `quantized` | `QuantizedBackend` | Quantized inference |

---

## 10. Quick Start Commands

```bash
# 1. Minimal GRPO (30 seconds on CPU)
python examples/minimal_grpo.py

# 2. SFT + RL hybrid
python examples/sft_plus_rl.py

# 3. YAML-driven GRPO
python -m hermes_agentic_rl.cli.main train-rl --config configs/echo_grpo_mvp.yaml

# 4. MT-GRPO standalone
python train_mtgrpo.py --backend tiny --n-iters 20

# 5. Letter counting smoke test
python scripts/smoke_letter_counting.py

# 6. Training validation suite
python scripts/training_validation.py
```

---

## 11. Version History (Config Fields)

- **v0.2**: Base GRPO/PPO with group normalization
- **v0.5**: Per-token advantage (REINFORCE++), interleaved SFT, batch generate
- **v0.6**: Checkpoint/resume, KL estimator (k1/k2/k3)
- **v0.7**: Best checkpoint, early stopping, advantage whitening (PPO)
- **v0.8**: Adaptive KL, reward normalization, ratio-based early stop
- **v0.9**: FSDP/DDP, AMP, gradient accumulation, EMA rollout, LR scheduler
- **v0.10**: vLLM rollout, LoRA hot-reload, pipeline rollouts
- **v0.11**: PRM co-training, memory reward shaping, dynamic reward balancer
- **v0.12**: Curriculum scheduler, staleness-adaptive TIS, OPD hint extraction

---

*This documentation covers the complete training surface of hermes-agentic-rl as of commit `1fe2d3a` (v1.0.0-rc1).*
