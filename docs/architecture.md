# Architecture (v0.6+)

## Layered design

```
┌─────────────────────────────────────────────────────────────┐
│ CLI (cli/)                                                  │
│   main.py      rollout / train / train-rl / offline         │
│   train_rl.py  YAML → trainer factory                       │
└─────────────┬───────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ Trainer (trainers/)                                         │
│   OnPolicyTrainer  shared rollout → loss → step skeleton    │
│   GRPOTrainer      thin wrapper → algos.GRPO                │
│   PPOTrainer       thin wrapper → algos.PPO (value head)    │
│   CheckpointMgr    model + optim + rng + stats bundle       │
└───────┬──────────────────────────────────────────────┬──────┘
        ↓                                              ↓
┌────────────────┐  ┌─────────────────┐  ┌────────────────┐
│ Algo (algos/)  │  │ Rewards         │  │ Backend        │
│   GRPO         │  │   RewardManager │  │   LLMBackend   │
│   PPO + GAE    │  │   components    │  │   tiny / hf    │
│   common/kl    │  │   RM, lagrangian│  │   value head   │
│   common/loss  │  └─────────────────┘  │   LoRA inject  │
└────────────────┘                       └────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ Env + MDP (envs/, mdp/, agent_loop/)                        │
│   BaseEnv / EchoEnv / SimToolEnv / LetterCountingEnv        │
│   CurriculumEnv (moving-avg level promotion)                │
│   PolicyAgentLoop / MultiTurnAgentLoop                      │
│   Observation / Action / PromptStateEncoder                 │
└─────────────────────────────────────────────────────────────┘
```

## Data flow (one training iter)

```
env.get_next_item()
    ↓
agent_loop_factory(policy) → MultiTurn/Policy loop
    ↓ generate(prompt_ids, temperature, seed)
Trajectory { runtime.rl = {prompt_ids, response_ids, old_logprobs} }
    ↓
RewardManager.evaluate(item, trajectory)  → RewardSummary{final_score, components[]}
    ↓
RolloutRecord { prompt_ids, response_ids, old_logprobs, reward, group_id }
    ↓  (N × prompts_per_iter × group_size records)
Algo.compute_loss(policy, ref_policy, batch)  → (loss_tensor, stats)
    ↓
optim.zero_grad(); loss.backward(); clip_grad; optim.step()
    ↓
CheckpointManager.save() [every N iters]
MetricsWriter(record) [jsonl / tb / wandb / dashboard]
```

## Extension points

| Want to… | Extend |
|---|---|
| Support a new LLM (DeepSeek, Qwen, SmolLM2) | `backends/*.py` — implement `LLMBackend` (generate / score / trainable_parameters) |
| Add an RL algorithm (DPO-RL, RLOO, GSPO) | `algos/*.py` — implement `BaseAlgo.compute_loss` |
| Add a new environment | `envs/*.py` — subclass `BaseEnv` (setup / get_next_item / format_prompt) |
| Add a reward component | `rewards/*.py` — implement `BaseReward.evaluate` |
| Add a metrics destination | `monitor/writers.py` — callable `(dict) -> None`; add to `MultiMetricsWriter` |
| Swap agent loop | `agent_loop/*.py` — subclass `BaseAgentLoop`; pass `agent_loop_factory` to trainer |
| Distribute rollouts | `distributed/mp_pool.py` — existing `MPRolloutPool`; or write a Ray variant |

## Why this shape

- **Single on-policy trainer** (`OnPolicyTrainer`) — GRPO and PPO share 95% of the control flow (rollout collection, advantage computation, loss.backward, checkpoint). Keeping one owner for that flow eliminates drift between the two algorithms.
- **Algo as a pure function** (`BaseAlgo.compute_loss`) — deterministic, side-effect-free; trainer owns the optimizer. This lets algos be unit-tested against hand-crafted `RolloutBatch` fixtures.
- **Backend protocol over inheritance** — any object implementing `generate / score / trainable_parameters` is a valid policy. Tiny model (30K params) exists for CPU-friendly end-to-end tests; HF backend is an opt-in extra.
- **Metadata-only coupling to Trajectory** — the RL loop reads `trajectory.metadata["runtime"]["rl"]` rather than importing Hermes-specific types. The same trainer runs on fake/mock runtimes, Hermes-Agent wrappers, and atropos envs.
- **Opt-in everything** — Lagrangian, LoRA, value head, reference policy, per-token advantage, distributed rollouts, multiple metrics backends, resumable checkpoints — all default-off. The v0.2 MVP config still runs verbatim.

## Package map

| Layer | Package | Key classes |
|---|---|---|
| Runtime | `runtime/` | `BaseRuntimeAdapter`, `HermesAIAgentWrapper`, `FakeRuntimeAdapter` |
| MDP | `mdp/` | `Observation`, `Action`, `PromptStateEncoder` |
| Agent loop | `agent_loop/` | `PolicyAgentLoop`, `MultiTurnAgentLoop` |
| Backend | `backends/` | `LLMBackend`, `TinyCausalLMBackend`, `HFCausalLMBackend` |
| Algo | `algos/` | `BaseAlgo`, `GRPO`, `PPO`, `common.{advantage, gae, kl, loss, reinforce_pp}` |
| Rewards | `rewards/` | `RewardManager`, `OutcomeReward`, `ToolcallReward`, `FileSystemVerifierReward`, `RewardModel`, `LagrangianController` |
| Trainer | `trainers/` | `OnPolicyTrainer`, `GRPOTrainer`, `PPOTrainer`, `CheckpointManager` |
| Offline | `offline/` | `BCTrainer`, `DPOTrainer`, `ReplayBuffer` |
| PEFT | `peft/` | `LoRALinear`, `inject_lora` |
| Env | `envs/` | `EchoTaskEnv`, `SimToolEnv`, `LetterCountingEnv`, `CurriculumEnv` |
| Eval | `eval/` | `EvalHarness`, `VersionManager`, `run_ab`, `paired_welch_t` |
| Monitor | `monitor/` | `LiveDashboard`, `JsonlMetricsWriter`, `TensorBoardMetricsWriter`, `WandbMetricsWriter`, `MultiMetricsWriter` |
| Distributed | `distributed/` | `MPRolloutPool` (stdlib multiprocessing) |
| Integration | `integrations/` | `AtroposEnvAdapter`, `HermesAPIServer`, `AtroposRewardComponent` |
| CLI | `cli/` | `main`, `train_rl`, `offline_cli` |
