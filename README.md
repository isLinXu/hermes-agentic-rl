# hermes-agentic-rl

> **0.6.0-dev** — Agentic RL framework for Hermes-Agent.
>
> 0.2 closed the rollout → reward → gradient → parameter-update loop (GRPO MVP).
> 0.3 added PPO + GAE, multi-turn tool-use, curriculum learning, and a stdlib
> dashboard. 0.4 added the full RLHF stack (LoRA, BC / DPO, Bradley-Terry RM,
> multi-process rollout, eval harness + A/B, version manager, Lagrangian
> safety). 0.5 added **atropos interop**: any
> [atropos](https://github.com/NousResearch/atropos) environment
> (GSM8K / math / tool-use / SWE-RL / 50+ more) can now drive a hermes trainer
> directly — zero HTTP server, zero vLLM, zero GPU required. **0.6 closes the
> production engineering gaps**: resumable checkpoints with `auto_resume`,
> Schulman K3 (unbiased, non-negative) KL estimator for GRPO/PPO, pluggable
> metrics backends (JSONL + TensorBoard + W&B with graceful degradation),
> and CLI-level RewardModel injection for RLHF fine-tuning.

## 0.6 feature matrix

| Area | Module | What it gives you |
|---|---|---|
| Resumable training | `trainers.checkpoint.CheckpointManager` + `OnPolicyTrainerConfig.{checkpoint_every, auto_resume, resume_from, keep_last_checkpoints}` | Full `{model, optimizer, rng, stats, config}` bundle with atomic writes + automatic pruning. `auto_resume: true` in YAML picks up where the last run left off, including stats history. |
| KL estimators | `algos.common.kl.kl_from_logprobs` + `GRPOConfig.kl_estimator`, `PPOConfig.kl_estimator` | K1 (legacy, unbiased, signed), K2 (low variance), **K3** (Schulman 2020: `exp(-r) - 1 + r`, unbiased AND non-negative). Recommended default for new runs: `kl_estimator: k3`. |
| Metrics backends | `monitor.writers.{JsonlMetricsWriter, TensorBoardMetricsWriter, WandbMetricsWriter, MultiMetricsWriter, build_writer_from_config}` | Pluggable `metrics:` YAML section composes jsonl / stdout / tensorboard / wandb writers. Each writer is isolation-fault-safe: if TB isn't installed or wandb.init wasn't called, the writer becomes a silent no-op — training never dies on a metrics IO error. |
| RewardModel CLI | `cli.train_rl._build_reward_model_component` | `reward_model:` YAML block loads a pre-trained Bradley-Terry RM head (`head_path: rm_head.pt`) and composes it into the RewardManager alongside existing components. Works with both Tiny and HF backends. |
| HF backend fixes | `backends.hf.HFCausalLMBackend._set_mode` | `score()` and `score_with_value()` now consistently use `model.train()` only when `torch.is_grad_enabled()` — dropout/LayerNorm behavior is finally deterministic across GRPO (score) and PPO (score_with_value) paths. |
| Tests | 120 → **143** (all green) | +5 checkpoint/resume, +7 KL estimators, +7 metrics writers, +4 RewardModel CLI. |

See `configs/echo_grpo_v06.yaml` for a full annotated v0.6 example.

## 0.5 interop layer: `integrations.atropos_env_import`

`HermesAPIServer` is a drop-in replacement for atropos's `APIServer` that
delegates all generation to any `LLMBackend` (Tiny / HF / your own). atropos's
`ServerManager.managed_server()` picks it up because it's structurally a
non-OpenAI server — so real `ManagedServer` (tokens + logprobs) works too.

```python
from atroposlib.envs.base import BaseEnvConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.integrations import (
    AtroposEnvAdapter, AtroposEnvAdapterConfig, AtroposRewardComponent,
)
from hermes_agentic_rl.trainers import GRPOTrainer, GRPOTrainerConfig
from my_pkg.my_atropos_env import MyEnv  # any atroposlib.envs.base.BaseEnv

backend = TinyCausalLMBackend(TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2))
env_cfg = BaseEnvConfig(group_size=4, tokenizer_name="gpt2", max_token_length=256)
adapter = AtroposEnvAdapter(env_cls=MyEnv, env_config=env_cfg, backend=backend,
                            adapter_config=AtroposEnvAdapterConfig(max_new_tokens=32))
rm = RewardManager([AtroposRewardComponent(adapter)])
trainer = GRPOTrainer(backend, adapter, rm,
                      cfg=GRPOTrainerConfig(n_iters=20, group_size=4,
                                            prompts_per_iter=4, lr=3e-4,
                                            max_new_tokens=32, seed=0))
trainer.train()
```

Two generation modes, auto-selected:

- **Direct**: backend & atropos share the same vocab (e.g. `HFCausalLMBackend`
  pointed at the same tokenizer). Token ids flow through unmodified. This is
  the high-fidelity path you'd use for real training.
- **Text-level bridge**: backend has a different vocab (e.g. `TinyCausalLM`
  with a char tokenizer). The server decodes HF prompt → re-encodes with the
  backend's tokenizer → runs `generate` → re-encodes response back to HF vocab
  so atropos consumers stay happy. Logprob alignment is approximate but the
  whole pipeline runs end-to-end on CPU with no real language model.

See `examples/run_atropos_env_with_hermes.py` for a runnable end-to-end demo.

## 0.4 feature matrix

| Area | Module | What it gives you |
|---|---|---|
| Distributed rollouts | `distributed.MPRolloutPool` | N-process worker pool via stdlib `multiprocessing`; learner broadcasts weights each iter, workers return rollout records. No Ray needed. |
| LoRA | `peft.{LoRAConfig, LoRALinear, inject_lora}` | Low-rank adapters injected into any `nn.Linear` in a backend; `save`/`load`/`merge_into_base` for zero-cost deploy. ~10% trainable params. |
| HF backend | `backends.HFCausalLMBackend` *(optional extra `hf`)* | Drop-in `LLMBackend` on top of `AutoModelForCausalLM` with value-head support for PPO. |
| Offline BC | `offline.BCTrainer` | NLL warm-start from a `ReplayBuffer` of `(prompt, response)` demonstrations. |
| Offline DPO | `offline.DPOTrainer` | Direct Preference Optimization on `DPOPair` buffers; frozen ref policy auto-snapshot. |
| Reward model | `rewards.reward_model.{RewardModel, RewardModelTrainer, RewardModelComponent}` | Bradley-Terry scalar RM, plugs into `RewardManager` as a component. |
| Eval harness | `eval.{EvalHarness, leaderboard_markdown}` | Deterministic N-rollout evaluation → aggregate metrics + per-component breakdown. |
| A/B testing | `eval.{run_ab, paired_welch_t}` | Paired Welch t-test with Gaussian p-value approx; no scipy required. |
| Version manager | `eval.VersionManager` | On-disk checkpoint registry + exclusive tags (`production` / `canary` / free-form). |
| Safety (RCPO) | `rewards.lagrangian.LagrangianController` | Cost-limited policy optimization with dual-variable updates; opt-in on any OnPolicyTrainer. |
| CLI | `offline` subcommand | `--config configs/offline_{bc,dpo,rm}_mvp.yaml offline` end-to-end. |
| Tests | 94 → **114** (all green) | +LoRA, +BC/DPO, +RM, +Eval/AB/VersionManager, +Lagrangian, +MP pool. |

Every v0.2/v0.3 API, config, and CLI remains 100% backward compatible.

---

## What's new in 0.2 (vs. 0.1)

v0.1 was a minimal rollout collector: it had no `nn.Module`, no optimizer, no
log-probs, and no gradients. `AtroposGrpoTrainer` despite its name only wrote
JSONL. The refactor in this release corrects all of that:

| Area | 0.1 state | 0.2-dev state |
|---|---|---|
| LLM backend | ❌ none | ✅ `backends.TinyCausalLMBackend` (CPU-friendly, ~30K params) + stable `LLMBackend` protocol |
| MDP semantics | ❌ only `get_next_item` | ✅ `mdp/` with `Observation`, `Action`, `PromptStateEncoder` |
| Agent loop | Black-box `AIAgent.run_conversation` | ✅ `PolicyAgentLoop` carries **token-level `old_logprobs`** end-to-end |
| RL algorithm | ❌ none | ✅ `algos.GRPO` — group-normalized advantage + clipped surrogate + optional KL-to-ref |
| Trainer | JSONL writer mislabeled as "GRPO" | ✅ `trainers.GRPOTrainer` — owns policy + AdamW + updates params |
| Exporter | fused into `AtroposGrpoTrainer` | ✅ split out to `exporters.AtroposJsonlExporter`; old name kept as deprecated alias |
| CLI | `rollout`, `train` | ✅ `rollout`, `train`, **new `train-rl`** |
| Tests | 67 | 82 (all green) |

## Quick start — real RL training (MVP)

```bash
pip install -e '.[rl]'    # adds torch
python -m hermes_agentic_rl.cli.main train-rl \
    --config configs/echo_grpo_mvp.yaml \
    --output outputs/train_rl_mvp
```

Expected output (60 iters, ~15s on CPU):

```
[train-rl] backend=tiny params=26624 iters=60 group=8 lr=0.005
[grpo] iter=0  mean_reward=0.0224 loss=-0.0000 ... clip_frac=0.0000
[grpo] iter=10 mean_reward=0.0315 ...
[grpo] iter=50 mean_reward=0.1050 ...
[train-rl] DONE last_mean_reward=0.0348 best=0.1117 delta=+0.0124
```

The MVP task is a trivial "echo" env: the policy must say a target string.
A random tiny policy gets ~0.02; after training, reward trends upward and
peaks 3–5× higher. This proves the full RL loop is mechanically correct.

To plug in a real LLM, implement `LLMBackend` over `transformers` (or any
other library) — the `generate / score / trainable_parameters` contract is
stable, and the rest of the stack (GRPO, rollout manager, reward manager) is
unchanged.

## Architecture

```
hermes_agentic_rl/
├── backends/        # LLMBackend protocol: generate, score (differentiable), params
│   ├── base.py
│   └── tiny.py      # self-contained 2-layer Transformer (CPU MVP)
├── mdp/             # Observation, Action, PromptStateEncoder
├── agent_loop/      # PolicyAgentLoop: exposes token-level old_logprobs
├── algos/           # GRPO + common (advantage, clipped surrogate)
│   ├── grpo.py
│   └── common/
├── trainers/
│   ├── base.py           # legacy exporter-style trainer interface
│   └── grpo_trainer.py   # ⭐ real GRPO trainer: owns policy + optimizer
├── exporters/
│   └── atropos_jsonl.py  # JSONL writer (the real home of the old name)
├── envs/
│   ├── terminal_task_env.py  # filesystem-verifier tasks (needs real Hermes)
│   └── echo_task_env.py      # ⭐ MVP learnable task, zero external deps
├── rewards/         # outcome, toolcall, filesystem_verifier, aggregate
├── runtime/         # fake + hermes adapters (AIAgent.run_conversation)
├── core/            # types, RolloutManager, RewardManager, TrainerBridge
└── cli/
    ├── main.py         # rollout / train / train-rl dispatch
    └── train_rl.py     # ⭐ new RL entrypoint
```

## Two CLIs, two purposes

| Command | Owns a policy? | Updates params? | Purpose |
|---|---|---|---|
| `rollout` | no | no | Generate one trajectory for inspection |
| `train` | no | **no** | Collect trajectories + rewards → JSONL (for downstream Atropos/verl/...) |
| `train-rl` | **yes** | **yes** | Real RL training with GRPO (the MVP) |

## GRPO briefing

Given a prompt and `G` sampled rollouts with rewards `r_1..r_G`:

```
A_i = (r_i − mean(r)) / (std(r) + ε)                 # group-normalized advantage
ratio_t = exp( logπ_new(a_t|s_t) − logπ_old(a_t|s_t) )
loss_i = −min( ratio_t · A_i,  clip(ratio_t, 1±ε) · A_i )
         + β · KL(π_new || π_ref)   (optional)
```

No value head. Zero-variance groups produce no gradient (design-correct).

## Running the old pipeline (0.1 back-compat)

```bash
# fake runtime, same as before
python -m hermes_agentic_rl.cli.main rollout \
    --config configs/terminal_grpo.yaml --output outputs/trajectory.json
python -m hermes_agentic_rl.cli.main train \
    --config configs/terminal_grpo.yaml
```

The default reward weights in `terminal_grpo.yaml` have been rebalanced to
`verifier 0.8 / outcome 0.1 / toolcall 0.1` to avoid the "say 'done' to win"
exploit that the old `0.7/0.3` setup allowed.

## Tests

```bash
python -m pytest tests -q   # 82 passed
```

New in 0.2-dev:
- `test_backends_tiny.py` — tokenizer round-trip, generate, differentiable score
- `test_algos_grpo.py` — advantage normalization, clipped surrogate, GRPO one-step
- `test_mvp_end_to_end.py` — params move, reward trends up, metadata flows through

## Real Hermes runtime

Same as before; see `docs/real-hermes-check.md`. A trainable backend for real
Hermes interactions (bypassing `AIAgent.run_conversation` to expose logπ) is
planned for 0.3.

## Atropos / tinker-atropos integration (preflight)

If you have local checkouts at repo root:

```
./atropos/
./tinker-atropos/
```

You can run a zero-side-effect integration preflight:

```bash
python -m hermes_agentic_rl.cli.main atropos-preflight
```

This prints a JSON report of which directories and python deps are available.
In particular, `tinker-atropos` training requires the `tinker` python package.

## What's new in 0.3-dev (vs. 0.2)

| Area | 0.2 | 0.3-dev |
|---|---|---|
| Algorithms | GRPO only | **PPO** (value-head + token-level GAE + clipped value loss) + GRPO |
| GRPO variants | DeepSeek default | +`loss_agg`: `mean_token` / `sum_token` / **Dr.GRPO** |
| Agent loop | single-turn only | +`MultiTurnAgentLoop` with `<tool_call>NAME(ARG)</tool_call>` protocol, per-turn records |
| Envs | echo, terminal | +`SimToolEnv` (zero-dep `calc` tool) + `CurriculumEnv` (moving-avg promotion/demotion) |
| Trainer | `GRPOTrainer` | `OnPolicyTrainer` base + `GRPOTrainer` / `PPOTrainer` thin subclasses |
| Backend | TinyCausalLM (~26K params) | +optional **value head** (~30K params, still CPU-friendly) |
| Monitoring | stdout only | +**pure-stdlib live dashboard** (`http.server` + Chart.js CDN), opt-in |
| CLI | `train-rl --config echo_grpo_mvp.yaml` | `algo: grpo | ppo`, `agent_loop: policy | multi_turn`, `environment: echo | sim_tool | curriculum`, `dashboard: {enabled, port}` |
| Tests | 82 | **94** (all green) |

## Quick start — PPO

```bash
python -m hermes_agentic_rl.cli.main train-rl \
    --config configs/echo_ppo_mvp.yaml \
    --output outputs/ppo_mvp
```

## Quick start — multi-turn tool-use with GRPO

```bash
python -m hermes_agentic_rl.cli.main train-rl \
    --config configs/sim_tool_grpo_multiturn.yaml
```

The tool protocol is string-based:

```
<tool_call>calc(3 + 4)</tool_call>
                ↓
<tool_result>7</tool_result>
```

`SimToolEnv` rewards three sub-skills with a dense decomposition
(`0.4·used_tool + 0.3·tool_result_correct + 0.3·final_answer_correct`), so
even a small policy gets a meaningful gradient signal.

## Quick start — curriculum learning

```bash
python -m hermes_agentic_rl.cli.main train-rl \
    --config configs/curriculum_grpo.yaml
```

Levels, window size, and promotion threshold are all YAML-configurable.
`env.observe(reward)` is called by the trainer after every rollout, so no
extra plumbing is needed.

## Live dashboard (opt-in)

Add to any config:

```yaml
dashboard:
  enabled: true
  host: 127.0.0.1
  port: 8765
```

Then open `http://127.0.0.1:8765/`. Four live line charts update every second:
mean_reward, loss, kl, value_loss. Zero Python dependencies — just
`http.server` and a CDN `<script src>` tag.

## Architecture (0.3)

```
hermes_agentic_rl/
├── backends/            # LLMBackend: generate / score / score_with_value (PPO)
├── mdp/                 # Observation, Action, PromptStateEncoder
├── agent_loop/
│   ├── policy_loop.py   # single-turn (v0.2)
│   └── multi_turn_loop.py  # ⭐ tool-use with per-turn RL metadata
├── algos/
│   ├── grpo.py          # +loss_agg (Dr.GRPO)
│   ├── ppo.py           # ⭐ GAE + clipped value loss
│   └── common/
│       ├── advantage.py # group_normalize_advantage
│       ├── gae.py       # ⭐ compute_gae, terminal_token_rewards
│       └── loss.py      # clipped_surrogate_loss, ⭐ clipped_value_loss
├── trainers/
│   ├── on_policy.py     # ⭐ shared rollout→loss→step skeleton
│   ├── grpo_trainer.py  # thin subclass
│   └── ppo_trainer.py   # ⭐ thin subclass, requires value-head backend
├── envs/
│   ├── echo_task_env.py
│   ├── sim_tool_env.py  # ⭐ tool-use task with safe AST eval
│   ├── curriculum.py    # ⭐ moving-avg level promotion
│   └── terminal_task_env.py
├── monitor/
│   └── dashboard.py     # ⭐ pure-stdlib HTTP + Chart.js CDN
└── cli/
    ├── main.py
    └── train_rl.py      # algo=grpo|ppo, env=echo|sim_tool|curriculum
```

## Roadmap

- [x] **P0** — rename `AtroposGrpoTrainer` → `AtroposJsonlExporter`, rebalance default rewards
- [x] **P1** — `backends/`, `mdp/`, `agent_loop/` — expose token-level signal
- [x] **P2** — GRPO algorithm + trainer + CLI (MVP)
- [x] **P3** — PPO (value head + GAE) + multi-turn tool-use + curriculum + live dashboard
- [x] **P4** — HF backend (GPT-2 / SmolLM2 / Qwen) + LoRA + MPS training
- [x] **P5** — Standalone `letter_counting` env + SFT warmup + GRPO on MPS
- [ ] **P6** — distributed rollout (Ray actor pool) + multi-process rollout
- [ ] **P7** — eval harness + A/B versioning + Lagrangian safety constraints
- [ ] **P8** — offline algos (BC / AWR / DPO) + reward model trainer

## 0.5 MPS Training Results

Real GRPO training on Apple Silicon MPS (24GB):

| Config | Value |
|--------|-------|
| Model | `openai-community/gpt2` (124M params) |
| Device | MPS (Apple Silicon) |
| Env | `letter_counting` (standalone, 10-tier adaptive difficulty) |
| SFT | 200 samples × 3 epochs, loss 0.77→0.30 |
| GRPO | 100 iters, group=4, prompts=4, lr=2e-5, temp=0.8 |
| Speed | ~7.2s/iter on MPS |

**Reward curve (GRPO after SFT):**
```
iter   0 | 0.1875 | #####
iter   5 | 0.5625 | ################
iter  10 | 0.2500 | #######
iter  15 | 0.0000 | #   ← mode collapse (format drift)
...
iter  60 | 0.0625 | #   ← brief recovery
```

**Key findings:**
- GRPO successfully increases reward from 0.19→0.56 in first 5 iters
- GPT-2 (no instruct tuning) drifts away from `<answer>` format → reward collapse
- SFT warmup is essential for sparse-reward tasks
- MPS is 2-3× faster than CPU for this workload

**Run it yourself:**
```bash
# Full pipeline: SFT warmup → GRPO on MPS
python scripts/train_mps.py

# GRPO only (from existing SFT checkpoint)
python scripts/train_mps_grpo.py

# Quick smoke test (tiny backend, no HF model)
python scripts/smoke_letter_counting.py
```
