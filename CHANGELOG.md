# Changelog

## 0.9.2 - 2026-05-28

Performance optimizations on top of 0.9.1 — backwards compatible.

- **HybridAlgo shared forward** (P0, ~40% GPU time savings on overlapping
  records). `RolloutBatch` gains opt-in fields `shared_new_logprobs` and
  `shared_logprobs_temperature`; `HybridAlgo` now runs `policy.score_batch`
  ONCE over the union of GRPO + OPD records and feeds the differentiable
  rows into both sub-algorithms. The autograd graph is preserved through
  the new `stack_cached_logprobs` helper. Each sub-algo (`GRPO`,
  `OPDAlgo`) reports a `shared_logprobs_cache_hit` flag in `stats.extra`,
  and Hybrid exposes `shared_forward_active`.
- **old_logprobs tensor cache** (P1). `RolloutRecord.metadata` now
  supports an optional `_old_logprobs_tensor` (1-D `torch.Tensor`) which
  `OnPolicyTrainer._prepare_update_batch` populates once per iteration.
  `update_epochs > 1` passes no longer rebuild
  `torch.tensor(record.old_logprobs)` on every minibatch. GRPO, PPO and
  OPD all consume the cache through the new `old_logprobs_tensor()` helper
  with transparent dtype/device migration.
- **Repository hygiene**: `.gitignore` now covers `.DS_Store` recursively,
  `.venv311/`, `.session_tmps/`, `.benchmarks/`, `.thumbs/`, `.uploads/`,
  `.coverage.bogon.*`, and explicitly excludes the legacy
  `atropos-main.zip` / `tinker-atropos-main.zip` snapshots (which should
  be removed via `git rm --cached`).
- **Tests**: new `tests/test_optimization_v091.py` covers the
  RolloutManager assistant-message extraction, TrainerBridge retry +
  cancellation, ToolcallReward weight blending, the shared
  on-policy-config builder, config schema validation,
  `stack_cached_logprobs` autograd-preservation, and the
  `old_logprobs_tensor` zero-copy path.

## 0.9.1 - 2026-05-27

Architectural refactor — backwards compatible.

- **Config**: introduce `build_shared_on_policy_config()` helper that copies
  overlapping fields from any trainer-config dataclass into
  `OnPolicyTrainerConfig`. Eliminates ~180 lines of repetitive field-by-field
  mapping in `GRPOTrainer`, `PPOTrainer`, and `GSPOTrainer` (each
  `__init__` shrinks by ~60 lines).
- **Config validation**: `load_config()` now runs a schema check by default
  via the new `validate_config()` (raises `ConfigValidationError` on
  unknown `runtime.integration`, non-positive `max_agent_turns`, invalid
  `trainer.n_iters` / `trainer.lr`, etc.). Pass `load_config(path,
  validate=False)` to opt out.
- **OnPolicyTrainer**: split the 200-line `__init__` into 12 focused
  `_setup_*` helpers (distributed wrapping, optimizer + LR scheduler, AMP,
  rollout backends, reference policy, agent loop, logging, reward & KL
  controllers, run state, checkpoint manager, batch generator, token budget,
  entropy scheduler). Public attributes and initialization order preserved.
- **Async training loop**: add native `OnPolicyTrainer.train_async()`. The
  sync `train()` is now a thin wrapper that raises a clear error if invoked
  from a running event loop (previously it silently broke in Jupyter). SFT
  helpers also expose `_maybe_run_*_sft_async()` variants.
- **RolloutManager**: `RolloutStep.assistant_message` is now populated from
  `messages[]` (explicit `turn_index` annotation preferred, positional
  fallback for legacy agent loops). Downstream rewards that need the
  textual assistant reply (LLM judge, coherence, length-style) now see it.
- **PPO**: `_prepare_update_batch` now runs a single
  `score_with_value_batch()` instead of per-record `score_with_value()`,
  with graceful fallback to the per-record loop when a backend does not
  implement the batched value path. Records with empty `response_ids` are
  filtered before the batched call.
- **ToolcallReward**: the previously-hardcoded 0.4 / 0.3 / 0.3 blend is
  exposed via `name_weight`, `schema_weight`, `value_weight` keyword
  arguments (auto-normalised to sum 1.0). Defaults preserved.
- **TrainerBridge**: gain retry-with-backoff, counters (`submitted`,
  `failed`, `retried`, `last_error`), and a `.metrics()` helper. `submit()`
  no longer retries on `CancelledError` / `KeyboardInterrupt`. Default
  `max_retries=1` keeps prior single-shot semantics.
- **HybridAlgo**: KL aggregation now weights each branch by `w_branch ×
  records_in_branch` (previously used unweighted record counts). Records
  that feed both GRPO and OPD branches are tracked under
  `extra.hybrid_double_forward_records` for future shared-forward
  optimization.
- Suppressed `metrics_sink` / `env.snapshot` exceptions now surface as
  Python warnings instead of being silently swallowed; training is still
  not interrupted.

## 0.9.0 - 2026-05-24

- Bump package metadata from `0.6.0.dev0` to `0.9.0`.
- Add RLOO, OPD, Hybrid, Factored GRPO, and Best-of-N algorithm variants.
- Add GSPO sequence-level GRPO variant with `algo: gspo` CLI support.
- Add `LengthPenaltyReward` and YAML `reward.components` wiring.
- Add `stability_preset` for standard/aggressive reward normalization, KL, and
  entropy-schedule combinations.
- Add opt-in EMA shadow policy rollout for local HF/Tiny trainers.
- Add vLLM rollout backend wiring with learner-to-rollout weight sync.
- Add distributed training scaffolding for FSDP, DDP, multiprocessing rollout
  pools, and Ray rollout actors.
- Add token-budget clipping, entropy scheduling, mixed curriculum support, and
  richer v1.0 optimization smoke coverage.
- Add PRM, LLM judge, next-state PRM, shaping, and letter-counting reward
  components.
- Propagate PRM/dense reward `token_rewards` from trajectory metadata into
  `RolloutRecord.metadata` for PPO GAE consumption.
- Add code sandbox, SWE-style Atropos integration hooks, and replay export
  utilities.
- Add opt-in trainer wiring for token-budget clipping and entropy scheduling.
- Fix PPO/GRPO ratio math under mixed precision by casting rollout log-probs
  to the learner log-prob dtype before subtraction.
- Fail fast when a generation-only backend is passed as the learner policy.
- Improve vLLM weight sync compatibility by preferring `apply_model_updates`
  when exposed and falling back to legacy driver-worker model loading.
- Add a stable GRPO recipe for format-collapse-sensitive runs.

## 0.8.0 - 2026-05-24

- Add batched backend scoring APIs for policy and value-head updates.
- Vectorize PPO/GRPO loss paths, clipped value loss, and GAE helpers.
- Add K3 KL estimation, adaptive KL control, target-KL early stopping, and
  running reward normalization.
- Add checkpoint resume state for running reward statistics and adaptive KL.

## 0.7.0 - 2026-05-24

- Add resumable `CheckpointManager` bundles with optimizer, RNG, stats, and
  config snapshots.
- Add best-checkpoint retention and trainer early stopping.
- Add interleaved/bootstrap SFT hooks for on-policy trainers.
- Add multi-turn agent credit assignment with `hybrid` as the default mode.
