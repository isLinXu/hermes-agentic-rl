# Changelog

## 0.11.0 - 2026-06-07

OPD reliability + process-reward parity with OpenClaw-RL. Closes the "OPD
silently degrades to GRPO" gap (deep-analysis A2/A3).

- **`rewards/opd_hint_extractor.py::OPDHintExtractor`** (A2, P0). An in-trainer
  judge that recovers OPD directive hints from the next-state signal for any
  record lacking one, run *before* the teacher fill. Previously `opd_hint` only
  existed if a specific env (`letter_counting`) or `NextStatePRMComponent`
  pre-populated it; in every other config the OPD/Hybrid branch was a no-op and
  silently degraded to plain GRPO. Now any env/reward that surfaces a next-state
  signal drives OPD. Supports rule-based, LLM-judge (OpenAI-compatible), and
  `letter_counting` extractors via the `opd.hint_extractor` YAML block. Adds a
  de-templating quality filter (rejects "be more helpful"-style low-info hints),
  per-iter `opd_hint_*` metrics including `opd_hint_effective_rate`, a
  `max_records` cost guard, and fail-soft judge-error handling. The trainer now
  propagates `runtime["next_state"]` onto records (`_extract_next_state`) so the
  extractor has a signal to work with.
- **`rewards/judge_cache.py::JudgeCache` + `cached_judge`** (A3). Content-
  addressed, bounded-LRU, thread-safe cache wrapping any sync/async judge so the
  m-vote PRM and OPD hint extractor stop re-querying identical
  `(response, next_state)` pairs — the cost prerequisite for affordable LLM
  judges. Opt-in via `opd.hint_extractor.cache`.
- **`rewards/process_reward.py::ProcessRewardAggregator`** (A3). Implements
  OpenClaw-RL's long-horizon objective `final = o + (1/m)·Σ rᵢ`: combines the
  terminal outcome with the mean majority-voted per-step next-state PRM reward.
  Reads `runtime["step_next_states"]`, gracefully falls back to the single
  `next_state`, and resolves the outcome via metadata key or callable.
- **Config**: `configs/letter_counting_hybrid_opd_extractor.yaml` demonstrates
  OPD firing through the in-trainer extractor (no bespoke hint plumbing).
- **Tests**: `tests/test_opd_hint_extractor.py` (extraction, quality filter,
  fail-soft, async, factory, full trainer closed loop) and
  `tests/test_process_reward_and_cache.py` (cache hit/miss/LRU, aggregation,
  fallback, clip).

## 0.10.0 - 2026-06-01

OPD teacher-logprob **closed loop** — the headline algorithmic gap from the
deep-analysis report is now closed (OpenClaw-RL §3.2). Also aligns
`pyproject.toml` version (was a stale `0.6.0.dev0`).

- **`rewards/opd_teacher.py::TeacherLogprobFiller`** (P0). Re-scores hinted
  records under a hint-enhanced context (`prompt + [HINT_START]{hint}[HINT_END]`)
  using the current policy as a self-distillation teacher, under
  `torch.no_grad()` and re-using the rollout `response_ids` for token
  alignment, then writes `metadata["teacher_logprobs"]`. Previously
  `next_state_prm.py` only wrote an empty placeholder, so the OPD branch was a
  no-op on real data and `HybridAlgo` silently degraded to plain GRPO.
- **Capability-axis-aware OPD weighting** (hermes extension beyond OpenClaw-RL's
  single global `w_opd`). The filler classifies each hint into a capability
  axis and stamps a per-record `opd_adv_scale`; `OPDAlgo` multiplies the
  token-level directive advantage by it and reports `opd_adv_scale_mean`.
- **Trainer wiring.** `OnPolicyTrainer` gains `opd_teacher_fill` /
  `opd_hint_template` / `opd_teacher_max_hint_tokens` /
  `opd_capability_axis_weights` config, instantiates the filler, runs it each
  iter after reward normalisation, propagates `opd_hint` into record metadata,
  and fan-outs `opd_teacher_*` metrics.
- **`hybrid` algo wired into `train-rl` CLI** via `HybridTrainer` /
  `HybridTrainerConfig` and an `opd:` YAML block. `algo: hybrid` is now a
  first-class CLI option alongside `grpo` / `ppo`.
- **`letter_counting_next_state` reward** derives a corrective next-state +
  directive hint from the verifiable ground truth, making the full OPD loop
  runnable end-to-end. Ships A/B configs
  (`letter_counting_hybrid_opd.yaml` vs `letter_counting_grpo_baseline.yaml`)
  and a CPU smoke config.
- **Pipelined (double-buffered) rollout/update** (P0-2, step 1). Opt-in
  `pipeline_rollouts` (requires a rollout pool): iteration N+1's rollouts are
  dispatched — and their weight snapshot broadcast — *before* iteration N's
  gradient update, so rollout overlaps the update (tolerates 1-step policy
  staleness; PPO/GRPO ratio clipping absorbs the lag). `OnPolicyTrainer._one_iter`
  is split into `_collect_for_iter` + `_update_on_records`, and
  `_collect_distributed` into `_dispatch_distributed` + `_drain_distributed`.
  Emits `rollout_staleness` (0 for BSP, ~1 in steady-state pipelined mode) and
  `policy_version` (completed-update count) per iter for observability.
- **Multi-stream unified training** (P0-3). New `MixedCurriculumEnv` keeps all
  task streams live and draws each item from a *weighted, adaptively-reweighted*
  mixture (difficulty-prioritised: struggling streams gain sampling weight,
  floored at `min_weight`), so heterogeneous task types train in a single
  optimizer step instead of separate phases. `CurriculumEnv.observe` now accepts
  a `level=` kwarg; the trainer's new `_observe_env_reward` helper forwards the
  per-item stream tag so reward is attributed to the right stream. Reachable via
  the CLI `environment.type: multi_stream` (per-stream reward routing by
  `_curriculum_level`); per-stream weights/counts/means surface under
  `env_snapshot`. Ships `configs/multi_stream_smoke.yaml`. Rollout records are
  tagged with `stream_level` and `_summarize_batch_metadata` emits per-stream
  `stream/{i}/{count,share,mean_reward,reward_std}` (plus `n_streams`) into the
  flattened metrics record, so per-stream performance is visible alongside the
  env snapshot.

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
