# `train-rl` YAML configuration reference

Top-level keys recognized by `hermes_agentic_rl.cli.train_rl.run_train_rl`:

```yaml
algo: grpo | ppo              # default "grpo"

backend:
  name: tiny | hf
  # tiny options
  dim: 32
  n_heads: 4
  n_layers: 2
  max_len: 128
  seed: 0
  with_value_head: false      # auto-forced true for PPO
  # hf options
  model_name_or_path: gpt2
  device: cpu                 # cpu | cuda | mps
  dtype: float32
  trust_remote_code: false

environment:
  type: echo | sim_tool | letter_counting | curriculum | hermes_reasoning_traces
  # echo / sim_tool / letter_counting
  dataset_path: path/to/tasks.jsonl   # optional
  dataset_size: 16                    # sim_tool only
  dataset_seed: 0
  max_level: 10                       # letter_counting only
  n_samples: 200                      # letter_counting only
  # hermes_reasoning_traces (HF dataset)
  dataset_name: lambda/hermes-agent-reasoning-traces
  dataset_config: kimi
  dataset_split: train
  dataset_limit: 12
  streaming: true
  shuffle: true
  seed: 0
  revision: main
  cache_dir: null
  history_window_messages: 12
  max_prompt_chars: 400             # useful for tiny/char-level backends
  include_system_prompt: true
  tool_call_format_hint: false
  assistant_response_prefix: ""     # optional fixed prefix appended to prompts
  assistant_response_suffix: ""     # optional fixed suffix restored for reward
  assistant_response_adapter: ""    # terminal_command_tool_call for command-only actions
  reward_weight: 1.0
  # curriculum
  levels:                             # list[dict] of sub-env specs
    - { type: echo }
    - { type: sim_tool, dataset_size: 16 }
  window: 20
  promote_threshold: 0.6
  demote_threshold: null
  allow_demote: false

agent_loop:
  type: policy | multi_turn           # default "policy"
  max_turns: 3                        # multi_turn only
  max_new_tokens_per_turn: 16         # multi_turn only

train_rl:
  # --- core ---
  n_iters: 30
  group_size: 4
  prompts_per_iter: 2
  lr: 5.0e-3
  max_new_tokens: 8
  temperature: 1.0
  clip_eps: 0.2
  grad_clip: 1.0
  log_every: 5
  save_every: 0                       # legacy flat state_dict dump
  seed: 0
  multi_turn: false
  output_dir: outputs/run_1           # overridden by --output CLI arg

  # --- reference policy / KL ---
  use_reference: false
  kl_coef: 0.0
  kl_estimator: k1 | k2 | k3          # default k3 (low-variance, non-negative)

  # --- loss aggregation ---
  loss_agg: mean_token | sum_token | dr_grpo
  max_len_for_dr_grpo: 256

  # --- GRPO-only ---
  entropy_coef: 0.0
  advantage_eps: 1.0e-6
  advantage_norm: group | batch | whiten
  per_token_advantage: false

  # --- PPO-only ---
  vf_coef: 0.5
  vf_clip_eps: 0.2
  gamma: 1.0
  lam: 0.95
  normalize_advantage: true
  whiten_advantage: false             # v0.7
  advantage_clip: 3.0                 # v0.7

  # --- interleaved SFT (GRPO) ---
  interleave_sft_every: 0
  interleave_sft_samples: 32
  interleave_sft_lr: 1.0e-4
  interleave_sft_epochs: 1
  interleave_sft_batch_size: 8
  bootstrap_sft_rounds: 0
  bootstrap_sft_samples: 32
  bootstrap_sft_lr: 1.0e-4
  bootstrap_sft_epochs: 1

  # --- batch generate ---
  batch_generate: false

  # --- v0.6: checkpoint / resume ---
  checkpoint_every: 0                 # iters between full ckpts; 0 disables
  keep_last_checkpoints: 3
  resume_from: null | int | "latest"
  auto_resume: false                  # True → pick newest ckpt if any

# v0.6: metrics backends — all optional, all isolation-fault-safe
metrics:
  jsonl: true                         # → <output_dir>/metrics.jsonl
  stdout: false
  tensorboard: true | "<path>"        # → <output_dir>/tb/
  wandb: true |                       # auto-calls wandb.init() if installed
    enabled: true
    prefix: train                     # metric namespace, e.g. train/loss
    project: hermes-agentic-rl
    name: echo-grpo-run
    group: ablation-a
    mode: online | offline | disabled
    tags: [grpo, echo]
    notes: short description
    finish_on_close: true             # default true
    config:                           # merged into wandb.init(config=...)
      sweep_id: debug-01

# v0.3: live dashboard (stdlib HTTP + Chart.js CDN)
dashboard:
  enabled: false
  host: 127.0.0.1
  port: 8765

# v0.6: learned reward model (Bradley-Terry RM head)
reward_model:
  enabled: false
  backend: tiny | hf
  # tiny options (same as top-level backend)
  dim: 32
  n_heads: 4
  n_layers: 2
  max_len: 256
  seed: 0
  # hf options
  model_name_or_path: gpt2
  device: cpu
  dtype: float32
  trust_remote_code: false
  # loading a pre-trained head
  head_path: path/to/rm_head.pt
  weight: 1.0
  freeze_base: true
```

## Minimal runnable example

See `configs/echo_grpo_mvp.yaml` (v0.2 MVP, still works verbatim).

## Annotated v0.6 example

See `configs/echo_grpo_v06.yaml` for a config that exercises:
- K3 KL estimator
- Checkpoint + auto-resume
- Pluggable metrics writers

## W&B Notes

- Recommended install: `pip install -e '.[rl,data,metrics]'`
- Online sync uses the standard `WANDB_API_KEY` environment variable.
- The trainer automatically pushes all scalar iteration metrics to W&B,
  including nested numeric fields such as lagrangian or multi-turn credit
  summaries when present.
- Final run summary metrics such as `last_mean_reward`, `best_mean_reward`,
  and `reward_delta` are written into the W&B run summary at the end.

## Session Replay Mining

`session-replay`, `online-cycle`, and `self-evolution-batch` can annotate replay
records with direction-aware mining metadata:

```yaml
replay_mining:
  enabled: true
  min_skill_reward: 0.5
  long_context_messages: 8
  long_context_chars: 3000
```

When enabled, each replay sample gets `metadata.replay_mining` plus a flat
`metadata.capability_axes` list. The annotation includes capability axes,
reasons, recommended uses, a usefulness score, and whether the turn looks like
a Skill candidate. Quality reports include aggregate counts under
`replay_mining`, and `self-evolution-batch` rolls these up into
`mined_replay_axes` and `skill_candidates`.

Use these fields to filter the same session trace pool into different
optimization queues: tool reliability replay, failure recovery replay, context
benchmark seeds, or candidate Skill exports.

`session-train-worker` can consume the same annotations:

```yaml
session_train_worker:
  train:
    replay_filter:
      require_any_capability_axes: [tool_use_reliability]
      require_any_recommended_uses: [tool_reliability_replay]
      require_skill_candidate: false
```

Supported filter keys are `require_any_capability_axes`,
`require_all_capability_axes`, `require_any_recommended_uses`,
`require_all_recommended_uses`, and `require_skill_candidate`.
`self-evolution-batch` automatically injects a `require_any_capability_axes`
filter from each direction's `objective.target_metrics` when a direction maps
to known capability axes.
Set `replay_mining: false` or `replay_mining.enabled: false` to keep replay
records unannotated; in that mode `self-evolution-batch` also skips automatic
directional replay filters.

## `eval-rl` Held-Out Evaluation

`eval-rl` reuses the same `backend`, `environment`, `agent_loop`, and
`metrics` blocks, then adds an `eval_rl` block:

```yaml
eval_rl:
  output_dir: outputs/hermes_eval
  split: val                         # train | val | test | all
  split_by: source_trace_id          # keeps trace turns together
  val_ratio: 0.2
  test_ratio: 0.1
  seed: 0
  n_rollouts: 32
  temperature: 0.0                   # greedy by default for reproducibility
  max_new_tokens: 128
  success_metric: reward             # reward | metadata/<key> | component/<name>
  success_threshold: 0.25
  rank_metric: mean_reward           # or success_rate / any numeric metric
  promotion_gate:
    fail_on_hold: false              # return exit code 3 when gate says hold
    min_reward_delta: 0.01
    min_success_rate_delta: 0.02
    min_rank_metric_delta: 0.01
    require_paired_winner: true
    max_p_value: 0.10
    required_capability_axes: [tool_use_reliability]
    min_capability_delta: 0.02
    capability_thresholds:
      task_success: 0.00
    max_capability_regression: 0.01
  capability_axes:                   # false disables; omitted uses defaults
    tool_use_reliability:
      description: Hermes tool-call structure and argument fidelity
      metrics:
        - metadata/tool_call_parse_ok
        - metadata/tool_name_match
        - metadata/argument_key_overlap
        - metadata/argument_value_similarity
    task_success:
      metrics:
        - { name: mean_reward, weight: 0.6 }
        - { name: success_rate, weight: 0.4 }
  policies:
    - name: baseline
    - name: rl_checkpoint
      checkpoint_path: outputs/run/checkpoints/iter_00023/model.pt
    - name: sweep_run
      checkpoint_dir: outputs/run/checkpoints
      max_checkpoints: 3
```

Outputs are `eval_summary.json`, `eval_rollouts.jsonl`, `leaderboard.md`,
`ranking.md`, `promotion.md`, and `capability_report.md`. The summary also
includes `best_policy`, `ranking`, `success_metric`, `success_threshold`,
`promotion_readout`, and `capability_report` so you can pick the strongest
checkpoint directly and see which agent capability moved. W&B/TensorBoard/JSONL
logging is controlled by the same `metrics:` block used for training.

`capability_axes` groups raw eval metrics into agent-level improvement axes.
This is where Hermes-style self-evolution becomes measurable beyond a single
reward number: tool reliability, task success, interaction control, and
signals useful for deciding whether an experience should become replay data or
a Skill. Omit the block to use the default axes, or set it to `false` to skip
the report.

`eval-gate` runs the same evaluation but forces `promotion_gate.fail_on_hold`
to `true`, so CI or release scripts can stop automatically when held-out
evidence does not justify promotion.
If `required_capability_axes`, `capability_thresholds`, or
`max_capability_regression` are set, the promotion gate also checks
`capability_report.deltas`, so a candidate cannot pass merely by improving the
aggregate reward while regressing an important agent capability.

## Benchmark Suite Scorecards

`benchmark-suite` runs multiple `eval-rl` configs and aggregates them into a
single release-style scorecard:

```yaml
benchmark_suite:
  name: hermes-agentic-scorecard
  output_dir: outputs/hermes_benchmark_suite
  fail_on_required_failure: true
  benchmarks:
    - name: hermes-tool-call-heldout
      config_path: configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml
      required: true
      weight: 1.0
      score_metric: mean_reward
      thresholds:
        min_score: 0.0
        min_success_rate: 0.0
    - name: prompt-context-retention
      config_path: configs/context_benchmark_eval_rl.yaml
      required: true
      weight: 1.0
      score_metric: metadata/context_required_fact_recall
      thresholds:
        min_score: 0.0
```

```bash
python -m hermes_agentic_rl.cli.main benchmark-suite \
  --config configs/benchmark_suite.yaml
```

Each benchmark writes its normal `eval_summary.json`, `promotion.md`, and
`capability_report.md` under the suite output directory. The suite then writes
`scorecard.json` and `scorecard.md` with per-benchmark status, best policy,
promotion recommendation, threshold checks, required pass rate, and weighted
score. Use `require_promotion: true` on a benchmark when the suite should fail
unless that benchmark's promotion gate recommends `promote`. If an individual
benchmark crashes or fails to write a valid summary, the suite still records the
error in the scorecard so CI can report the broken benchmark directly.
`config_path` entries may be absolute, relative to the suite file, or relative
to the command's current working directory.

## Skill Candidate Export

`skill-export` converts replay records tagged by `metadata.replay_mining` into
reviewable Skill candidates:

```yaml
skill_export:
  input_path: outputs/hermes_self_evolution_batch/tool_use_reliability/replay.jsonl
  output_dir: outputs/hermes_skill_candidates
  require_skill_candidate: true
  min_reward: 0.25
  group_by: primary_axis             # primary_axis | recommended_use | single
  max_examples_per_skill: 8
  quality_min_examples: 2
  quality_min_mean_reward: 0.25
  quality_min_mean_usefulness: 0.5
  quality_min_axis_consistency: 0.6
  quality_min_validation_examples: 1
  quality_max_negative_signal_ratio: 0.25
  quality_ready_min_score: 0.75
  quality_blocked_max_score: 0.35
```

```bash
python -m hermes_agentic_rl.cli.main skill-export \
  --config configs/skill_export.yaml
```

Each exported candidate directory contains `SKILL.md`, `manifest.json`, and
`validation.jsonl`. New `session-replay` outputs include compact
`metadata.source_turn` evidence so the generated `SKILL.md` can cite the user
task, observed assistant behavior, feedback, reward, and capability axes.
The exporter also writes `quality_report.json` beside `summary.json`, and each
candidate `manifest.json` includes a `quality` block. Statuses are deliberately
review-oriented: `ready_for_review` means all configured checks passed,
`draft` means the candidate has promise but needs more evidence, and `blocked`
means the candidate failed enough checks that it should not be promoted without
new traces.

The quality gate is heuristic and explainable. It checks sample count, mean
reward, mean replay-usefulness score, dominant capability-axis consistency,
validation examples, and the ratio of negative feedback signals. For Hermes
traces, axis consistency treats capability labels as multi-label: repeated
tool-use traces can still pass even when they also carry `skill_learning` or
`self_evolution_signal`.

## Online Self-Evolution

`online-self-evolve` is the higher-level closed-loop orchestrator. It reuses
the existing `online-cycle` stages, then optionally exports Skill candidates
and runs `eval-gate`:

```yaml
online_self_evolve:
  output_dir: outputs/hermes_online_self_evolve
  stages:
    online_cycle: true
    skill_export: true
    eval_gate: false
  skill_export:
    input_path: outputs/hermes_online_self_evolve/replay.jsonl
    output_dir: outputs/hermes_online_self_evolve/skill_candidates
    require_skill_candidate: true
    min_reward: 0.25
    quality_min_examples: 2
    quality_ready_min_score: 0.75
  eval_gate:
    enabled: false
    config_path: configs/context_benchmark_eval_rl.yaml
    output_dir: outputs/hermes_online_self_evolve/eval_gate
```

```bash
python -m hermes_agentic_rl.cli.main online-self-evolve \
  --config configs/online_self_evolve.yaml \
  --once \
  --limit 1
```

Outputs include `online_self_evolve_summary.json` and
`online_self_evolve_report.md`, which summarize sessions, replay records, Skill
candidates, Skill quality status counts, optional promotion-gate results, and
stage artifact paths.

## Hermes Reasoning Traces

- `environment.type: hermes_reasoning_traces` loads
  `lambda/hermes-agent-reasoning-traces` from Hugging Face.
- Each assistant turn in a trace becomes one training item with a full
  conversation prefix as the prompt and the real assistant reply as the target.
- For small experiments, set `streaming: true` + `dataset_limit` so the loader
  can use Hugging Face's rows API fallback without pulling the whole dataset.
- For terminal-command stages, set
  `assistant_response_adapter: terminal_command_tool_call`. The loader trains on
  only the extracted `arguments.command` string, while reward/eval wraps model
  output back into a Hermes `<tool_call>` block before scoring parse validity,
  tool-name match, argument-key overlap, and command similarity.
- For command-action eval, prefer
  `success_metric: metadata/argument_value_similarity` with an explicit
  threshold. This keeps command-content success separate from the fixed wrapper
  structure that the adapter already guarantees.

## Context Benchmark

`environment.type: context_benchmark` creates a lightweight benchmark for the
`prompt_context` capability axis. It stresses long-context fact retention,
user-constraint preservation, tool-result summarization, distractor avoidance,
and concise synthesis:

```yaml
environment:
  type: context_benchmark
  dataset_size: 12
  dataset_seed: 0
  noise_blocks: 10
  max_response_chars: 360
eval_rl:
  success_metric: metadata/context_required_fact_recall
  rank_metric: metadata/context_required_fact_recall
  promotion_gate:
    required_capability_axes: [prompt_context]
    min_capability_delta: 0.01
    max_capability_regression: 0.02
```

The reward component writes metadata including
`context_required_fact_recall`, `context_constraint_satisfaction`,
`context_tool_summary_retention`, `context_distractor_avoidance`,
`context_precision`, and `context_compression_ok`. These feed the default
`prompt_context` capability axis and can be used directly as `success_metric`
or `rank_metric`.
- This works best with `bootstrap_sft_rounds` or `interleave_sft_every` to warm
  start on the real traces before continuing RL updates.
- See `configs/hermes_reasoning_traces_grpo_smoke.yaml` for a conservative
  real-dataset smoke setup with W&B enabled.

## Interleaved / Bootstrap SFT Notes

- `interleave_sft_*` and `bootstrap_sft_*` are available on on-policy runs
  (`grpo` and `ppo`).
- They require the active environment to implement `build_supervised_samples(item)`.
- In-tree environments with teacher samples today: `echo`, `sim_tool`, `letter_counting`, and `curriculum` (delegates to the active sub-env).
- For tool-call tasks, ensure `agent_loop.max_new_tokens_per_turn` is long enough to emit the full tool syntax; `configs/sim_tool_grpo_multiturn.yaml` uses `48` on purpose.
- `multi_turn_credit` also supports `judge_weight` and `teacher_weight`. When teacher samples are available, the trainer can derive turn-local similarity shaping from the expected assistant response, which is especially useful for early tool-call and answer-format learning.
