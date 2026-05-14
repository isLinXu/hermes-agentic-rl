# Hermes-Agentic-RL Optimization Plan

This plan distills the supplied Hermes Agent articles into a concrete roadmap
for this repository. The shared theme is clear: Hermes-style self-evolution is
not only model-weight training. It is a closed loop across Skills, memory,
prompt construction, context management, harness execution, replay data, and
held-out evaluation.

## Source Themes

The five articles point to the same design center from different angles:

| Source | Useful idea for this repo |
|---|---|
| Hermes architecture deep dive | Treat self-improvement as a native closed loop, not an afterthought. |
| Source-code self-evolution analysis | Improve how the agent uses the model through persistent files, prompt strategy, and judge feedback, even before changing weights. |
| Skills system analysis | Make reusable procedures first-class training/eval artifacts, with progressive loading and low token overhead. |
| Context compression tutorial | Measure context and interaction control, not only task reward. |
| Prompt / Context / Harness practice | Combine external self-evolution with RL so behavior and model weights can both improve. |

## Design Direction

Hermes-agentic-rl should optimize an agent along explicit capability axes:

| Axis | What it measures | Current signals |
|---|---|---|
| `task_success` | Does the policy solve held-out tasks? | `mean_reward`, `success_rate` |
| `tool_use_reliability` | Does it emit valid Hermes tool calls and correct arguments? | `tool_call_parse_ok`, `tool_name_match`, argument metrics |
| `interaction_control` | Does it finish cleanly and preserve eval signal quality? | `finished_naturally_rate`, `success_metric_found_rate` |
| `self_evolution_signal` | Is the experience useful for replay, Skills, or future self-improvement? | `success_score_mean`, `mean_reward` |
| `prompt_context` | Does the prompt/context strategy preserve the right information? | planned context-retention and compression metrics |
| `skill_learning` | Can repeated successful patterns become reusable Skills or datasets? | planned Skill export and replay-mining metrics |

The important shift is that checkpoint promotion should answer "which agent
capability improved?" rather than only "did reward go up?".

## Implementation Roadmap

1. Capability-axis evaluation.
   Implemented in this batch. `eval-rl` now writes `capability_report.md` and
   embeds `capability_report` in `eval_summary.json`. `self-evolution-batch`
   also maps each direction's `objective.target_metrics` into inferred
   capability axes.

2. Direction-aware replay mining.
   Extend session replay reports so each replay sample records why it is useful:
   tool reliability, recovery, context preservation, or Skill-worthy procedure.
   This makes self-evolution datasets easier to filter and compare.

3. Skill export loop.
   Add an exporter that converts repeated high-quality sessions into
   Skill-style Markdown candidates with metadata, validation samples, and
   failure cases. The goal is external agent improvement before or alongside
   RL weight updates.

4. Context and prompt harness benchmarks.
   Add held-out scenarios that stress long context, compression, memory recall,
   and tool-result summarization. These should produce metrics that can feed the
   `prompt_context` capability axis.

5. RL promotion with capability thresholds.
   Extend `promotion_gate` so a run can require improvements on selected
   capability axes, not only reward/success-rate deltas.

6. Online self-evolution cycle.
   Combine real Hermes sessions, replay mining, Skill candidate export,
   train/eval, W&B reports, and `eval-gate` into one repeatable loop.

## Immediate Result

The first step is now in place: evaluation output includes capability axes, and
batch self-evolution directions are annotated with the axes they target. This
gives the next training run a more useful diagnosis surface:

```bash
python -m hermes_agentic_rl.cli.main eval-rl \
  --config configs/hermes_reasoning_traces_eval_rl_terminal_command_stage2.yaml
```

Review:

- `eval_summary.json` -> `capability_report`
- `capability_report.md`
- W&B summary -> `capability_report`
- `outputs/hermes_self_evolution_batch/batch_summary.json` -> `capability_axes`
