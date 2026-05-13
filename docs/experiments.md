# Experiment Notes

This page keeps run-specific results out of the README. Use it to track
training/eval snapshots, W&B links, and caveats that may change as configs and
datasets evolve.

## Held-Out Checkpoint Sweep

- Split: `val`, grouped by `source_trace_id`; 32 rollouts from 36 held-out
  trace groups.
- W&B: [hermes-reasoning-traces-parquet-mps-terminal-curriculum-eval](https://wandb.ai/linxu/hermes-agentic-rl/runs/zkiuwvrh).
- Baseline: `mean_reward=0.2621`, `success_rate=0.6250`,
  `similarity=0.5606`.
- `filtered_hybrid:iter_00016`: `mean_reward=0.2444`,
  `success_rate=0.3750`, paired winner=`baseline`.
- Best checkpoint: `filtered_hybrid:iter_00023`, `mean_reward=0.2714`,
  `success_rate=0.8750`, `similarity=0.6758`.
- Paired A/B for best: `mean_diff=+0.0093`, `approx_p=0.0152`,
  winner=`candidate`.
- Caveat: `tool_call_parse_ok`, `tool_name_match`, and argument overlap are
  still `0.0`, so this shows reward/similarity lift, not reliable executable
  tool-call success yet.

## Stage-2 Command-Action Sweep

- Train W&B run:
  [terminal-command-stage2](https://wandb.ai/linxu/hermes-agentic-rl/runs/n15aytgt).
- Eval W&B run:
  [terminal-command-stage2-eval](https://wandb.ai/linxu/hermes-agentic-rl/runs/zjqc00ep).
- Baseline under the adapter: `mean_reward=0.3813`,
  `similarity=0.0662`, `success_rate=0.0000`,
  `tool_call_parse_ok=1.0000`, `tool_name_match=1.0000`,
  `argument_key_overlap=1.0000`.
- Best regular checkpoint sweep result: `terminal_command_stage2:iter_00016`,
  `mean_reward=0.4301`, `similarity=0.1397`.
- Targeted eval of the train-best checkpoint
  `checkpoints_best/iter_00021`: `mean_reward=0.4437`,
  `similarity=0.1602`, `success_rate=0.3438`,
  `mean_reward` delta `+0.0624` over baseline.
- Promotion gate snapshot for the held-out stage-2 config:
  recommendation=`promote`, paired winner=`candidate`,
  `approx_p=1.2849863395558714e-09`.
- Interpretation: the adapter now guarantees executable Hermes terminal
  structure, and the command-content success metric
  (`metadata/argument_value_similarity >= 0.2`) shows a real lift over baseline.
  It is still not solved: exact matches remain `0.0000`, and generation still
  tends to hit the token cap instead of stopping naturally.

## MPS Training Snapshots

### Temperature / Grad-Clip Fix

- Dataset: `/Users/gatilin/Downloads/train.parquet`, 7055 parquet rows.
- Extraction: first 128 rows expanded to 962 assistant-turn samples.
- Device: `mps:0`.
- W&B: [hermes-reasoning-traces-parquet-mps-tempfix-short](https://wandb.ai/linxu/hermes-agentic-rl/runs/ierc0gjk).
- Result: `last_mean_reward=0.12571`, `best_mean_reward=0.13030`,
  `clip_frac=0.0`, `approx_kl≈3.1e-5`.
- Interpretation: the policy update is stable, but
  `parse_ok/name_match/argument_key_overlap` are still `0.0`.

### Terminal Curriculum

- Extraction: first 128 rows, filtered to terminal-command tool calls only.
- Device: `mps:0`.
- W&B: [hermes-reasoning-traces-parquet-mps-terminal-curriculum](https://wandb.ai/linxu/hermes-agentic-rl/runs/sq96yy60).
- Result: `last_mean_reward=0.28246`, `best_mean_reward=0.29632`.
- Interpretation: the fixed JSON scaffold and terminal-only curriculum raise
  reward substantially and improve partial structure, but the tiny model still
  does not close the full tool-call cleanly.

### Prefill Validation

- Extraction: first 32 rows, same filtered tool-call target shape.
- W&B: [hermes-reasoning-traces-parquet-mps-prefill-check](https://wandb.ai/linxu/hermes-agentic-rl/runs/q7i857sk).
- Result: `last_mean_reward=0.13507`, `tool_call_present_mean=1.0`,
  `partial_tool_call_score_mean=0.09125`, while full JSON/tool-name metrics
  remain `0.0`.
- Interpretation: the scaffold fixes the "no tool-call structure at all"
  failure mode, but more SFT/RL is still needed to learn valid JSON and tool
  names.

## Live Checks

- `hermes-preflight`: `repo_source=subproject`, `missing=[]`.
- `train-rl` real dataset smoke with W&B sync.
- `train-rl` local parquet MPS run with W&B sync.
- `online-cycle` NewAPI smoke with real Hermes, worker update, and
  self-evolution export.
