# Self-Evolution Export

`session-eval-export` turns Hermes session traces into an evaluation dataset
that matches the `task_input` / `expected_behavior` shape used by
`hermes-agent-self-evolution`.

## Input

- Session JSONL from `session-sidecar`
- Trajectory JSON from `rollout`
- Session bundles with `messages` or `metadata.messages`

## Output

- `train.jsonl`
- `val.jsonl`
- `holdout.jsonl`
- `manifest.json`

## Example

```bash
python -m hermes_agentic_rl.cli.main session-eval-export \
  --config configs/session_eval_export_hermes.yaml
```
