# Training loop reference

Single authoritative account of what `OnPolicyTrainer.train()` does in a
typical iteration. Both `GRPOTrainer` and `PPOTrainer` inherit this skeleton.

## Pseudocode (one iter)

```
async def _one_iter(it):
    if it == 0:
        await env.setup()
    if lagrangian is not None:
        lagrangian.begin_iter()                    # reset per-iter cost accum.

    # --- rollout collection ---
    if rollout_pool is None:
        records = []
        for _ in range(prompts_per_iter):
            item = await env.get_next_item()
            # for _ in range(group_size): fresh PolicyAgentLoop → Trajectory
            records.extend(await _collect_group(item))
    else:
        records = await _collect_distributed()     # MPRolloutPool broadcast+drain

    batch = RolloutBatch(records=records)

    # --- loss & step ---
    optim.zero_grad()
    loss, stats = algo.compute_loss(policy, ref_policy, batch)
    if lagrangian is not None:
        loss = lagrangian.penalty_term(loss)       # + λ · mean_cost
    if loss.requires_grad:
        loss.backward()
        clip_grad_norm_(policy.trainable_parameters(), grad_clip)
        optim.step()

    return stats
```

## Per-iter side effects (outer loop)

| Stage | Trigger | Action |
|---|---|---|
| After loss.step | Always | Lagrangian dual update (`λ ← max(0, λ + lr·(E[cost] - limit))`) |
| After metrics aggregation | `log_every` | Default logger prints `[train] iter=… mean_reward=… loss=…` |
| After logger | Always (if configured) | `metrics_sink(record)` → `MultiMetricsWriter` → JSONL / TB / W&B / dashboard |
| Per N iters | `save_every > 0` (legacy) | `torch.save(model.state_dict(), …)` |
| Per N iters | `checkpoint_every > 0` (v0.6) | `CheckpointManager.save(CheckpointState(...))` full bundle |
| Start of train() | `auto_resume=true` OR `resume_from=…` | `CheckpointManager.load_*` → restore model / optim / RNG / stats |
| End of train() | `checkpoint_every > 0` | Final flush at `iter = n_iters - 1` |

## Rollout record (per rollout)

`RolloutRecord` is what the `Algo` consumes; every optimizer step sees:

- `prompt_ids: list[int]` — context fed to the policy.
- `response_ids: list[int]` — sampled tokens.
- `old_logprobs: list[float]` — logπ_old at sampling time (needed for PPO ratio).
- `reward: float` — scalar aggregate from `RewardManager`.
- `group_id: str` — used by GRPO to group rollouts sharing a prompt.
- `metadata: dict` — reward components, final_output, turn_index, etc.

The `agent_loop_factory` is responsible for populating this metadata into
`trajectory.metadata["runtime"]["rl"]`. If it's missing, the trainer raises
rather than silently producing degenerate records.

## Resumption semantics

`auto_resume` uses `CheckpointManager.load_latest()` which picks the
*newest* `iter_XXXXX` directory. Resumption restores:

1. model `state_dict` into `policy.model`
2. optimizer state into `self._optim` (best-effort; mismatches are ignored)
3. torch / python / numpy RNG states
4. stats history (`self.stats.iters`)
5. `best_reward` + `best_iteration`
6. `self._start_iter = ckpt.iteration + 1` — training skips the finished iter

If `resume_from=N` refers to a non-existent iter, a `RuntimeError` is raised
immediately during trainer construction — we don't silently start over.

## KL estimators

`kl_estimator ∈ {k1, k2, k3}` controls how `KL(π_new ‖ π_ref)` is approximated:

```
r = logπ_new - logπ_ref

k1:  mean(r)                      # unbiased, signed (legacy)
k2:  mean(0.5 * r²)               # biased, ≥ 0, low-variance
k3:  mean(exp(-r) - 1 + r)        # unbiased AND ≥ 0  ← recommended
```

Default is `k1` for v0.5 backward compatibility. New configs should use `k3`.

## Advantage normalization

| Algo | Config | Semantics |
|---|---|---|
| GRPO | `advantage_norm="group"` (default) | `A_i = (r_i − mean_g) / std_g` per prompt |
| GRPO | `advantage_norm="batch"` | z-score across all rollouts in the iter |
| GRPO | `advantage_norm="whiten"` | batch z-score + clip to ±3 |
| PPO  | `normalize_advantage=true` (default) | batch z-score on GAE output |
| PPO  | `whiten_advantage=true` + `advantage_clip=3.0` (v0.7) | above + clip |
