# ADR 0003 — Resumable checkpoints as a first-class feature (v0.6)

Date: 2026-05-10
Status: Accepted

## Context

Pre-v0.6 `save_every` only dumped the model's `state_dict()` to a flat
`.pt` file. This lets you *reload the model* but not *resume training*:
optimizer momentum, learning rate schedules, RNG streams, reward history —
all lost. Long RL runs are often 6+ hours on a laptop MPS; a crash sent you
back to iter 0.

Goals:
1. One switch (`auto_resume: true`) and YAML continues from the latest ckpt.
2. Crash-safe writes (atomic temp+rename).
3. Automatic pruning (`keep_last_checkpoints`).
4. No extra dependencies (no MosaicML Composer, no Accelerate).

## Decision

Introduce `CheckpointManager` at `trainers/checkpoint.py` with:

```
{output_dir}/checkpoints/
    iter_00010/
        model.pt              # policy.model.state_dict()
        optimizer.pt          # self._optim.state_dict()
        trainer_state.json    # iteration, best_reward, stats history
        rng_state.pt          # torch + python + numpy RNG
        config.yaml           # dataclass snapshot (read-only; for audit)
```

`OnPolicyTrainer` lazily instantiates the manager when any of
`checkpoint_every > 0 / resume_from != None / auto_resume = true` is set.
`_maybe_resume()` runs once in `__init__` and sets `self._start_iter`.

## Consequences

**Pros:**
- Drop-in: existing configs don't change. New fields default to "off".
- `metrics.jsonl` already captures full stats history, so after resume the
  merged JSONL is the audit trail of everything that happened.
- CI's end-to-end smoke now verifies resume semantics (5 tests).

**Cons:**
- Checkpoints are not small for HF backends (a GPT-2 model is ~475MB).
  We rely on `keep_last_checkpoints` (default 3) to bound disk use.
- `torch.save` in PyTorch 2.x rejects temp names starting with `.` — we
  use `tmp_<name>_<uniq>.partial` siblings for atomic replace.
- RNG state contains `numpy._reconstruct` which is not allowlisted by
  `weights_only=True`. We trust our own writer and load RNG with
  `weights_only=False`.

**Alternatives considered:**
- MosaicML Composer — pulls in too many dependencies for a library whose
  core runtime dep is only `PyYAML`.
- HF Accelerate — would force the backend abstraction to leak HF-ness.
- Write-through to Version Manager (ADR 0004) — confusing role overlap.
