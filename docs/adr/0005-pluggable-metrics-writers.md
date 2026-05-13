# ADR 0005 — Pluggable metrics writers with graceful degradation (v0.6)

Date: 2026-05-10
Status: Accepted

## Context

v0.3 introduced a `metrics_sink: Callable[[dict], None]` on
`OnPolicyTrainerConfig`. This was enough for a single live-dashboard
recorder, but users wanted:
- Persistent JSONL for post-mortem analysis (without a running server).
- TensorBoard scalars for publication / sharing.
- W&B integration that doesn't break the training loop if wandb isn't
  installed or init fails.
- Multiple destinations simultaneously (JSONL + dashboard).

## Decision

Introduce `monitor/writers.py` with four concrete writers and a composition
primitive:

```
JsonlMetricsWriter        stdlib; appends one JSON line per iter
StdoutMetricsWriter       stdlib; prints formatted line
TensorBoardMetricsWriter  optional torch.utils.tensorboard; no-op if missing
WandbMetricsWriter        optional wandb.init/log; auto-init if configured
MultiMetricsWriter        composes N writers, isolates failures
```

`build_writer_from_config(cfg, output_dir, extra)` maps the YAML
`metrics:` block to a `MultiMetricsWriter`. Each writer's `__call__` is
wrapped in try/except inside `MultiMetricsWriter` — **metric IO must never
kill training**.

## Consequences

**Pros:**
- `pyproject.toml` runtime deps stay at `PyYAML>=6.0`. TB/W&B are
  optional extras; their absence silently disables the writer.
- Dashboard, JSONL, and TB can all run in the same training.
- W&B can now be fully configured from YAML, including project/name/tags and
  run config capture, without requiring user code to call `wandb.init()`.
- User-defined callables (for research experiments) compose cleanly via
  `MultiMetricsWriter([my_callback, JsonlMetricsWriter(…)])`.

**Cons:**
- Graceful degradation means a typo in the YAML (`wandbd: true`) is
  silently ignored. We accept this; adding a schema validator would
  double the config surface.
- Each writer's failure is isolated, but we don't log the failure — if
  TB stops working mid-run, the only signal is an incomplete log. The
  training run itself is never interrupted, which is the intended trade.

**Alternatives considered:**
- A formal "backend registry" à la Keras. Overkill for 4 writers.
- Direct imports of tensorboard in the trainer. Breaks "PyYAML is the
  only runtime dep" goal.
