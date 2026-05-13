# FAQ

## How do I know RL improved the model?

Use `eval-rl` on a held-out grouped split, then compare baseline and checkpoint
metrics on the same trace groups. Training reward alone is not enough. The
generated `eval_summary.json` also includes a `promotion_readout` block that
summarizes the strongest candidate against the baseline in one place, and
`promotion.md` gives you the same decision in a quick human-readable form.

## Why is the Apple MPS path still mentioned?

The MPS configs are the fastest local path for interactive iteration on Apple
Silicon. They are useful for development even when the real deployment target
is a remote GPU runner.

## Why is the lock file not the full training stack?

`requirements-lock.txt` is intentionally scoped to the engineering toolchain
and docs stack. Training-time extras such as `torch`, `datasets`, `wandb`, and
`transformers` stay optional so the base lock stays lighter and easier to audit.
