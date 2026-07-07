# Configuration

## YAML Config

The primary configuration interface is a YAML file loaded by
`hermes_agentic_rl.yaml_config`. See `docs/configuration.md` for the full
field reference.

## Pydantic Validation

When pydantic is installed, configs are validated against a schema before
training begins:

```python
from hermes_agentic_rl.config_validation import validate_config

# Raises with human-friendly error if invalid
validated = validate_config(yaml_dict)
```

### Validation Rules

| Field | Constraint |
|-------|-----------|
| `backend` | Must be `tiny`, `hf`, `vllm`, or `mock` |
| `n_iters` | Must be >= 1 |
| `group_size` | Must be >= 1 (warning if < 2: GRPO degenerates to REINFORCE) |
| `lr` | Must be > 0 |
| `lora_hot_reload` | Requires `vllm_rollout_model` to be set |
| `staleness_adaptive_tis.interpolation` | Must be `linear` or `exp` |
| `staleness_adaptive_tis.min_rho_clip` | Must be <= `max_rho_clip` |
| `quantization.format` | Must be `gptq`, `awq`, or `gguf` |
| `rewards.composer.aggregator` | Must be `weighted_sum`, `mean`, `max`, or `gated` |

### Fallback (no pydantic)

When pydantic is not installed, a lightweight dict-based validator checks
the same constraints. The validated config is returned as a plain dict.

## Dataclass Config

For programmatic use, trainer configs are dataclasses with `__post_init__`
cross-field validation:

```python
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainerConfig

cfg = GRPOTrainerConfig(
    n_iters=100,
    group_size=8,
    use_reference=True,
    adaptive_kl=True,
    target_kl=0.05,
)
# __post_init__ validates cross-field constraints
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HERMES_CACHE_DIR` | HuggingFace dataset cache directory |
| `HERMES_LOG_LEVEL` | Logging level (default: INFO) |
| `HERMES_DISABLE_TQDM` | Set to `1` to disable progress bars |
