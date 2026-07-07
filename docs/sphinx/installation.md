# Installation

## Prerequisites

- Python 3.11 or 3.12
- PyTorch (CPU or CUDA)

## Install

```bash
# Core package (training + eval)
pip install -e '.[rl,data,metrics]'

# With development tooling (ruff, mypy, pytest, etc.)
pip install -e '.[dev]'

# With documentation tooling (Sphinx, MyST)
pip install -e '.[docs]'

# Everything
pip install -e '.[rl,data,metrics,dev,docs,benchmark]'
```

## Optional Dependencies

| Extra | Description |
|-------|-------------|
| `rl` | RL training stack (torch, transformers) |
| `data` | Dataset loading (datasets, pyarrow) |
| `metrics` | Evaluation metrics (wandb, tensorboard) |
| `dev` | Development tooling (ruff, mypy, pytest, pytest-cov) |
| `docs` | Sphinx + MyST documentation |
| `benchmark` | pytest-benchmark for micro-benchmarks |

## Verify Installation

```python
import hermes_agentic_rl
print(hermes_agentic_rl.__version__)
```
