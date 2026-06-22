# hermes-agentic-rl Roadmap

> Last updated: 2026-06-21 · Current version: **0.11.0** · Target stable: **1.0.0**

This document defines what is considered **stable API** today, what may change
before v1.0, and the engineering milestones on the path to a production-grade
agentic-RL framework.

---

## Versioning policy (pre-1.0)

| Range | Meaning |
|---|---|
| **0.x.y** | Minor releases may add features or fix bugs; **patch** releases are backward compatible within the same minor. **Minor** releases may introduce deprecations with warnings. |
| **1.0.0** | Core API frozen (see below). Breaking changes require a major bump and a documented migration guide. |

Deprecation rule: any symbol marked deprecated in release **0.N** will be
removed no earlier than **0.N+2** (at least one minor release of warning).

---

## Stable API boundary (target for v1.0)

These surfaces are intended to remain stable after 1.0:

### Data contracts (`hermes_agentic_rl.core.types`)

- `Trajectory`, `RolloutStep`, `RewardResult`, `RewardSummary`, `TrainSample`

### Algorithm interface (`hermes_agentic_rl.algos.base`)

- `RolloutRecord`, `RolloutBatch`, `AlgoUpdateStats`
- `BaseAlgo.compute_loss(policy, ref, batch) → (loss, stats)`

### Backend protocol (`hermes_agentic_rl.backends.base`)

- `LLMBackend`: `generate`, `score`, `score_batch`, `trainable_parameters`

### Trainer skeleton (`hermes_agentic_rl.trainers.on_policy`)

- `OnPolicyTrainerConfig` field names and semantics
- Checkpoint/resume file layout (`model.pt`, `optim.pt`, `rng.pt`, `stats.json`)

### CLI entry points (`hermes_agentic_rl.cli.main`)

| Command | Exit-code semantics |
|---|---|
| `train-rl` | 0 success, 1 error, 2 config/runtime |
| `eval-gate` | 0 pass, **3 block promotion**, 4 verifier fail |
| `online-cycle` | 0 success, 1 error |

### Config schema (`hermes_agentic_rl.config` + `config_models`)

- Top-level keys: `runtime`, `environment`, `trainer`, `reward`, `backend`, `train_rl`
- `runtime.integration` enum
- Pydantic models in `config_models.py` (optional `[config]` extra)

---

## Explicitly *not* stable (may change before 1.0)

- Internal trainer helpers (`trainers/_rollout_helpers.py`, private methods on `OnPolicyTrainer`)
- Hermes runtime adapter hacks (`runtime/hermes_adapter.py`)
- Atropos integration shims (`integrations/atropos_env_import.py`)
- Deprecated aliases: `AtroposGrpoTrainer` → use `AtroposJsonlExporter` or `GRPOTrainer`
- YAML keys under `train_rl` that are not documented in `docs/configuration.md`

---

## Phase 1 — Engineering hardening (v0.12 – v0.15)

| Item | Status | Notes |
|---|---|---|
| Mypy 0 errors on `hermes_agentic_rl/` | ✅ Done | 177 modules, CI gate |
| Pydantic config validation | ✅ Done | `[config]` extra, dict fallback |
| Algorithm micro-benchmarks | ✅ Done | `tests/benchmarks/`, CI step |
| CI benchmark job | ✅ Done | `--benchmark-disable` in test matrix |
| Mock-backend smoke tests | ✅ Done | `tests/test_mock_backend_core.py` |
| API stability doc (this file) | ✅ Done | |
| Progressive mypy strict | 🔄 In progress | `mypy-strict.ini` on 3 reward modules |
| Sphinx API docs refresh | 📋 Planned | auto-doc remaining public modules |
| Docker multi-stage + compose | ✅ Done | `Dockerfile`, `docker-compose.yml` |

## Phase 2 — Capability expansion (v0.16 – v0.20)

- Tensor / pipeline / expert parallel for large models
- Quantized rollout backends (GPTQ / AWQ / GGUF)
- Fault-tolerant distributed pool auto-recovery
- Optuna / Ray Tune hyperparameter search
- Multimodal reward components

## Phase 3 — Ecosystem (v1.0+)

- v1.0 release with frozen core API
- Kubernetes operator + Helm chart
- Community tutorials and case studies
- Optional enterprise support track

---

## How to propose a breaking change

1. Open an issue tagged `api-breaking` describing migration path.
2. Add a `DeprecationWarning` in the current minor release.
3. Update this roadmap and `CHANGELOG.md`.
4. Remove deprecated code only after the grace period above.

---

## Related documents

- [Configuration reference](configuration.md)
- [Architecture overview](architecture.md)
- [Engineering workflow (Sphinx)](sphinx/engineering.md)
- [CHANGELOG](../CHANGELOG.md)
