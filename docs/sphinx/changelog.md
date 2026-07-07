# Changelog

## v0.12.0 (engineering hardening)

### Type Safety
- Fixed all 22 mypy errors across the codebase (0 errors, 191 files)
- Enhanced `ruler.py` type annotations with `TemplateFactory` alias
- Fixed `quantized.py` override signatures to match base `LLMBackend`
- Fixed `yaml_config.py` reward_manager union type handling

### CI/CD
- Added dedicated `benchmark` CI job with real timing (not `--benchmark-disable`)
- Updated mypy CI command to use `--ignore-missing-imports`
- Added `feat/**` branches to CI trigger

### Configuration Validation
- Added Pydantic v2 config validation layer (`config_validation.py`)
- Schema covers: backend, training, rewards, RULER, curriculum,
  staleness-TIS, LoRA hot-reload, quantization, client-server
- Fallback validator when pydantic is unavailable
- 30 test cases covering valid/invalid configs

### Documentation
- Expanded Sphinx docs with 7 API reference pages (core, trainers, rewards,
  envs, distributed, backends, peft, tools)
- Added installation, quickstart, configuration, and architecture guides
- Enhanced `conf.py` with MyST extensions, intersphinx, furo theme support
- Updated `engineering.md` to reflect current toolchain

### Modules (from prior commits)
- `peft/lora_hot_reload.py` — LoRA delta merge + vLLM sync
- `client_server/` — decoupled rollout architecture
- `envs/mcp_tool_env.py` — MCP tool-calling environment
- `algos/common/staleness_adaptive_tis.py` — dynamic rho_clip
- `backends/quantized.py` — GPTQ/AWQ/GGUF inference
- `tuning/hparam_search.py` — Optuna TPE/Bayesian search
- `benchmarks/perf_suite.py` — 7 benchmark scenarios
- `distributed/fault_tolerant_pool.py` — elastic scaling
- `rewards/multimodal.py` — CLIP-based image-text similarity
- `distributed/model_parallel.py` — TP/PP/EP interface

## v0.11.0

- OpenPipe/ART architecture comparison report
- Fable5TraceEnv adapter
- GRPOTrainerConfig @dataclass(slots=True) with 83 fields
- TrainingOrchestrator extraction
- RULER auto-reward integration
- YAML configuration entry point
- `_compat.py` API freeze markers
- Example scripts: `minimal_grpo.py`, `sft_plus_rl.py`
