"""Integrations with external RL / environment ecosystems.

Currently supported:
  - atropos_env_import: wrap any `atroposlib.envs.base.BaseEnv` as a
    hermes `envs.BaseEnv` + provide an in-process fake APIServer so the
    atropos env's ``self.server.chat_completion(...)`` calls are served
    by a hermes ``LLMBackend`` (Tiny or HF), with zero HTTP / zero vLLM.
  - hermes_repo: local-source discovery for the upstream hermes-agent repo,
    including subproject, environment-variable, and explicit config lookup.
  - atropos_repo: local-source discovery for Atropos and Tinker-Atropos,
    including subproject, local, vendor, and environment-variable lookup.

Everything in this package is **optional**: importing a submodule fails
gracefully if the external dependency is missing.
"""

from hermes_agentic_rl.integrations.atropos_env_import import (
    AtroposEnvAdapter,
    AtroposEnvAdapterConfig,
    AtroposRewardComponent,
    AtroposUnavailableError,
    HermesAPIServer,
)
from hermes_agentic_rl.integrations.atropos_repo import (
    ATROPOS_REPO_ENV,
    TINKER_ATROPOS_REPO_ENV,
    ExternalRepoResolution,
    prepare_atropos_imports,
    prepare_tinker_atropos_imports,
    resolve_atropos_repo,
    resolve_tinker_atropos_repo,
)
from hermes_agentic_rl.integrations.hermes_preflight import (
    HermesPreflightResult,
    run_hermes_preflight,
)
from hermes_agentic_rl.integrations.hermes_repo import (
    HERMES_AGENT_REPO_ENV,
    HermesRepoResolution,
    prepare_hermes_imports,
    resolve_hermes_repo,
)

__all__ = [
    "ATROPOS_REPO_ENV",
    "HERMES_AGENT_REPO_ENV",
    "TINKER_ATROPOS_REPO_ENV",
    "AtroposEnvAdapter",
    "AtroposEnvAdapterConfig",
    "AtroposRewardComponent",
    "AtroposUnavailableError",
    "ExternalRepoResolution",
    "HermesAPIServer",
    "HermesPreflightResult",
    "HermesRepoResolution",
    "prepare_atropos_imports",
    "prepare_hermes_imports",
    "prepare_tinker_atropos_imports",
    "resolve_atropos_repo",
    "resolve_hermes_repo",
    "resolve_tinker_atropos_repo",
    "run_hermes_preflight",
]
