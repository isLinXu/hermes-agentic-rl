"""Integrations with external RL / environment ecosystems.

Currently supported:
  - atropos_env_import: wrap any `atroposlib.envs.base.BaseEnv` as a
    hermes `envs.BaseEnv` + provide an in-process fake APIServer so the
    atropos env's ``self.server.chat_completion(...)`` calls are served
    by a hermes ``LLMBackend`` (Tiny or HF), with zero HTTP / zero vLLM.
  - hermes_repo: local-source discovery for the upstream hermes-agent repo,
    including subproject, environment-variable, and explicit config lookup.

Everything in this package is **optional**: importing a submodule fails
gracefully if the external dependency is missing.
"""

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
    "HERMES_AGENT_REPO_ENV",
    "HermesPreflightResult",
    "HermesRepoResolution",
    "prepare_hermes_imports",
    "resolve_hermes_repo",
    "run_hermes_preflight",
]

try:
    from hermes_agentic_rl.integrations.atropos_env_import import (
        AtroposEnvAdapter,
        AtroposEnvAdapterConfig,
        AtroposRewardComponent,
        AtroposUnavailableError,
        HermesAPIServer,
    )

    __all__.extend(
        [
            "AtroposEnvAdapter",
            "AtroposEnvAdapterConfig",
            "AtroposRewardComponent",
            "AtroposUnavailableError",
            "HermesAPIServer",
        ]
    )
except Exception:
    pass
