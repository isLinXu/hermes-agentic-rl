"""Integrations with external RL / environment ecosystems.

Currently supported:
  - atropos_env_import: wrap any `atroposlib.envs.base.BaseEnv` as a
    hermes `envs.BaseEnv` + provide an in-process fake APIServer so the
    atropos env's ``self.server.chat_completion(...)`` calls are served
    by a hermes ``LLMBackend`` (Tiny or HF), with zero HTTP / zero vLLM.

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

__all__ = [
    "AtroposEnvAdapter",
    "AtroposEnvAdapterConfig",
    "AtroposRewardComponent",
    "AtroposUnavailableError",
    "HermesAPIServer",
]
