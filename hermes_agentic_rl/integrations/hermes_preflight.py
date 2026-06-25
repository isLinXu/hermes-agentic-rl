from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_agentic_rl.integrations.hermes_repo import prepare_hermes_imports


@dataclass(frozen=True, slots=True)
class HermesPreflightResult:
    base_dir: Path
    repo_path: Path | None
    repo_source: str | None
    python_ok: dict[str, bool]
    missing: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_dir": str(self.base_dir),
            "repo_path": str(self.repo_path) if self.repo_path else None,
            "repo_source": self.repo_source,
            "python_ok": dict(self.python_ok),
            "missing": list(self.missing),
        }


@dataclass(frozen=True, slots=True)
class ModuleProbeResult:
    name: str
    exists: bool
    error_type: str | None = None


def _probe_module(name: str) -> ModuleProbeResult:
    try:
        return ModuleProbeResult(name=name, exists=importlib.util.find_spec(name) is not None)
    except ValueError:
        return ModuleProbeResult(name=name, exists=True)
    except Exception as exc:
        return ModuleProbeResult(name=name, exists=False, error_type=type(exc).__name__)


def run_hermes_preflight(base_dir: Path) -> HermesPreflightResult:
    base_dir = base_dir.resolve()
    resolution = prepare_hermes_imports(base_dir=base_dir)
    probes = {
        "run_agent": _probe_module("run_agent"),
        "environments.agent_loop": _probe_module("environments.agent_loop"),
        "tools": _probe_module("tools"),
        "wandb": _probe_module("wandb"),
        "torch": _probe_module("torch"),
    }

    python_ok = {
        name: probe.exists for name, probe in probes.items()
    }

    missing: list[str] = []
    if resolution.repo_path is None:
        missing.append("local_dir:subprojects/hermes-agent")
    elif not resolution.is_present:
        missing.append(f"local_dir_missing:{resolution.repo_path}")
    elif not resolution.has_run_agent:
        missing.append(f"missing_run_agent:{resolution.repo_path}")

    for module_name in ("run_agent", "environments.agent_loop"):
        probe = probes[module_name]
        if probe.exists:
            continue
        if probe.error_type:
            missing.append(f"python_probe_error:{module_name}:{probe.error_type}")
            continue
        missing.append(f"python:{module_name}")

    return HermesPreflightResult(
        base_dir=base_dir,
        repo_path=resolution.repo_path,
        repo_source=resolution.source,
        python_ok=python_ok,
        missing=missing,
    )
