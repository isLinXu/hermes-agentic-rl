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


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return True


def run_hermes_preflight(base_dir: Path) -> HermesPreflightResult:
    base_dir = base_dir.resolve()
    resolution = prepare_hermes_imports(base_dir=base_dir)

    python_ok = {
        "run_agent": _module_exists("run_agent"),
        "environments.agent_loop": _module_exists("environments.agent_loop"),
        "tools": _module_exists("tools"),
        "wandb": _module_exists("wandb"),
        "torch": _module_exists("torch"),
    }

    missing: list[str] = []
    if resolution.repo_path is None:
        missing.append("local_dir:subprojects/hermes-agent")
    elif not resolution.is_present:
        missing.append(f"local_dir_missing:{resolution.repo_path}")
    elif not resolution.has_run_agent:
        missing.append(f"missing_run_agent:{resolution.repo_path}")

    if not python_ok["run_agent"]:
        missing.append("python:run_agent")
    if not python_ok["environments.agent_loop"]:
        missing.append("python:environments.agent_loop")

    return HermesPreflightResult(
        base_dir=base_dir,
        repo_path=resolution.repo_path,
        repo_source=resolution.source,
        python_ok=python_ok,
        missing=missing,
    )
