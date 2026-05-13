from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_agentic_rl.integrations.atropos_repo import (
    ExternalRepoResolution,
    prepare_atropos_imports,
    prepare_tinker_atropos_imports,
)


@dataclass(frozen=True, slots=True)
class AtroposPreflightResult:
    """Preflight result for local Atropos/Tinker-Atropos integration."""

    base_dir: Path
    atropos: ExternalRepoResolution
    tinker_atropos: ExternalRepoResolution
    python_ok: dict[str, bool]
    missing: list[str]

    @property
    def atropos_dir(self) -> Path | None:
        return self.atropos.repo_path if self.atropos.has_marker else None

    @property
    def tinker_atropos_dir(self) -> Path | None:
        return self.tinker_atropos.repo_path if self.tinker_atropos.has_marker else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_dir": str(self.base_dir),
            "atropos_dir": str(self.atropos_dir) if self.atropos_dir else None,
            "tinker_atropos_dir": str(self.tinker_atropos_dir) if self.tinker_atropos_dir else None,
            "atropos_source": self.atropos.source,
            "tinker_atropos_source": self.tinker_atropos.source,
            "checked_paths": {
                "atropos": [str(path) for path in self.atropos.checked_paths],
                "tinker_atropos": [
                    str(path) for path in self.tinker_atropos.checked_paths
                ],
            },
            "python_ok": dict(self.python_ok),
            "missing": list(self.missing),
        }


def _module_exists(name: str) -> bool:
    importlib.invalidate_caches()
    return importlib.util.find_spec(name) is not None


def run_atropos_preflight(base_dir: Path) -> AtroposPreflightResult:
    """Check whether local Atropos + tinker-atropos are present and importable.

    This does NOT start training. It only validates:
    - local repo directories exist
    - python can import atroposlib + tinker_atropos.config
    - key runtime deps for tinker-atropos trainer are available (tinker, wandb)
    """

    base_dir = base_dir.resolve()
    atropos = prepare_atropos_imports(base_dir=base_dir)
    tinker_atropos = prepare_tinker_atropos_imports(base_dir=base_dir)

    python_ok = {
        "atroposlib": _module_exists("atroposlib"),
        "tinker_atropos.config": _module_exists("tinker_atropos.config"),
        # Required by tinker-atropos trainer:
        "tinker": _module_exists("tinker"),
        # Optional but commonly used:
        "wandb": _module_exists("wandb"),
        "torch": _module_exists("torch"),
        "transformers": _module_exists("transformers"),
    }

    missing: list[str] = []
    if not atropos.has_marker:
        missing.append("local_dir:atropos")
    if not tinker_atropos.has_marker:
        missing.append("local_dir:tinker-atropos")
    if not python_ok["atroposlib"]:
        missing.append("python:atroposlib")
    if not python_ok["tinker_atropos.config"]:
        missing.append("python:tinker_atropos")
    if not python_ok["tinker"]:
        missing.append("python:tinker (required for TinkerAtroposTrainer)")

    return AtroposPreflightResult(
        base_dir=base_dir,
        atropos=atropos,
        tinker_atropos=tinker_atropos,
        python_ok=python_ok,
        missing=missing,
    )
