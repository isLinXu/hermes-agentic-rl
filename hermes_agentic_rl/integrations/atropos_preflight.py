from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AtroposPreflightResult:
    """Preflight result for local Atropos/Tinker-Atropos integration."""

    base_dir: Path
    atropos_dir: Path | None
    tinker_atropos_dir: Path | None
    python_ok: dict[str, bool]
    missing: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_dir": str(self.base_dir),
            "atropos_dir": str(self.atropos_dir) if self.atropos_dir else None,
            "tinker_atropos_dir": str(self.tinker_atropos_dir) if self.tinker_atropos_dir else None,
            "python_ok": dict(self.python_ok),
            "missing": list(self.missing),
        }


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _maybe_add_sys_path(path: Path) -> None:
    # Ensure local vendored repos can be imported without pip install.
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def run_atropos_preflight(base_dir: Path) -> AtroposPreflightResult:
    """Check whether local Atropos + tinker-atropos are present and importable.

    This does NOT start training. It only validates:
    - local repo directories exist
    - python can import atroposlib + tinker_atropos.config
    - key runtime deps for tinker-atropos trainer are available (tinker, wandb)
    """

    base_dir = base_dir.resolve()
    atropos_dir = (base_dir / "atropos") if (base_dir / "atropos").exists() else None
    tinker_atropos_dir = (
        (base_dir / "tinker-atropos") if (base_dir / "tinker-atropos").exists() else None
    )

    if atropos_dir is not None:
        _maybe_add_sys_path(atropos_dir)
    if tinker_atropos_dir is not None:
        _maybe_add_sys_path(tinker_atropos_dir)

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
    if atropos_dir is None:
        missing.append("local_dir:atropos")
    if tinker_atropos_dir is None:
        missing.append("local_dir:tinker-atropos")
    if not python_ok["atroposlib"]:
        missing.append("python:atroposlib")
    if not python_ok["tinker_atropos.config"]:
        missing.append("python:tinker_atropos")
    if not python_ok["tinker"]:
        missing.append("python:tinker (required for TinkerAtroposTrainer)")

    return AtroposPreflightResult(
        base_dir=base_dir,
        atropos_dir=atropos_dir.resolve() if atropos_dir else None,
        tinker_atropos_dir=tinker_atropos_dir.resolve() if tinker_atropos_dir else None,
        python_ok=python_ok,
        missing=missing,
    )

