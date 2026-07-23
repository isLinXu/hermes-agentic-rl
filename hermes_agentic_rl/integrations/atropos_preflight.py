from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ATROPOS_RELATIVE_PATHS = (
    Path("subprojects/atropos"),
    Path("atropos"),
)
_TINKER_ATROPOS_RELATIVE_PATHS = (
    Path("subprojects/tinker-atropos"),
    Path("tinker-atropos"),
)


@dataclass(frozen=True, slots=True)
class AtroposMissingDetail:
    """Structured detail for a single preflight missing item."""

    kind: str
    repo: str | None = None
    path: Path | None = None
    module: str | None = None
    required: bool | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.repo is not None:
            payload["repo"] = self.repo
        if self.path is not None:
            payload["path"] = str(self.path)
        if self.module is not None:
            payload["module"] = self.module
        if self.required is not None:
            payload["required"] = self.required
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class AtroposPreflightResult:
    """Preflight result for local Atropos/Tinker-Atropos integration."""

    base_dir: Path
    atropos_dir: Path | None
    tinker_atropos_dir: Path | None
    python_ok: dict[str, bool]
    missing: list[str]
    missing_details: list[AtroposMissingDetail]

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_dir": str(self.base_dir),
            "atropos_dir": str(self.atropos_dir) if self.atropos_dir else None,
            "tinker_atropos_dir": str(self.tinker_atropos_dir) if self.tinker_atropos_dir else None,
            "python_ok": dict(self.python_ok),
            "missing": list(self.missing),
            "missing_details": [detail.as_dict() for detail in self.missing_details],
        }


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return True
    except ModuleNotFoundError:
        return False
    except Exception:
        return False


def _maybe_add_sys_path(path: Path) -> None:
    # Ensure local subprojects can be imported without pip install.
    p = str(path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _first_existing(base_dir: Path, candidates: tuple[Path, ...]) -> Path | None:
    for relative in candidates:
        candidate = (base_dir / relative).resolve()
        if candidate.exists():
            return candidate
    return None


def run_atropos_preflight(base_dir: Path) -> AtroposPreflightResult:
    """Check whether local Atropos + tinker-atropos are present and importable.

    This does NOT start training. It only validates:
    - local repo directories exist
    - python can import atroposlib + tinker_atropos.config
    - key runtime deps for tinker-atropos trainer are available (tinker, wandb)
    """

    base_dir = base_dir.resolve()
    atropos_dir = _first_existing(base_dir, _ATROPOS_RELATIVE_PATHS)
    tinker_atropos_dir = _first_existing(base_dir, _TINKER_ATROPOS_RELATIVE_PATHS)

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
    missing_details: list[AtroposMissingDetail] = []
    if atropos_dir is None:
        missing.append("local_dir:subprojects/atropos")
        missing_details.append(
            AtroposMissingDetail(
                kind="local_dir_missing",
                repo="atropos",
                path=(base_dir / "subprojects" / "atropos").resolve(),
            )
        )
    if tinker_atropos_dir is None:
        missing.append("local_dir:subprojects/tinker-atropos")
        missing_details.append(
            AtroposMissingDetail(
                kind="local_dir_missing",
                repo="tinker-atropos",
                path=(base_dir / "subprojects" / "tinker-atropos").resolve(),
            )
        )
    if not python_ok["atroposlib"]:
        missing.append("python:atroposlib")
        missing_details.append(
            AtroposMissingDetail(
                kind="python_module_missing",
                module="atroposlib",
                required=True,
                reason="required for local Atropos integration",
            )
        )
    if not python_ok["tinker_atropos.config"]:
        missing.append("python:tinker_atropos.config")
        missing_details.append(
            AtroposMissingDetail(
                kind="python_module_missing",
                module="tinker_atropos.config",
                required=True,
                reason="required for local tinker-atropos config import",
            )
        )
    if not python_ok["tinker"]:
        missing.append("python:tinker")
        missing_details.append(
            AtroposMissingDetail(
                kind="python_module_missing",
                module="tinker",
                required=True,
                reason="required for TinkerAtroposTrainer",
            )
        )

    return AtroposPreflightResult(
        base_dir=base_dir,
        atropos_dir=atropos_dir,
        tinker_atropos_dir=tinker_atropos_dir,
        python_ok=python_ok,
        missing=missing,
        missing_details=missing_details,
    )
