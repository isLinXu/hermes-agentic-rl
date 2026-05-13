from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERMES_AGENT_REPO_ENV = "HERMES_AGENT_REPO"
_DEFAULT_RELATIVE_REPO_PATHS = (
    Path("subprojects/hermes-agent"),
    Path("hermes-agent"),
    Path("vendor/hermes-agent"),
)


@dataclass(frozen=True, slots=True)
class HermesRepoResolution:
    base_dir: Path
    repo_path: Path | None
    source: str | None
    checked_paths: tuple[Path, ...]

    @property
    def is_explicit(self) -> bool:
        return self.source in {"config", "env"}

    @property
    def is_present(self) -> bool:
        return bool(self.repo_path and self.repo_path.exists())

    @property
    def has_run_agent(self) -> bool:
        return bool(self.repo_path and (self.repo_path / "run_agent.py").exists())

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_dir": str(self.base_dir),
            "repo_path": str(self.repo_path) if self.repo_path else None,
            "source": self.source,
            "checked_paths": [str(path) for path in self.checked_paths],
            "is_explicit": self.is_explicit,
            "is_present": self.is_present,
            "has_run_agent": self.has_run_agent,
        }


def _coerce_path(raw_path: Any, base_dir: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_hermes_repo(
    runtime_cfg: dict[str, Any] | None = None,
    *,
    base_dir: Path | None = None,
) -> HermesRepoResolution:
    base_dir = (base_dir or Path.cwd()).resolve()
    runtime_cfg = dict(runtime_cfg or {})

    configured = runtime_cfg.get("repo_path")
    if configured:
        repo_path = _coerce_path(configured, base_dir)
        return HermesRepoResolution(
            base_dir=base_dir,
            repo_path=repo_path,
            source="config",
            checked_paths=(repo_path,),
        )

    env_repo = os.getenv(HERMES_AGENT_REPO_ENV)
    if env_repo:
        repo_path = _coerce_path(env_repo, base_dir)
        return HermesRepoResolution(
            base_dir=base_dir,
            repo_path=repo_path,
            source="env",
            checked_paths=(repo_path,),
        )

    checked_paths: list[Path] = []
    for relative in _DEFAULT_RELATIVE_REPO_PATHS:
        candidate = (base_dir / relative).resolve()
        checked_paths.append(candidate)
        if candidate.exists():
            return HermesRepoResolution(
                base_dir=base_dir,
                repo_path=candidate,
                source="subproject",
                checked_paths=tuple(checked_paths),
            )

    return HermesRepoResolution(
        base_dir=base_dir,
        repo_path=None,
        source=None,
        checked_paths=tuple(checked_paths),
    )


def prepare_hermes_imports(
    runtime_cfg: dict[str, Any] | None = None,
    *,
    base_dir: Path | None = None,
) -> HermesRepoResolution:
    resolution = resolve_hermes_repo(runtime_cfg, base_dir=base_dir)
    if resolution.repo_path and resolution.repo_path.is_dir():
        repo_path = str(resolution.repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    return resolution
