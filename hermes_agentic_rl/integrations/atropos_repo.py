from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ATROPOS_REPO_ENV = "ATROPOS_REPO"
TINKER_ATROPOS_REPO_ENV = "TINKER_ATROPOS_REPO"

_ATROPOS_MARKERS = (Path("atroposlib"), Path("environments"))
_TINKER_ATROPOS_MARKERS = (Path("tinker_atropos"),)

_DEFAULT_ATROPOS_PATHS = (
    Path("subprojects/atropos"),
    Path("subprojects/hermes-agent/atropos"),
    Path("atropos"),
    Path("vendor/atropos"),
)
_DEFAULT_TINKER_ATROPOS_PATHS = (
    Path("subprojects/tinker-atropos"),
    Path("subprojects/hermes-agent/tinker-atropos"),
    Path("tinker-atropos"),
    Path("vendor/tinker-atropos"),
)


@dataclass(frozen=True, slots=True)
class ExternalRepoResolution:
    name: str
    env_var: str
    base_dir: Path
    repo_path: Path | None
    source: str | None
    checked_paths: tuple[Path, ...]
    marker_paths: tuple[Path, ...]

    @property
    def is_explicit(self) -> bool:
        return self.source in {"config", "env"}

    @property
    def is_present(self) -> bool:
        return bool(self.repo_path and self.repo_path.exists())

    @property
    def has_marker(self) -> bool:
        return any(path.exists() for path in self.marker_paths)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "env_var": self.env_var,
            "base_dir": str(self.base_dir),
            "repo_path": str(self.repo_path) if self.repo_path else None,
            "source": self.source,
            "checked_paths": [str(path) for path in self.checked_paths],
            "marker_paths": [str(path) for path in self.marker_paths],
            "is_explicit": self.is_explicit,
            "is_present": self.is_present,
            "has_marker": self.has_marker,
        }


def _coerce_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _marker_paths(repo_path: Path, markers: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(repo_path / marker for marker in markers)


def _has_markers(repo_path: Path, markers: tuple[Path, ...]) -> bool:
    return any(path.exists() for path in _marker_paths(repo_path, markers))


def _source_for_default_path(relative_path: Path) -> str:
    first_part = relative_path.parts[0] if relative_path.parts else ""
    if first_part == "subprojects":
        return "subproject"
    if first_part == "vendor":
        return "vendor"
    return "local"


def _resolve_external_repo(
    *,
    name: str,
    env_var: str,
    default_paths: tuple[Path, ...],
    markers: tuple[Path, ...],
    repo_path: str | Path | None = None,
    base_dir: Path | None = None,
) -> ExternalRepoResolution:
    root = (base_dir or Path.cwd()).resolve()

    if repo_path:
        candidate = _coerce_path(repo_path, root)
        return ExternalRepoResolution(
            name=name,
            env_var=env_var,
            base_dir=root,
            repo_path=candidate,
            source="config",
            checked_paths=(candidate,),
            marker_paths=_marker_paths(candidate, markers),
        )

    env_repo = os.getenv(env_var)
    if env_repo:
        candidate = _coerce_path(env_repo, root)
        return ExternalRepoResolution(
            name=name,
            env_var=env_var,
            base_dir=root,
            repo_path=candidate,
            source="env",
            checked_paths=(candidate,),
            marker_paths=_marker_paths(candidate, markers),
        )

    checked_paths: list[Path] = []
    for relative in default_paths:
        candidate = (root / relative).resolve()
        checked_paths.append(candidate)
        if candidate.exists() and _has_markers(candidate, markers):
            return ExternalRepoResolution(
                name=name,
                env_var=env_var,
                base_dir=root,
                repo_path=candidate,
                source=_source_for_default_path(relative),
                checked_paths=tuple(checked_paths),
                marker_paths=_marker_paths(candidate, markers),
            )

    return ExternalRepoResolution(
        name=name,
        env_var=env_var,
        base_dir=root,
        repo_path=None,
        source=None,
        checked_paths=tuple(checked_paths),
        marker_paths=(),
    )


def resolve_atropos_repo(
    repo_path: str | Path | None = None,
    *,
    base_dir: Path | None = None,
) -> ExternalRepoResolution:
    return _resolve_external_repo(
        name="atropos",
        env_var=ATROPOS_REPO_ENV,
        default_paths=_DEFAULT_ATROPOS_PATHS,
        markers=_ATROPOS_MARKERS,
        repo_path=repo_path,
        base_dir=base_dir,
    )


def resolve_tinker_atropos_repo(
    repo_path: str | Path | None = None,
    *,
    base_dir: Path | None = None,
) -> ExternalRepoResolution:
    return _resolve_external_repo(
        name="tinker-atropos",
        env_var=TINKER_ATROPOS_REPO_ENV,
        default_paths=_DEFAULT_TINKER_ATROPOS_PATHS,
        markers=_TINKER_ATROPOS_MARKERS,
        repo_path=repo_path,
        base_dir=base_dir,
    )


def _prepend_import_path(path: Path | None) -> None:
    if not path or not path.is_dir():
        return
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def prepare_atropos_imports(
    repo_path: str | Path | None = None,
    *,
    base_dir: Path | None = None,
) -> ExternalRepoResolution:
    resolution = resolve_atropos_repo(repo_path, base_dir=base_dir)
    _prepend_import_path(resolution.repo_path)
    return resolution


def prepare_tinker_atropos_imports(
    repo_path: str | Path | None = None,
    *,
    base_dir: Path | None = None,
) -> ExternalRepoResolution:
    resolution = resolve_tinker_atropos_repo(repo_path, base_dir=base_dir)
    _prepend_import_path(resolution.repo_path)
    return resolution
