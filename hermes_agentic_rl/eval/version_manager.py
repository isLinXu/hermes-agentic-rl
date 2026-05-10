"""Policy version manager.

Keeps a directory of named checkpoints (``state_dict`` files) with metadata
and a lightweight tag system (``production`` / ``canary`` / free-form).

Layout::

    {root}/
      index.json               # list of VersionInfo
      v1_init/policy.pt
      v2_ppo/policy.pt
      ...

All methods are pure-stdlib + torch.save/torch.load.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VersionInfo:
    name: str
    path: str
    created_at: float
    algo: str | None = None
    parent: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class VersionManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.versions: list[VersionInfo] = []
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            for d in data:
                self.versions.append(VersionInfo(**d))

    def _write_index(self) -> None:
        self.index_path.write_text(
            json.dumps([asdict(v) for v in self.versions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_version(
        self,
        name: str,
        state_dict: dict[str, Any],
        *,
        algo: str | None = None,
        parent: str | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> VersionInfo:
        import torch

        if any(v.name == name for v in self.versions):
            raise ValueError(f"version '{name}' already exists")
        subdir = self.root / name
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / "policy.pt"
        torch.save(state_dict, path)
        info = VersionInfo(
            name=name,
            path=str(path),
            created_at=time.time(),
            algo=algo,
            parent=parent,
            metrics=dict(metrics or {}),
            tags=list(tags or []),
        )
        self.versions.append(info)
        self._write_index()
        return info

    def load_version(self, name: str) -> dict[str, Any]:
        import torch

        info = self.get(name)
        return torch.load(info.path, map_location="cpu")  # noqa: torch-load-unsafe

    def get(self, name: str) -> VersionInfo:
        for v in self.versions:
            if v.name == name:
                return v
        raise KeyError(f"version '{name}' not found")

    def list_versions(self) -> list[VersionInfo]:
        return list(self.versions)

    def set_tag(self, name: str, tag: str, *, exclusive: bool = False) -> None:
        """Attach ``tag`` to ``name``. If ``exclusive``, remove it from all others first."""
        info = self.get(name)
        if exclusive:
            for v in self.versions:
                if v.name != name and tag in v.tags:
                    v.tags.remove(tag)
        if tag not in info.tags:
            info.tags.append(tag)
        self._write_index()

    def unset_tag(self, name: str, tag: str) -> None:
        info = self.get(name)
        if tag in info.tags:
            info.tags.remove(tag)
        self._write_index()

    def find_by_tag(self, tag: str) -> list[VersionInfo]:
        return [v for v in self.versions if tag in v.tags]

    def promote(self, name: str) -> None:
        """Shorthand: set tag 'production' exclusively on ``name``."""
        self.set_tag(name, "production", exclusive=True)
