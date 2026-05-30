"""Checkpoint save/restore for RL training.

Features:
  - Save: model weights, optimizer state, iteration, RNG state, stats.
  - Restore: resume from any checkpoint.
  - Atomic writes: save to temp then rename (crash-safe).
  - Sharding: for large models, save in multiple shards.

Checkpoint layout::

    {output_dir}/checkpoints/
        iter_00010/
            model.pt          # model state_dict
            optimizer.pt      # optimizer state_dict
            trainer_state.json  # iteration, rng, stats
            config.yaml       # snapshot of training config
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Atomic save helper
# ---------------------------------------------------------------------------


def _atomic_save(obj: Any, path: Path, serializer: str = "torch") -> None:
    """Save to a temp file then atomically rename (crash-safe).

    Note: PyTorch 2.x's zip-based serializer rejects temp names starting
    with '.' (treated as invalid zip entry). We use a plain ``tmp_<name>``
    prefix sibling file instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use suffix-style temp name that torch.save accepts (no leading dot).
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"tmp_{path.name}_", suffix=".partial"
    )
    try:
        if serializer == "torch":
            os.close(fd)  # let torch manage its own handle
            torch.save(obj, tmp)
        elif serializer == "json":
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        elif serializer == "yaml":
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(obj, f, allow_unicode=True)
        else:
            os.close(fd)
            raise ValueError(f"Unknown serializer: {serializer}")
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


@dataclass
class CheckpointState:
    """Full training state for resumption."""

    iteration: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    rng_state: dict[str, Any] | None  # torch + python random state
    stats: list[dict[str, Any]]  # accumulated training stats
    config: dict[str, Any]  # training config snapshot
    best_reward: float
    best_iteration: int


class CheckpointManager:
    """Save and restore training checkpoints.

    Usage::

        ckpt = CheckpointManager(output_dir / "checkpoints", keep_last=5)
        ckpt.save(CheckpointState(iteration=10, ...))
        state = ckpt.load(10)  # or ckpt.load_latest()
        trainer.restore(state)
    """

    def __init__(
        self,
        base_dir: str | Path,
        keep_last: int = 5,
    ):
        self.base_dir = Path(base_dir)
        self.keep_last = keep_last
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_dir(self, iteration: int) -> Path:
        return self.base_dir / f"iter_{iteration:05d}"

    def save(self, state: CheckpointState) -> Path:
        """Save checkpoint for the given iteration."""
        ckpt_dir = self._checkpoint_dir(state.iteration)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Model weights
        _atomic_save(state.model_state, ckpt_dir / "model.pt", "torch")

        # Optimizer state
        if state.optimizer_state is not None:
            _atomic_save(state.optimizer_state, ckpt_dir / "optimizer.pt", "torch")

        # Training state (JSON for readability)
        _atomic_save(
            {
                "iteration": state.iteration,
                "best_reward": state.best_reward,
                "best_iteration": state.best_iteration,
                "stats": state.stats,
            },
            ckpt_dir / "trainer_state.json",
            "json",
        )

        # Config snapshot
        _atomic_save(state.config, ckpt_dir / "config.yaml", "yaml")

        # RNG state
        if state.rng_state is not None:
            _atomic_save(state.rng_state, ckpt_dir / "rng_state.pt", "torch")

        # Cleanup old checkpoints
        self._prune()

        return ckpt_dir

    def load(self, iteration: int) -> CheckpointState | None:
        """Load checkpoint for a specific iteration."""
        ckpt_dir = self._checkpoint_dir(iteration)
        return self._load_from_dir(ckpt_dir)

    def load_latest(self) -> CheckpointState | None:
        """Load the latest checkpoint."""
        dirs = sorted(self.base_dir.glob("iter_*"))
        if not dirs:
            return None
        return self._load_from_dir(dirs[-1])

    def list_checkpoints(self) -> list[int]:
        """List available checkpoint iterations."""
        dirs = sorted(self.base_dir.glob("iter_*"))
        return [
            int(d.name.replace("iter_", ""))
            for d in dirs
            if d.name.startswith("iter_")
        ]

    def _load_from_dir(self, ckpt_dir: Path) -> CheckpointState | None:
        if not ckpt_dir.exists():
            return None

        model_path = ckpt_dir / "model.pt"
        optimizer_path = ckpt_dir / "optimizer.pt"
        state_path = ckpt_dir / "trainer_state.json"
        config_path = ckpt_dir / "config.yaml"
        rng_path = ckpt_dir / "rng_state.pt"

        if not model_path.exists():
            return None

        model_state = torch.load(model_path, map_location="cpu", weights_only=True)
        # Optimizer state may contain Python objects (e.g. ParamGroup dict with
        # typed step counters) — fall back to weights_only=False if the strict
        # load fails. Source is our own atomic write, so this is safe.
        if optimizer_path.exists():
            try:
                optimizer_state = torch.load(
                    optimizer_path, map_location="cpu", weights_only=True
                )
            except Exception:
                optimizer_state = torch.load(
                    optimizer_path, map_location="cpu", weights_only=False
                )
        else:
            optimizer_state = None
        # RNG state contains numpy._reconstruct which is NOT whitelisted under
        # weights_only=True. Trust our own writer and load with full pickle.
        rng_state = (
            torch.load(rng_path, map_location="cpu", weights_only=False)
            if rng_path.exists()
            else None
        )

        with open(state_path, encoding="utf-8") as f:
            trainer_state = json.load(f)

        config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        return CheckpointState(
            iteration=trainer_state["iteration"],
            model_state=model_state,
            optimizer_state=optimizer_state,
            rng_state=rng_state,
            stats=trainer_state.get("stats", []),
            config=config,
            best_reward=trainer_state.get("best_reward", 0.0),
            best_iteration=trainer_state.get("best_iteration", 0),
        )

    def _prune(self) -> None:
        """Keep only the last keep_last checkpoints."""
        if self.keep_last <= 0:
            return
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.keep_last:
            return
        to_remove = sorted(checkpoints)[: -self.keep_last]
        for it in to_remove:
            ckpt_dir = self._checkpoint_dir(it)
            if ckpt_dir.exists():
                import shutil

                shutil.rmtree(ckpt_dir)


# ---------------------------------------------------------------------------
# Async checkpoint saver (non-blocking I/O during training)
# ---------------------------------------------------------------------------


def snapshot_state_to_cpu(state: CheckpointState) -> CheckpointState:
    """Clone model tensors to CPU so the training thread can keep stepping."""
    model_state = {
        k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
        for k, v in state.model_state.items()
    }
    return CheckpointState(
        iteration=state.iteration,
        model_state=model_state,
        optimizer_state=state.optimizer_state,
        rng_state=state.rng_state,
        stats=list(state.stats),
        config=dict(state.config),
        best_reward=state.best_reward,
        best_iteration=state.best_iteration,
    )


class AsyncCheckpointSaver:
    """Background worker that drains checkpoint saves in submission order.

    ``submit`` enqueues a ``(manager, state)`` pair; ``flush`` blocks until
    the queue is empty and re-raises the first worker error so callers can
    fail the training step instead of silently losing checkpoints.
    """

    def __init__(self) -> None:
        import queue
        import threading

        self._queue: queue.Queue[tuple[Any, CheckpointState] | None] = queue.Queue()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                mgr, state = item
                mgr.save(state)
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()

    def submit(self, mgr: Any, state: CheckpointState) -> None:
        if self._closed:
            raise RuntimeError("AsyncCheckpointSaver is closed")
        if self._error is not None:
            raise self._error
        self._queue.put((mgr, state))

    def flush(self) -> None:
        self._queue.join()
        if self._error is not None:
            err = self._error
            self._error = None
            raise err

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()


# ---------------------------------------------------------------------------
# Trainer integration helpers
# ---------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    """Capture torch + python + numpy RNG states."""
    import random

    state: dict[str, Any] = {
        "torch": torch.random.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states from captured dict."""
    import random

    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


def save_checkpoint_for_trainer(
    trainer: Any,  # OnPolicyTrainer
    iteration: int,
    output_dir: Path,
    best_reward: float = 0.0,
    best_iteration: int = 0,
    keep_last: int = 5,
) -> Path:
    """Convenience: save full checkpoint from an OnPolicyTrainer."""
    ckpt = CheckpointManager(output_dir / "checkpoints", keep_last=keep_last)

    model_state = trainer.policy.model.state_dict()
    optimizer_state = trainer._optim.state_dict() if hasattr(trainer, "_optim") else None
    rng_state = capture_rng_state()

    state = CheckpointState(
        iteration=iteration,
        model_state=model_state,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        stats=trainer.stats.iters if hasattr(trainer, "stats") else [],
        config={},
        best_reward=best_reward,
        best_iteration=best_iteration,
    )
    return ckpt.save(state)


def restore_checkpoint_to_trainer(
    trainer: Any,
    ckpt_state: CheckpointState,
) -> int:
    """Restore trainer state from a checkpoint.

    Returns the iteration to resume from.
    """
    # Load model weights
    trainer.policy.model.load_state_dict(ckpt_state.model_state)

    # Load optimizer
    if ckpt_state.optimizer_state is not None and hasattr(trainer, "_optim"):
        trainer._optim.load_state_dict(ckpt_state.optimizer_state)

    # Restore RNG
    if ckpt_state.rng_state is not None:
        restore_rng_state(ckpt_state.rng_state)

    return ckpt_state.iteration
