"""Client-Server API layer — decouples training (server) from rollout (client).

Inspired by OpenPipe/ART's client-server architecture, this module provides
a thin protocol layer that separates the **learner** (training process with
gradient computation) from the **rollout worker** (inference-only generation).

The design has three components:

1. **TrainingServer** — owns the model, optimizer, and training loop.
   Exposes ``get_weights()`` / ``set_weights()`` for weight synchronization
   and ``train_step()`` for gradient updates.

2. **RolloutClient** — a lightweight proxy that calls the server for weight
   sync and delegates generation to a rollout backend (vLLM, HF, etc.).
   This allows the rollout worker to run in a separate process or machine.

3. **WeightSyncProtocol** — defines the serialization format for weight
   exchange. Supports full-state-dict and delta-only modes.

Usage (single-process mode)::

    from hermes_agentic_rl.client_server import TrainingServer, RolloutClient

    server = TrainingServer(model, optimizer)
    client = RolloutClient(rollout_backend, server)  # in-process
    client.sync_weights()
    outputs = client.generate(prompts)

Usage (multi-process, via shared file / RPC)::

    # Server side:
    server = TrainingServer(model, optimizer)
    server.serve_weights(path="/shared/weights.pt")

    # Client side:
    client = RolloutClient(rollout_backend, weight_source="/shared/weights.pt")
    client.sync_weights()
"""

from __future__ import annotations

import logging
import pickle
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SyncMode(Enum):
    """Weight synchronization mode."""

    FULL = "full"        # Send entire state dict
    DELTA = "delta"       # Send only changed tensors (requires base)
    METADATA = "metadata"  # Send only shapes/dtypes (for sanity checks)


@dataclass(slots=True)
class WeightSnapshot:
    """A versioned weight snapshot for sync.

    Attributes
    ----------
    state_dict : dict[str, torch.Tensor]
        The model weights (CPU tensors).
    version : int
        Monotonically increasing version number.
    timestamp : float
        Unix timestamp of snapshot creation.
    metadata : dict[str, Any]
        Extra info (e.g., training step, loss, lr).
    """

    state_dict: dict[str, torch.Tensor]
    version: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_tensors(self) -> int:
        return len(self.state_dict)

    @property
    def total_bytes(self) -> int:
        return sum(v.element_size() * v.numel() for v in self.state_dict.values())


class WeightSink(Protocol):
    """Protocol for receiving weight updates (client side)."""

    def load_weights_from(self, snapshot: WeightSnapshot) -> None: ...


class WeightSource(Protocol):
    """Protocol for providing weight updates (server side)."""

    def get_current_weights(self) -> WeightSnapshot: ...


class TrainingServer:
    """Server-side: owns model + optimizer, provides weight snapshots.

    Parameters
    ----------
    model : nn.Module
        The trainable model (learner).
    optimizer : torch.optim.Optimizer | None
        The optimizer. If None, ``train_step`` will raise.
    sync_mode : SyncMode
        How weights are serialized for transfer. Default: FULL.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        sync_mode: SyncMode = SyncMode.FULL,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.sync_mode = sync_mode
        self._version = 1  # initial weights are version 1
        self._last_snapshot: WeightSnapshot | None = None
        self._training_step = 0
        self._training_metrics: dict[str, float] = {}

    def get_weights(self) -> WeightSnapshot:
        """Return a snapshot of current model weights (CPU tensors)."""
        return self.get_current_weights()

    def get_current_weights(self) -> WeightSnapshot:
        """Return a snapshot of current model weights (CPU tensors).

        This is the ``WeightSource`` protocol method called by ``RolloutClient``.
        The version number only increases when weights actually change
        (after ``train_step`` or ``load_weights``), not on every call.
        """
        cpu_state = {
            k: v.detach().cpu()
            for k, v in self.model.state_dict().items()
        }
        snapshot = WeightSnapshot(
            state_dict=cpu_state,
            version=self._version,
            timestamp=time.time(),
            metadata={
                "training_step": self._training_step,
                "sync_mode": self.sync_mode.value,
                **self._training_metrics,
            },
        )
        self._last_snapshot = snapshot
        return snapshot

    def train_step(self, loss: torch.Tensor) -> float:
        """Run one optimizer step. Returns the loss value."""
        if self.optimizer is None:
            raise RuntimeError("TrainingServer.train_step requires an optimizer")
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = _compute_grad_norm(self.model)
        self.optimizer.step()
        self._training_step += 1
        self._version += 1  # weights changed
        loss_val = float(loss.detach())
        self._training_metrics = {
            "loss": loss_val,
            "grad_norm": float(grad_norm),
        }
        return loss_val

    @property
    def version(self) -> int:
        return self._version

    @property
    def training_step(self) -> int:
        return self._training_step

    def serve_weights(self, path: str | Path) -> None:
        """Write current weights to a file for out-of-process sync."""
        snapshot = self.get_current_weights()
        _write_snapshot(snapshot, Path(path))
        logger.debug("TrainingServer: wrote weights v%d to %s", snapshot.version, path)

    def load_weights(self, path: str | Path) -> None:
        """Load weights from a file (e.g., for checkpoint resume)."""
        snapshot = _read_snapshot(Path(path))
        self.model.load_state_dict(snapshot.state_dict)
        self._version = snapshot.version
        logger.info("TrainingServer: loaded weights v%d from %s", snapshot.version, path)


class RolloutClient:
    """Client-side: syncs weights from server and delegates generation.

    Parameters
    ----------
    rollout_backend : Any
        An object with ``generate()`` and ``sync_weights_from()`` methods
        (e.g., ``VLLMRolloutBackend``, ``HFRolloutBackend``).
    weight_source : WeightSource | str | Path | None
        Either an in-process ``TrainingServer``, or a file path for
        out-of-process weight sync. If None, no weight sync happens.
    sync_interval : float
        Minimum seconds between weight syncs (0 = always sync). This
        prevents overwhelming the network/file system with frequent syncs.
    """

    def __init__(
        self,
        rollout_backend: Any,
        weight_source: WeightSource | str | Path | None = None,
        *,
        sync_interval: float = 0.0,
    ) -> None:
        self.rollout_backend = rollout_backend
        self._weight_source = weight_source
        self._sync_interval = sync_interval
        self._last_sync_time: float = 0.0
        self._current_version: int = 0

    def sync_weights(self, force: bool = False) -> bool:
        """Pull latest weights from the server.

        Returns True if a sync actually happened, False if skipped
        due to ``sync_interval`` throttling.
        """
        if self._weight_source is None:
            return False

        now = time.time()
        if not force and (now - self._last_sync_time) < self._sync_interval:
            return False

        if isinstance(self._weight_source, (str, Path)):
            snapshot = _read_snapshot(Path(self._weight_source))
        else:
            snapshot = self._weight_source.get_current_weights()

        if snapshot.version == self._current_version and not force:
            return False  # already up to date

        # Push to rollout backend.
        sync_fn = getattr(self.rollout_backend, "sync_weights_from", None)
        if sync_fn is not None:
            sync_fn(snapshot.state_dict)
        else:
            # Fall back to load_state_dict if available.
            load_fn = getattr(self.rollout_backend, "load_state_dict", None)
            if load_fn is not None:
                load_fn(snapshot.state_dict)
            else:
                logger.warning("RolloutClient: backend has no sync_weights_from or load_state_dict")

        self._current_version = snapshot.version
        self._last_sync_time = now
        logger.debug(
            "RolloutClient: synced to weight version %d (%d tensors, %.2f MB)",
            snapshot.version,
            snapshot.num_tensors,
            snapshot.total_bytes / 1e6,
        )
        return True

    def generate(self, prompts: list[str] | list[list[int]], **kwargs: Any) -> Any:
        """Generate responses using the rollout backend."""
        self.sync_weights()
        gen_fn = getattr(self.rollout_backend, "generate", None)
        if gen_fn is not None:
            return gen_fn(prompts, **kwargs)
        gen_batch_fn = getattr(self.rollout_backend, "generate_batch", None)
        if gen_batch_fn is not None:
            return gen_batch_fn(prompts, **kwargs)
        raise AttributeError(
            f"Rollout backend {type(self.rollout_backend).__name__}"
            " has no generate/generate_batch method"
        )

    @property
    def weight_version(self) -> int:
        return self._current_version


# ── Serialization helpers ──

_SNAPSHOT_MAGIC = b"HARL\x00\x01"  # Hermes Agentic RL snapshot format v1


def _write_snapshot(snapshot: WeightSnapshot, path: Path) -> None:
    """Write a WeightSnapshot to disk using torch.save for portability."""
    payload = {
        "state_dict": snapshot.state_dict,
        "version": snapshot.version,
        "timestamp": snapshot.timestamp,
        "metadata": snapshot.metadata,
    }
    torch.save(payload, path)


def _read_snapshot(path: Path) -> WeightSnapshot:
    """Read a WeightSnapshot from disk."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return WeightSnapshot(
        state_dict=payload["state_dict"],
        version=payload["version"],
        timestamp=payload["timestamp"],
        metadata=payload.get("metadata", {}),
    )


def _compute_grad_norm(model: nn.Module) -> float:
    """Compute the global gradient norm (L2)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total**0.5


# ── Convenience: ClientServerPair for single-process mode ──


@dataclass(slots=True)
class ClientServerPair:
    """Bundles a TrainingServer and RolloutClient for convenience.

    In single-process mode, the client's weight_source is the server
    itself, so weight sync is just a dict copy.
    """

    server: TrainingServer
    client: RolloutClient

    @classmethod
    def create(
        cls,
        model: nn.Module,
        rollout_backend: Any,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        sync_mode: SyncMode = SyncMode.FULL,
        sync_interval: float = 0.0,
    ) -> ClientServerPair:
        server = TrainingServer(model, optimizer, sync_mode=sync_mode)
        client = RolloutClient(
            rollout_backend,
            weight_source=server,
            sync_interval=sync_interval,
        )
        return cls(server=server, client=client)

    def step_and_sync(self, loss: torch.Tensor) -> float:
        """Convenience: train step + weight sync in one call."""
        loss_val = self.server.train_step(loss)
        self.client.sync_weights(force=True)
        return loss_val
