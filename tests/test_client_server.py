"""Client-Server API layer tests: weight sync, train step, file-based sync."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.client_server import (
    ClientServerPair,
    RolloutClient,
    SyncMode,
    TrainingServer,
    WeightSnapshot,
)


class _MockRolloutBackend:
    """Captures synced weights and generates dummy outputs."""

    def __init__(self) -> None:
        self.synced_weights: list[dict[str, Any]] = []
        self.generate_calls: int = 0

    def sync_weights_from(self, state_dict: dict[str, Any]) -> None:
        self.synced_weights.append({k: v.clone() for k, v in state_dict.items()})

    def generate(self, prompts, **kwargs):
        self.generate_calls += 1
        return ["dummy_output"] * len(prompts)


def _make_model():
    return TinyCausalLMBackend(TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2)).model


def test_training_server_get_weights_returns_snapshot():
    model = _make_model()
    server = TrainingServer(model)
    snap = server.get_weights()
    assert isinstance(snap, WeightSnapshot)
    assert snap.version == 1
    assert snap.num_tensors > 0
    assert snap.total_bytes > 0


def test_training_server_version_increments():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    server = TrainingServer(model, optimizer)
    v1 = server.get_weights().version
    # train_step increments version
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss = model(ids).sum()
    server.train_step(loss)
    v2 = server.get_weights().version
    assert v2 == v1 + 1


def test_training_server_train_step_updates_metrics():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    server = TrainingServer(model, optimizer)

    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss = model(ids).sum()
    loss_val = server.train_step(loss)
    assert isinstance(loss_val, float)
    assert server.training_step == 1


def test_training_server_train_step_without_optimizer_raises():
    model = _make_model()
    server = TrainingServer(model, optimizer=None)
    with pytest.raises(RuntimeError, match="requires an optimizer"):
        server.train_step(torch.tensor(1.0))


def test_rollout_client_syncs_from_server():
    model = _make_model()
    server = TrainingServer(model)
    mock = _MockRolloutBackend()
    client = RolloutClient(mock, weight_source=server)

    synced = client.sync_weights(force=True)
    assert synced is True
    assert len(mock.synced_weights) == 1
    assert client.weight_version > 0


def test_rollout_client_skip_sync_same_version():
    model = _make_model()
    server = TrainingServer(model)
    mock = _MockRolloutBackend()
    client = RolloutClient(mock, weight_source=server)

    client.sync_weights(force=True)

    # Without force and same version, should skip.
    synced = client.sync_weights()
    assert synced is False


def test_rollout_client_no_source_returns_false():
    mock = _MockRolloutBackend()
    client = RolloutClient(mock, weight_source=None)
    assert client.sync_weights() is False


def test_rollout_client_sync_interval_throttle():
    model = _make_model()
    server = TrainingServer(model)
    mock = _MockRolloutBackend()
    client = RolloutClient(mock, weight_source=server, sync_interval=100.0)

    # First sync goes through.
    assert client.sync_weights(force=True) is True
    # Second sync within interval — skipped.
    assert client.sync_weights() is False


def test_rollout_client_generate_triggers_sync():
    model = _make_model()
    server = TrainingServer(model)
    mock = _MockRolloutBackend()
    client = RolloutClient(mock, weight_source=server)

    result = client.generate(["hello"])
    assert mock.generate_calls == 1
    assert len(result) == 1


def test_rollout_client_generate_no_sync_method():
    """If backend has no sync_weights_from, falls back to load_state_dict."""
    model = _make_model()
    server = TrainingServer(model)

    class _FallbackBackend:
        def __init__(self):
            self.loaded = []
        def load_state_dict(self, sd):
            self.loaded.append(sd)
        def generate(self, prompts, **kw):
            return ["ok"]

    backend = _FallbackBackend()
    client = RolloutClient(backend, weight_source=server)
    client.sync_weights(force=True)
    assert len(backend.loaded) == 1


def test_file_based_weight_sync(tmp_path: Path):
    """Server writes to file, client reads from file."""
    model = _make_model()
    server = TrainingServer(model)
    weight_file = tmp_path / "weights.pt"

    server.serve_weights(weight_file)
    assert weight_file.exists()

    mock = _MockRolloutBackend()
    client = RolloutClient(mock, weight_source=weight_file)
    synced = client.sync_weights(force=True)
    assert synced is True
    assert len(mock.synced_weights) == 1


def test_server_load_weights_roundtrip(tmp_path: Path):
    model = _make_model()
    server = TrainingServer(model)
    weight_file = tmp_path / "weights.pt"
    server.serve_weights(weight_file)

    # Modify model weights, then load back.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    server.load_weights(weight_file)
    assert server.version > 0


def test_client_server_pair_create_and_step():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    mock = _MockRolloutBackend()

    pair = ClientServerPair.create(
        model, mock, optimizer, sync_mode=SyncMode.FULL,
    )

    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss = model(ids).sum()
    loss_val = pair.step_and_sync(loss)
    assert isinstance(loss_val, float)
    assert pair.server.training_step == 1
    assert len(mock.synced_weights) == 1  # synced after step


def test_weight_snapshot_metadata():
    model = _make_model()
    server = TrainingServer(model)
    snap = server.get_weights()
    assert "training_step" in snap.metadata
    assert "sync_mode" in snap.metadata
