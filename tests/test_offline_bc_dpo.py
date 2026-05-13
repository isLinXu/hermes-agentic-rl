"""Offline trainer tests: BC + DPO + ReplayBuffer IO."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.offline import (
    BCConfig,
    BCTrainer,
    DPOConfig,
    DPOPair,
    DPOTrainer,
    ReplayBuffer,
    TrainSample,
)


def _small_backend(seed: int = 0) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(seed=seed, dim=16, n_heads=2, n_layers=1, max_len=32)
    )


def test_replay_buffer_jsonl_roundtrip(tmp_path: Path):
    buf = ReplayBuffer()
    buf.add_sample(TrainSample(prompt_ids=[1, 2, 3], response_ids=[4, 5], reward=0.7))
    buf.add_pair(DPOPair(prompt_ids=[1, 2], chosen_ids=[3, 4], rejected_ids=[5, 6]))
    p = tmp_path / "buf.jsonl"
    buf.save_jsonl(p)
    loaded = ReplayBuffer.load_jsonl(p)
    assert len(list(loaded.iter_samples())) == 1
    assert len(list(loaded.iter_dpo_pairs())) == 1


def test_bc_trainer_reduces_nll():
    backend = _small_backend(seed=0)
    tok = backend.tokenizer
    prompt = [tok.bos_id] + tok.encode("A?")
    target = tok.encode("hi")
    buf = ReplayBuffer()
    # add the same sample 8 times (small overfit)
    for _ in range(8):
        buf.add_sample(TrainSample(prompt_ids=prompt, response_ids=target, reward=1.0))
    trainer = BCTrainer(
        backend,
        buf,
        cfg=BCConfig(n_epochs=5, batch_size=2, lr=0.01, log_every=1000, seed=0),
    )
    stats = trainer.train()
    assert len(stats.steps) > 0
    first_nll = stats.steps[0]["nll"]
    last_nll = stats.steps[-1]["nll"]
    assert last_nll < first_nll  # training should reduce NLL


def test_dpo_trainer_improves_preference_accuracy():
    backend = _small_backend(seed=0)
    tok = backend.tokenizer
    prompt = [tok.bos_id] + tok.encode("Q?")
    chosen = tok.encode("yes")
    rejected = tok.encode("no")
    buf = ReplayBuffer()
    for _ in range(6):
        buf.add_pair(DPOPair(prompt_ids=prompt, chosen_ids=chosen, rejected_ids=rejected))
    trainer = DPOTrainer(
        backend,
        buf,
        cfg=DPOConfig(n_epochs=5, batch_size=2, lr=0.01, beta=0.5, log_every=1000, seed=0),
    )
    stats = trainer.train()
    first = stats.steps[0]
    last = stats.steps[-1]
    # losses should drop; accuracy should rise
    assert last["loss"] <= first["loss"]
    assert last["acc"] >= first["acc"]


def test_dpo_needs_preference_pairs():
    backend = _small_backend(seed=0)
    buf = ReplayBuffer()
    buf.add_sample(TrainSample(prompt_ids=[1], response_ids=[2], reward=0.5))
    with pytest.raises(RuntimeError, match="DPOPair"):
        DPOTrainer(backend, buf, cfg=DPOConfig()).train()
