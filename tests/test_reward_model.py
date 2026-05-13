"""Reward-model trainer tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.offline import DPOPair, ReplayBuffer
from hermes_agentic_rl.rewards.reward_model import (
    RewardModel,
    RewardModelConfig,
    RewardModelTrainer,
)


def _small_backend(seed: int = 0) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(seed=seed, dim=16, n_heads=2, n_layers=1, max_len=32)
    )


def test_reward_model_separates_chosen_from_rejected():
    backend = _small_backend(seed=0)
    rm = RewardModel(backend, freeze_base=True)
    tok = backend.tokenizer
    prompt = [tok.bos_id] + tok.encode("Q?")
    chosen = tok.encode("yes")
    rejected = tok.encode("no")
    buf = ReplayBuffer()
    for _ in range(8):
        buf.add_pair(DPOPair(prompt_ids=prompt, chosen_ids=chosen, rejected_ids=rejected))
    trainer = RewardModelTrainer(
        rm, buf, cfg=RewardModelConfig(n_epochs=5, batch_size=2, lr=0.05, log_every=1000, seed=0)
    )
    stats = trainer.train()
    # By the end, accuracy should be high (the RM has a trivial binary choice)
    assert stats.steps[-1]["acc"] >= 0.5
    # Score ordering must reflect the preference
    with torch.no_grad():
        r_w = rm.score_pair(prompt, chosen).item()
        r_l = rm.score_pair(prompt, rejected).item()
    assert r_w > r_l
