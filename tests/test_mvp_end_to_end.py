"""End-to-end MVP: TinyBackend + EchoEnv + GRPOTrainer.

Verifies:
  (a) params actually move (gradient flows)
  (b) reward trend goes up (learning happens)
  (c) echo reward reaches a meaningful level after ~40 iters
"""

from __future__ import annotations

from statistics import mean
from types import MethodType

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


def test_grpo_mvp_params_move_and_reward_trends_up():
    torch.manual_seed(0)
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=24, n_heads=2, n_layers=2, seed=0))
    params_before = [p.detach().clone() for p in backend.trainable_parameters()]

    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=rm,
        cfg=GRPOTrainerConfig(
            n_iters=40,
            group_size=6,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=8,
            temperature=1.0,
            log_every=100,  # silence
            seed=42,
        ),
    )
    stats = trainer.train()

    # (a) params moved
    params_after = list(backend.trainable_parameters())
    drift = sum(
        (a - b).abs().sum().item() for a, b in zip(params_before, params_after, strict=False)
    )
    assert drift > 0, f"no gradient reached the policy (drift={drift})"

    rewards = [r["mean_reward"] for r in stats.iters]
    assert len(rewards) == 40

    # (b) trend: late window mean > early window mean (lenient — MVP with
    #     30K params is noisy, but should be directionally positive)
    early = mean(rewards[:10])
    late = mean(rewards[-10:])
    assert late > early, f"reward did not improve: early={early:.4f} late={late:.4f}"

    # (c) non-trivial best
    assert stats.best_reward() > 0.03, f"best reward too low: {stats.best_reward():.4f}"


def test_agent_loop_emits_rl_metadata():
    """Non-regression: PolicyAgentLoop must carry prompt_ids / response_ids / old_logprobs."""
    import asyncio

    from hermes_agentic_rl.agent_loop.policy_loop import PolicyAgentLoop

    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=2, seed=0))
    loop = PolicyAgentLoop(backend=backend, max_new_tokens=4, temperature=1.0, seed=0)
    result = asyncio.run(loop.run("hello"))
    rl = result["metadata"]["rl"]
    assert set(rl.keys()) >= {"prompt_ids", "response_ids", "old_logprobs"}
    assert len(rl["response_ids"]) == len(rl["old_logprobs"])
    assert len(rl["response_ids"]) <= 4


def test_grpo_batch_generate_path_runs(monkeypatch):
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=2, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])

    calls: list[int] = []
    from hermes_agentic_rl.backends.batch_generate import BatchRolloutGenerator

    original = BatchRolloutGenerator.generate

    def _spy(self, prompt_ids_list, seed=None):
        calls.append(len(prompt_ids_list))
        return original(self, prompt_ids_list, seed=seed)

    monkeypatch.setattr(BatchRolloutGenerator, "generate", _spy)

    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=rm,
        cfg=GRPOTrainerConfig(
            n_iters=1,
            group_size=4,
            prompts_per_iter=1,
            lr=5e-3,
            max_new_tokens=6,
            temperature=1.0,
            log_every=100,
            seed=7,
            batch_generate=True,
        ),
    )
    stats = trainer.train()

    assert calls == [4]
    assert len(stats.iters) == 1


def test_grpo_update_epochs_preserve_group_boundaries():
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=2, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=rm,
        cfg=GRPOTrainerConfig(
            n_iters=1,
            group_size=4,
            prompts_per_iter=2,
            lr=5e-3,
            max_new_tokens=6,
            temperature=1.0,
            log_every=100,
            seed=7,
            update_epochs=2,
            minibatch_size=3,
        ),
    )

    calls: list[tuple[int, list[int]]] = []
    original = trainer.algo.compute_loss

    def _spy(self, policy, ref_policy, batch):
        calls.append((len(batch.records), sorted(len(v) for v in batch.by_group().values())))
        return original(policy, ref_policy, batch)

    trainer.algo.compute_loss = MethodType(_spy, trainer.algo)
    stats = trainer.train()

    assert len(stats.iters) == 1
    rec = stats.iters[0]
    assert rec["n_optimizer_steps"] == 4
    assert rec["update_epochs"] == 2
    assert rec["n_minibatches"] == 4
    assert len(calls) == 4
    assert all(batch_size == 4 for batch_size, _ in calls)
    assert all(group_sizes == [4] for _, group_sizes in calls)


def test_deprecated_atropos_shim_still_works():
    """Back-compat: the old AtroposGrpoTrainer name keeps working with a warning."""
    import tempfile
    import warnings
    from pathlib import Path

    from hermes_agentic_rl.trainers.atropos_grpo import AtroposGrpoTrainer

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "t.jsonl"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exporter = AtroposGrpoTrainer(output_path=out)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        # Still usable as an exporter.
        assert hasattr(exporter, "submit_sync")
