from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.runtime.errors import RuntimeConfigurationError
from hermes_agentic_rl.trainers.ema import EMAModel
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


def test_ema_model_blends_parameters() -> None:
    policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    ema = EMAModel(policy, tau=0.5)
    shadow_param = next(ema.shadow.model.parameters()).detach().clone()

    with torch.no_grad():
        next(policy.model.parameters()).add_(2.0)

    ema.update(policy)

    updated_shadow_param = next(ema.shadow.model.parameters()).detach()
    assert torch.allclose(updated_shadow_param, shadow_param + 1.0)


def test_grpo_trainer_uses_ema_backend_for_local_rollout() -> None:
    policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])

    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        reward_manager=rm,
        cfg=GRPOTrainerConfig(
            n_iters=1,
            group_size=4,
            prompts_per_iter=1,
            lr=5e-3,
            max_new_tokens=4,
            log_every=100,
            use_ema_rollout=True,
            ema_tau=0.1,
        ),
    )

    assert trainer._rollout_backend() is not policy
    stats = trainer.train()

    assert stats.iters[0]["ema_rollout"] == 1.0
    assert stats.iters[0]["ema_tau"] == 0.1


def test_ema_rollout_rejects_vllm_rollout_mix() -> None:
    policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])

    with pytest.raises(RuntimeConfigurationError, match="EMA rollout"):
        GRPOTrainer(
            policy=policy,
            env=env,
            reward_manager=rm,
            cfg=GRPOTrainerConfig(
                use_ema_rollout=True,
                vllm_rollout_model="tiny-rollout",
            ),
        )
