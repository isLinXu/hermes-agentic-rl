"""Tests for backend-algorithm compatibility and FSDP-aware checkpointing."""

from __future__ import annotations

import pytest

from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


class _NonTrainableBackend:
    """Mock backend that claims to be non-trainable (like VLLMRolloutBackend)."""

    is_trainable_flag = False
    supports_value_head_flag = False

    def is_trainable(self) -> bool:
        return self.is_trainable_flag

    def supports_value_head(self) -> bool:
        return self.supports_value_head_flag

    @property
    def tokenizer(self):
        return None

    def generate(self, *a, **kw):
        ...

    def score(self, *a, **kw):
        ...

    def score_batch(self, *a, **kw):
        ...

    def trainable_parameters(self):
        return []


class TestBackendCompat:
    """Backend-algorithm compatibility checks."""

    def test_grpo_rejects_non_trainable_backend(self):
        """GRPOTrainer should reject VLLMRolloutBackend as policy."""
        non_trainable = _NonTrainableBackend()
        env = EchoTaskEnv(build_default_echo_dataset())
        reward = RewardManager([EchoRewardComponent(weight=1.0)])
        cfg = GRPOTrainerConfig(n_iters=1, group_size=2, prompts_per_iter=1)

        with pytest.raises(RuntimeError, match="not trainable"):
            GRPOTrainer(
                policy=non_trainable,  # type: ignore[arg-type]
                env=env,
                reward_manager=reward,
                cfg=cfg,
            )

    def test_grpo_accepts_trainable_backend(self):
        """GRPOTrainer should accept TinyCausalLMBackend."""
        backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
        env = EchoTaskEnv(build_default_echo_dataset())
        reward = RewardManager([EchoRewardComponent(weight=1.0)])
        cfg = GRPOTrainerConfig(n_iters=1, group_size=2, prompts_per_iter=1, max_new_tokens=4)

        trainer = GRPOTrainer(
            policy=backend,
            env=env,
            reward_manager=reward,
            cfg=cfg,
        )
        assert trainer.algo_name == "grpo"

    def test_gspo_rejects_non_trainable_backend(self):
        """GSPOTrainer should reject non-trainable backends."""
        from hermes_agentic_rl.trainers.gspo_trainer import GSPOTrainer, GSPOTrainerConfig

        non_trainable = _NonTrainableBackend()
        env = EchoTaskEnv(build_default_echo_dataset())
        reward = RewardManager([EchoRewardComponent(weight=1.0)])
        cfg = GSPOTrainerConfig(n_iters=1, group_size=2, prompts_per_iter=1)

        with pytest.raises(RuntimeError, match="not trainable"):
            GSPOTrainer(
                policy=non_trainable,  # type: ignore[arg-type]
                env=env,
                reward_manager=reward,
                cfg=cfg,
            )

    def test_ppo_rejects_non_value_head_backend(self):
        """PPOTrainer should reject backends without value head."""
        from hermes_agentic_rl.trainers.ppo_trainer import PPOTrainer, PPOTrainerConfig

        backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
        assert not backend.supports_value_head()

        env = EchoTaskEnv(build_default_echo_dataset())
        reward = RewardManager([EchoRewardComponent(weight=1.0)])
        cfg = PPOTrainerConfig(n_iters=1, group_size=2, prompts_per_iter=1)

        with pytest.raises(RuntimeError, match="value-head"):
            PPOTrainer(
                policy=backend,
                env=env,
                reward_manager=reward,
                cfg=cfg,
            )


class TestFSDPCheckpointOps:
    """FSDP-aware checkpoint operations (mocked, no real FSDP needed)."""

    def test_save_full_checkpoint_uses_fsdp_path(self, tmp_path, monkeypatch):
        """When _fsdp_enabled is True, save_full_checkpoint should use gather_fsdp_state_dict."""
        from hermes_agentic_rl.trainers._checkpoint_ops import save_full_checkpoint

        backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
        env = EchoTaskEnv(build_default_echo_dataset())
        reward = RewardManager([EchoRewardComponent(weight=1.0)])
        cfg = GRPOTrainerConfig(
            n_iters=1, group_size=2, prompts_per_iter=1, max_new_tokens=4,
            checkpoint_every=1, output_dir=tmp_path,
        )
        trainer = GRPOTrainer(policy=backend, env=env, reward_manager=reward, cfg=cfg)

        # Force FSDP flag
        trainer._fsdp_enabled = True  # type: ignore[attr-defined]

        # Mock CheckpointManager
        saved_states: list = []

        class MockMgr:
            def save(self, state):
                saved_states.append(state)

        trainer._ckpt_manager = MockMgr()  # type: ignore[attr-defined]

        # Mock gather_fsdp_state_dict to return normal state_dict
        original_sd = backend.model.state_dict()

        def mock_gather(model):
            return {k: v.clone() for k, v in original_sd.items()}

        import hermes_agentic_rl.trainers._checkpoint_ops as ckpt_ops
        monkeypatch.setattr(
            "hermes_agentic_rl.trainers.distributed.gather_fsdp_state_dict",
            mock_gather,
        )

        save_full_checkpoint(trainer, 0)

        assert len(saved_states) == 1
        assert saved_states[0].iteration == 0

    def test_save_full_checkpoint_non_fsdp_uses_direct_state_dict(self, tmp_path):
        """When _fsdp_enabled is False, save_full_checkpoint uses model.state_dict() directly."""
        from hermes_agentic_rl.trainers._checkpoint_ops import save_full_checkpoint

        backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
        env = EchoTaskEnv(build_default_echo_dataset())
        reward = RewardManager([EchoRewardComponent(weight=1.0)])
        cfg = GRPOTrainerConfig(
            n_iters=1, group_size=2, prompts_per_iter=1, max_new_tokens=4,
            checkpoint_every=1, output_dir=tmp_path,
        )
        trainer = GRPOTrainer(policy=backend, env=env, reward_manager=reward, cfg=cfg)

        # FSDP not enabled
        assert not getattr(trainer, "_fsdp_enabled", False)

        saved_states: list = []

        class MockMgr:
            def save(self, state):
                saved_states.append(state)

        trainer._ckpt_manager = MockMgr()  # type: ignore[attr-defined]

        save_full_checkpoint(trainer, 0)

        assert len(saved_states) == 1
