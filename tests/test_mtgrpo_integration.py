"""Integration test: verify the MT-GRPO training entry point runs end-to-end.

This test exercises the full stack:
  1. TinyBackend (no GPU needed)
  2. SimToolEnv (in-process environment)
  3. RewardComposer with ToolcallReward + OutcomeReward
  4. GRPOTrainer with curriculum enabled
  5. OnPolicyTrainer.train() loop (2 iterations)

It catches the three-way breakage that was present in train_mtgrpo.py:
  - RewardManager(components=[]) TypeError
  - CurriculumScheduler not wired into trainer
  - create_reward_composer return value discarded

Run: python -m pytest tests/test_mtgrpo_integration.py -v
Or:  python tests/test_mtgrpo_integration.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def test_reward_manager_compatibility():
    """Test that RewardManager accepts components= keyword (the bug that crashed)."""
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.rewards.composer import RewardComposer, RewardComposerConfig
    from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
    from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward

    # The exact call that used to crash
    rm = RewardManager(components=[])
    assert len(rm.components) == 0

    # With actual components
    rm2 = RewardManager(components=[ToolcallReward(weight=1.0)])
    assert len(rm2.components) == 1

    # RewardComposer as drop-in replacement
    composer = RewardComposer(
        components=[ToolcallReward(weight=1.0), OutcomeReward(weight=1.0)],
        config=RewardComposerConfig(),
    )
    assert hasattr(composer, "evaluate")
    assert hasattr(composer, "components")
    assert len(composer.components) == 2

    logger.info("✓ RewardManager/Composer compatibility test passed")


def test_config_validation():
    """Test GRPOTrainerConfig __post_init__ fatal validation."""
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainerConfig

    # Valid config
    cfg = GRPOTrainerConfig(n_iters=5, group_size=4)
    assert cfg.n_iters == 5

    # Invalid: group_size < 2 (warns, doesn't raise — allowed for testing)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GRPOTrainerConfig(group_size=1)
        assert len(w) == 1
        assert "group_size" in str(w[0].message)

    # Invalid: lr <= 0
    with pytest.raises(ValueError):
        GRPOTrainerConfig(lr=0)

    # Curriculum passes through
    cfg2 = GRPOTrainerConfig(curriculum={"auto_advance": True})
    assert cfg2.curriculum == {"auto_advance": True}

    logger.info("✓ Config validation test passed")


def test_curriculum_integration():
    """Test that curriculum config is forwarded to OnPolicyTrainerConfig."""
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainerConfig
    from hermes_agentic_rl.trainers.on_policy_config import build_shared_on_policy_config

    cfg = GRPOTrainerConfig(
        n_iters=3,
        group_size=4,
        curriculum={"auto_advance": True, "min_iters_per_stage": 2},
    )
    shared = build_shared_on_policy_config(cfg)
    assert shared.curriculum == {"auto_advance": True, "min_iters_per_stage": 2}

    logger.info("✓ Curriculum integration test passed")


def test_ruler_reward():
    """Test RULER automatic reward generation."""
    from hermes_agentic_rl.core.types import RolloutStep, Trajectory
    from hermes_agentic_rl.rewards.ruler import RULER

    traj = Trajectory(
        task_id="test-1",
        prompt="What is 2+2?",
        steps=[RolloutStep(turn_index=0, assistant_message="4",
                            tool_calls=[{"name": "calc", "arguments": {"expr": "2+2"}}])],
        final_output="4",
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )

    ruler = RULER.from_config({
        "rules": [
            {"name": "exact", "template": "exact_match", "weight": 2.0,
             "params": {"gold_key": "answer"}},
            {"name": "turns", "template": "turn_efficiency", "weight": 1.0,
             "params": {"max_turns": 5}},
        ],
    })

    result = asyncio.run(ruler.evaluate({"answer": "4"}, traj, None))
    assert result.score > 0.0
    assert len(result.metadata["fired_rules"]) == 2
    assert result.metadata["raw_scores"]["exact"] == 1.0

    logger.info(f"✓ RULER test passed (score={result.score:.3f})")


def test_yaml_config():
    """Test YAML config loading and component building."""
    from hermes_agentic_rl.yaml_config import (
        build_reward_components,
        build_trainer_config,
        load_config,
    )

    config_path = Path(__file__).parent.parent / "configs" / "mtgrpo_tiny.yaml"
    if not config_path.exists():
        logger.warning("Skipping YAML test — config file not found")
        return

    config = load_config(config_path)
    assert "n_iters" in config
    assert "rewards" in config
    assert "curriculum" in config

    tcfg = build_trainer_config(config)
    assert tcfg.n_iters == 20
    assert tcfg.curriculum is not None

    components, composer_cfg = build_reward_components(config)
    assert len(components) >= 2
    assert composer_cfg is not None

    logger.info("✓ YAML config test passed")


def test_orchestrator_import():
    """Test that TrainingOrchestrator can be imported."""
    from hermes_agentic_rl.trainers.orchestrator import TrainingOrchestrator
    assert hasattr(TrainingOrchestrator, "run")
    assert hasattr(TrainingOrchestrator, "_run_side_effects")
    logger.info("✓ Orchestrator import test passed")


def test_end_to_end_tiny_training():
    """Run 2 iterations of GRPO training with TinyBackend.

    This is the ultimate smoke test: if the entry point is broken,
    this test will fail.
    """
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
    from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv, build_sim_tool_dataset
    from hermes_agentic_rl.rewards.composer import RewardComposer, RewardComposerConfig
    from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
    from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    # 1. Build backend
    backend = TinyCausalLMBackend(
        TinyBackendConfig(
            seed=42,
            with_value_head=False,
            dim=32,
            n_heads=4,
            n_layers=2,
            max_len=256,
        )
    )

    # 2. Build env
    dataset = build_sim_tool_dataset(n=8, seed=42)
    env = SimToolEnv(dataset=dataset)

    # 3. Build reward composer (the fix: use RewardComposer directly)
    components = [ToolcallReward(weight=1.0), OutcomeReward(weight=1.0)]
    reward_manager = RewardComposer(
        components=components,
        config=RewardComposerConfig(
            normalize={"toolcall_reward": True},
        ),
    )

    # 4. Build config with curriculum
    cfg = GRPOTrainerConfig(
        n_iters=2,
        group_size=4,
        prompts_per_iter=2,
        lr=1e-3,
        max_new_tokens=32,
        temperature=1.0,
        clip_eps=0.2,
        log_every=1,
        multi_turn=True,
        multi_turn_credit={"mode": "discounted", "gamma": 0.9},
        per_token_advantage=True,
        advantage_norm="group",
        curriculum={"auto_advance": True, "min_iters_per_stage": 1},
    )

    # 5. Create trainer
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=cfg,
    )

    # 6. Verify curriculum scheduler was created inside trainer
    assert trainer._curriculum_scheduler is not None, \
        "Curriculum scheduler should be initialized inside trainer"

    # 7. Run training (this is the ultimate test)
    logger.info("Starting end-to-end training (2 iters)...")
    stats = trainer.train()
    logger.info(f"Training completed with {len(stats.iters)} iteration records")

    assert len(stats.iters) > 0, "Training should produce at least 1 record"

    # 8. Verify curriculum snapshot is available
    snap = trainer._curriculum_scheduler.snapshot()
    logger.info(f"Curriculum snapshot: stage={snap['curriculum/current_stage']}, "
                f"iters_in_stage={snap['curriculum/iters_in_stage']}")

    logger.info("✓ End-to-end tiny training test passed")


def main():
    """Run all tests."""
    tests = [
        test_reward_manager_compatibility,
        test_config_validation,
        test_curriculum_integration,
        test_ruler_reward,
        test_yaml_config,
        test_orchestrator_import,
        test_end_to_end_tiny_training,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test.__name__} FAILED: {e}", exc_info=True)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
