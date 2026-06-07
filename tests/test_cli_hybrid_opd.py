"""CLI + end-to-end coverage for the hybrid (GRPO + OPD) closed loop.

Verifies:
  * ``_build_hybrid`` parses the ``train_rl:`` + ``opd:`` blocks.
  * The ``letter_counting_next_state`` reward emits an ``opd_hint`` on a wrong
    answer (the directive next-state signal).
  * Running ``HybridTrainer`` on letter_counting actually fires the OPD branch
    (teacher-logprob fill -> n_opd > 0) — no longer a no-op.
  * ``run_train_rl`` drives the whole thing from a YAML config.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.cli.train_rl import _build_hybrid, run_train_rl
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.envs.letter_counting import (
    LetterCountingConfig,
    LetterCountingEnv,
    LetterCountingNextStateReward,
)

REPO = Path(__file__).resolve().parents[1]


def _traj(final_output: str) -> Trajectory:
    return Trajectory(
        task_id="t0",
        prompt="q",
        steps=[],
        final_output=final_output,
        finished_naturally=True,
        turns_used=1,
        metadata={"runtime": {"rl": {}}},
    )


def test_build_hybrid_parses_opd_block():
    cfg = {
        "train_rl": {"n_iters": 5, "w_rl": 1.0, "w_opd": 2.0},
        "opd": {
            "teacher_fill": True,
            "adv_diff_clip": 1.5,
            "max_hint_tokens": 64,
            "capability_axis_weights": {"task_success": 1.5},
        },
    }
    hc = _build_hybrid(cfg, output_dir=None)
    assert hc.w_opd == 2.0
    assert hc.opd_teacher_fill is True
    assert hc.opd_adv_diff_clip == 1.5
    assert hc.opd_teacher_max_hint_tokens == 64
    assert hc.opd_capability_axis_weights == {"task_success": 1.5}


def test_next_state_reward_emits_hint_on_wrong_answer():
    item = {
        "correct_counts": {"e": 3},
        "target_letters": ["e"],
        "instruction": "How many e's...",
    }
    traj = _traj("<answer>1</answer>")
    reward = LetterCountingNextStateReward(weight=1.0)
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert result.score == 0.0  # wrong
    rl = traj.metadata["runtime"]["rl"]
    assert "opd_hint" in rl
    assert "<answer>3</answer>" in rl["opd_hint"]


def test_next_state_reward_no_hint_on_correct_answer():
    item = {"correct_counts": {"e": 1}, "target_letters": ["e"], "instruction": "q"}
    traj = _traj("<answer>1</answer>")
    reward = LetterCountingNextStateReward(weight=1.0)
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert result.score == 1.0  # correct
    assert "opd_hint" not in traj.metadata["runtime"]["rl"]


def test_hybrid_trainer_fires_opd_on_letter_counting():
    from hermes_agentic_rl.trainers.hybrid_trainer import (
        HybridTrainer,
        HybridTrainerConfig,
    )

    env = LetterCountingEnv(LetterCountingConfig(starting_level=2, max_level=2, seed=1))
    rm = RewardManager([LetterCountingNextStateReward(weight=1.0)])
    cfg = HybridTrainerConfig(
        n_iters=2,
        group_size=4,
        prompts_per_iter=1,
        max_new_tokens=8,
        lr=5e-3,
        log_every=100,
        seed=3,
        use_reference=True,
        opd_teacher_fill=True,
        opd_capability_axis_weights={"task_success": 1.5},
    )
    from hermes_agentic_rl.backends.tiny import (
        TinyBackendConfig,
        TinyCausalLMBackend,
    )

    backend = TinyCausalLMBackend(TinyBackendConfig(dim=32, n_heads=4, n_layers=2, seed=0))
    trainer = HybridTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
    stats = trainer.train()

    # A tiny random policy is almost always wrong -> hints emitted -> OPD fires.
    fired = any(it.get("n_opd", 0) > 0 for it in stats.iters)
    filled = any(it.get("opd_teacher_n_filled", 0) > 0 for it in stats.iters)
    assert fired, "OPD branch never fired despite wrong answers + hints"
    assert filled, "teacher-logprob filler never populated teacher_logprobs"


def test_run_train_rl_hybrid_smoke(tmp_path):
    config = REPO / "configs" / "letter_counting_hybrid_opd_smoke.yaml"
    assert config.exists()
    rc = run_train_rl(str(config), output_dir=str(tmp_path / "run"))
    assert rc == 0
    summary = tmp_path / "run" / "train_rl_summary.json"
    assert summary.exists()
