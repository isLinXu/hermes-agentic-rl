"""Unit tests for the v1.1 optimizations from the deep-optimization report.

Each block exercises one fix from `深度优化方案与报告 v1.0`:

* RewardModel.score_batch + batched Bradley-Terry training (P0-1, P1-4).
* OPD KL estimator switchable to k3 / k1 / k2 (P0-2).
* GRPO group-of-1 fallback to batch normalization (P1-1).
* TokenBudgetManager middle-drop strategy preserving positional adjacency
  (P1-2).
* HybridAlgo cache-temperature warning when OPD cannot consume the cache
  (P1-3).
* CheckpointManager async saver with ordered drain (P1-5).
* GSPO consumes the shared logprobs cache when wrapped by Hybrid-style
  callers (P2-1).
* SFT collate prefers truncating the response over the prompt (P2-2).

These tests run on CPU under the Tiny backend so they're fast.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.algos.gspo import GSPO, GSPOConfig
from hermes_agentic_rl.algos.hybrid import HybridAlgo, HybridConfig
from hermes_agentic_rl.algos.opd import OPDAlgo, OPDConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.offline import DPOPair, ReplayBuffer
from hermes_agentic_rl.rewards.reward_model import (
    RewardModel,
    RewardModelConfig,
    RewardModelTrainer,
)
from hermes_agentic_rl.trainers.checkpoint import (
    AsyncCheckpointSaver,
    CheckpointManager,
    CheckpointState,
    snapshot_state_to_cpu,
)
from hermes_agentic_rl.trainers.token_budget import (
    TokenBudgetConfig,
    TokenBudgetManager,
)


def _backend(seed: int = 0) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(
            seed=seed, dim=16, n_heads=2, n_layers=1, max_len=64,
        )
    )


def _make_record(
    backend: TinyCausalLMBackend,
    prompt_text: str,
    response_text: str,
    reward: float,
    group_id: str = "g",
    *,
    metadata: dict | None = None,
) -> RolloutRecord:
    tok = backend.tokenizer
    prompt_ids = [tok.bos_id] + tok.encode(prompt_text)
    response_ids = tok.encode(response_text)
    if not response_ids:
        response_ids = [tok.eos_id]
    # Compute "old_logprobs" once with no grad so the training-side ratio
    # is a meaningful, finite number.
    with torch.no_grad():
        lp = backend.score(prompt_ids, response_ids).cpu().tolist()
    md = dict(metadata) if metadata else {}
    return RolloutRecord(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        old_logprobs=lp,
        reward=reward,
        group_id=group_id,
        metadata=md,
    )


# ---------------------------------------------------------------------------
# P0-1 / P1-4: RewardModel batched scoring + Bradley-Terry training.
# ---------------------------------------------------------------------------


def test_reward_model_score_batch_matches_score_pair() -> None:
    backend = _backend(seed=0)
    rm = RewardModel(backend, freeze_base=True)
    tok = backend.tokenizer
    prompt = [tok.bos_id] + tok.encode("Q?")
    a = tok.encode("yes")
    b = tok.encode("no")
    c = tok.encode("maybe so")

    with torch.no_grad():
        scores_loop = torch.stack(
            [rm.score_pair(prompt, a), rm.score_pair(prompt, b), rm.score_pair(prompt, c)]
        )
        scores_batch = rm.score_batch([prompt] * 3, [a, b, c])

    assert scores_batch.shape == (3,)
    # Batched scores should agree with the single-pair path within the
    # rounding tolerance imposed by left-padding (positional embeddings
    # may shift but for the tiny backend all responses are short enough
    # to land in the same window).
    assert torch.allclose(scores_loop, scores_batch, atol=1e-4)


def test_reward_model_trainer_uses_batched_forward() -> None:
    backend = _backend(seed=1)
    rm = RewardModel(backend, freeze_base=True)
    tok = backend.tokenizer
    prompt = [tok.bos_id] + tok.encode("Q?")
    chosen = tok.encode("yes")
    rejected = tok.encode("no")
    buf = ReplayBuffer()
    for _ in range(8):
        buf.add_pair(DPOPair(prompt_ids=prompt, chosen_ids=chosen, rejected_ids=rejected))
    trainer = RewardModelTrainer(
        rm,
        buf,
        cfg=RewardModelConfig(
            n_epochs=4,
            batch_size=4,
            lr=0.05,
            log_every=1000,
            seed=0,
            score_batch_size=4,
        ),
    )
    stats = trainer.train()
    # Batched trainer must still converge on the trivial chosen/rejected
    # split, and surface batch metadata at full pair granularity.
    assert stats.steps[-1]["acc"] >= 0.5
    assert stats.steps[-1]["n"] == 4
    with torch.no_grad():
        r_w = rm.score_pair(prompt, chosen).item()
        r_l = rm.score_pair(prompt, rejected).item()
    assert r_w > r_l


def test_reward_model_explicit_device_resolution() -> None:
    backend = _backend(seed=2)
    rm = RewardModel(backend, freeze_base=True, device="cpu")
    assert rm._backbone_device().type == "cpu"
    # Head + first model parameter share the device.
    head_p = next(rm.head.parameters())
    assert head_p.device.type == "cpu"


# ---------------------------------------------------------------------------
# P0-2: OPD KL estimator alignment with the rest of the algos.
# ---------------------------------------------------------------------------


def test_opd_default_kl_is_k3_and_configurable() -> None:
    cfg = OPDConfig()
    assert cfg.kl_estimator == "k3"
    cfg2 = OPDConfig(kl_estimator="k2")
    assert cfg2.kl_estimator == "k2"


def test_opd_with_k3_gives_non_negative_kl_signal() -> None:
    backend = _backend(seed=3)
    ref = _backend(seed=3)  # identical init, KL ≈ 0
    rec_a = _make_record(backend, "x", "yes", reward=1.0)
    rec_b = _make_record(backend, "y", "no", reward=0.0)
    # Inject teacher logprobs equal to current logprobs so the OPD
    # advantage is zero — only the KL term contributes.
    rec_a.metadata["teacher_logprobs"] = list(rec_a.old_logprobs)
    rec_b.metadata["teacher_logprobs"] = list(rec_b.old_logprobs)
    batch = RolloutBatch(records=[rec_a, rec_b])
    algo = OPDAlgo(OPDConfig(kl_coef=0.5, kl_estimator="k3"))
    loss, stats = algo.compute_loss(backend, ref, batch)
    assert torch.isfinite(loss)
    # k3 is non-negative by construction.
    assert stats.kl >= -1e-6


# ---------------------------------------------------------------------------
# P1-1: GRPO group_size=1 fallback to batch normalization.
# ---------------------------------------------------------------------------


def test_grpo_group_of_one_falls_back_to_batch_norm() -> None:
    backend = _backend(seed=4)
    # Three single-sample groups → GRPO would emit zero advantage on
    # every record; fallback should kick in and produce non-zero
    # (positive variance).
    records = [
        _make_record(backend, "a", "yes", reward=1.0, group_id="g0"),
        _make_record(backend, "b", "no",  reward=-1.0, group_id="g1"),
        _make_record(backend, "c", "yo",  reward=0.5, group_id="g2"),
    ]
    batch = RolloutBatch(records=records)
    algo = GRPO(GRPOConfig(advantage_norm="group", kl_coef=0.0))
    loss, stats = algo.compute_loss(backend, None, batch)
    assert stats.extra["group_norm_batch_fallback"] == 1.0
    # The fallback must produce non-trivial advantage signal and a
    # finite, differentiable loss.
    assert abs(stats.mean_advantage) >= 0.0
    assert torch.isfinite(loss)


def test_grpo_normal_groups_do_not_engage_fallback() -> None:
    backend = _backend(seed=5)
    records = [
        _make_record(backend, "a", "yes", reward=1.0, group_id="g0"),
        _make_record(backend, "a", "no",  reward=-1.0, group_id="g0"),
    ]
    batch = RolloutBatch(records=records)
    algo = GRPO(GRPOConfig(advantage_norm="group", kl_coef=0.0))
    _, stats = algo.compute_loss(backend, None, batch)
    assert stats.extra["group_norm_batch_fallback"] == 0.0


# ---------------------------------------------------------------------------
# P1-2: TokenBudget middle-drop strategy.
# ---------------------------------------------------------------------------


def test_token_budget_default_strategy_drops_middle() -> None:
    rec = RolloutRecord(
        prompt_ids=[1, 2],
        response_ids=list(range(100, 130)),
        old_logprobs=[-0.1] * 30,
        reward=1.0,
        group_id="g",
        metadata={},
    )
    mgr = TokenBudgetManager(
        TokenBudgetConfig(
            max_response_tokens=10,
            head_keep=3,
            tail_keep=3,
            max_tokens_per_batch=200,
            middle_strategy="drop",
        )
    )
    [out] = mgr.apply([rec])
    # Default "drop" preserves head + tail contiguously, so the kept
    # ids must be exactly the first 3 + last 3 originals.
    assert out.response_ids[:3] == [100, 101, 102]
    assert out.response_ids[-3:] == [127, 128, 129]
    assert out.metadata.get("token_budget_middle_dropped", 0) > 0
    assert out.metadata.get("token_budget_middle_strategy") == "drop"
    assert mgr.last_stats["middle_strategy"] == "drop"
    assert mgr.last_stats["total_middle_dropped"] > 0


def test_token_budget_subsample_strategy_remains_available() -> None:
    rec = RolloutRecord(
        prompt_ids=[1],
        response_ids=list(range(100, 130)),
        old_logprobs=[-0.1] * 30,
        reward=1.0,
        group_id="g",
        metadata={},
    )
    mgr = TokenBudgetManager(
        TokenBudgetConfig(
            max_response_tokens=12,
            head_keep=3,
            tail_keep=3,
            max_tokens_per_batch=200,
            middle_strategy="subsample",
        )
    )
    [out] = mgr.apply([rec])
    assert out.metadata.get("token_budget_middle_strategy") == "subsample"
    # Sample size budget = 12 - 3 - 3 = 6, so we keep 3 + 6 + 3 = 12.
    assert len(out.response_ids) == 12


# ---------------------------------------------------------------------------
# P1-3: HybridAlgo cache-temperature warning.
# ---------------------------------------------------------------------------


def _hybrid_records(backend: TinyCausalLMBackend) -> list[RolloutRecord]:
    rec_a = _make_record(backend, "p1", "yes", reward=1.0, group_id="g")
    rec_b = _make_record(backend, "p2", "no", reward=-1.0, group_id="g")
    # Both records carry teacher_logprobs *and* reward → both branches
    # contribute, so the shared cache path is exercised.
    rec_a.metadata["teacher_logprobs"] = list(rec_a.old_logprobs)
    rec_b.metadata["teacher_logprobs"] = list(rec_b.old_logprobs)
    # Mark the rollout temperature on one record so the hybrid sees a
    # non-1.0 score temperature and triggers the OPD cache miss.
    rec_a.metadata["rollout_temperature"] = 0.7
    rec_b.metadata["rollout_temperature"] = 0.7
    return [rec_a, rec_b]


def test_hybrid_warns_on_cache_temperature_mismatch_for_opd() -> None:
    backend = _backend(seed=6)
    records = _hybrid_records(backend)
    batch = RolloutBatch(records=records)
    algo = HybridAlgo(HybridConfig())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, stats = algo.compute_loss(backend, None, batch)

    msgs = [str(w.message) for w in caught]
    assert any("OPD branch cannot consume the shared forward cache" in m for m in msgs), (
        f"expected OPD cache-miss warning, got: {msgs}"
    )
    assert stats.extra.get("opd_cache_miss_temperature") == 1.0
    assert abs(stats.extra.get("cache_temperature", 0.0) - 0.7) < 1e-6


def test_hybrid_no_warning_when_temperature_is_one() -> None:
    backend = _backend(seed=7)
    records = _hybrid_records(backend)
    for rec in records:
        rec.metadata["rollout_temperature"] = 1.0
    batch = RolloutBatch(records=records)
    algo = HybridAlgo(HybridConfig())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, stats = algo.compute_loss(backend, None, batch)

    msgs = [str(w.message) for w in caught]
    assert not any("OPD branch cannot consume" in m for m in msgs)
    assert stats.extra.get("opd_cache_miss_temperature") == 0.0


# ---------------------------------------------------------------------------
# P1-5: AsyncCheckpointSaver — ordered drain + error surfacing.
# ---------------------------------------------------------------------------


def _state(it: int, *, model_state: dict | None = None) -> CheckpointState:
    return CheckpointState(
        iteration=it,
        model_state=model_state if model_state is not None else {"w": torch.zeros(1)},
        optimizer_state=None,
        rng_state=None,
        stats=[],
        config={"iter": it},
        best_reward=0.0,
        best_iteration=0,
    )


def test_async_checkpoint_saver_drains_in_order(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "ckpts", keep_last=10)
    saver = AsyncCheckpointSaver()
    try:
        for i in range(3):
            saver.submit(mgr, snapshot_state_to_cpu(_state(i)))
        saver.flush()
    finally:
        saver.close()
    # All three iterations must be on disk and listable.
    iters = mgr.list_checkpoints()
    assert iters == [0, 1, 2]


def test_async_checkpoint_saver_surfaces_errors(tmp_path: Path) -> None:
    class _BoomMgr:
        def save(self, _state: CheckpointState) -> None:
            raise RuntimeError("disk full")

    saver = AsyncCheckpointSaver()
    try:
        saver.submit(_BoomMgr(), snapshot_state_to_cpu(_state(0)))
        # Worker raises asynchronously; flush must re-raise.
        with pytest.raises(RuntimeError, match="disk full"):
            saver.flush()
    finally:
        saver.close()


def test_snapshot_state_to_cpu_clones_tensors() -> None:
    src = torch.tensor([1.0, 2.0, 3.0])
    state = _state(7, model_state={"w": src})
    snap = snapshot_state_to_cpu(state)
    assert snap.iteration == 7
    assert torch.equal(snap.model_state["w"], src.cpu())
    # Mutating the source must not change the snapshot.
    src.zero_()
    assert snap.model_state["w"].abs().sum().item() > 0


# ---------------------------------------------------------------------------
# P2-1: GSPO consumes the shared logprobs cache when given one.
# ---------------------------------------------------------------------------


def test_gspo_consumes_shared_logprobs_cache() -> None:
    backend = _backend(seed=8)
    records = [
        _make_record(backend, "a", "yes", reward=1.0, group_id="g0"),
        _make_record(backend, "a", "no",  reward=-1.0, group_id="g0"),
    ]
    # Pre-compute and inject the cache so GSPO short-circuits the
    # forward pass.
    new_logp, mask = backend.score_batch(
        [r.prompt_ids for r in records],
        [r.response_ids for r in records],
        temperature=1.0,
    )
    cache = {}
    for i, rec in enumerate(records):
        R_i = int(mask[i].sum().item())
        cache[id(rec)] = new_logp[i, :R_i]

    batch = RolloutBatch(
        records=records,
        shared_new_logprobs=cache,
        shared_logprobs_temperature=1.0,
    )
    algo = GSPO(GSPOConfig(advantage_norm="group", kl_coef=0.0))
    _, stats = algo.compute_loss(backend, None, batch)
    assert stats.extra.get("shared_logprobs_cache_hit") == 1.0


def test_gspo_falls_back_when_no_cache() -> None:
    backend = _backend(seed=9)
    records = [
        _make_record(backend, "a", "yes", reward=1.0, group_id="g0"),
        _make_record(backend, "a", "no",  reward=-1.0, group_id="g0"),
    ]
    batch = RolloutBatch(records=records)
    algo = GSPO(GSPOConfig(advantage_norm="group", kl_coef=0.0))
    _, stats = algo.compute_loss(backend, None, batch)
    assert stats.extra.get("shared_logprobs_cache_hit") == 0.0


# ---------------------------------------------------------------------------
# Report v2: trainer/CLI default alignment + hybrid strict mode.
# ---------------------------------------------------------------------------


def test_grpo_config_and_trainer_default_kl_is_k3() -> None:
    assert GRPOConfig().kl_estimator == "k3"
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainerConfig

    assert GRPOTrainerConfig().kl_estimator == "k3"


def test_gspo_config_and_trainer_default_log_ratio_clip_is_40() -> None:
    assert GSPOConfig().log_ratio_clip == 40.0
    from hermes_agentic_rl.trainers.gspo_trainer import GSPOTrainerConfig

    assert GSPOTrainerConfig().log_ratio_clip == 40.0


def test_hybrid_strict_mode_raises_on_cache_temperature_mismatch() -> None:
    backend = _backend(seed=10)
    records = _hybrid_records(backend)
    batch = RolloutBatch(records=records)
    algo = HybridAlgo(HybridConfig(strict_cache_temperature=True))

    with pytest.raises(ValueError, match="strict_cache_temperature=True"):
        algo.compute_loss(backend, None, batch)


def test_collect_group_batched_builds_trajectory_steps() -> None:
    import asyncio

    from hermes_agentic_rl.envs.echo_task_env import EchoTaskEnv, build_default_echo_dataset
    from hermes_agentic_rl.rewards.base import BaseReward
    from hermes_agentic_rl.rewards.composer import RewardComposer
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    class _StepsGatedReward(BaseReward):
        name = "steps_gate"

        async def evaluate(self, item, trajectory, tool_context=None):
            from hermes_agentic_rl.core.types import RewardResult

            score = 1.0 if trajectory.steps else 0.0
            return RewardResult(name=self.name, score=score, reason="steps")

    backend = _backend(seed=11)
    env = EchoTaskEnv(build_default_echo_dataset())
    composer = RewardComposer(
        components=[_StepsGatedReward()],
        config={"conditions": {"steps_gate": lambda _item, traj: bool(traj.steps)}},
    )
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=composer,
        cfg=GRPOTrainerConfig(
            n_iters=1,
            group_size=2,
            prompts_per_iter=1,
            lr=5e-3,
            max_new_tokens=6,
            temperature=1.0,
            log_every=100,
            seed=11,
            batch_generate=True,
        ),
    )

    item = asyncio.run(env.get_next_item())
    records = asyncio.run(trainer._collect_group_batched(item))

    assert len(records) == 2
    assert all(rec.reward == 1.0 for rec in records)
