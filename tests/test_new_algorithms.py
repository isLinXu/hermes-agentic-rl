"""Tests for new algorithms: BestOfN, RejectionSampler, FactoredGRPO,
OPDAlgo, HybridAlgo, and NextStatePRM.

These modules were implemented but had no test coverage.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# BestOfN / RejectionSampler
# ---------------------------------------------------------------------------
from hermes_agentic_rl.algos.best_of_n import (
    BestOfN,
    BestOfNConfig,
    BestOfNResult,
    PreferencePair,
    RejectionSample,
    RejectionSampler,
    RejectionSamplerConfig,
)
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.echo_task_env import EchoTaskEnv
from hermes_agentic_rl.rewards.base import BaseReward


class _SequencePolicy:
    def __init__(self, torch_module) -> None:
        self.param = torch_module.nn.Parameter(torch_module.zeros(()))
        self.torch = torch_module

    def score_batch(self, prompt_ids_list, response_ids_list, temperature=1.0):
        del prompt_ids_list, temperature
        B = len(response_ids_list)
        T = max((len(ids) for ids in response_ids_list), default=0)
        logp = self.param.expand(B, T).clone()
        mask = self.torch.zeros(B, T, dtype=self.torch.bool)
        for i, ids in enumerate(response_ids_list):
            if ids:
                mask[i, : len(ids)] = True
        return logp * mask.to(logp.dtype), mask


def test_gspo_sequence_ratio_uses_old_seq_logprob_metadata():
    torch = pytest.importorskip("torch")
    from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
    from hermes_agentic_rl.algos.gspo import GSPO, GSPOConfig

    policy = _SequencePolicy(torch)
    records = [
        RolloutRecord(
            prompt_ids=[1],
            response_ids=[2, 3],
            old_logprobs=[99.0, 99.0],
            reward=1.0,
            group_id="g",
            metadata={"old_seq_logprob": 0.0},
        ),
        RolloutRecord(
            prompt_ids=[1],
            response_ids=[4],
            old_logprobs=[99.0],
            reward=2.0,
            group_id="g",
            metadata={"old_seq_logprob": 0.0},
        ),
    ]
    algo = GSPO(GSPOConfig(advantage_norm="none", kl_coef=0.0))

    loss, stats = algo.compute_loss(policy, None, RolloutBatch(records))

    assert stats.extra["algo"] == "gspo"
    assert stats.policy_loss == pytest.approx(-1.5, abs=1e-6)
    loss.backward()
    assert policy.param.grad is not None
    assert float(policy.param.grad.item()) == pytest.approx(-2.0, abs=1e-6)


class _PassthroughReward(BaseReward):
    """Always give score 0.5 so BestOfN can score valid rollouts."""

    name = "passthrough"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(self, item: dict, traj: Trajectory, tool_context: Any) -> RewardResult:
        return RewardResult(score=0.5, weight=self.weight, name=self.name, reason="fixed")


# EchoTaskEnv items need {"instruction": "...", "target": "..."}
_ECHO_ITEMS = [
    {"task_id": "t0", "instruction": "Say: hello", "target": "hello"},
    {"task_id": "t1", "instruction": "Say: world", "target": "world"},
]


def _make_bon(n: int = 3, max_new_tokens: int = 8) -> BestOfN:
    pytest.importorskip("torch")
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=42))
    env = EchoTaskEnv(_ECHO_ITEMS)
    rm = RewardManager([_PassthroughReward()])
    return BestOfN(
        policy=backend,
        env=env,
        reward_manager=rm,
        cfg=BestOfNConfig(n=n, max_new_tokens=max_new_tokens, seed_base=0),
    )


def test_best_of_n_result_shape():
    bon = _make_bon(n=3)
    result = asyncio.run(bon.best(_ECHO_ITEMS[0]))
    assert isinstance(result, BestOfNResult)
    assert 0.0 <= result.winner_score <= 1.0
    assert len(result.all_records) <= 3
    assert len(result.all_scores) == len(result.all_records)


def test_best_of_n_winner_is_highest_score():
    bon = _make_bon(n=4)
    result = asyncio.run(bon.best(_ECHO_ITEMS[0]))
    if len(result.all_scores) >= 2:
        assert result.all_scores[0] >= result.all_scores[-1]
    assert result.winner_score == (result.all_scores[0] if result.all_scores else 0.0)


def test_best_of_n_preference_pairs_ordered():
    bon = _make_bon(n=4)
    pairs = asyncio.run(bon.preference_pairs(_ECHO_ITEMS[0]))
    for pair in pairs:
        assert isinstance(pair, PreferencePair)
        assert pair.chosen_score >= pair.rejected_score
        assert isinstance(pair.prompt_ids, list)
        assert isinstance(pair.chosen_ids, list)
        assert isinstance(pair.rejected_ids, list)


def test_best_of_n_empty_valid_fallback():
    """When ALL rollouts produce empty responses, BestOfN returns fallback."""
    pytest.importorskip("torch")
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=42))
    env = EchoTaskEnv(_ECHO_ITEMS)
    rm = RewardManager([_PassthroughReward()])
    bon = BestOfN(
        policy=backend,
        env=env,
        reward_manager=rm,
        cfg=BestOfNConfig(n=2, max_new_tokens=1, min_response_tokens=9999),
    )
    result = asyncio.run(bon.best(_ECHO_ITEMS[0]))
    assert result.winner_score == 0.0
    assert result.all_records == []


def test_rejection_sampler_threshold():
    bon = _make_bon(n=4)
    sampler = RejectionSampler(bon, cfg=RejectionSamplerConfig(threshold=0.0))
    samples = asyncio.run(sampler.sample(_ECHO_ITEMS[0]))
    for s in samples:
        assert isinstance(s, RejectionSample)
        assert s.score >= 0.0


def test_rejection_sampler_high_threshold_rejects_all():
    bon = _make_bon(n=3)
    sampler = RejectionSampler(bon, cfg=RejectionSamplerConfig(threshold=2.0))
    samples = asyncio.run(sampler.sample(_ECHO_ITEMS[0]))
    assert samples == []


def test_rejection_sampler_percentile_mode():
    bon = _make_bon(n=4)
    sampler = RejectionSampler(
        bon, cfg=RejectionSamplerConfig(threshold=0.5, percentile_mode=True, max_accept=2)
    )
    samples = asyncio.run(sampler.sample(_ECHO_ITEMS[0]))
    assert len(samples) <= 2


def test_rejection_sampler_build_sft_dataset():
    bon = _make_bon(n=2)
    sampler = RejectionSampler(bon, cfg=RejectionSamplerConfig(threshold=0.0))
    dataset = asyncio.run(sampler.build_sft_dataset(_ECHO_ITEMS[:2]))
    assert isinstance(dataset, list)


# ---------------------------------------------------------------------------
# FactoredGRPO
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.factored import (
    DEFAULT_HEADS,
    FactoredConfig,
    FactoredGRPO,
    _normalize_weights,
)
from hermes_agentic_rl.algos.grpo import GRPOConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend


def _make_tiny() -> TinyCausalLMBackend:
    return TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))


def _make_factored_records(b: TinyCausalLMBackend, n: int = 4) -> list[RolloutRecord]:
    records = []
    prompt = b.tokenizer.encode("factored test")
    for i in range(n):
        out = b.generate(prompt, max_new_tokens=3, seed=i)
        factored_meta = {
            "action_type": {
                "response_ids": list(out.response_ids),
                "old_logprobs": list(out.logprobs),
            },
            "tool_id": {
                "response_ids": list(out.response_ids),
                "old_logprobs": list(out.logprobs),
            },
        }
        records.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=float(i) / n,
                group_id="g0",
                metadata={"factored": factored_meta},
            )
        )
    return records


def test_normalize_weights_sums_to_one():
    w = _normalize_weights({"a": 1.0, "b": 3.0})
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_normalize_weights_all_zero():
    w = _normalize_weights({"a": 0.0})
    assert abs(w["a"] - 1.0) < 1e-9


def test_factored_grpo_with_factored_records():
    b = _make_tiny()
    records = _make_factored_records(b, n=4)
    algo = FactoredGRPO(grpo_cfg=GRPOConfig(kl_coef=0.0))
    loss, stats = algo.compute_loss(b, None, RolloutBatch(records))

    assert stats.n_records == 4
    assert stats.extra["n_factored"] == 4
    assert stats.extra["n_flat_fallback"] == 0
    assert stats.extra["algo"] == "factored_grpo"
    if loss.requires_grad:
        loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in b.trainable_parameters()
    )
    assert has_grad


def test_factored_grpo_flat_fallback_for_missing_metadata():
    b = _make_tiny()
    prompt = b.tokenizer.encode("flat test")
    records = [
        RolloutRecord(
            prompt_ids=list(prompt),
            response_ids=list(b.generate(prompt, max_new_tokens=3, seed=i).response_ids),
            old_logprobs=list(b.generate(prompt, max_new_tokens=3, seed=i).logprobs),
            reward=float(i),
            group_id="g0",
        )
        for i in range(3)
    ]
    algo = FactoredGRPO()
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.extra["n_flat_fallback"] == 3
    assert stats.extra["n_factored"] == 0


def test_factored_grpo_empty_batch():
    b = _make_tiny()
    algo = FactoredGRPO()
    loss, stats = algo.compute_loss(b, None, RolloutBatch([]))
    assert stats.n_records == 0
    assert float(loss.item()) == 0.0


def test_factored_grpo_mixed_records():
    """Some records have factored metadata, some don't."""
    b = _make_tiny()
    prompt = b.tokenizer.encode("mixed")
    records: list[RolloutRecord] = []
    for i in range(4):
        out = b.generate(prompt, max_new_tokens=3, seed=i)
        meta: dict[str, Any] = {}
        if i % 2 == 0:
            meta["factored"] = {
                "action_type": {
                    "response_ids": list(out.response_ids),
                    "old_logprobs": list(out.logprobs),
                }
            }
        records.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=float(i),
                group_id="g0",
                metadata=meta,
            )
        )
    algo = FactoredGRPO()
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.extra["n_factored"] == 2
    assert stats.extra["n_flat_fallback"] == 2


# ---------------------------------------------------------------------------
# OPDAlgo
# ---------------------------------------------------------------------------

from hermes_agentic_rl.algos.opd import (
    OPDAlgo,
    OPDConfig,
    OPDJudge,
    extract_hint_text,
    wrap_hint,
)


def test_wrap_and_extract_hint():
    wrapped = wrap_hint("Use tool X")
    assert "[HINT_START]" in wrapped
    assert extract_hint_text(wrapped) == "Use tool X"


def test_extract_hint_text_missing():
    assert extract_hint_text("no hints here") is None


def test_opd_judge_extracts_hint():
    async def _judge(resp: str, ns: str) -> str:
        return f"[HINT_START]fix: {ns}[HINT_END]"

    judge = OPDJudge(_judge)
    hint = asyncio.run(judge.extract_hint("my response", "error msg"))
    assert hint is not None and "fix:" in hint


def test_opd_judge_returns_none_for_no_hint():
    async def _judge(resp: str, ns: str) -> None:
        return None

    judge = OPDJudge(_judge)
    hint = asyncio.run(judge.extract_hint("resp", "state"))
    assert hint is None


def _make_opd_records(
    b: TinyCausalLMBackend, n: int = 3, with_hints: bool = True
) -> list[RolloutRecord]:
    prompt = b.tokenizer.encode("opd test")
    records = []
    for i in range(n):
        out = b.generate(prompt, max_new_tokens=4, seed=i)
        meta: dict[str, Any] = {}
        if with_hints:
            meta["teacher_logprobs"] = [lp + 0.1 for lp in out.logprobs]
        records.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=float(i),
                group_id="g0",
                metadata=meta,
            )
        )
    return records


def test_opd_algo_with_hints_produces_gradient():
    b = _make_tiny()
    records = _make_opd_records(b, n=3, with_hints=True)
    algo = OPDAlgo(OPDConfig(kl_coef=0.0))
    loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.n_records == 3
    if loss.requires_grad:
        loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in b.trainable_parameters()
    )
    assert has_grad


def test_opd_algo_skip_missing_hints():
    b = _make_tiny()
    records = _make_opd_records(b, n=3, with_hints=False)
    algo = OPDAlgo(OPDConfig(skip_missing_hints=True))
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.extra.get("n_opd_records", 0) == 0


def test_opd_algo_fallback_without_hints():
    b = _make_tiny()
    records = _make_opd_records(b, n=3, with_hints=False)
    algo = OPDAlgo(OPDConfig(skip_missing_hints=False, kl_coef=0.0))
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.n_records == 3


def test_opd_algo_empty_batch():
    b = _make_tiny()
    algo = OPDAlgo()
    _loss, stats = algo.compute_loss(b, None, RolloutBatch([]))
    assert stats.n_records == 0


# ---------------------------------------------------------------------------
# HybridAlgo
# ---------------------------------------------------------------------------

from hermes_agentic_rl.algos.hybrid import HybridAlgo, HybridConfig


def _make_hybrid_records(
    b: TinyCausalLMBackend,
    n_grpo: int = 3,
    n_opd: int = 2,
) -> list[RolloutRecord]:
    prompt = b.tokenizer.encode("hybrid test")
    records: list[RolloutRecord] = []

    for i in range(n_grpo):
        out = b.generate(prompt, max_new_tokens=3, seed=i)
        records.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=float(i + 1),
                group_id="g0",
            )
        )

    for j in range(n_opd):
        out = b.generate(prompt, max_new_tokens=3, seed=n_grpo + j)
        records.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=0.0,
                group_id="g1",
                metadata={"teacher_logprobs": [lp + 0.05 for lp in out.logprobs]},
            )
        )

    return records


def test_hybrid_algo_partitions_records():
    b = _make_tiny()
    records = _make_hybrid_records(b, n_grpo=3, n_opd=2)
    algo = HybridAlgo()
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.n_records == 5
    assert "n_grpo" in stats.extra and "n_opd" in stats.extra
    assert stats.extra["n_grpo"] >= 3


def test_hybrid_algo_gradient_flows():
    b = _make_tiny()
    records = _make_hybrid_records(b, n_grpo=3, n_opd=2)
    algo = HybridAlgo()
    loss, _stats = algo.compute_loss(b, None, RolloutBatch(records))
    if loss.requires_grad:
        loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in b.trainable_parameters()
    )
    assert has_grad


def test_hybrid_algo_grpo_only_when_no_hints():
    b = _make_tiny()
    prompt = b.tokenizer.encode("grpo only")
    records = [
        RolloutRecord(
            prompt_ids=list(prompt),
            response_ids=list(b.generate(prompt, max_new_tokens=3, seed=i).response_ids),
            old_logprobs=list(b.generate(prompt, max_new_tokens=3, seed=i).logprobs),
            reward=float(i + 1),
            group_id="g0",
        )
        for i in range(3)
    ]
    algo = HybridAlgo()
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.extra["n_opd"] == 0
    assert stats.extra["n_grpo"] == 3


def test_hybrid_algo_empty_batch():
    b = _make_tiny()
    algo = HybridAlgo()
    loss, stats = algo.compute_loss(b, None, RolloutBatch([]))
    assert stats.n_records == 0
    assert float(loss.item()) == 0.0


def test_hybrid_algo_w_rl_zero():
    b = _make_tiny()
    records = _make_hybrid_records(b, n_grpo=3, n_opd=2)
    algo = HybridAlgo(HybridConfig(w_rl=0.0, w_opd=1.0))
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.extra["n_grpo"] == 0


def test_hybrid_algo_w_opd_zero():
    b = _make_tiny()
    records = _make_hybrid_records(b, n_grpo=3, n_opd=2)
    algo = HybridAlgo(HybridConfig(w_rl=1.0, w_opd=0.0))
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.extra["n_opd"] == 0


# ---------------------------------------------------------------------------
# NextStatePRM
# ---------------------------------------------------------------------------

from hermes_agentic_rl.rewards.next_state_prm import (
    BAD,
    GOOD,
    NEUTRAL,
    NextStateJudge,
    NextStatePRM,
    NextStatePRMComponent,
    NextStatePRMConfig,
    PRMVote,
)


def _make_judge(vote: int = GOOD, hint: str | None = None) -> NextStateJudge:
    hint_block = f"[HINT_START]{hint}[HINT_END]" if hint else ""

    async def _fn(resp: str, ns: str) -> str:
        label = {GOOD: "GOOD", BAD: "BAD", NEUTRAL: "NEUTRAL"}[vote]
        return f"{label} {hint_block}"

    return NextStateJudge(_fn)


def test_next_state_judge_parse_good():
    assert asyncio.run(_make_judge(GOOD).judge("r", "s")).vote == GOOD


def test_next_state_judge_parse_bad():
    assert asyncio.run(_make_judge(BAD).judge("r", "s")).vote == BAD


def test_next_state_judge_parse_neutral():
    assert asyncio.run(_make_judge(NEUTRAL).judge("r", "s")).vote == NEUTRAL


def test_next_state_judge_extracts_hint():
    vote = asyncio.run(_make_judge(BAD, hint="Try again with X").judge("r", "s"))
    assert vote.hint == "Try again with X"


def test_next_state_prm_majority_good():
    prm = NextStatePRM(_make_judge(GOOD), NextStatePRMConfig(m=3))
    score, _ = asyncio.run(prm.score("resp", "state"))
    assert score == GOOD


def test_next_state_prm_majority_bad():
    prm = NextStatePRM(_make_judge(BAD), NextStatePRMConfig(m=3))
    score, _ = asyncio.run(prm.score("resp", "state"))
    assert score == BAD


def test_next_state_prm_tie():
    """1 GOOD + 2 BAD with m=3 → BAD majority (2 ≥ ceil(3/2)=2)."""
    call_count = 0

    async def _fn(resp: str, ns: str) -> str:
        nonlocal call_count
        call_count += 1
        return "GOOD" if call_count == 1 else "BAD"

    prm = NextStatePRM(NextStateJudge(_fn), NextStatePRMConfig(m=3))
    score, _ = asyncio.run(prm.score("resp", "state"))
    assert score == BAD


def test_next_state_prm_hint_passthrough():
    prm = NextStatePRM(
        _make_judge(GOOD, hint="Use tool call format"),
        NextStatePRMConfig(m=1, hint_min_length=5),
    )
    score, hint = asyncio.run(prm.score("resp", "state"))
    assert score == GOOD
    assert hint is not None and "tool call" in hint


def test_next_state_prm_missing_next_state():
    prm = NextStatePRM(_make_judge(GOOD), NextStatePRMConfig(at_least_one=True))
    score, _hint = asyncio.run(prm.score("resp", None))
    assert score == 0


def _make_traj(next_state: str | None = None) -> Trajectory:
    meta: dict[str, Any] = {}
    if next_state is not None:
        meta = {"runtime": {"next_state": next_state}}
    return Trajectory(
        task_id="t0",
        prompt="test prompt",
        steps=[],
        final_output="my response",
        finished_naturally=True,
        turns_used=1,
        metadata=meta,
    )


def test_next_state_prm_component_good():
    comp = NextStatePRMComponent(
        NextStatePRM(_make_judge(GOOD), NextStatePRMConfig(m=1)), weight=1.0
    )
    result = asyncio.run(comp.evaluate({}, _make_traj("feedback"), None))
    assert result.score == float(GOOD)
    assert result.name == "next_state_prm"


def test_next_state_prm_component_bad():
    comp = NextStatePRMComponent(
        NextStatePRM(_make_judge(BAD), NextStatePRMConfig(m=1)), weight=1.0
    )
    result = asyncio.run(comp.evaluate({}, _make_traj("wrong"), None))
    assert result.score == float(BAD)


def test_next_state_prm_component_no_next_state():
    """Trajectory with no runtime.next_state → at_least_one fallback → 0."""
    comp = NextStatePRMComponent(
        NextStatePRM(_make_judge(GOOD), NextStatePRMConfig(m=1, at_least_one=True)),
        weight=1.0,
    )
    result = asyncio.run(comp.evaluate({}, _make_traj(None), None))
    assert result.score == 0.0
