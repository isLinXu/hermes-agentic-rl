from __future__ import annotations

from hermes_agentic_rl.algos.base import RolloutRecord
from hermes_agentic_rl.trainers.token_budget import TokenBudgetConfig, TokenBudgetManager


class _CharTokenizer:
    vocab_size = 256
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [ord(ch) for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if i > 2)


def _record(text: str) -> RolloutRecord:
    ids = _CharTokenizer().encode(text)
    return RolloutRecord(
        prompt_ids=[1, 2, 3],
        response_ids=ids,
        old_logprobs=[-0.1] * len(ids),
        reward=1.0,
        group_id="g",
        metadata={},
    )


def test_token_budget_truncates_and_keeps_logprobs_aligned() -> None:
    rec = _record("abcdefghijklmnopqrstuvwxyz")
    mgr = TokenBudgetManager(
        TokenBudgetConfig(
            max_response_tokens=12,
            head_keep=4,
            tail_keep=4,
            max_tokens_per_batch=100,
        ),
        tokenizer=_CharTokenizer(),
    )

    [out] = mgr.apply([rec])

    assert len(out.response_ids) <= 12
    assert len(out.old_logprobs) == len(out.response_ids)
    assert out.metadata["token_budget_truncated"] is True
    assert mgr.last_stats["n_truncated"] == 1


def test_token_budget_tail_does_not_start_inside_tool_call() -> None:
    tokenizer = _CharTokenizer()
    rec = _record('AAAA<tool_call>{"a": 1}</tool_call>BBBBBBBB')
    mgr = TokenBudgetManager(
        TokenBudgetConfig(
            max_response_tokens=24,
            head_keep=4,
            tail_keep=25,
            max_tokens_per_batch=100,
            protect_tool_call_boundaries=True,
        ),
        tokenizer=tokenizer,
    )

    [out] = mgr.apply([rec])
    decoded = tokenizer.decode(out.response_ids)

    assert '{"a": 1}</tool_call>' not in decoded
    assert decoded.endswith("BBBBBBBB")
    assert len(out.old_logprobs) == len(out.response_ids)


def test_token_budget_can_be_disabled_for_tool_call_boundary_protection() -> None:
    tokenizer = _CharTokenizer()
    text = 'AAAA<tool_call>{"a": 1}</tool_call>BBBBBBBB'
    ids = tokenizer.encode(text)
    nominal = TokenBudgetManager._safe_tail_start(
        ids,
        25,
        tokenizer=tokenizer,
        protect_tool_call_boundaries=False,
    )

    assert nominal == len(ids) - 25


def test_token_budget_empty_batch() -> None:
    mgr = TokenBudgetManager(TokenBudgetConfig(max_tokens_per_batch=10))

    assert mgr.apply([]) == []
    assert mgr.last_stats["n_orig"] == 0
