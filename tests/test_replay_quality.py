from __future__ import annotations

from hermes_agentic_rl.collectors.replay_quality import apply_record_quality_filters


def test_quality_filters_reject_reward_bounds_and_low_diversity() -> None:
    records = [
        {
            "prompt_ids": [1, 2],
            "response_ids": [3, 4, 5],
            "reward": 0.8,
            "metadata": {"session_id": "good"},
        },
        {
            "prompt_ids": [1, 2],
            "response_ids": [3, 4, 5],
            "reward": -0.1,
            "metadata": {"session_id": "low-reward"},
        },
        {
            "prompt_ids": [1, 2],
            "response_ids": [7, 7, 7, 7],
            "reward": 0.6,
            "metadata": {"session_id": "repetitive"},
        },
    ]

    valid, rejected = apply_record_quality_filters(
        records,
        {
            "min_reward": 0.0,
            "min_unique_response_tokens": 2,
            "max_response_token_repetition": 0.75,
        },
    )

    assert [row["metadata"]["session_id"] for row in valid] == ["good"]
    assert {row["reason"] for row in rejected} == {
        "reward_below_min",
        "response_low_token_diversity",
    }


def test_quality_filters_require_nested_metadata_keys() -> None:
    records = [
        {
            "prompt_ids": [1, 2],
            "response_ids": [3, 4],
            "reward": 0.2,
            "metadata": {"session_id": "sess-1", "source_turn": {"assistant_message": "ok"}},
        },
        {
            "prompt_ids": [1, 2],
            "response_ids": [3, 4],
            "reward": 0.2,
            "metadata": {"session_id": "sess-2", "source_turn": {}},
        },
    ]

    valid, rejected = apply_record_quality_filters(
        records,
        {"require_metadata_keys": ["session_id", "source_turn.assistant_message"]},
    )

    assert [row["metadata"]["session_id"] for row in valid] == ["sess-1"]
    assert [row["reason"] for row in rejected] == ["missing_required_metadata"]
