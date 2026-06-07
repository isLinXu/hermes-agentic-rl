from __future__ import annotations

import pytest

from hermes_agentic_rl.core.dataset import select_items_by_group, split_dataset


def test_split_dataset_preserves_tiny_datasets_for_train() -> None:
    splits = split_dataset([{"id": 1}, {"id": 2}], seed=1)

    assert [item["id"] for item in splits.train] == [2, 1]
    assert splits.val == []
    assert splits.test == []


def test_split_dataset_can_reuse_existing_order() -> None:
    splits = split_dataset(
        [{"id": idx} for idx in range(4)],
        train_ratio=0.5,
        val_ratio=0.25,
        shuffle=False,
    )

    assert [item["id"] for item in splits.train] == [0, 1]
    assert [item["id"] for item in splits.val] == [2]
    assert [item["id"] for item in splits.test] == [3]


def test_select_items_by_group_keeps_groups_together_with_index_fallback() -> None:
    rows = [
        {"task_id": "a-0", "source_trace_id": "a"},
        {"task_id": "a-1", "source_trace_id": "a"},
        {"task_id": "b-0", "source_trace_id": "b"},
        {"task_id": "b-1", "source_trace_id": "b"},
        {"payload": "no explicit group"},
    ]

    selected, info = select_items_by_group(
        rows,
        split="test",
        group_key="source_trace_id",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
    )

    assert info["test_groups"] == 1
    assert info["selected_groups"] == 1
    if selected and "source_trace_id" in selected[0]:
        group = selected[0]["source_trace_id"]
        assert all(row.get("source_trace_id") == group for row in selected)


def test_select_items_by_group_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        select_items_by_group([], split="dev")
