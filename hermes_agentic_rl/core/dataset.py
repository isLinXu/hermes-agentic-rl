from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class DatasetSplits:
    train: list[Any]
    val: list[Any]
    test: list[Any]

    def as_tuple(self) -> tuple[list[Any], list[Any], list[Any]]:
        return self.train, self.val, self.test


def _validate_ratio(name: str, value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError(f"{name} must be a non-negative finite ratio")
    return ratio


def _ratio_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    return max(1, round(total * ratio))


def _floor_ratio_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    return max(1, int(total * ratio))


def split_dataset(
    items: Sequence[T],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float | None = None,
    seed: int | None = None,
    shuffle: bool = True,
) -> DatasetSplits:
    """Split a sequence into train/validation/test partitions.

    For tiny datasets fewer than three rows, all rows stay in train so callers
    never emit empty train data while pretending to have meaningful holdouts.
    """
    rows = list(items)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if len(rows) < 3:
        return DatasetSplits(train=rows, val=[], test=[])

    train_ratio = _validate_ratio("train_ratio", train_ratio)
    val_ratio = _validate_ratio("val_ratio", val_ratio)
    if test_ratio is None:
        test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    test_ratio = _validate_ratio("test_ratio", test_ratio)
    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("at least one split ratio must be positive")

    total = len(rows)
    n_train = _floor_ratio_count(total, train_ratio)
    n_val = _floor_ratio_count(total, val_ratio)
    if n_train + n_val >= total:
        n_val = max(0, total - n_train - 1)
    n_test = max(0, total - n_train - n_val)

    return DatasetSplits(
        train=rows[:n_train],
        val=rows[n_train : n_train + n_val],
        test=rows[n_train + n_val : n_train + n_val + n_test],
    )


def _group_id(item: dict[str, Any], index: int, group_key: str) -> str:
    return str(item.get(group_key) or item.get("task_id") or index)


def select_items_by_group(
    items: Sequence[dict[str, Any]],
    *,
    split: str = "val",
    group_key: str = "source_trace_id",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select train/val/test/all items while keeping groups intact."""
    split = str(split).lower()
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be one of: train, val, test, all")

    rows = list(items)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(rows):
        groups[_group_id(item, idx, group_key)].append(idx)

    if split == "all":
        return rows, {
            "split": split,
            "group_key": group_key,
            "total_items": len(rows),
            "total_groups": len(groups),
            "selected_items": len(rows),
            "selected_groups": len(groups),
        }

    val_ratio = _validate_ratio("val_ratio", val_ratio)
    test_ratio = _validate_ratio("test_ratio", test_ratio)

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    total_groups = len(group_ids)
    n_test = min(total_groups, _ratio_count(total_groups, test_ratio))
    n_val = min(total_groups - n_test, _ratio_count(total_groups, val_ratio))
    n_train = max(0, total_groups - n_val - n_test)

    split_ids = {
        "train": set(group_ids[:n_train]),
        "val": set(group_ids[n_train : n_train + n_val]),
        "test": set(group_ids[n_train + n_val :]),
    }
    selected_ids = split_ids[split]
    selected_indices = {idx for group_id in selected_ids for idx in groups.get(group_id, [])}
    selected = [item for idx, item in enumerate(rows) if idx in selected_indices]
    return selected, {
        "split": split,
        "group_key": group_key,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_items": len(rows),
        "total_groups": total_groups,
        "train_groups": len(split_ids["train"]),
        "val_groups": len(split_ids["val"]),
        "test_groups": len(split_ids["test"]),
        "selected_items": len(selected),
        "selected_groups": len(selected_ids),
    }
