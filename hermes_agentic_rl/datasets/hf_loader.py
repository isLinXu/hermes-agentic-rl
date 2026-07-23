from __future__ import annotations

import json
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

_HF_ROWS_MAX_PAGE_SIZE = 100


def _load_hf_rows_via_server(
    repo_id: str,
    *,
    config_name: str | None,
    split: str,
    limit: int,
    shuffle: bool,
    seed: int,
    timeout: float = 30.0,
    retries: int = 2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        page_size = min(_HF_ROWS_MAX_PAGE_SIZE, limit - len(rows))
        query = {
            "dataset": repo_id,
            "split": split,
            "offset": offset,
            "length": page_size,
        }
        if config_name:
            query["config"] = config_name
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(query)
        last_error: Exception | None = None
        payload: dict[str, Any] | None = None
        for attempt in range(max(1, retries + 1)):
            try:
                with urlopen(url, timeout=timeout) as response:
                    payload = json.load(response)
                break
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    raise
                time.sleep(min(2.0**attempt, 8.0))
        if payload is None:
            raise RuntimeError(f"failed to load HF rows page: {last_error}")
        page_rows = [dict(entry["row"]) for entry in payload.get("rows", []) if "row" in entry]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


def load_hf_dataset(
    repo_id: str,
    *,
    config_name: str | None = None,
    split: str = "train",
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    streaming: bool = False,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    rows_api_only: bool = False,
    rows_api_timeout: float = 30.0,
    rows_api_retries: int = 2,
) -> list[dict[str, Any]]:
    """Load a Hugging Face dataset split into a list of plain dict rows."""
    if streaming and limit is not None:
        try:
            rows = _load_hf_rows_via_server(
                repo_id,
                config_name=config_name,
                split=split,
                limit=limit,
                shuffle=shuffle,
                seed=seed,
                timeout=rows_api_timeout,
                retries=rows_api_retries,
            )
            items: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                item = dict(row)
                item.setdefault(
                    "task_id",
                    f"{repo_id.replace('/', '__')}::{config_name or 'default'}::{split}::{index}",
                )
                items.append(item)
            if items:
                return items
        except Exception as exc:
            if rows_api_only:
                raise RuntimeError(
                    "HF rows API loading failed and `rows_api_only` is enabled; "
                    "not falling back to full parquet download."
                ) from exc

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "HF dataset loading requires either the optional `datasets` package "
            "(`pip install -e '.[data]'`) or `streaming: true` together with a "
            "finite `dataset_limit` so the rows API fallback can be used."
        ) from exc

    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
    }
    if config_name:
        kwargs["name"] = config_name
    if revision:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    dataset = load_dataset(repo_id, **kwargs)
    if shuffle and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)

    rows_iter: Iterable[dict[str, Any]] = dataset if streaming else [dict(row) for row in dataset]

    if limit is not None:
        if streaming:
            from itertools import islice

            rows_iter = islice(rows_iter, limit)
        else:
            rows_iter = list(rows_iter)[:limit]

    loaded_items: list[dict[str, Any]] = []
    for index, row in enumerate(rows_iter):
        item = dict(row)
        item.setdefault(
            "task_id",
            f"{repo_id.replace('/', '__')}::{config_name or 'default'}::{split}::{index}",
        )
        loaded_items.append(item)
    return loaded_items
