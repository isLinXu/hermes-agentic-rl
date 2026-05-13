from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    items: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            item.setdefault("task_id", f"generated-{index}")
            items.append(item)
    return items
