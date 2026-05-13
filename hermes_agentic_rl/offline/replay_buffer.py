"""Offline data containers and JSONL IO.

A single-sample view::

    TrainSample(prompt_ids, response_ids, reward=..., advantage=...)

A preference pair (for DPO)::

    DPOPair(prompt_ids, chosen_ids, rejected_ids)

JSONL schema (both forms supported):

    {"prompt_ids": [...], "response_ids": [...], "reward": 0.8}          # BC/AWR
    {"prompt_ids": [...], "chosen_ids": [...], "rejected_ids": [...]}    # DPO
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainSample:
    prompt_ids: list[int]
    response_ids: list[int]
    reward: float = 0.0
    advantage: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class DPOPair:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


class ReplayBuffer:
    """Simple append-only buffer with JSONL round-trip.

    Mixed storage (TrainSample + DPOPair) is supported; iterate with
    ``iter_samples()`` / ``iter_dpo_pairs()`` to get typed views.
    """

    def __init__(self) -> None:
        self.samples: list[TrainSample] = []
        self.pairs: list[DPOPair] = []

    def add_sample(self, sample: TrainSample) -> None:
        self.samples.append(sample)

    def add_pair(self, pair: DPOPair) -> None:
        self.pairs.append(pair)

    def __len__(self) -> int:
        return len(self.samples) + len(self.pairs)

    def iter_samples(self) -> Iterator[TrainSample]:
        yield from self.samples

    def iter_dpo_pairs(self) -> Iterator[DPOPair]:
        yield from self.pairs

    # --- JSONL IO ---

    def save_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as h:
            for s in self.samples:
                h.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            for pr in self.pairs:
                h.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> ReplayBuffer:
        buf = cls()
        with Path(path).open("r", encoding="utf-8") as h:
            for line in h:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "chosen_ids" in obj and "rejected_ids" in obj:
                    buf.add_pair(
                        DPOPair(
                            prompt_ids=list(obj["prompt_ids"]),
                            chosen_ids=list(obj["chosen_ids"]),
                            rejected_ids=list(obj["rejected_ids"]),
                            metadata=dict(obj.get("metadata", {})),
                        )
                    )
                else:
                    buf.add_sample(
                        TrainSample(
                            prompt_ids=list(obj["prompt_ids"]),
                            response_ids=list(obj["response_ids"]),
                            reward=float(obj.get("reward", 0.0)),
                            advantage=obj.get("advantage"),
                            metadata=dict(obj.get("metadata", {})),
                        )
                    )
        return buf

    @classmethod
    def from_samples(cls, samples: Iterable[TrainSample]) -> ReplayBuffer:
        b = cls()
        for s in samples:
            b.add_sample(s)
        return b

    @classmethod
    def from_pairs(cls, pairs: Iterable[DPOPair]) -> ReplayBuffer:
        b = cls()
        for p in pairs:
            b.add_pair(p)
        return b
