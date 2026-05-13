from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_agentic_rl.collectors.conversation_collector import (
    collect_session_turn_samples,
)
from hermes_agentic_rl.collectors.session_judge import judge_session_turn_sample
from hermes_agentic_rl.collectors.trajectory_adapter import (
    session_turn_sample_to_train_sample,
    trajectory_to_session_turn_samples,
)
from hermes_agentic_rl.core.trajectory import trajectory_from_dict


class HFTokenizerAdapter:
    def __init__(self, hf_tok: Any) -> None:
        self._tok = hf_tok
        self.vocab_size = int(getattr(hf_tok, "vocab_size", len(hf_tok)))
        self.pad_id = int(
            hf_tok.pad_token_id
            if hf_tok.pad_token_id is not None
            else (hf_tok.eos_token_id if hf_tok.eos_token_id is not None else 0)
        )
        self.eos_id = int(hf_tok.eos_token_id) if hf_tok.eos_token_id is not None else self.pad_id
        self.bos_id = int(
            hf_tok.bos_token_id if hf_tok.bos_token_id is not None else self.pad_id
        )

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(self._tok.encode(text, add_special_tokens=False))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=True))


def build_tokenizer_from_config(cfg: dict[str, Any]) -> Any:
    backend_cfg = cfg.get("backend", cfg) or {}
    name = str(backend_cfg.get("name", "tiny"))
    if name == "tiny":
        from hermes_agentic_rl.backends.tiny import TinyTokenizer

        return TinyTokenizer(extra_chars=str(backend_cfg.get("extra_chars", "")))
    if name == "hf":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(backend_cfg.get("model_name_or_path", "gpt2")),
            trust_remote_code=bool(backend_cfg.get("trust_remote_code", False)),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
        return HFTokenizerAdapter(tokenizer)
    raise ValueError(f"unsupported backend.name for replay export: {name!r}")


def append_jsonl(path: str | Path, payloads: list[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(target)


def record_to_session_turn_samples(record: Any) -> list[Any]:
    if isinstance(record, dict) and "task_id" in record and "prompt" in record and "turns_used" in record:
        return trajectory_to_session_turn_samples(trajectory_from_dict(record))

    if isinstance(record, dict):
        messages = record.get("messages")
        if not isinstance(messages, list):
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                messages = metadata.get("messages")
        if not isinstance(messages, list):
            raise ValueError("session record must contain `messages` or trajectory-shaped payload")

        runtime = record.get("runtime")
        runtime_task_id = None
        if isinstance(runtime, dict):
            runtime_task_id = runtime.get("task_id") or runtime.get("session_id")
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and runtime_task_id is None:
            runtime_meta = metadata.get("runtime")
            if isinstance(runtime_meta, dict):
                runtime_task_id = runtime_meta.get("task_id") or runtime_meta.get("session_id")
        return collect_session_turn_samples(
            [dict(m) for m in messages if isinstance(m, dict)],
            session_id=str(record.get("session_id") or runtime_task_id or record.get("task_id") or "session"),
            task_id=record.get("task_id"),
        )

    raise ValueError("unsupported session record type")


def replay_jsonl_payloads_from_record(
    record: Any,
    *,
    tokenizer: Any,
    judge_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    turn_samples = record_to_session_turn_samples(record)
    payloads: list[dict[str, Any]] = []
    for sample in turn_samples:
        reward_summary = judge_session_turn_sample(sample, judge_config)
        train_sample = session_turn_sample_to_train_sample(
            sample,
            tokenizer=tokenizer,
            reward_summary=reward_summary,
        )
        payloads.append(
            {
                "prompt_ids": list(train_sample.prompt_ids),
                "response_ids": list(train_sample.response_ids),
                "reward": float(train_sample.reward),
                "advantage": train_sample.advantage,
                "metadata": dict(train_sample.metadata),
            }
        )
    return payloads
