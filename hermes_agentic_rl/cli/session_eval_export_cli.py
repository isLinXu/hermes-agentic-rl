from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []

    if target.suffix in {".jsonl", ".jl"}:
        records: list[dict[str, Any]] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    payload = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
        return [item for item in payload["sessions"] if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def _first_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _extract_prompt(record: dict[str, Any]) -> str:
    for key in ("prompt", "instruction", "task_input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("prompt")
        if isinstance(value, str) and value.strip():
            return value.strip()
        messages = metadata.get("messages")
        if isinstance(messages, list):
            prompt = _first_user_message([msg for msg in messages if isinstance(msg, dict)])
            if prompt:
                return prompt

    messages = record.get("messages")
    if isinstance(messages, list):
        prompt = _first_user_message([msg for msg in messages if isinstance(msg, dict)])
        if prompt:
            return prompt

    task_id = record.get("task_id")
    return str(task_id or "").strip()


def _extract_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if isinstance(messages, list):
        return [msg for msg in messages if isinstance(msg, dict)]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        messages = metadata.get("messages")
        if isinstance(messages, list):
            return [msg for msg in messages if isinstance(msg, dict)]
    return []


def _extract_expected_behavior(record: dict[str, Any]) -> str:
    final_output = record.get("final_output")
    if not isinstance(final_output, str) or not final_output.strip():
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            final_output = metadata.get("final_output")
    if not isinstance(final_output, str) or not final_output.strip():
        final_output = "Resolve the task successfully and answer clearly."
    else:
        final_output = final_output.strip()

    turns_used = record.get("turns_used")
    runtime = {}
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("runtime"), dict):
        runtime = dict(metadata["runtime"])
    runtime_name = str(runtime.get("integration") or "hermes").strip()
    tool_calls = 0
    for step in record.get("steps", []) if isinstance(record.get("steps"), list) else []:
        if isinstance(step, dict):
            tool_calls += len(step.get("tool_calls") or [])
    if not tool_calls:
        tool_calls = sum(
            len(msg.get("tool_calls") or [])
            for msg in _extract_messages(record)
            if isinstance(msg, dict)
        )

    return (
        f"Produce the observed successful outcome for this Hermes session. "
        f"Use tools only when needed, keep the response aligned with the user intent, "
        f"and finish with: {final_output}. "
        f"Observed turns_used={turns_used if turns_used is not None else 'unknown'}, "
        f"tool_calls={tool_calls}, runtime={runtime_name}."
    )


def _difficulty_for(record: dict[str, Any]) -> str:
    turns_used = int(record.get("turns_used", 0) or 0)
    tool_calls = 0
    for step in record.get("steps", []) if isinstance(record.get("steps"), list) else []:
        if isinstance(step, dict):
            tool_calls += len(step.get("tool_calls") or [])
    if not tool_calls:
        tool_calls = sum(
            len(msg.get("tool_calls") or [])
            for msg in _extract_messages(record)
            if isinstance(msg, dict)
        )
    if turns_used >= 5 or tool_calls >= 4:
        return "hard"
    if turns_used >= 3 or tool_calls >= 2:
        return "medium"
    return "easy"


def _category_for(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        runtime = metadata.get("runtime")
        if isinstance(runtime, dict):
            integration = str(runtime.get("integration") or "").strip()
            if integration:
                return integration
        entrypoint = str(metadata.get("entrypoint") or "").strip()
        if entrypoint:
            return entrypoint
    if _extract_messages(record):
        return "session-trace"
    return "general"


def _record_reward(record: dict[str, Any]) -> float | None:
    reward = record.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        reward = metadata.get("reward")
        if isinstance(reward, (int, float)):
            return float(reward)
    return None


def _build_examples(records: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        prompt = _extract_prompt(record)
        expected = _extract_expected_behavior(record)
        if not prompt or not expected:
            continue
        raw_metadata = record.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        examples.append(
            {
                "task_input": prompt,
                "expected_behavior": expected,
                "difficulty": _difficulty_for(record),
                "category": _category_for(record),
                "source": source,
                "session_id": record.get("session_id") or metadata.get("session_id"),
                "task_id": record.get("task_id") or metadata.get("task_id"),
                "reward": _record_reward(record),
                "turns_used": record.get("turns_used") or metadata.get("turns_used"),
                "final_output": record.get("final_output") or metadata.get("final_output"),
            }
        )
    return examples


def _split_examples(
    examples: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = list(examples)
    if not items:
        return [], [], []

    if len(items) < 3:
        return items, [], []

    n_total = len(items)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    n_holdout = max(0, n_total - n_train - n_val)
    return (
        items[:n_train],
        items[n_train:n_train + n_val],
        items[n_train + n_val:n_train + n_val + n_holdout],
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


def _write_manifest(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def run_session_eval_export_config(
    cfg: dict[str, Any],
    *,
    input_path: str | None = None,
    output_path: str | None = None,
) -> int:
    effective_input = input_path or cfg.get("input_path")
    if not effective_input:
        raise ValueError("session-eval-export requires `input_path` in config or --input")

    effective_output = output_path or cfg.get("output_path") or cfg.get("output_dir")
    if not effective_output:
        raise ValueError("session-eval-export requires `output_path` or `output_dir`")

    train_ratio = float(cfg.get("train_ratio", 0.5))
    val_ratio = float(cfg.get("val_ratio", 0.25))
    source = str(cfg.get("source", "hermes-session"))
    min_reward = cfg.get("min_reward")
    max_examples = cfg.get("max_examples")
    seed = cfg.get("seed")

    records = _load_records(effective_input)
    if min_reward is not None:
        threshold = float(min_reward)
        filtered_records = []
        for record in records:
            reward = _record_reward(record)
            if reward is None or reward >= threshold:
                filtered_records.append(record)
        records = filtered_records
    if max_examples is not None:
        try:
            limit = max(0, int(max_examples))
        except (TypeError, ValueError):
            limit = 0
        if limit:
            records = records[:limit]

    examples = _build_examples(records, source=source)
    if seed is not None:
        random.Random(int(seed)).shuffle(examples)
    else:
        random.shuffle(examples)
    train, val, holdout = _split_examples(
        examples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    output = Path(effective_output)
    if output.suffix in {".jsonl", ".jl"}:
        _write_jsonl(output, examples)
        manifest_path = output.with_suffix(".manifest.json")
        manifest = _write_manifest(
            manifest_path,
            {
                "input_path": str(effective_input),
                "output_path": str(output),
                "samples": len(examples),
                "source": source,
            },
        )
        print(
            "[session-eval-export] "
            f"wrote {len(examples)} examples to {output} (manifest {manifest})"
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.jsonl"
    val_path = output / "val.jsonl"
    holdout_path = output / "holdout.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    _write_jsonl(holdout_path, holdout)
    _write_manifest(
        output / "manifest.json",
        {
            "input_path": str(effective_input),
            "output_dir": str(output),
            "source": source,
            "samples": {
                "train": len(train),
                "val": len(val),
                "holdout": len(holdout),
                "total": len(examples),
            },
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "min_reward": min_reward,
            "max_examples": max_examples,
        },
    )
    mean_reward = 0.0
    rewards: list[float] = [
        float(item["reward"])
        for item in examples
        if isinstance(item.get("reward"), (int, float))
    ]
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
    print(
        "[session-eval-export] "
        f"wrote train={len(train)} val={len(val)} holdout={len(holdout)} "
        f"to {output} mean_reward={mean_reward:.4f}"
    )
    return 0


def run_session_eval_export(
    config_path: str | Path,
    *,
    input_path: str | None = None,
    output_path: str | None = None,
) -> int:
    cfg = _load_config(config_path)
    return run_session_eval_export_config(
        cfg,
        input_path=input_path,
        output_path=output_path,
    )
