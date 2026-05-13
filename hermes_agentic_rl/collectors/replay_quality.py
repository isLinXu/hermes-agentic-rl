from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl_records_best_effort(path: str | Path) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    target = Path(path)
    records: list[Any] = []
    rejected: list[dict[str, Any]] = []
    lines_seen = 0
    with target.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith("\n"):
                stripped = raw_line.strip()
                if stripped:
                    rejected.append(
                        {
                            "kind": "partial_json_line",
                            "line_no": line_no,
                            "path": str(target),
                            "raw_line": stripped,
                            "error": "line_missing_newline",
                        }
                    )
                    lines_seen += 1
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            lines_seen += 1
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                rejected.append(
                    {
                        "kind": "invalid_json_line",
                        "line_no": line_no,
                        "path": str(target),
                        "raw_line": stripped,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return records, rejected, {"lines_seen": lines_seen, "loaded_records": len(records)}


def normalize_replay_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompt_ids = _normalize_token_ids(record.get("prompt_ids"))
        response_ids = _normalize_token_ids(record.get("response_ids"))
        reward_raw = record.get("reward", 0.0)
        try:
            reward = float(reward_raw)
        except (TypeError, ValueError):
            reward = float("nan")

        reason = None
        if prompt_ids is None:
            reason = "invalid_prompt_ids"
        elif response_ids is None:
            reason = "invalid_response_ids"
        elif not response_ids:
            reason = "empty_response_ids"
        elif not math.isfinite(reward):
            reason = "non_finite_reward"

        if reason is not None:
            rejected.append(
                {
                    "kind": "invalid_replay_record",
                    "index": idx,
                    "reason": reason,
                    "record": record,
                }
            )
            continue

        normalized = dict(record)
        normalized["prompt_ids"] = prompt_ids
        normalized["response_ids"] = response_ids
        normalized["reward"] = reward
        valid.append(normalized)
    return valid, rejected


def apply_record_quality_filters(
    records: list[dict[str, Any]],
    cfg: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = cfg or {}
    min_prompt_tokens = max(0, int(config.get("min_prompt_tokens", 1)))
    min_response_tokens = max(0, int(config.get("min_response_tokens", 1)))
    max_prompt_tokens = max(0, int(config.get("max_prompt_tokens", 0)))
    max_response_tokens = max(0, int(config.get("max_response_tokens", 0)))
    max_abs_reward = float(config.get("max_abs_reward", 0.0))
    dedupe_within_scan = bool(config.get("dedupe_within_scan", False))

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for idx, record in enumerate(records):
        prompt_ids = list(record.get("prompt_ids") or [])
        response_ids = list(record.get("response_ids") or [])
        reward = float(record.get("reward", 0.0))
        reason = None
        if len(prompt_ids) < min_prompt_tokens:
            reason = "prompt_too_short"
        elif len(response_ids) < min_response_tokens:
            reason = "response_too_short"
        elif max_prompt_tokens > 0 and len(prompt_ids) > max_prompt_tokens:
            reason = "prompt_too_long"
        elif max_response_tokens > 0 and len(response_ids) > max_response_tokens:
            reason = "response_too_long"
        elif max_abs_reward > 0 and abs(reward) > max_abs_reward:
            reason = "reward_out_of_range"
        elif dedupe_within_scan:
            fingerprint = record_fingerprint(record)
            if fingerprint in seen_fingerprints:
                reason = "duplicate_record"
            else:
                seen_fingerprints.add(fingerprint)

        if reason is not None:
            rejected.append(
                {
                    "kind": "quality_filtered_replay_record",
                    "index": idx,
                    "reason": reason,
                    "record": record,
                }
            )
            continue
        valid.append(record)
    return valid, rejected


def summarize_rejections(rejected: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    for row in rejected:
        by_kind[str(row.get("kind") or "unknown")] += 1
        reason = row.get("reason")
        if reason is not None:
            by_reason[str(reason)] += 1
    return {
        "count": len(rejected),
        "by_kind": dict(sorted(by_kind.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


def record_fingerprint(record: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "prompt_ids": list(record.get("prompt_ids") or []),
            "response_ids": list(record.get("response_ids") or []),
            "reward": float(record.get("reward", 0.0)),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_token_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        out.append(int(item))
    return out
