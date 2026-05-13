from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.datasets.hf_loader import load_hf_dataset
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.rewards.base import BaseReward

DEFAULT_REPO_ID = "lambda/hermes-agent-reasoning-traces"
DEFAULT_CONFIG_NAME = "kimi"
DEFAULT_SPLIT = "train"


@dataclass(slots=True)
class HermesReasoningTracesConfig:
    repo_id: str = DEFAULT_REPO_ID
    config_name: str = DEFAULT_CONFIG_NAME
    split: str = DEFAULT_SPLIT
    dataset_path: str | None = None
    limit: int | None = None
    shuffle: bool = True
    seed: int = 0
    streaming: bool = False
    revision: str | None = None
    cache_dir: str | None = None
    rows_api_only: bool = False
    rows_api_timeout: float = 30.0
    rows_api_retries: int = 2
    history_window_messages: int = 0
    max_prompt_chars: int = 0
    min_target_chars: int = 0
    max_target_chars: int = 0
    require_target_substring: str | None = None
    max_assistant_turn_index: int | None = None
    include_system_prompt: bool = True
    tool_call_format_hint: bool = False
    assistant_response_prefix: str = ""
    assistant_response_suffix: str = ""
    assistant_response_adapter: str = ""
    reward_weight: float = 1.0
    reward_mode: str = "similarity"
    tool_call_reward_weight: float = 0.75
    text_reward_weight: float = 0.25


def _role(message: dict[str, Any]) -> str:
    raw = str(message.get("role") or message.get("from") or "").strip().lower()
    mapping = {
        "gpt": "assistant",
        "assistant": "assistant",
        "human": "user",
        "user": "user",
        "tool": "tool",
        "function": "tool",
        "system": "system",
    }
    return mapping.get(raw, raw or "unknown")


def _message_text(message: dict[str, Any]) -> str:
    for key in ("value", "content", "text", "message"):
        value = message.get(key)
        if value is not None:
            return str(value)
    return ""


def _format_tools(tools: Any) -> str:
    tools = _decode_json_string(tools)
    if not tools:
        return ""
    try:
        rendered = json.dumps(tools, ensure_ascii=False, indent=2)
    except Exception:
        rendered = repr(tools)
    return "Available tools:\n" + rendered


def _format_tool_call_hint(tools: Any) -> str:
    tool_names = _tool_names(tools)
    allowed = f" Allowed tool names: {', '.join(tool_names)}." if tool_names else ""
    return (
        "Response format guidance:\n"
        "Return only the next assistant message. For tool-use turns, use exactly:\n"
        "<think>\n"
        "</think>\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {}}\n'
        "</tool_call>\n"
        f"Do not add prose outside the XML-style tags.{allowed}"
    )


def _format_terminal_command_hint(tools: Any) -> str:
    tool_names = _tool_names(tools)
    allowed = f" Available tool names: {', '.join(tool_names)}." if tool_names else ""
    return (
        "Response format guidance:\n"
        "Return only the terminal command string to execute. Do not include "
        "<tool_call> tags, JSON, Markdown fences, or explanatory prose."
        f"{allowed}"
    )


def _format_response_hint(tools: Any, *, response_adapter: str = "") -> str:
    if response_adapter == "terminal_command_tool_call":
        return _format_terminal_command_hint(tools)
    return _format_tool_call_hint(tools)


def _tool_names(tools: Any) -> list[str]:
    tools = _decode_json_string(tools)
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        function = tool.get("function")
        if not isinstance(name, str) and isinstance(function, dict):
            name = function.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return sorted(set(names))


def _format_conversation(
    messages: list[dict[str, Any]],
    *,
    include_system_prompt: bool,
) -> str:
    lines: list[str] = []
    for msg in messages:
        role = _role(msg)
        if role == "system" and not include_system_prompt:
            continue
        content = _message_text(msg).strip()
        if not content:
            continue
        if role == "assistant":
            heading = "Assistant"
        elif role == "user":
            heading = "User"
        elif role == "tool":
            tool_name = msg.get("name") or msg.get("tool_name") or msg.get("tool")
            heading = f"Tool[{tool_name}]" if tool_name else "Tool"
        elif role == "system":
            heading = "System"
        else:
            heading = role.capitalize()
        lines.append(f"{heading}:\n{content}")
    return "\n\n".join(lines).strip()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _truncate_instruction(text: str, *, max_prompt_chars: int) -> str:
    if max_prompt_chars <= 0 or len(text) <= max_prompt_chars:
        return text
    marker = "\n\n[Earlier context truncated]\n\n"
    tail_budget = max(32, max_prompt_chars - len(marker))
    return marker + text[-tail_budget:]


def _trace_similarity(prediction: str, target: str) -> float:
    pred_norm = _normalize_text(prediction)
    target_norm = _normalize_text(target)
    if not target_norm:
        return 0.0
    if pred_norm == target_norm:
        return 1.0
    return SequenceMatcher(None, pred_norm, target_norm).ratio()


def _format_prompt_with_response_prefix(instruction: str, response_prefix: str) -> str:
    if not response_prefix:
        return instruction
    separator = "" if instruction.endswith(("\n", " ")) else "\n"
    return f"{instruction}{separator}{response_prefix}"


def _prediction_with_response_prefix(prediction: str, response_prefix: str) -> str:
    if not response_prefix:
        return prediction
    if prediction.startswith(response_prefix):
        return prediction
    return f"{response_prefix}{prediction}"


def _prediction_with_response_suffix(prediction: str, response_suffix: str) -> str:
    if not response_suffix:
        return prediction
    if prediction.endswith(response_suffix):
        return prediction
    return f"{prediction}{response_suffix}"


def _strip_response_suffix(target: str, response_suffix: str) -> str:
    if not response_suffix:
        return target
    if target.endswith(response_suffix):
        return target[: -len(response_suffix)]
    stripped_target = target.rstrip()
    stripped_suffix = response_suffix.rstrip()
    if stripped_suffix and stripped_target.endswith(stripped_suffix):
        return stripped_target[: -len(stripped_suffix)]
    return target


def _strip_response_prefix(target: str, response_prefix: str) -> str:
    if not response_prefix:
        return target
    if target.startswith(response_prefix):
        return target[len(response_prefix) :]
    stripped_target = target.lstrip()
    stripped_prefix = response_prefix.lstrip()
    if stripped_target.startswith(stripped_prefix):
        return stripped_target[len(stripped_prefix) :]
    return target


def _strip_response_affixes(
    target: str,
    *,
    response_prefix: str = "",
    response_suffix: str = "",
) -> str:
    stripped = _strip_response_prefix(target, response_prefix)
    return _strip_response_suffix(stripped, response_suffix)


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_LEGACY_TOOL_CALL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\((.*)\)\s*$",
    re.DOTALL,
)


def _safe_json_loads(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _stringify_argument_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _normalize_tool_arguments(value: Any) -> dict[str, Any]:
    decoded = _decode_json_string(value)
    if isinstance(decoded, dict):
        return decoded
    if decoded is None or decoded == "":
        return {}
    if isinstance(decoded, list):
        return {"items": decoded}
    return {"arg": decoded}


def _extract_tool_name_and_arguments(value: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    function = value.get("function")
    if isinstance(function, dict):
        func_name = function.get("name")
        arguments = function.get("arguments", value.get("arguments"))
        return (
            str(func_name).strip() if isinstance(func_name, str) and func_name.strip() else None,
            _normalize_tool_arguments(arguments),
        )

    for name_key in ("name", "tool_name", "tool"):
        name = value.get(name_key)
        if isinstance(name, str) and name.strip():
            return name.strip(), _normalize_tool_arguments(value.get("arguments", value.get("args")))
    return None, _normalize_tool_arguments(value.get("arguments", value.get("args")))


def _tool_call_from_object(value: dict[str, Any], *, raw: str) -> dict[str, Any]:
    name, arguments = _extract_tool_name_and_arguments(value)
    return {
        "raw": raw,
        "name": name,
        "arguments": arguments,
        "parse_ok": bool(name),
        "object_like": True,
        "format": "json",
    }


def _tool_calls_from_block(block: str) -> list[dict[str, Any]]:
    parsed = _safe_json_loads(block)
    if isinstance(parsed, list):
        calls = []
        for item in parsed:
            if isinstance(item, dict):
                calls.append(_tool_call_from_object(item, raw=block))
        if calls:
            return calls
    if isinstance(parsed, dict):
        return [_tool_call_from_object(parsed, raw=block)]

    legacy_match = _LEGACY_TOOL_CALL_RE.match(block)
    if legacy_match:
        name, arg = legacy_match.groups()
        return [
            {
                "raw": block,
                "name": name.strip(),
                "arguments": {"arg": arg.strip()},
                "parse_ok": True,
                "object_like": False,
                "format": "legacy",
            }
        ]

    return [
        {
            "raw": block,
            "name": None,
            "arguments": {},
            "parse_ok": False,
            "object_like": False,
            "format": "unknown",
        }
    ]


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text or ""):
        block = match.group(1).strip()
        if block:
            calls.extend(_tool_calls_from_block(block))
    return calls


def _terminal_command_from_tool_call(text: str) -> str | None:
    for call in _extract_tool_calls(text):
        name = str(call.get("name") or "").strip().lower()
        args = _normalize_tool_arguments(call.get("arguments"))
        command = args.get("command") if isinstance(args, dict) else None
        if isinstance(command, str) and (name == "terminal" or not name):
            return command
    return None


def _terminal_command_from_prediction(prediction: str) -> str:
    parsed_command = _terminal_command_from_tool_call(prediction)
    if parsed_command is not None:
        return parsed_command.strip()
    text = str(prediction or "").strip()
    text = re.sub(r"</?tool_call>", "", text, flags=re.IGNORECASE).strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    return text


def _wrap_terminal_command_tool_call(command: str) -> str:
    payload = {
        "name": "terminal",
        "arguments": {"command": _terminal_command_from_prediction(command)},
    }
    return (
        "<think>\n"
        "</think>\n"
        "<tool_call>\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "</tool_call>"
    )


def _adapt_prediction_response(
    prediction: str,
    *,
    response_prefix: str = "",
    response_suffix: str = "",
    response_adapter: str = "",
) -> str:
    if response_adapter == "terminal_command_tool_call":
        return _wrap_terminal_command_tool_call(prediction)
    prediction = _prediction_with_response_prefix(prediction, response_prefix)
    return _prediction_with_response_suffix(prediction, response_suffix)


def _flatten_arguments(value: Any, *, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_arguments(child, prefix=child_key))
        return out
    return {prefix or "arg": _stringify_argument_value(value)}


def _argument_key_overlap(pred_args: dict[str, Any], target_args: dict[str, Any]) -> float:
    pred_keys = set(_flatten_arguments(pred_args))
    target_keys = set(_flatten_arguments(target_args))
    if not target_keys:
        return 1.0 if not pred_keys else 0.0
    if not pred_keys:
        return 0.0
    return len(pred_keys & target_keys) / len(pred_keys | target_keys)


def _argument_value_similarity(pred_args: dict[str, Any], target_args: dict[str, Any]) -> float:
    pred_flat = _flatten_arguments(pred_args)
    target_flat = _flatten_arguments(target_args)
    if not target_flat:
        return 1.0 if not pred_flat else 0.0
    common_keys = sorted(set(pred_flat) & set(target_flat))
    if common_keys:
        return sum(
            _trace_similarity(pred_flat[key], target_flat[key])
            for key in common_keys
        ) / len(common_keys)
    return 0.5 * _trace_similarity(
        _stringify_argument_value(pred_args),
        _stringify_argument_value(target_args),
    )


def _partial_tool_call_structure(prediction: str) -> tuple[float, dict[str, float]]:
    lowered = prediction.lower()
    open_tag = 1.0 if "<tool_call" in lowered else 0.0
    close_tag = 1.0 if "</tool_call>" in lowered else 0.0
    json_braces = 1.0 if "{" in prediction and "}" in prediction else 0.0
    name_key = 1.0 if '"name"' in lowered or "'name'" in lowered else 0.0
    arguments_key = 1.0 if '"arguments"' in lowered or "'arguments'" in lowered else 0.0
    components = {
        "partial_open_tag": open_tag,
        "partial_close_tag": close_tag,
        "partial_json_braces": json_braces,
        "partial_name_key": name_key,
        "partial_arguments_key": arguments_key,
    }
    score = (
        0.08 * open_tag
        + 0.08 * close_tag
        + 0.06 * json_braces
        + 0.06 * name_key
        + 0.06 * arguments_key
    )
    return max(0.0, min(0.34, score)), components


def _score_tool_call_pair(
    prediction_call: dict[str, Any],
    target_call: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    pred_name = str(prediction_call.get("name") or "").strip()
    target_name = str(target_call.get("name") or "").strip()
    pred_args = _normalize_tool_arguments(prediction_call.get("arguments"))
    target_args = _normalize_tool_arguments(target_call.get("arguments"))

    components = {
        "tool_call_present": 1.0,
        "tool_call_parse_ok": 1.0
        if prediction_call.get("parse_ok") and prediction_call.get("object_like")
        else 0.0,
        "tool_name_match": 1.0
        if pred_name and target_name and pred_name.lower() == target_name.lower()
        else 0.0,
        "argument_key_overlap": _argument_key_overlap(pred_args, target_args),
        "argument_value_similarity": _argument_value_similarity(pred_args, target_args),
    }
    weights = {
        "tool_call_present": 0.15,
        "tool_call_parse_ok": 0.20,
        "tool_name_match": 0.25,
        "argument_key_overlap": 0.15,
        "argument_value_similarity": 0.25,
    }
    score = sum(components[key] * weights[key] for key in weights)
    return max(0.0, min(1.0, score)), components


def _structured_tool_call_score(prediction: str, target: str) -> tuple[float, dict[str, Any]]:
    target_calls = _extract_tool_calls(target)
    prediction_calls = _extract_tool_calls(prediction)
    metadata: dict[str, Any] = {
        "target_tool_call_count": len(target_calls),
        "prediction_tool_call_count": len(prediction_calls),
    }
    if not target_calls:
        score = _trace_similarity(prediction, target)
        metadata["fallback_similarity"] = score
        return score, metadata
    if not prediction_calls:
        partial_score, partial_metadata = _partial_tool_call_structure(prediction)
        metadata.update(
            {
                "tool_call_present": partial_metadata["partial_open_tag"],
                "tool_call_parse_ok": 0.0,
                "tool_name_match": 0.0,
                "argument_key_overlap": 0.0,
                "argument_value_similarity": 0.0,
                "partial_tool_call_score": partial_score,
                **partial_metadata,
            }
        )
        return partial_score, metadata

    best_scores: list[float] = []
    best_components: list[dict[str, float]] = []
    for target_call in target_calls:
        pair_scores = [
            _score_tool_call_pair(prediction_call, target_call)
            for prediction_call in prediction_calls
        ]
        best_score, best_component = max(pair_scores, key=lambda pair: pair[0])
        best_scores.append(best_score)
        best_components.append(best_component)

    mean_score = sum(best_scores) / len(best_scores)
    for key in best_components[0]:
        metadata[key] = sum(component[key] for component in best_components) / len(best_components)
    metadata["tool_call_score"] = mean_score
    return mean_score, metadata


def load_hermes_reasoning_trace_turns(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    split: str = DEFAULT_SPLIT,
    dataset_path: str | None = None,
    limit: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    streaming: bool = False,
    revision: str | None = None,
    cache_dir: str | None = None,
    rows_api_only: bool = False,
    rows_api_timeout: float = 30.0,
    rows_api_retries: int = 2,
    history_window_messages: int = 0,
    max_prompt_chars: int = 0,
    min_target_chars: int = 0,
    max_target_chars: int = 0,
    require_target_substring: str | None = None,
    max_assistant_turn_index: int | None = None,
    include_system_prompt: bool = True,
    tool_call_format_hint: bool = False,
    assistant_response_prefix: str = "",
    assistant_response_suffix: str = "",
    assistant_response_adapter: str = "",
) -> list[dict[str, Any]]:
    if dataset_path:
        rows = _load_local_trace_rows(
            dataset_path,
            limit=limit if not shuffle else None,
        )
        if shuffle:
            import random

            random.Random(seed).shuffle(rows)
        if limit is not None:
            rows = rows[:limit]
    else:
        rows = load_hf_dataset(
            repo_id,
            config_name=config_name,
            split=split,
            limit=limit,
            shuffle=shuffle,
            seed=seed,
            streaming=streaming,
            revision=revision,
            cache_dir=cache_dir,
            rows_api_only=rows_api_only,
            rows_api_timeout=rows_api_timeout,
            rows_api_retries=rows_api_retries,
    )
    items: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        row = _normalize_trace_row(row)
        conversations = row.get("conversations") or []
        if not isinstance(conversations, list):
            continue
        tools = row.get("tools")
        trace_prefix = []
        assistant_turn = 0
        row_task_id = str(row.get("task_id") or row.get("id") or f"trace-{row_idx}")
        for msg_idx, msg in enumerate(conversations):
            if not isinstance(msg, dict):
                continue
            role = _role(msg)
            if role != "assistant":
                trace_prefix.append(msg)
                continue
            target = _message_text(msg).strip()
            if not target:
                trace_prefix.append(msg)
                continue
            if max_assistant_turn_index is not None and assistant_turn > max_assistant_turn_index:
                trace_prefix.append(msg)
                assistant_turn += 1
                continue
            if not _target_passes_filters(
                target,
                min_target_chars=min_target_chars,
                max_target_chars=max_target_chars,
                require_target_substring=require_target_substring,
            ):
                trace_prefix.append(msg)
                assistant_turn += 1
                continue
            prompt_messages = list(trace_prefix)
            if history_window_messages > 0:
                prompt_messages = prompt_messages[-history_window_messages:]
            prompt_parts: list[str] = []
            category = row.get("category")
            subcategory = row.get("subcategory")
            task = row.get("task")
            if category or subcategory or task:
                meta_bits = [f"category={category}" if category else None,
                             f"subcategory={subcategory}" if subcategory else None,
                             f"task={task}" if task else None]
                prompt_parts.append("Trace metadata:\n" + "\n".join(bit for bit in meta_bits if bit))
            tools_block = _format_tools(tools)
            if tools_block:
                prompt_parts.append(tools_block)
            if tool_call_format_hint:
                prompt_parts.append(
                    _format_response_hint(
                        tools,
                        response_adapter=assistant_response_adapter,
                    )
                )
            conversation_block = _format_conversation(
                prompt_messages,
                include_system_prompt=include_system_prompt,
            )
            if conversation_block:
                prompt_parts.append("Conversation so far:\n" + conversation_block)
            instruction = "\n\n".join(prompt_parts).strip()
            if not instruction:
                trace_prefix.append(msg)
                continue
            instruction = instruction + "\n\nAssistant:"
            instruction = _truncate_instruction(
                instruction,
                max_prompt_chars=max_prompt_chars,
            )
            if assistant_response_adapter == "terminal_command_tool_call":
                command_target = _terminal_command_from_tool_call(target)
                if not command_target or not command_target.strip():
                    trace_prefix.append(msg)
                    assistant_turn += 1
                    continue
                target_response = _prediction_with_response_suffix(
                    command_target.strip(),
                    assistant_response_suffix,
                )
            else:
                target_response = _strip_response_affixes(
                    target,
                    response_prefix=assistant_response_prefix,
                    response_suffix=assistant_response_suffix,
                )
            items.append(
                {
                    "task_id": f"{row_task_id}::assistant::{assistant_turn}",
                    "source_trace_id": row_task_id,
                    "assistant_turn_index": assistant_turn,
                    "message_index": msg_idx,
                    "instruction": instruction,
                    "target_response": target_response,
                    "target_response_full": target,
                    "response_prefix": assistant_response_prefix,
                    "response_suffix": assistant_response_suffix,
                    "response_adapter": assistant_response_adapter,
                    "category": category,
                    "subcategory": subcategory,
                    "task": task,
                    "tools": tools,
                    "raw_conversations": conversations,
                }
            )
            assistant_turn += 1
            trace_prefix.append(msg)
    return items


def _target_passes_filters(
    target: str,
    *,
    min_target_chars: int = 0,
    max_target_chars: int = 0,
    require_target_substring: str | None = None,
) -> bool:
    if min_target_chars > 0 and len(target) < min_target_chars:
        return False
    if max_target_chars > 0 and len(target) > max_target_chars:
        return False
    return not (require_target_substring and require_target_substring not in target)


def _decode_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _normalize_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["tools"] = _decode_json_string(normalized.get("tools"))
    normalized["conversations"] = _decode_json_string(normalized.get("conversations"))
    return normalized


def _load_local_trace_rows(path: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    local_path = Path(path)
    if local_path.suffix.lower() == ".parquet":
        return _load_local_parquet_trace_rows(local_path, limit=limit)

    rows: list[dict[str, Any]] = []
    with local_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _load_local_parquet_trace_rows(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Reading local parquet reasoning traces requires `pyarrow` "
            "(`pip install pyarrow` or install the data extra)."
        ) from exc

    rows: list[dict[str, Any]] = []
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            if isinstance(row, dict):
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


class HermesReasoningTraceReward(BaseReward):
    name = "hermes_reasoning_trace_match"

    def __init__(
        self,
        weight: float = 1.0,
        *,
        reward_mode: str = "similarity",
        tool_call_reward_weight: float = 0.75,
        text_reward_weight: float = 0.25,
    ) -> None:
        self.weight = weight
        self.reward_mode = reward_mode
        self.tool_call_reward_weight = tool_call_reward_weight
        self.text_reward_weight = text_reward_weight
        if self.reward_mode not in {"similarity", "tool_call", "hybrid"}:
            raise ValueError(
                "HermesReasoningTraceReward reward_mode must be one of "
                "'similarity', 'tool_call', or 'hybrid'"
            )
        if self.reward_mode == "hybrid" and (
            self.tool_call_reward_weight + self.text_reward_weight
        ) <= 0:
            raise ValueError("hybrid reward weights must have a positive sum")

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        target = str(item.get("target_response_full") or item.get("target_response") or "").strip()
        raw_prediction = str(trajectory.final_output or "").strip()
        response_prefix = str(item.get("response_prefix") or "")
        response_suffix = str(item.get("response_suffix") or "")
        response_adapter = str(item.get("response_adapter") or "")
        prediction = _adapt_prediction_response(
            raw_prediction,
            response_prefix=response_prefix,
            response_suffix=response_suffix,
            response_adapter=response_adapter,
        ).strip()
        if not target:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="missing target_response",
                weight=self.weight,
            )
        comparison_prediction = prediction
        comparison_target = target
        if response_adapter == "terminal_command_tool_call":
            command_target = str(item.get("target_response") or "").strip()
            comparison_prediction = _terminal_command_from_prediction(raw_prediction)
            comparison_target = command_target or _terminal_command_from_tool_call(target) or target
        exact_match = _normalize_text(comparison_prediction) == _normalize_text(comparison_target)
        text_score = _trace_similarity(comparison_prediction, comparison_target)
        tool_call_score, tool_metadata = _structured_tool_call_score(prediction, target)
        has_target_tool_call = bool(tool_metadata["target_tool_call_count"])

        if exact_match:
            score = 1.0
        elif self.reward_mode == "similarity" or not has_target_tool_call:
            score = text_score
        elif self.reward_mode == "tool_call":
            score = tool_call_score
        else:
            denom = self.tool_call_reward_weight + self.text_reward_weight
            score = (
                self.tool_call_reward_weight * tool_call_score
                + self.text_reward_weight * text_score
            ) / denom
        return RewardResult(
            name=self.name,
            score=score,
            reason=(
                f"mode={self.reward_mode} score={score:.4f} "
                f"text_similarity={text_score:.4f} tool_call={tool_call_score:.4f}"
            ),
            weight=self.weight,
            metadata={
                "exact_match": exact_match,
                "prediction_chars": len(prediction),
                "target_chars": len(target),
                "similarity": text_score,
                "reward_mode": self.reward_mode,
                "tool_call_reward_weight": self.tool_call_reward_weight,
                "text_reward_weight": self.text_reward_weight,
                "response_adapter": response_adapter,
                **tool_metadata,
            },
        )


class HermesReasoningTraceEnv(BaseEnv):
    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        reward_weight: float = 1.0,
        reward_mode: str = "similarity",
        tool_call_reward_weight: float = 0.75,
        text_reward_weight: float = 0.25,
        assistant_response_prefix: str = "",
        assistant_response_suffix: str = "",
        assistant_response_adapter: str = "",
    ) -> None:
        if not items:
            raise ValueError("HermesReasoningTraceEnv requires at least one item")
        self.items = items
        self._reward = HermesReasoningTraceReward(
            weight=reward_weight,
            reward_mode=reward_mode,
            tool_call_reward_weight=tool_call_reward_weight,
            text_reward_weight=text_reward_weight,
        )
        self._assistant_response_prefix = assistant_response_prefix
        self._assistant_response_suffix = assistant_response_suffix
        self._assistant_response_adapter = assistant_response_adapter
        self._cursor = 0

    @property
    def reward_component(self) -> HermesReasoningTraceReward:
        return self._reward

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> HermesReasoningTracesConfig:
        return HermesReasoningTracesConfig(
            repo_id=str(cfg.get("dataset_name", DEFAULT_REPO_ID)),
            config_name=str(cfg.get("dataset_config", DEFAULT_CONFIG_NAME)),
            split=str(cfg.get("dataset_split", DEFAULT_SPLIT)),
            dataset_path=cfg.get("dataset_path"),
            limit=(
                int(cfg["dataset_limit"])
                if cfg.get("dataset_limit") not in {None, ""}
                else None
            ),
            shuffle=bool(cfg.get("shuffle", True)),
            seed=int(cfg.get("seed", 0)),
            streaming=bool(cfg.get("streaming", False)),
            revision=cfg.get("revision"),
            cache_dir=cfg.get("cache_dir"),
            rows_api_only=bool(cfg.get("rows_api_only", False)),
            rows_api_timeout=float(cfg.get("rows_api_timeout", 30.0)),
            rows_api_retries=int(cfg.get("rows_api_retries", 2)),
            history_window_messages=int(cfg.get("history_window_messages", 0)),
            max_prompt_chars=int(cfg.get("max_prompt_chars", 0)),
            min_target_chars=int(cfg.get("min_target_chars", 0)),
            max_target_chars=int(cfg.get("max_target_chars", 0)),
            require_target_substring=cfg.get("require_target_substring"),
            max_assistant_turn_index=(
                int(cfg["max_assistant_turn_index"])
                if cfg.get("max_assistant_turn_index") not in {None, ""}
                else None
            ),
            include_system_prompt=bool(cfg.get("include_system_prompt", True)),
            reward_weight=float(cfg.get("reward_weight", 1.0)),
            reward_mode=str(cfg.get("reward_mode", "similarity")),
            tool_call_reward_weight=float(cfg.get("tool_call_reward_weight", 0.75)),
            text_reward_weight=float(cfg.get("text_reward_weight", 0.25)),
            tool_call_format_hint=bool(cfg.get("tool_call_format_hint", False)),
            assistant_response_prefix=str(cfg.get("assistant_response_prefix", "")),
            assistant_response_suffix=str(cfg.get("assistant_response_suffix", "")),
            assistant_response_adapter=str(cfg.get("assistant_response_adapter", "")),
        )

    @classmethod
    def from_hf_dataset(cls, cfg: dict[str, Any]) -> HermesReasoningTraceEnv:
        spec = cls.from_config(cfg)
        items = load_hermes_reasoning_trace_turns(
            repo_id=spec.repo_id,
            config_name=spec.config_name,
            split=spec.split,
            dataset_path=spec.dataset_path,
            limit=spec.limit,
            shuffle=spec.shuffle,
            seed=spec.seed,
            streaming=spec.streaming,
            revision=spec.revision,
            cache_dir=spec.cache_dir,
            rows_api_only=spec.rows_api_only,
            rows_api_timeout=spec.rows_api_timeout,
            rows_api_retries=spec.rows_api_retries,
            history_window_messages=spec.history_window_messages,
            max_prompt_chars=spec.max_prompt_chars,
            min_target_chars=spec.min_target_chars,
            max_target_chars=spec.max_target_chars,
            require_target_substring=spec.require_target_substring,
            max_assistant_turn_index=spec.max_assistant_turn_index,
            include_system_prompt=spec.include_system_prompt,
            tool_call_format_hint=spec.tool_call_format_hint,
            assistant_response_prefix=spec.assistant_response_prefix,
            assistant_response_suffix=spec.assistant_response_suffix,
            assistant_response_adapter=spec.assistant_response_adapter,
        )
        return cls(
            items,
            reward_weight=spec.reward_weight,
            reward_mode=spec.reward_mode,
            tool_call_reward_weight=spec.tool_call_reward_weight,
            text_reward_weight=spec.text_reward_weight,
            assistant_response_prefix=spec.assistant_response_prefix,
            assistant_response_suffix=spec.assistant_response_suffix,
            assistant_response_adapter=spec.assistant_response_adapter,
        )

    async def setup(self) -> None:
        return None

    async def get_next_item(self) -> dict[str, Any]:
        item = self.items[self._cursor % len(self.items)]
        self._cursor += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        instruction = str(item.get("instruction") or "")
        response_prefix = str(item.get("response_prefix") or self._assistant_response_prefix or "")
        return _format_prompt_with_response_prefix(instruction, response_prefix)

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        return [await self._reward.evaluate(item, trajectory, tool_context)]

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        instruction = str(item.get("instruction") or "").strip()
        response = str(item.get("target_response") or "")
        response_prefix = str(item.get("response_prefix") or self._assistant_response_prefix or "")
        response_suffix = str(item.get("response_suffix") or self._assistant_response_suffix or "")
        response_adapter = str(item.get("response_adapter") or self._assistant_response_adapter or "")
        instruction = self.format_prompt(item)
        if not instruction or not response.strip():
            return []
        if response_adapter != "terminal_command_tool_call":
            response = _strip_response_affixes(
                response,
                response_prefix=response_prefix,
                response_suffix=response_suffix,
            ).strip()
        return [
            SupervisedSample(
                instruction=instruction,
                response=response,
                prompt_suffix="",
                metadata={
                    "source_trace_id": item.get("source_trace_id"),
                    "assistant_turn_index": item.get("assistant_turn_index"),
                    "message_index": item.get("message_index"),
                    "category": item.get("category"),
                    "subcategory": item.get("subcategory"),
                    "task": item.get("task"),
                },
            )
        ]
