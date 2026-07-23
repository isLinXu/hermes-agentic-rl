"""Fable-5 Agent Trace environment for RL training.

Loads the Glint-Research/Fable-5-traces dataset from HuggingFace and converts
each row into a training item for tool-use policy learning and reasoning
distillation.

Dataset: https://huggingface.co/datasets/Glint-Research/Fable-5-traces

Two loading modes:
  1. Flat merged JSONL (fable5_cot_merged.jsonl) -- default, efficient
  2. Pi Agent trace config (pi_agent) -- richer, includes session metadata

Each training item contains:
  - instruction: the conversation context (truncated to max_prompt_chars)
  - target_response: the expected assistant action (tool call or text)
  - output_type: tool_use or text
  - tool_name / tool_arguments: for tool_use rows
  - cot: reasoning trace (for optional SFT warm-start)

Reward modes:
  - similarity: text similarity (SequenceMatcher ratio)
  - tool_call: structured tool-call scoring (name + arguments)
  - hybrid: weighted combination (default for tool_use rows)
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.datasets.hf_loader import load_hf_dataset
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.rewards.base import BaseReward

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Glint-Research/Fable-5-traces"
DEFAULT_CONFIG_NAME = "pi_agent"
DEFAULT_SPLIT = "train"
DEFAULT_MERGED_URL = (
    "https://huggingface.co/datasets/Glint-Research/Fable-5-traces/"
    "resolve/main/fable5_cot_merged.jsonl"
)
DEFAULT_LIMIT = 500


@dataclass(slots=True)
class Fable5TraceConfig:
    """Configuration for Fable-5 trace loading and reward."""

    repo_id: str = DEFAULT_REPO_ID
    config_name: str = DEFAULT_CONFIG_NAME
    split: str = DEFAULT_SPLIT
    limit: int | None = DEFAULT_LIMIT
    shuffle: bool = True
    seed: int = 42
    streaming: bool = True
    cache_dir: str | None = None
    use_flat_jsonl: bool = True
    flat_jsonl_url: str = DEFAULT_MERGED_URL
    flat_jsonl_path: str | None = None
    max_prompt_chars: int = 4096
    max_target_chars: int = 8192
    min_target_chars: int = 10
    include_cot_in_target: bool = True
    reward_mode: str = "hybrid"
    tool_call_reward_weight: float = 0.7
    text_reward_weight: float = 0.3
    output_type_bonus: float = 0.1


# --- text helpers ---

def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[Earlier context truncated]\n\n"
    tail_budget = max(32, max_chars - len(marker))
    return marker + text[-tail_budget:]


def _text_similarity(prediction: str, target: str) -> float:
    pred_norm = _normalize_text(prediction)
    target_norm = _normalize_text(target)
    if not target_norm:
        return 0.0
    if pred_norm == target_norm:
        return 1.0
    return SequenceMatcher(None, pred_norm, target_norm).ratio()


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


# Tool-call block regex: matches the outermost JSON object
# We use a balanced-brace scanner instead of regex for nested JSON
_TOOL_CALL_BLOCK_RE = re.compile(r"\{", re.DOTALL)


def _extract_tool_call_from_completion(completion: str) -> dict[str, Any] | None:
    """Try to parse a structured tool call from the completion text."""
    # Try direct JSON parse first
    stripped = completion.strip()
    if stripped and stripped[0] == "{":
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # Scan for the first balanced JSON object containing a "tool" or "name" key
    for match in _TOOL_CALL_BLOCK_RE.finditer(completion):
        start = match.start()
        depth = 0
        for i in range(start, len(completion)):
            if completion[i] == "{":
                depth += 1
            elif completion[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = completion[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and (
                            "tool" in parsed or "name" in parsed
                        ):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def _load_local_jsonl(path: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
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


def _download_flat_jsonl(url: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Download the flat merged JSONL from HuggingFace."""
    rows: list[dict[str, Any]] = []
    with urlopen(url, timeout=60.0) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    return rows


# --- row converters ---

def _flat_row_to_item(row: dict[str, Any], cfg: Fable5TraceConfig) -> dict[str, Any]:
    """Convert a flat-JSONL row into a training item."""
    uid = str(row.get("uid") or row.get("source_file") or "unknown")
    context = str(row.get("context") or "")
    cot = str(row.get("cot") or "")
    output_type = str(row.get("output_type") or "text").strip()
    output = row.get("output")
    completion = str(row.get("completion") or "")

    instruction = _truncate(context, cfg.max_prompt_chars)

    if output_type == "tool_use" and isinstance(output, dict):
        tool_name = str(output.get("tool") or output.get("name") or "")
        tool_input = output.get("input") or output.get("arguments") or {}
        target_response = json.dumps(
            {"name": tool_name, "arguments": tool_input},
            ensure_ascii=False,
        )
        target_full = completion if cfg.include_cot_in_target else target_response
    elif output_type == "text":
        target_response = str(output) if output else completion
        target_full = completion if cfg.include_cot_in_target else target_response
    else:
        target_response = completion
        target_full = completion

    target_full = _truncate(target_full, cfg.max_target_chars)

    if len(target_response.strip()) < cfg.min_target_chars:
        return {}

    return {
        "task_id": f"fable5::{uid}",
        "instruction": instruction,
        "target_response": target_response,
        "target_response_full": target_full,
        "output_type": output_type,
        "tool_name": (output.get("tool") if isinstance(output, dict) else None),
        "tool_arguments": (output.get("input") if isinstance(output, dict) else None),
        "cot": cot,
        "completion": completion,
        "model": str(row.get("model") or ""),
        "session": str(row.get("session") or ""),
        "source_file": str(row.get("source_file") or ""),
        "origin": str(row.get("origin") or ""),
    }


def _pi_agent_row_to_item(
    row: dict[str, Any], cfg: Fable5TraceConfig
) -> list[dict[str, Any]]:
    """Convert a pi_agent config row into one or more training items."""
    items: list[dict[str, Any]] = []
    messages = _decode_json_string(row.get("messages")) or []
    session_id = str(row.get("session_id") or "")
    metadata = row.get("metadata") or {}

    context_parts: list[str] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        reasoning = str(msg.get("reasoning_content") or "").strip()
        tool_calls = msg.get("tool_calls") or []

        if role == "assistant":
            instruction = _truncate("\n\n".join(context_parts), cfg.max_prompt_chars)

            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function") or {}
                    tool_name = str(func.get("name") or "")
                    tool_args = func.get("arguments") or {}
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            pass
                    target_response = json.dumps(
                        {"name": tool_name, "arguments": tool_args},
                        ensure_ascii=False,
                    )
                    target_full = (
                        (reasoning + "\n" + target_response).strip()
                        if cfg.include_cot_in_target
                        else target_response
                    )
                    target_full = _truncate(target_full, cfg.max_target_chars)
                    if len(target_response.strip()) < cfg.min_target_chars:
                        continue
                    items.append(
                        {
                            "task_id": f"fable5::{session_id}#{i}",
                            "instruction": instruction,
                            "target_response": target_response,
                            "target_response_full": target_full,
                            "output_type": "tool_use",
                            "tool_name": tool_name,
                            "tool_arguments": tool_args,
                            "cot": reasoning,
                            "completion": content or target_response,
                            "model": str(metadata.get("model") or ""),
                            "session": session_id,
                            "source_file": str(row.get("file_path") or ""),
                            "origin": str(metadata.get("trace_type") or "pi"),
                        }
                    )
            elif content:
                target_full = (
                    (reasoning + "\n" + content).strip()
                    if cfg.include_cot_in_target
                    else content
                )
                target_full = _truncate(target_full, cfg.max_target_chars)
                if len(content.strip()) < cfg.min_target_chars:
                    continue
                items.append(
                    {
                        "task_id": f"fable5::{session_id}#{i}",
                        "instruction": instruction,
                        "target_response": content,
                        "target_response_full": target_full,
                        "output_type": "text",
                        "tool_name": None,
                        "tool_arguments": None,
                        "cot": reasoning,
                        "completion": content,
                        "model": str(metadata.get("model") or ""),
                        "session": session_id,
                        "source_file": str(row.get("file_path") or ""),
                        "origin": str(metadata.get("trace_type") or "pi"),
                    }
                )

        heading = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            role, role.capitalize() if role else "Unknown"
        )
        context_parts.append(f"{heading}:\n{content}")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function") or {}
                tc_str = json.dumps(
                    {
                        "name": func.get("name"),
                        "arguments": func.get("arguments"),
                    },
                    ensure_ascii=False,
                )
                context_parts.append(f"ToolCall:\n{tc_str}")

    return items


# --- public loader ---

def load_fable5_traces(cfg: Fable5TraceConfig) -> list[dict[str, Any]]:
    """Load Fable-5 traces and convert to training items."""
    raw_rows: list[dict[str, Any]] = []

    if cfg.use_flat_jsonl:
        if cfg.flat_jsonl_path:
            logger.info("Loading Fable-5 flat JSONL from local: %s", cfg.flat_jsonl_path)
            raw_rows = _load_local_jsonl(cfg.flat_jsonl_path, limit=cfg.limit)
        else:
            logger.info("Downloading Fable-5 flat JSONL from HuggingFace...")
            raw_rows = _download_flat_jsonl(cfg.flat_jsonl_url, limit=cfg.limit)

        items: list[dict[str, Any]] = [_flat_row_to_item(row, cfg) for row in raw_rows]
        items = [item for item in items if item]
    else:
        logger.info("Loading Fable-5 via HF datasets API (config=%s)...", cfg.config_name)
        raw_rows = load_hf_dataset(
            cfg.repo_id,
            config_name=cfg.config_name,
            split=cfg.split,
            limit=cfg.limit,
            shuffle=False,
            seed=cfg.seed,
            streaming=cfg.streaming,
            cache_dir=cfg.cache_dir,
        )
        items_raw: list[dict[str, Any]] = []
        for row in raw_rows:
            items_raw.extend(_pi_agent_row_to_item(row, cfg))
        items = items_raw

    if cfg.shuffle and items:
        rng = random.Random(cfg.seed)
        rng.shuffle(items)

    logger.info("Loaded %d Fable-5 training items", len(items))
    return items


# --- reward ---

class Fable5TraceReward(BaseReward):
    """Reward component for Fable-5 trace matching."""

    name = "fable5_trace_reward"

    def __init__(
        self,
        weight: float = 1.0,
        *,
        reward_mode: str = "hybrid",
        tool_call_reward_weight: float = 0.7,
        text_reward_weight: float = 0.3,
        output_type_bonus: float = 0.1,
    ) -> None:
        self.weight = weight
        self.reward_mode = reward_mode
        self.tool_call_reward_weight = tool_call_reward_weight
        self.text_reward_weight = text_reward_weight
        self.output_type_bonus = output_type_bonus
        if self.reward_mode not in {"similarity", "tool_call", "hybrid"}:
            raise ValueError(
                "Fable5TraceReward reward_mode must be 'similarity', "
                "'tool_call', or 'hybrid'"
            )

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del tool_context

        target = str(
            item.get("target_response")
            or item.get("target_response_full")
            or ""
        ).strip()
        prediction = str(trajectory.final_output or "").strip()
        output_type = str(item.get("output_type") or "text")

        if not target:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="missing target_response",
                weight=self.weight,
            )

        if _normalize_text(prediction) == _normalize_text(target):
            return RewardResult(
                name=self.name,
                score=1.0,
                reason="exact_match",
                weight=self.weight,
                metadata={"exact_match": True},
            )

        text_score = _text_similarity(prediction, target)
        tool_score, tool_meta = self._structured_tool_score(prediction, target, item)

        if self.reward_mode == "similarity":
            score = text_score
        elif self.reward_mode == "tool_call":
            score = tool_score
        else:
            denom = self.tool_call_reward_weight + self.text_reward_weight
            if output_type == "tool_use":
                score = (
                    self.tool_call_reward_weight * tool_score
                    + self.text_reward_weight * text_score
                ) / max(denom, 1e-8)
            else:
                score = text_score

        type_bonus = 0.0
        if self.output_type_bonus > 0:
            pred_has_tool = bool(_extract_tool_call_from_completion(prediction))
            if (output_type == "tool_use" and pred_has_tool) or (
                output_type == "text" and not pred_has_tool
            ):
                type_bonus = self.output_type_bonus
        score = min(1.0, score + type_bonus)

        return RewardResult(
            name=self.name,
            score=score,
            reason=(
                f"mode={self.reward_mode} score={score:.4f} "
                f"text={text_score:.4f} tool={tool_score:.4f} "
                f"type_bonus={type_bonus:.2f}"
            ),
            weight=self.weight,
            metadata={
                "exact_match": False,
                "similarity": text_score,
                "tool_call_score": tool_score,
                "output_type": output_type,
                "type_bonus": type_bonus,
                "reward_mode": self.reward_mode,
                **tool_meta,
            },
        )

    @staticmethod
    def _structured_tool_score(
        prediction: str,
        target: str,
        item: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        target_name = str(item.get("tool_name") or "")
        target_args = item.get("tool_arguments") or {}

        pred_call = _extract_tool_call_from_completion(prediction)
        if pred_call is None:
            pred_call = {}

        pred_name = ""
        pred_args: dict[str, Any] = {}
        if pred_call:
            pred_name = str(pred_call.get("name") or pred_call.get("tool") or "")
            pred_args = pred_call.get("arguments") or pred_call.get("input") or {}

        name_score = 1.0 if pred_name and pred_name.lower() == target_name.lower() else 0.0

        if isinstance(target_args, dict) and isinstance(pred_args, dict):
            if not target_args:
                args_score = 1.0 if not pred_args else 0.5
            else:
                target_keys = set(target_args.keys())
                pred_keys = set(pred_args.keys())
                key_overlap = len(target_keys & pred_keys) / max(len(target_keys), 1)
                value_matches = sum(
                    1
                    for k in target_keys & pred_keys
                    if str(target_args[k]).strip() == str(pred_args[k]).strip()
                )
                value_score = value_matches / max(len(target_keys), 1)
                args_score = 0.5 * key_overlap + 0.5 * value_score
        else:
            args_score = 0.0

        total = 0.5 * name_score + 0.5 * args_score
        meta = {
            "target_tool_name": target_name,
            "pred_tool_name": pred_name,
            "name_match": name_score,
            "args_match": args_score,
            "target_tool_call_count": 1 if target_name else 0,
        }
        return total, meta


# --- environment ---

class Fable5TraceEnv(BaseEnv):
    """Round-robin Fable-5 trace environment."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        reward_weight: float = 1.0,
        reward_mode: str = "hybrid",
        tool_call_reward_weight: float = 0.7,
        text_reward_weight: float = 0.3,
        output_type_bonus: float = 0.1,
    ) -> None:
        if not items:
            raise ValueError("Fable5TraceEnv requires at least one item")
        self.items = items
        self._reward = Fable5TraceReward(
            weight=reward_weight,
            reward_mode=reward_mode,
            tool_call_reward_weight=tool_call_reward_weight,
            text_reward_weight=text_reward_weight,
            output_type_bonus=output_type_bonus,
        )
        self._cursor = 0

    @property
    def reward_component(self) -> Fable5TraceReward:
        return self._reward

    @classmethod
    def from_config(cls, cfg_dict: dict[str, Any]) -> Fable5TraceEnv:
        """Build env from a config dict (e.g. from YAML)."""
        cfg = Fable5TraceConfig(
            repo_id=str(cfg_dict.get("repo_id", DEFAULT_REPO_ID)),
            config_name=str(cfg_dict.get("config_name", DEFAULT_CONFIG_NAME)),
            split=str(cfg_dict.get("split", DEFAULT_SPLIT)),
            limit=(
                int(cfg_dict["limit"])
                if cfg_dict.get("limit") not in {None, ""}
                else DEFAULT_LIMIT
            ),
            shuffle=bool(cfg_dict.get("shuffle", True)),
            seed=int(cfg_dict.get("seed", 42)),
            streaming=bool(cfg_dict.get("streaming", True)),
            cache_dir=cfg_dict.get("cache_dir"),
            use_flat_jsonl=bool(cfg_dict.get("use_flat_jsonl", True)),
            flat_jsonl_url=str(cfg_dict.get("flat_jsonl_url", DEFAULT_MERGED_URL)),
            flat_jsonl_path=cfg_dict.get("flat_jsonl_path"),
            max_prompt_chars=int(cfg_dict.get("max_prompt_chars", 4096)),
            max_target_chars=int(cfg_dict.get("max_target_chars", 8192)),
            min_target_chars=int(cfg_dict.get("min_target_chars", 10)),
            include_cot_in_target=bool(cfg_dict.get("include_cot_in_target", True)),
            reward_mode=str(cfg_dict.get("reward_mode", "hybrid")),
            tool_call_reward_weight=float(
                cfg_dict.get("tool_call_reward_weight", 0.7)
            ),
            text_reward_weight=float(cfg_dict.get("text_reward_weight", 0.3)),
            output_type_bonus=float(cfg_dict.get("output_type_bonus", 0.1)),
        )
        items = load_fable5_traces(cfg)
        return cls(
            items,
            reward_weight=float(cfg_dict.get("reward_weight", 1.0)),
            reward_mode=cfg.reward_mode,
            tool_call_reward_weight=cfg.tool_call_reward_weight,
            text_reward_weight=cfg.text_reward_weight,
            output_type_bonus=cfg.output_type_bonus,
        )

    async def setup(self) -> None:
        self._cursor = 0

    async def get_next_item(self) -> dict[str, Any]:
        item = self.items[self._cursor % len(self.items)]
        self._cursor += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        return str(item.get("instruction") or "")

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        return [await self._reward.evaluate(item, trajectory, tool_context)]

    def build_supervised_samples(
        self, item: dict[str, Any]
    ) -> list[SupervisedSample]:
        instruction = str(item.get("instruction") or "").strip()
        response = str(
            item.get("target_response") or item.get("completion") or ""
        ).strip()
        if not instruction or not response:
            return []
        return [
            SupervisedSample(
                instruction=instruction,
                response=response,
                prompt_suffix="",
                metadata={
                    "source_trace_id": item.get("task_id"),
                    "output_type": item.get("output_type"),
                    "tool_name": item.get("tool_name"),
                    "model": item.get("model"),
                    "session": item.get("session"),
                },
            )
        ]
