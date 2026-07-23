"""Multi-turn agent loop with tool-use.

Design (correctness-first):
  - Each turn has its own (prompt_prefix, response, logprobs) triple. The
    ``prompt_prefix`` is the exact token sequence the policy conditioned on
    at that turn: initial prompt + all assistant tokens generated so far +
    all tool-result tokens injected so far.
  - We expose BOTH a "flat" view (concatenated response_ids + logprobs) for
    single-turn-style trainers, AND a structured ``turns`` list for
    multi-turn-aware trainers that want to score each segment under its
    true context.
  - Tool-result tokens are treated as environment observations (no gradient,
    not counted in ``response_ids``). They live inside each turn's
    ``prompt_prefix`` so ``score()`` at training time uses the identical
    context that rollout saw.

Tool protocol (string-based so any tokenizer works):

    <tool_call>NAME(ARG)</tool_call>

If the assistant response contains this substring, the tool is executed and
the result injected as ``<tool_result>...</tool_result>``. If absent, the
loop terminates with that turn's output as the final answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from hermes_agentic_rl.agent_loop.base import BaseAgentLoop
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.mdp.state_encoder import PromptStateEncoder

ToolFn = Callable[[str, str], str]
"""Signature: fn(tool_name, tool_arg) -> tool_result_text."""

DEFAULT_TOOL_RESULT_TEMPLATE = "\n<tool_result>{result}</tool_result>\n"

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^)]*?)\s*\)\s*</tool_call>"
)


class MultiTurnAgentLoop(BaseAgentLoop):
    """Generate → optionally call a tool → feed result back → repeat."""

    def __init__(
        self,
        backend: LLMBackend,
        tools: Mapping[str, ToolFn],
        *,
        encoder: PromptStateEncoder | None = None,
        max_turns: int = 3,
        max_new_tokens_per_turn: int = 16,
        temperature: float = 1.0,
        seed: int | None = None,
        stop_strings: list[str] | None = None,
        tool_result_template: str = DEFAULT_TOOL_RESULT_TEMPLATE,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.encoder = encoder or PromptStateEncoder(backend.tokenizer)
        self.max_turns = max_turns
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        self.temperature = temperature
        self.seed = seed
        self.stop_strings = list(stop_strings or [])
        self.tool_result_template = tool_result_template

    async def run(self, prompt: str) -> dict[str, Any]:
        obs = self.encoder.encode({"instruction": prompt})
        initial_prompt_ids = list(obs.prompt_ids)

        working_ids = list(initial_prompt_ids)
        turns: list[dict[str, Any]] = []

        tool_calls_per_turn: list[list[dict[str, Any]]] = []
        tool_results_per_turn: list[list[dict[str, Any]]] = []
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        final_text = ""
        finished = False

        for turn in range(self.max_turns):
            prompt_prefix = list(working_ids)  # exact context this turn saw
            gen = self.backend.generate(
                prompt_ids=working_ids,
                max_new_tokens=self.max_new_tokens_per_turn,
                temperature=self.temperature,
                seed=(self.seed + turn * 7919) if self.seed is not None else None,
                stop_strings=self.stop_strings,
            )
            assistant_ids = list(gen.response_ids)
            assistant_text = self.backend.tokenizer.decode(assistant_ids)
            working_ids.extend(assistant_ids)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text,
            }
            messages.append(assistant_message)

            turns.append(
                {
                    "prompt_prefix_ids": prompt_prefix,
                    "response_ids": assistant_ids,
                    "old_logprobs": list(gen.logprobs),
                    "assistant_text": assistant_text,
                }
            )

            m = TOOL_CALL_RE.search(assistant_text)
            if m is None:
                final_text = assistant_text
                finished = gen.finished
                tool_calls_per_turn.append([])
                tool_results_per_turn.append([])
                break

            tool_name, tool_arg = m.group(1), m.group(2).strip()
            tool_fn = self.tools.get(tool_name)
            assistant_message["tool_calls"] = [{"name": tool_name, "arg": tool_arg}]
            try:
                tool_result_text = (
                    tool_fn(tool_name, tool_arg)
                    if tool_fn is not None
                    else f"error:unknown_tool({tool_name})"
                )
            except Exception as exc:  # defensive — tools never crash rollout
                tool_result_text = f"error:{type(exc).__name__}:{exc}"
            tool_calls_per_turn.append([{"name": tool_name, "arg": tool_arg}])
            tool_results_per_turn.append([{"name": tool_name, "result": tool_result_text}])
            messages.append({"role": "tool", "name": tool_name, "content": tool_result_text})

            injected = self.tool_result_template.format(result=tool_result_text)
            inject_ids = self.backend.tokenizer.encode(injected)
            working_ids.extend(inject_ids)
        else:
            # max_turns reached without a terminal no-tool-call turn.
            if turns:
                final_text = turns[-1]["assistant_text"]
            finished = False

        # Flat view (matches single-turn PolicyAgentLoop contract). This is
        # used by trainers that haven't been taught multi-turn yet. It is
        # mathematically sound iff the trainer uses either:
        #   (a) per-turn records (see `turns` metadata), OR
        #   (b) a score() call whose prompt === full_context_ids truncated
        #       just before each response segment.
        flat_response_ids: list[int] = []
        flat_logprobs: list[float] = []
        for t in turns:
            flat_response_ids.extend(t["response_ids"])
            flat_logprobs.extend(t["old_logprobs"])

        return {
            "messages": messages,
            "tool_calls": tool_calls_per_turn,
            "tool_results": tool_results_per_turn,
            "final_output": final_text,
            "finished_naturally": finished,
            "turns_used": max(1, len(turns)),
            "metadata": {
                "runtime": "multi_turn_agent_loop",
                "prompt": prompt,
                "rl": {
                    "prompt_ids": list(initial_prompt_ids),
                    "response_ids": list(flat_response_ids),
                    "old_logprobs": list(flat_logprobs),
                    "temperature": self.temperature,
                    "multi_turn": True,
                    "full_context_ids": list(working_ids),
                    "n_turns": len(turns),
                    # Per-turn records: the trainer can use these to score
                    # each response segment under its true rollout context.
                    "turns": [
                        {
                            "prompt_prefix_ids": list(t["prompt_prefix_ids"]),
                            "response_ids": list(t["response_ids"]),
                            "old_logprobs": list(t["old_logprobs"]),
                        }
                        for t in turns
                    ],
                },
            },
        }
