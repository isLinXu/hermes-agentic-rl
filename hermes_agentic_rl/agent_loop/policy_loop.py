from __future__ import annotations

from typing import Any

from hermes_agentic_rl.agent_loop.base import BaseAgentLoop
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.mdp.state_encoder import PromptStateEncoder


class PolicyAgentLoop(BaseAgentLoop):
    """Minimal "generate one response" loop for the MVP.

    Contract:
      - calls `backend.generate(prompt_ids, ...)` once
      - packages the decoded text as a trivial assistant message
      - emits an RL metadata block with prompt_ids / response_ids /
        old_logprobs so the GRPO trainer can recompute logπ_new and form the
        ratio.

    Multi-turn tool-calling can be added later by extending this loop; for the
    MVP the important property is that the `old_logprobs` survive intact from
    rollout to training.
    """

    def __init__(
        self,
        backend: LLMBackend,
        encoder: PromptStateEncoder | None = None,
        max_new_tokens: int = 24,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.backend = backend
        self.encoder = encoder or PromptStateEncoder(backend.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed

    async def run(self, prompt: str) -> dict[str, Any]:
        obs = self.encoder.encode({"instruction": prompt})
        gen = self.backend.generate(
            prompt_ids=obs.prompt_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
        response_text = self.backend.tokenizer.decode(gen.response_ids)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text},
        ]
        return {
            "messages": messages,
            "tool_calls": [],
            "tool_results": [],
            "final_output": response_text,
            "finished_naturally": gen.finished,
            "turns_used": 1,
            "metadata": {
                "runtime": "policy_agent_loop",
                "prompt": prompt,
                "rl": {
                    "prompt_ids": list(obs.prompt_ids),
                    "response_ids": list(gen.response_ids),
                    "old_logprobs": list(gen.logprobs),
                    "temperature": self.temperature,
                },
            },
        }
