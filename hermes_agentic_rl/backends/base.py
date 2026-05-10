from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class BackendUnavailableError(RuntimeError):
    """Raised when a backend's dependencies (torch, model weights...) are missing."""


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Minimal tokenizer contract. All backends share this surface."""

    vocab_size: int
    pad_id: int
    eos_id: int

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


@dataclass(slots=True)
class GenerationOutput:
    """Result of a single `generate` call.

    - `response_ids`: sampled token ids (does not include prompt)
    - `logprobs`: log π(a_t | s_t) under the sampling policy, same length as
      response_ids. This is the "old logπ" used for PPO/GRPO ratio.
    - `finished`: whether generation hit EOS naturally (vs. max_new_tokens)
    - `metadata`: free-form (temperature, sampler, etc.)
    """

    response_ids: list[int]
    logprobs: list[float]
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """Unified LLM backend contract for rollout + training.

    Contract:
      - `generate()` is the INFERENCE path (no grad). It produces sampled ids
        and the corresponding log-probs under the sampling policy.
      - `score()` is the TRAINING path (with grad). Given a (prompt, response)
        pair, it returns a per-response-token log π vector as a torch tensor
        that supports `.backward()`.
      - `trainable_parameters()` returns an iterable of nn.Parameter so the
        optimizer can be built externally.

    This dual-path design mirrors verl/HybridFlow: rollout workers hold a
    frozen copy, learner holds the trainable copy, and `score()` is what
    closes the PPO/GRPO loop.
    """

    tokenizer: TokenizerProtocol

    @abstractmethod
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> GenerationOutput:
        raise NotImplementedError

    @abstractmethod
    def score(self, prompt_ids: list[int], response_ids: list[int]) -> Any:
        """Return per-response-token log-probs as a differentiable tensor.

        Shape: `[len(response_ids)]`. The caller (trainer) will combine these
        into the PPO/GRPO surrogate loss.
        """
        raise NotImplementedError

    @abstractmethod
    def trainable_parameters(self) -> Any:
        raise NotImplementedError

    # Optional: some backends (frozen reference models) override this.
    def is_trainable(self) -> bool:
        return True

    def supports_value_head(self) -> bool:
        """True iff `score_with_value()` is implemented and a critic exists."""
        return False

    def score_with_value(
        self, prompt_ids: list[int], response_ids: list[int]
    ) -> tuple[Any, Any, Any]:
        """Return differentiable (per-token logπ, per-token entropy, per-token value).

        Default implementation raises; PPO-capable backends override this.
        """
        raise NotImplementedError("this backend has no value head")
