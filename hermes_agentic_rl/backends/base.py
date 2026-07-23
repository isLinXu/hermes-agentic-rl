from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from hermes_agentic_rl._compat import stable


@stable
class BackendUnavailableError(RuntimeError):
    """Raised when a backend's dependencies (torch, model weights...) are missing."""


@stable
@runtime_checkable
class TokenizerProtocol(Protocol):
    """Minimal tokenizer contract. All backends share this surface."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@stable
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
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@stable
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
        stop_strings: list[str] | None = None,
    ) -> GenerationOutput:
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        temperature: float = 1.0,
    ) -> Any:
        """Return per-response-token log-probs as a differentiable tensor.

        Shape: `[len(response_ids)]`. The caller (trainer) will combine these
        into the PPO/GRPO surrogate loss.
        """
        raise NotImplementedError

    def score_temperature(self) -> float:
        """Temperature used when scoring rollout trajectories.

        By default we treat the rollout policy as temperature 1.0. Backends
        that log temperature-scaled rollout logprobs should override this
        behavior by accepting a matching ``temperature`` argument on score().
        """
        return 1.0

    @abstractmethod
    def trainable_parameters(self) -> Any:
        raise NotImplementedError

    def num_parameters(self) -> int:
        total = 0
        for param in self.trainable_parameters():
            if hasattr(param, "numel"):
                total += int(param.numel())
        return total

    # Optional: some backends (frozen reference models) override this.
    def is_trainable(self) -> bool:
        return True

    def supports_value_head(self) -> bool:
        """True iff `score_with_value()` is implemented and a critic exists."""
        return False

    def set_gradient_checkpointing(self, enabled: bool) -> bool:
        """Best-effort toggle for model gradient checkpointing.

        Backends that support activation checkpointing should override this
        and return True when the toggle was applied. The default keeps the
        base protocol backward compatible for small/testing backends.
        """
        _ = enabled
        return False

    def score_with_value(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        temperature: float = 1.0,
    ) -> tuple[Any, Any, Any]:
        """Return differentiable (per-token logπ, per-token entropy, per-token value).

        Default implementation raises; PPO-capable backends override this.
        """
        raise NotImplementedError("this backend has no value head")

    # ------------------------------------------------------------------
    # v0.8 batch API (PERFORMANCE CRITICAL)
    # ------------------------------------------------------------------
    #
    # The single-record ``score()`` / ``score_with_value()`` methods run one
    # forward per (prompt, response) pair, which burns 90% of training time
    # in Python-level loop overhead on GPU. These batched variants pad a
    # list of (prompt, response) pairs into a single [B, T] forward pass.
    #
    # Default implementation falls back to the per-record API so existing
    # backends keep working; fast backends (Tiny, HF) override directly.

    def score_batch(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
        temperature: float = 1.0,
    ) -> tuple[Any, Any]:
        """Batched counterpart of ``score``.

        Returns:
            logprobs: FloatTensor [B, T_max], differentiable per-token logπ,
                      zero-padded to T_max.
            mask:     BoolTensor  [B, T_max], True at valid response positions.

        Padding convention: responses are right-padded with zeros. Prompts
        are left-padded internally (see each backend's implementation) so
        the last T tokens always correspond to the response.
        """
        import torch as _torch

        per_record: list[_torch.Tensor] = [
            self.score(prompt_ids_list[i], response_ids_list[i], temperature=temperature)
            for i in range(len(prompt_ids_list))
        ]
        if not per_record:
            empty = _torch.zeros(0, 0)
            return empty, _torch.zeros(0, 0, dtype=_torch.bool)
        T_max = max(t.numel() for t in per_record)
        if T_max == 0:
            B = len(per_record)
            return (
                _torch.zeros(B, 0, dtype=per_record[0].dtype, device=per_record[0].device),
                _torch.zeros(B, 0, dtype=_torch.bool, device=per_record[0].device),
            )
        device = per_record[0].device
        dtype = per_record[0].dtype
        B = len(per_record)
        out = _torch.zeros(B, T_max, dtype=dtype, device=device)
        mask = _torch.zeros(B, T_max, dtype=_torch.bool, device=device)
        # IMPORTANT: do NOT .detach() — we need the graph to flow through.
        for i, t in enumerate(per_record):
            n = t.numel()
            if n == 0:
                continue
            out[i, :n] = t
            mask[i, :n] = True
        return out, mask

    def score_with_value_batch(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
        temperature: float = 1.0,
    ) -> tuple[Any, Any, Any, Any]:
        """Batched counterpart of ``score_with_value``.

        Returns:
            logprobs: [B, T_max]
            entropies: [B, T_max]
            values: [B, T_max]
            mask: [B, T_max] bool
        """
        if not self.supports_value_head():
            raise NotImplementedError("this backend has no value head")
        import torch as _torch

        trips: list[tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]] = [
            self.score_with_value(
                prompt_ids_list[i],
                response_ids_list[i],
                temperature=temperature,
            )
            for i in range(len(prompt_ids_list))
        ]
        if not trips:
            empty = _torch.zeros(0, 0)
            return empty, empty, empty, _torch.zeros(0, 0, dtype=_torch.bool)
        T_max = max(t[0].numel() for t in trips)
        device = trips[0][0].device
        dtype = trips[0][0].dtype
        B = len(trips)
        if T_max == 0:
            empty = _torch.zeros(B, 0, dtype=dtype, device=device)
            return empty, empty, empty, _torch.zeros(B, 0, dtype=_torch.bool, device=device)
        logp = _torch.zeros(B, T_max, dtype=dtype, device=device)
        ent = _torch.zeros(B, T_max, dtype=dtype, device=device)
        val = _torch.zeros(B, T_max, dtype=dtype, device=device)
        mask = _torch.zeros(B, T_max, dtype=_torch.bool, device=device)
        for i, (lp, e, v) in enumerate(trips):
            n = lp.numel()
            if n == 0:
                continue
            logp[i, :n] = lp
            ent[i, :n] = e
            val[i, :n] = v
            mask[i, :n] = True
        return logp, ent, val, mask

    # ------------------------------------------------------------------
    # v0.8 capability flags (for weight sync, device-aware ops)
    # ------------------------------------------------------------------

    @property
    def device(self) -> Any:
        """Device of the first trainable parameter (or CPU fallback)."""
        import torch as _torch

        for p in self.trainable_parameters():
            if hasattr(p, "device"):
                return p.device
        return _torch.device("cpu")

    @property
    def dtype(self) -> Any:
        """Dtype of the first trainable parameter (or float32 fallback)."""
        import torch as _torch

        for p in self.trainable_parameters():
            if hasattr(p, "dtype"):
                return p.dtype
        return _torch.float32
