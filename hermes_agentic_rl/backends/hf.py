"""Hugging Face Causal LM backend.

Optional dependency: ``transformers``. If missing, construction raises
``BackendUnavailableError``; imports of this module never fail.

Contract parity with ``TinyCausalLMBackend``:
  - ``generate(prompt_ids, max_new_tokens, temperature, seed)`` → GenerationOutput
  - ``score(prompt_ids, response_ids)`` → torch.Tensor[response_len]
  - ``score_with_value(...)`` → only when ``with_value_head=True`` and an
    external ``value_head`` nn.Module is attached (PPO path).
  - ``trainable_parameters()`` → params with requires_grad=True (so LoRA users
    can freeze the base automatically).

This is intentionally kept minimal. The heavy lifting (tokenizer adaptation,
chat templates, flash-attn, KV cache) is out of scope for this framework —
plug it in by subclassing if you need it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from hermes_agentic_rl.backends.base import (
    BackendUnavailableError,
    GenerationOutput,
    LLMBackend,
    TokenizerProtocol,
)


@dataclass(slots=True)
class HFBackendConfig:
    model_name_or_path: str = "sshleifer/tiny-gpt2"  # tiny default that real tests can replace
    device: str = "cpu"
    dtype: str = "float32"
    with_value_head: bool = False
    value_head_dim: int | None = None  # inferred from hidden_size if None
    trust_remote_code: bool = False
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)


class _HFTokenizerAdapter:
    """Adapts a HF PreTrainedTokenizer to our minimal TokenizerProtocol."""

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


class HFCausalLMBackend(LLMBackend):
    """HF transformers AutoModelForCausalLM backend."""

    def __init__(self, cfg: HFBackendConfig | None = None) -> None:
        if not _HAS_TORCH:
            raise BackendUnavailableError("HFCausalLMBackend requires torch")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover — exercised only when transformers absent
            raise BackendUnavailableError(
                "HFCausalLMBackend requires `transformers`. "
                "Install with: pip install -e '.[hf]'"
            ) from exc

        self.cfg = cfg or HFBackendConfig()
        self._hf_tok = AutoTokenizer.from_pretrained(
            self.cfg.model_name_or_path,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if self._hf_tok.pad_token_id is None:
            self._hf_tok.pad_token = self._hf_tok.eos_token or "<|pad|>"
        self.tokenizer: TokenizerProtocol = _HFTokenizerAdapter(self._hf_tok)
        dtype = getattr(torch, self.cfg.dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self.cfg.trust_remote_code,
            **self.cfg.extra_model_kwargs,
        ).to(self.cfg.device)

        self.value_head: nn.Module | None = None
        if self.cfg.with_value_head:
            hidden = int(
                self.cfg.value_head_dim
                or getattr(self.model.config, "hidden_size", None)
                or getattr(self.model.config, "n_embd", 768)
            )
            self.value_head = nn.Linear(hidden, 1, bias=True).to(self.cfg.device)

    # ---- helpers ----

    def _to_tensor(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor(ids, dtype=torch.long, device=self.cfg.device)

    # ---- LLMBackend API ----

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> GenerationOutput:
        self.model.eval()
        # MPS: torch.Generator(device='mps') is broken (RuntimeError: Placeholder
        # storage not allocated). Use global torch.manual_seed() which is
        # reproducible on MPS for multinomial. For CUDA/CPU we also prefer this
        # for simplicity.
        if seed is not None:
            torch.manual_seed(int(seed))
        ids = list(prompt_ids)
        response: list[int] = []
        logprobs: list[float] = []
        finished = False
        for _ in range(max_new_tokens):
            x = self._to_tensor(ids).unsqueeze(0)
            logits = self.model(x).logits[:, -1, :]
            if temperature <= 0:
                next_id = int(torch.argmax(logits, dim=-1).item())
                logp = float(torch.log_softmax(logits, dim=-1)[0, next_id].item())
            else:
                logits = logits / max(temperature, 1e-6)
                logp_all = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
                logp = float(logp_all[0, next_id].item())
            ids.append(next_id)
            response.append(next_id)
            logprobs.append(logp)
            if next_id == self.tokenizer.eos_id:
                finished = True
                break
        return GenerationOutput(
            response_ids=response,
            logprobs=logprobs,
            finished=finished,
            metadata={"temperature": temperature, "seed": seed},
        )

    def _set_mode(self) -> None:
        """Put the model in train() when grad is enabled, eval() otherwise.

        This makes dropout/LayerNorm behavior consistent between ``score``
        (GRPO) and ``score_with_value`` (PPO), and aligns with the caller's
        autograd context (e.g. ``@torch.no_grad`` → eval, training path →
        train).
        """
        self.model.train(torch.is_grad_enabled())

    def _forward_with_hidden(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits [B,T,V], last_hidden_state [B,T,D])."""
        self._set_mode()
        out = self.model(ids, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # type: ignore[union-attr]
        return out.logits, hidden

    def score(self, prompt_ids: list[int], response_ids: list[int]) -> torch.Tensor:
        full = list(prompt_ids) + list(response_ids)
        R = len(response_ids)
        if R == 0 or len(full) < 2:
            return torch.zeros(0, device=self.cfg.device)
        self._set_mode()
        x = self._to_tensor(full[:-1]).unsqueeze(0)
        targets = self._to_tensor(full[1:])
        logits = self.model(x).logits.squeeze(0)
        logp_all = torch.log_softmax(logits, dim=-1)
        per_tok = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return per_tok[-R:]

    def score_with_value(
        self, prompt_ids: list[int], response_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.value_head is None:
            raise RuntimeError("HFCausalLMBackend: with_value_head=True is required")
        full = list(prompt_ids) + list(response_ids)
        R = len(response_ids)
        if R == 0 or len(full) < 2:
            empty = torch.zeros(0, device=self.cfg.device)
            return empty, empty, empty
        x = self._to_tensor(full[:-1]).unsqueeze(0)
        targets = self._to_tensor(full[1:])
        logits, hidden = self._forward_with_hidden(x)
        logits = logits.squeeze(0)
        hidden = hidden.squeeze(0)
        logp_all = torch.log_softmax(logits, dim=-1)
        per_tok = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        ent_all = -(probs * logp_all).sum(dim=-1)
        values = self.value_head(hidden).squeeze(-1)
        return per_tok[-R:], ent_all[-R:], values[-R:]

    def trainable_parameters(self) -> Any:
        base_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.value_head is not None:
            base_params += list(self.value_head.parameters())
        return base_params

    def supports_value_head(self) -> bool:
        return self.value_head is not None

    def is_trainable(self) -> bool:
        return any(p.requires_grad for p in self.model.parameters())
