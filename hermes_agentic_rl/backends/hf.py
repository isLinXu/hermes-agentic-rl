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
from typing import Any, cast

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
    flash_attention: bool = False
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)


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
        self.bos_id = int(hf_tok.bos_token_id if hf_tok.bos_token_id is not None else self.pad_id)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(self._tok.encode(text, add_special_tokens=False))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=True))


def _response_matches_stop(
    tokenizer: TokenizerProtocol,
    response_ids: list[int],
    stop_strings: list[str] | None,
) -> bool:
    if not stop_strings:
        return False
    text = tokenizer.decode(response_ids)
    return any(stop and text.endswith(stop) for stop in stop_strings)


def _logprob_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits
    return logits / max(float(temperature), 1e-6)


class HFCausalLMBackend(LLMBackend):
    """HF transformers AutoModelForCausalLM backend."""

    def __init__(self, cfg: HFBackendConfig | None = None) -> None:
        if not _HAS_TORCH:
            raise BackendUnavailableError("HFCausalLMBackend requires torch")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover — exercised only when transformers absent
            raise BackendUnavailableError(
                "HFCausalLMBackend requires `transformers`. Install with: pip install -e '.[hf]'"
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
        extra_kw = dict(self.cfg.extra_model_kwargs)
        if self.cfg.flash_attention:
            extra_kw["attn_implementation"] = "flash_attention_2"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self.cfg.trust_remote_code,
            **extra_kw,
        ).to(self.cfg.device)
        self._gradient_checkpointing_enabled = False
        self._gradient_checkpointing_prev_use_cache: Any = None

        self.value_head: nn.Module | None = None
        if self.cfg.with_value_head:
            inferred_hidden = (
                self.cfg.value_head_dim
                if self.cfg.value_head_dim is not None
                else getattr(self.model.config, "hidden_size", None)
                or getattr(self.model.config, "n_embd", 768)
            )
            hidden = int(cast(int | str | float, inferred_hidden))
            self.value_head = nn.Linear(hidden, 1, bias=True).to(self.cfg.device)

    # ---- helpers ----

    def _to_tensor(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor(ids, dtype=torch.long, device=self.cfg.device)

    def set_gradient_checkpointing(self, enabled: bool) -> bool:
        """Toggle HF activation checkpointing and preserve ``use_cache``.

        Transformers models generally require ``config.use_cache=False`` while
        gradient checkpointing is enabled. Generation/rollout is faster and
        safer with the original cache setting, so the trainer toggles this
        around the update path.
        """
        if not hasattr(self.model, "gradient_checkpointing_enable"):
            return False

        config = getattr(self.model, "config", None)
        if enabled:
            if self._gradient_checkpointing_enabled:
                return True
            if config is not None and hasattr(config, "use_cache"):
                self._gradient_checkpointing_prev_use_cache = bool(config.use_cache)
                config.use_cache = False
            self.model.gradient_checkpointing_enable()
            enable_inputs = getattr(self.model, "enable_input_require_grads", None)
            if callable(enable_inputs):
                try:
                    enable_inputs()
                except Exception:
                    pass
            self._gradient_checkpointing_enabled = True
            return True

        if not self._gradient_checkpointing_enabled:
            return True
        disable = getattr(self.model, "gradient_checkpointing_disable", None)
        if callable(disable):
            disable()
        if (
            config is not None
            and hasattr(config, "use_cache")
            and self._gradient_checkpointing_prev_use_cache is not None
        ):
            config.use_cache = bool(self._gradient_checkpointing_prev_use_cache)
        self._gradient_checkpointing_prev_use_cache = None
        self._gradient_checkpointing_enabled = False
        return True

    # ---- LLMBackend API ----

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
        stop_strings: list[str] | None = None,
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
                policy_logits = _logprob_logits(logits, temperature)
                logp_all = torch.log_softmax(policy_logits, dim=-1)
                probs = torch.softmax(policy_logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
                logp = float(logp_all[0, next_id].item())
            ids.append(next_id)
            response.append(next_id)
            logprobs.append(logp)
            if next_id == self.tokenizer.eos_id:
                finished = True
                break
            if _response_matches_stop(self.tokenizer, response, stop_strings):
                finished = True
                break
        return GenerationOutput(
            response_ids=response,
            logprobs=logprobs,
            finished=finished,
            metadata={"temperature": temperature, "seed": seed, "stop_strings": stop_strings or []},
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

    def score(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        full = list(prompt_ids) + list(response_ids)
        R = len(response_ids)
        if R == 0 or len(full) < 2:
            return torch.zeros(0, device=self.cfg.device)
        self._set_mode()
        x = self._to_tensor(full[:-1]).unsqueeze(0)
        targets = self._to_tensor(full[1:])
        logits = self.model(x).logits.squeeze(0)
        policy_logits = _logprob_logits(logits, temperature)
        logp_all = torch.log_softmax(policy_logits, dim=-1)
        per_tok = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return per_tok[-R:]

    def score_batch(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """True batched score: one forward over a [B, T_in] padded batch."""
        logp, _ent, _val, mask = self._score_core_batch(
            prompt_ids_list,
            response_ids_list,
            need_value=False,
            need_entropy=False,
            temperature=temperature,
        )
        return logp, mask

    def score_with_value_batch(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.value_head is None:
            raise RuntimeError("score_with_value_batch requires with_value_head=True")
        return self._score_core_batch(
            prompt_ids_list,
            response_ids_list,
            need_value=True,
            need_entropy=True,
            temperature=temperature,
        )

    def _score_core_batch(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
        *,
        need_value: bool,
        need_entropy: bool,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.cfg.device
        self._set_mode()
        B = len(prompt_ids_list)
        assert len(response_ids_list) == B, "prompt/response batch size mismatch"
        if B == 0:
            empty = torch.zeros(0, 0, device=device)
            return empty, empty, empty, torch.zeros(0, 0, dtype=torch.bool, device=device)

        fulls: list[list[int]] = []
        R_list: list[int] = []
        for p_ids, r_ids in zip(prompt_ids_list, response_ids_list, strict=True):
            full = list(p_ids) + list(r_ids)
            fulls.append(full)
            R_list.append(min(len(r_ids), max(0, len(full) - 1)))

        if all(R == 0 for R in R_list) or all(len(f) < 2 for f in fulls):
            empty = torch.zeros(B, 0, device=device)
            return empty, empty, empty, torch.zeros(B, 0, dtype=torch.bool, device=device)

        L_max = max(len(f) for f in fulls)
        T_in = L_max - 1
        R_max = max(R_list)
        pad_id = int(self.tokenizer.pad_id)

        inp = torch.full((B, T_in), pad_id, dtype=torch.long, device=device)
        tgt = torch.full((B, T_in), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros(B, T_in, dtype=torch.long, device=device)
        pad_mask_in = torch.zeros(B, T_in, dtype=torch.bool, device=device)

        for i, full in enumerate(fulls):
            if len(full) < 2:
                continue
            inp_ids = full[:-1]
            tgt_ids = full[1:]
            k = len(inp_ids)
            inp[i, :k] = torch.tensor(inp_ids, dtype=torch.long, device=device)
            tgt[i, :k] = torch.tensor(tgt_ids, dtype=torch.long, device=device)
            attn_mask[i, :k] = 1
            pad_mask_in[i, :k] = True

        if need_value:
            if self.value_head is None:
                raise RuntimeError("value head is required for value scoring")
            out = self.model(inp, attention_mask=attn_mask, output_hidden_states=True)
            logits = out.logits  # [B, T_in, V]
            hidden = out.hidden_states[-1]  # [B, T_in, D]
            values_full = self.value_head(hidden).squeeze(-1)
        else:
            logits = self.model(inp, attention_mask=attn_mask).logits  # [B, T_in, V]
            values_full = torch.zeros(B, T_in, dtype=logits.dtype, device=device)

        policy_logits = _logprob_logits(logits, temperature)
        logp_all = torch.log_softmax(policy_logits, dim=-1)
        per_tok_logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T_in]
        if need_entropy:
            probs = torch.softmax(policy_logits, dim=-1)
            ent_all = -(probs * logp_all).sum(dim=-1)
        else:
            ent_all = torch.zeros_like(per_tok_logp)

        per_tok_logp = per_tok_logp * pad_mask_in.to(per_tok_logp.dtype)
        ent_all = ent_all * pad_mask_in.to(ent_all.dtype)
        values_full = values_full * pad_mask_in.to(values_full.dtype)

        logp_out = torch.zeros(B, R_max, dtype=per_tok_logp.dtype, device=device)
        ent_out = torch.zeros(B, R_max, dtype=ent_all.dtype, device=device)
        val_out = torch.zeros(B, R_max, dtype=values_full.dtype, device=device)
        mask_out = torch.zeros(B, R_max, dtype=torch.bool, device=device)

        for i, full in enumerate(fulls):
            R = R_list[i]
            if R <= 0 or len(full) < 2:
                continue
            k = len(full) - 1
            start = k - R
            logp_out[i, :R] = per_tok_logp[i, start:k]
            ent_out[i, :R] = ent_all[i, start:k]
            val_out[i, :R] = values_full[i, start:k]
            mask_out[i, :R] = True

        return logp_out, ent_out, val_out, mask_out

    def score_with_value(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        temperature: float = 1.0,
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
        policy_logits = _logprob_logits(logits, temperature)
        logp_all = torch.log_softmax(policy_logits, dim=-1)
        per_tok = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.softmax(policy_logits, dim=-1)
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

    def clone_frozen(self) -> HFCausalLMBackend:
        import copy

        other = HFCausalLMBackend(self.cfg)
        other.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        for p in other.model.parameters():
            p.requires_grad_(False)
        if self.value_head is not None and other.value_head is not None:
            other.value_head.load_state_dict(copy.deepcopy(self.value_head.state_dict()))
            for p in other.value_head.parameters():
                p.requires_grad_(False)
        other.model.eval()
        if other.value_head is not None:
            other.value_head.eval()
        return other
