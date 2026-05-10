"""Tiny self-contained Causal LM backend.

Purpose: provide a real, trainable, CPU-friendly LLM so the GRPO loop can be
exercised end-to-end without HF/transformers/vLLM dependencies. This is the
canonical backend for MVP smoke tests and unit tests.

Architecture:
  - Character-level tokenizer over a small printable ASCII subset plus special
    tokens <pad>/<bos>/<eos>/<tool>/<arg>.
  - 2-layer Transformer decoder with tiny hidden size (default 32).
  - AdamW-ready nn.Parameters.

Intentionally NOT production-capable. Swap in `HFCausalLMBackend` for real
models; the contract (`generate` / `score` / `trainable_parameters`) is
identical.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hermes_agentic_rl.backends.base import (
    GenerationOutput,
    LLMBackend,
    TokenizerProtocol,
)

# ----------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------


_DEFAULT_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?:;'\"()[]{}/_-+=\n\t"
)


class TinyTokenizer(TokenizerProtocol):
    """Char-level tokenizer with a few special ids."""

    def __init__(self, extra_chars: str = "") -> None:
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        chars = list(_DEFAULT_CHARS + extra_chars)
        # dedupe while preserving order
        seen: set[str] = set()
        uniq_chars: list[str] = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                uniq_chars.append(c)
        tokens = specials + uniq_chars
        self._stoi: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self._itos: list[str] = tokens
        self.pad_id = self._stoi["<pad>"]
        self.bos_id = self._stoi["<bos>"]
        self.eos_id = self._stoi["<eos>"]
        self.unk_id = self._stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self._stoi.get(ch, self.unk_id) for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out: list[str] = []
        for i in ids:
            if 0 <= int(i) < len(self._itos):
                tok = self._itos[int(i)]
                if tok in {"<pad>", "<bos>", "<eos>", "<unk>"}:
                    # skip specials in human-readable decode
                    continue
                out.append(tok)
        return "".join(out)


# ----------------------------------------------------------------------------
# Tiny Transformer
# ----------------------------------------------------------------------------


class _CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 256,
        with_value_head: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([_Block(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.value_head: nn.Linear | None = nn.Linear(dim, 1, bias=True) if with_value_head else None

    def _trunk(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert self.max_len >= T, f"sequence {T} exceeds max_len {self.max_len}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)  # [B, T, D]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T] → logits [B, T, V]
        return self.head(self._trunk(idx))

    def forward_with_value(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits [B,T,V], values [B,T])."""
        h = self._trunk(idx)
        logits = self.head(h)
        if self.value_head is None:
            values = torch.zeros(idx.shape, device=idx.device, dtype=logits.dtype)
        else:
            values = self.value_head(h).squeeze(-1)  # [B, T]
        return logits, values


# ----------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class TinyBackendConfig:
    dim: int = 32
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 256
    device: str = "cpu"
    dtype: str = "float32"
    seed: int | None = 0
    extra_chars: str = ""
    tokenizer: TinyTokenizer | None = field(default=None)
    with_value_head: bool = False


class TinyCausalLMBackend(LLMBackend):
    """Self-contained, trainable LLM backend. CPU-friendly by default.

    Parameters roughly fit in <100K params for the default config — this is
    intentional: the MVP must prove the RL loop works *mechanically* (gradients
    flow, rewards affect params, advantages reduce loss), not that it can learn
    a useful LLM. For real learning, swap in HFCausalLMBackend.
    """

    def __init__(self, cfg: TinyBackendConfig | None = None) -> None:
        self.cfg = cfg or TinyBackendConfig()
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        self.tokenizer = self.cfg.tokenizer or TinyTokenizer(extra_chars=self.cfg.extra_chars)
        self.model = TinyCausalLM(
            vocab_size=self.tokenizer.vocab_size,
            dim=self.cfg.dim,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            max_len=self.cfg.max_len,
            with_value_head=self.cfg.with_value_head,
        ).to(self.cfg.device)
        self._lora_adapter: Any | None = None

    # ---- helpers ----

    def _to_tensor(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor(ids, dtype=torch.long, device=self.cfg.device)

    # ---- LLMBackend API ----

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> GenerationOutput:
        self.model.eval()
        gen = torch.Generator(device=self.cfg.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()

        ids = list(prompt_ids)
        response: list[int] = []
        logprobs: list[float] = []
        finished = False
        for _ in range(max_new_tokens):
            if len(ids) > self.cfg.max_len:
                ids = ids[-self.cfg.max_len :]
            x = self._to_tensor(ids).unsqueeze(0)  # [1, T]
            logits = self.model(x)[:, -1, :]  # [1, V]
            if temperature <= 0:
                # greedy
                next_id = int(torch.argmax(logits, dim=-1).item())
                logp = float(torch.log_softmax(logits, dim=-1)[0, next_id].detach().item())
            else:
                logits = logits / max(temperature, 1e-6)
                logp_dist = torch.log_softmax(logits, dim=-1)
                prob = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(prob, 1, generator=gen).item())
                logp = float(logp_dist[0, next_id].detach().item())
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

    def score(self, prompt_ids: list[int], response_ids: list[int]) -> torch.Tensor:
        """Differentiable per-response-token log-prob tensor. Shape: [R]."""
        logp, _ent, _val = self._score_core(prompt_ids, response_ids, need_value=False, need_entropy=False)
        return logp

    def score_with_value(
        self, prompt_ids: list[int], response_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable (per-token logπ, per-token entropy, per-token value).

        All tensors have shape ``[R]`` (response length). Value tensor is a
        differentiable critic output aligned with response tokens (the value
        at position t is V(s_t) predicted from the prompt+response prefix
        one step before emitting token t).

        Raises ``RuntimeError`` if the backend has no value head.
        """
        if self.model.value_head is None:
            raise RuntimeError(
                "score_with_value() requires with_value_head=True; "
                "rebuild backend with TinyBackendConfig(with_value_head=True)"
            )
        return self._score_core(prompt_ids, response_ids, need_value=True, need_entropy=True)

    def _score_core(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        *,
        need_value: bool,
        need_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.train()
        full = list(prompt_ids) + list(response_ids)
        if len(full) > self.cfg.max_len:
            full = full[-self.cfg.max_len :]
        R = len(response_ids)
        zero = torch.zeros(0, device=self.cfg.device)
        if len(full) < 2 or R == 0:
            return zero, zero, zero
        x = self._to_tensor(full[:-1]).unsqueeze(0)  # [1, T-1]
        targets = self._to_tensor(full[1:])
        if need_value:
            logits, values = self.model.forward_with_value(x)
            logits = logits.squeeze(0)       # [T-1, V]
            values = values.squeeze(0)       # [T-1]
        else:
            logits = self.model(x).squeeze(0)
            values = torch.zeros(logits.shape[0], device=self.cfg.device)
        logp_all = torch.log_softmax(logits, dim=-1)
        per_tok_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [T-1]

        # entropy per position (over next-token distribution)
        if need_entropy:
            probs = torch.softmax(logits, dim=-1)
            ent_all = -(probs * logp_all).sum(dim=-1)  # [T-1]
        else:
            ent_all = torch.zeros_like(per_tok_logp)

        # Align response portion: the last R positions correspond to response tokens.
        logp_r = per_tok_logp[-R:]
        val_r = values[-R:]
        ent_r = ent_all[-R:]
        return logp_r, ent_r, val_r

    def trainable_parameters(self) -> Any:
        if self._lora_adapter is not None:
            return list(self._lora_adapter.parameters())
        return [p for p in self.model.parameters() if p.requires_grad]

    def enable_lora(self, cfg: Any = None) -> Any:
        """Inject LoRA into the model; returns the LoRAAdapter handle.

        ``cfg`` is a ``hermes_agentic_rl.peft.lora.LoRAConfig`` (typed ``Any``
        here to keep ``peft`` an optional lazy import).

        After this call, ``trainable_parameters()`` yields only LoRA params
        and ``self._lora_adapter`` is set. The base weights are frozen.
        Safe to call exactly once per backend instance.
        """
        if self._lora_adapter is not None:
            raise RuntimeError("LoRA already enabled on this backend")
        from hermes_agentic_rl.peft.lora import LoRAConfig, inject_lora

        lcfg = cfg or LoRAConfig()
        self._lora_adapter = inject_lora(self.model, lcfg)
        return self._lora_adapter

    # ---- convenience ----

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def clone_frozen(self) -> TinyCausalLMBackend:
        """Return a deep-copied backend with grads disabled (for reference policy)."""
        import copy

        other = TinyCausalLMBackend(self.cfg)
        other.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        for p in other.model.parameters():
            p.requires_grad_(False)
        other.model.eval()
        return other

    def is_trainable(self) -> bool:
        return any(p.requires_grad for p in self.model.parameters())

    def supports_value_head(self) -> bool:
        return self.model.value_head is not None

    def sync_from(self, other: TinyCausalLMBackend) -> None:
        self.model.load_state_dict(other.model.state_dict())
