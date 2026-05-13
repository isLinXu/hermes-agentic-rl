"""LLM backend unified protocol.

A backend exposes:
  - tokenizer (encode/decode)
  - generate(prompt_ids, max_new_tokens) → (response_ids, logprobs)
  - score(prompt_ids, response_ids) → per-token logπ (for PPO/GRPO recompute)
  - score_with_value(...) → differentiable per-token (logπ, entropy, value)
  - trainable_parameters()

Implementations:
  - TinyCausalLMBackend: self-contained 2-layer Transformer, CPU-friendly,
    the canonical MVP backend for smoke tests and unit tests.
  - HFCausalLMBackend: AutoModelForCausalLM (optional, needs transformers).
  - VLLMRolloutBackend: vLLM offline engine for fast batched generation
    (generation-only; training stays on HF backend).
"""

from hermes_agentic_rl.backends.base import (
    BackendUnavailableError,
    GenerationOutput,
    LLMBackend,
    TokenizerProtocol,
)
from hermes_agentic_rl.backends.tiny import (
    TinyBackendConfig,
    TinyCausalLMBackend,
    TinyTokenizer,
)

try:  # optional
    from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend

    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False

try:  # optional
    from hermes_agentic_rl.backends.vllm_backend import (
        VLLMRolloutBackend,
        VLLMRolloutConfig,
    )

    _HAS_VLLM = True
except Exception:  # pragma: no cover
    _HAS_VLLM = False

__all__ = [
    "BackendUnavailableError",
    "GenerationOutput",
    "LLMBackend",
    "TinyBackendConfig",
    "TinyCausalLMBackend",
    "TinyTokenizer",
    "TokenizerProtocol",
]
if _HAS_HF:
    __all__.extend(["HFBackendConfig", "HFCausalLMBackend"])
if _HAS_VLLM:
    __all__.extend(["VLLMRolloutBackend", "VLLMRolloutConfig"])
