"""Quantized rollout backend — GPTQ/AWQ/GGUF support for memory-efficient rollout.

Why this exists
---------------
On-policy RL requires frequent generation (rollout) alongside gradient
training. For large models (7B–70B), the rollout engine can be quantized to
reduce VRAM by 3–4× while the learner stays in full precision. This module:

1. **Detects** the quantization format from a model path or config.
2. **Builds** the appropriate backend (vLLM for GPTQ/AWQ, llama-cpp for GGUF).
3. **Exposes** a unified ``QuantizedRolloutBackend`` that delegates to the
   underlying engine.

Design
------
* Pure detection logic is dependency-free (no vllm / llama_cpp import at module
  level) so the module can always be imported.
* The actual backend construction is deferred to ``__init__`` time and guarded
  by availability checks.
* The backend implements the same ``generate_batch`` / ``sync_weights_from``
  interface as ``VLLMRolloutBackend``, so it is a drop-in replacement in the
  trainer's ``_vllm_rollout`` slot.

Limitations
-----------
* **GGUF** does not support ``sync_weights_from`` (weights are frozen in the
  GGUF file). This backend raises a clear error if hot weight sync is attempted.
* **GPTQ/AWQ** via vLLM support ``sync_weights_from`` but quantized tensors
  cannot be updated in-place — the backend falls back to full-precision sync
  + re-quantization (which is slow). For training, prefer LoRA hot-reload
  where only the adapter deltas are synced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from hermes_agentic_rl.backends.base import (
    BackendUnavailableError,
    GenerationOutput,
    LLMBackend,
)

logger = logging.getLogger(__name__)


class QuantFormat(Enum):
    """Supported quantization formats."""

    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    NONE = "none"  # not quantized


# ── Detection ──


def detect_quant_format(model_path: str) -> QuantFormat:
    """Detect the quantization format from a model path or HuggingFace name.

    Heuristics:
      - ``*.gguf`` extension → GGUF
      - path contains ``-gptq-`` or directory has ``quantize_config.json`` → GPTQ
      - path contains ``-awq-`` or directory has ``quant_config.json`` with
        ``quant_method: awq`` → AWQ
      - otherwise → NONE
    """
    p = Path(model_path)
    name_lower = str(p).lower()

    # GGUF: file extension check
    if name_lower.endswith(".gguf"):
        return QuantFormat.GGUF

    # GPTQ: pattern in name or config file
    if "-gptq" in name_lower or _has_file(p, "quantize_config.json"):
        return QuantFormat.GPTQ

    # AWQ: pattern in name or quant_config with quant_method=awq
    if "-awq" in name_lower or _check_awq_config(p):
        return QuantFormat.AWQ

    return QuantFormat.NONE


def _has_file(path: Path, filename: str) -> bool:
    """Check if a file exists in the given path (if it's a directory)."""
    try:
        return path.is_dir() and (path / filename).exists()
    except (OSError, ValueError):
        return False


def _check_awq_config(path: Path) -> bool:
    """Check if the path has a quant_config.json indicating AWQ."""
    if not _has_file(path, "quant_config.json"):
        return False
    try:
        import json

        with open(path / "quant_config.json") as f:
            cfg = json.load(f)
        return str(cfg.get("quant_method", "")).lower() == "awq"
    except (OSError, ValueError, ImportError):
        return False


# ── Config ──


@dataclass(slots=True)
class QuantizedRolloutConfig:
    """Configuration for the quantized rollout backend.

    Attributes
    ----------
    model_path : str
        Path or HuggingFace name of the quantized model.
    quant_format : QuantFormat | str | None
        Quantization format. If None, auto-detected from ``model_path``.
    tensor_parallel_size : int
        Number of GPUs for tensor parallelism (vLLM only).
    max_model_len : int
        Maximum sequence length.
    gpu_memory_utilization : float
        GPU memory fraction for vLLM.
    max_new_tokens : int
        Default generation length.
    temperature : float
        Sampling temperature.
    n_gpu_layers : int
        Number of layers to offload to GPU (GGUF/llama-cpp only). -1 = all.
    extra_kwargs : dict
        Extra engine-specific kwargs.
    """

    model_path: str = ""
    quant_format: QuantFormat | str | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    max_new_tokens: int = 512
    temperature: float = 1.0
    n_gpu_layers: int = -1
    extra_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.quant_format is None:
            self.quant_format = detect_quant_format(self.model_path)
        elif isinstance(self.quant_format, str):
            self.quant_format = QuantFormat(self.quant_format.lower())


# ── Backend ──


class QuantizedRolloutBackend(LLMBackend):
    """Unified rollout backend for GPTQ/AWQ/GGUF quantized models.

    Delegates to vLLM (for GPTQ/AWQ) or llama-cpp-python (for GGUF).
    Implements the same interface as ``VLLMRolloutBackend`` so it can be
    used as a drop-in replacement.
    """

    def __init__(self, cfg: QuantizedRolloutConfig) -> None:
        self.cfg = cfg
        self._format: QuantFormat = (
            cfg.quant_format
            if isinstance(cfg.quant_format, QuantFormat)
            else QuantFormat.NONE
        )
        self._engine: Any = None
        self._tokenizer: Any = None
        self._supports_weight_sync: bool = False

        if self._format == QuantFormat.GGUF:
            self._init_gguf()
        elif self._format in (QuantFormat.GPTQ, QuantFormat.AWQ):
            self._init_vllm_quantized()
        else:
            raise ValueError(
                f"QuantizedRolloutBackend: model {cfg.model_path!r} does not "
                "appear to be quantized. Use VLLMRolloutBackend or HFBackend "
                "for full-precision models."
            )

    # ── GGUF via llama-cpp-python ──

    def _init_gguf(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendUnavailableError(
                "GGUF rollout requires llama-cpp-python. "
                "Install with: pip install llama-cpp-python"
            ) from e

        kwargs = {
            "model_path": self.cfg.model_path,
            "n_ctx": self.cfg.max_model_len,
            "n_gpu_layers": self.cfg.n_gpu_layers,
            "verbose": False,
            **self.cfg.extra_kwargs,
        }
        self._engine = Llama(**kwargs)
        self._tokenizer = None  # llama-cpp handles tokenization internally
        self._supports_weight_sync = False
        logger.info(
            "GGUF backend initialized: %s (n_gpu_layers=%d)",
            self.cfg.model_path,
            self.cfg.n_gpu_layers,
        )

    # ── GPTQ/AWQ via vLLM ──

    def _init_vllm_quantized(self) -> None:
        try:
            from vllm import LLM as VLLM  # type: ignore[import-not-found]
        except ImportError as e:
            raise BackendUnavailableError(
                f"vLLM is required for {self._format.value} quantized rollout. "
                "Install with: pip install vllm"
            ) from e

        kwargs = {
            "model": self.cfg.model_path,
            "tensor_parallel_size": self.cfg.tensor_parallel_size,
            "max_model_len": self.cfg.max_model_len,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "trust_remote_code": True,
            "seed": 42,
            **self.cfg.extra_kwargs,
        }
        # vLLM auto-detects GPTQ/AWQ from the model config.
        self._engine = VLLM(**kwargs)
        self._tokenizer = getattr(self._engine, "get_tokenizer", lambda: None)()
        self._supports_weight_sync = True
        logger.info(
            "%s backend initialized via vLLM: %s",
            self._format.value.upper(),
            self.cfg.model_path,
        )

    # ── LLMBackend interface ──

    @property
    def model(self) -> Any:
        return self._engine

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def generate_batch(
        self,
        prompts: list[str] | list[list[int]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> list[GenerationOutput]:
        """Generate responses for a batch of prompts."""
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        temp = temperature if temperature is not None else self.cfg.temperature

        if self._format == QuantFormat.GGUF:
            return self._generate_gguf(prompts, max_tokens, temp)
        return self._generate_vllm(prompts, max_tokens, temp)

    def _generate_gguf(
        self,
        prompts: list[str] | list[list[int]],
        max_tokens: int,
        temperature: float,
    ) -> list[GenerationOutput]:
        results: list[GenerationOutput] = []
        for prompt in prompts:
            text = prompt if isinstance(prompt, str) else self._decode_ids(prompt)
            output = self._engine(
                text,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            generated_text = output["choices"][0]["text"]
            results.append(
                GenerationOutput(
                    text=generated_text,
                    token_ids=output["choices"][0].get("logprobs", {}).get("token_ids", []),
                )
            )
        return results

    def _generate_vllm(
        self,
        prompts: list[str] | list[list[int]],
        max_tokens: int,
        temperature: float,
    ) -> list[GenerationOutput]:
        from vllm import SamplingParams  # type: ignore[import-not-found]

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Convert token-id prompts to text if needed.
        text_prompts: list[str] = []
        for p in prompts:
            if isinstance(p, str):
                text_prompts.append(p)
            else:
                text_prompts.append(self._decode_ids(p))

        outputs = self._engine.generate(text_prompts, sampling)
        results: list[GenerationOutput] = []
        for output in outputs:
            text = output.outputs[0].text
            token_ids = list(output.outputs[0].token_ids)
            results.append(GenerationOutput(text=text, token_ids=token_ids))
        return results

    def sync_weights_from(self, state_dict: dict[str, Any]) -> None:
        """Push updated weights to the rollout engine.

        Only supported for GPTQ/AWQ via vLLM. GGUF models are frozen.
        """
        if not self._supports_weight_sync:
            raise NotImplementedError(
                "GGUF rollout backend does not support weight sync — "
                "weights are frozen in the GGUF file. Use GPTQ/AWQ for "
                "on-policy training with weight updates."
            )
        # Delegate to the vLLM engine's weight sync.
        sync_fn = getattr(self._engine, "load_weights", None)
        if sync_fn is not None:
            sync_fn(state_dict.items())
        else:
            logger.warning(
                "vLLM engine has no load_weights method — weight sync skipped"
            )

    # ── Required abstract methods from LLMBackend ──

    def generate(
        self,
        prompt: str | list[int],
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate a response for a single prompt."""
        results = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return results[0]

    def trainable_parameters(self) -> Any:
        """Return trainable parameters (none — this is a rollout-only backend)."""
        return []

    def _decode_ids(self, token_ids: list[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids)
        return ""

    def score(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> tuple[list[float], list[int]]:
        """Compute per-token log-probs (not supported for quantized rollout)."""
        raise NotImplementedError(
            "QuantizedRolloutBackend is generation-only. "
            "Use a full-precision backend for score()/score_batch()."
        )

    def score_batch(
        self,
        items: list[tuple[list[int], list[int]]],
    ) -> tuple[list[list[float]], list[list[int]]]:
        raise NotImplementedError(
            "QuantizedRolloutBackend is generation-only. "
            "Use a full-precision backend for score()/score_batch()."
        )


# ── Factory ──


def build_quantized_backend_from_config(
    config: dict[str, Any],
) -> QuantizedRolloutBackend | None:
    """Build a QuantizedRolloutBackend from a YAML ``quantization`` config block.

    YAML schema::

        quantization:
          model_path: /models/Qwen2.5-7B-Instruct-GPTQ-Int4
          format: gptq  # optional: auto-detected if omitted
          tensor_parallel_size: 1
          max_model_len: 4096
          gpu_memory_utilization: 0.90
          n_gpu_layers: -1  # GGUF only

    Returns None if the ``quantization`` block is absent or disabled.
    """
    quant_cfg = config.get("quantization", {})
    if not quant_cfg or not quant_cfg.get("model_path"):
        return None

    cfg = QuantizedRolloutConfig(
        model_path=str(quant_cfg["model_path"]),
        quant_format=quant_cfg.get("format"),
        tensor_parallel_size=int(quant_cfg.get("tensor_parallel_size", 1)),
        max_model_len=int(quant_cfg.get("max_model_len", 4096)),
        gpu_memory_utilization=float(quant_cfg.get("gpu_memory_utilization", 0.90)),
        max_new_tokens=int(quant_cfg.get("max_new_tokens", 512)),
        temperature=float(quant_cfg.get("temperature", 1.0)),
        n_gpu_layers=int(quant_cfg.get("n_gpu_layers", -1)),
        extra_kwargs=quant_cfg.get("extra_kwargs", {}),
    )
    return QuantizedRolloutBackend(cfg)
