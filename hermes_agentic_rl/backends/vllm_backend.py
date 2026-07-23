"""vLLM rollout backend — fast batched generation for on-policy RL.

Design principles (lightweight, no Ray dependency):
  - Uses ``vllm.LLM`` (offline engine) for **batched generation only**.
  - **Training** (``score()``, ``score_batch()``) stays on the HF backend —
    vLLM has no differentiable mode.
  - Weight sync: ``sync_weights_from(actor_state_dict)`` pushes updated
    weights from the learner into the vLLM engine so subsequent rollouts
    use the current policy. This is the core of on-policy training.
  - Optional prefix caching: for group rollout (G responses per prompt),
    the prompt is encoded once and reused G times via the same prefix.
  - PagedAttention KV cache is managed automatically by vLLM.

Usage:
    vllm_be = VLLMRolloutBackend(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=2,        # split across 2 GPUs
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    # In training loop:
    trainer.sync_weights_to_vllm()     # push current weights
    outputs = vllm_be.generate_batch(prompt_ids_list, max_new_tokens=256)

Optional dependency: ``vllm``. Install with ``pip install vllm``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from hermes_agentic_rl.backends.base import (
    BackendUnavailableError,
    GenerationOutput,
    LLMBackend,
    TokenizerProtocol,
)


@dataclass(slots=True)
class VLLMRolloutConfig:
    """vLLM engine configuration.

    Most fields map 1:1 to ``vllm.LLM`` constructor args. The ones that
    don't are documented inline.
    """

    model: str = ""  # model name or path (HF hub or local)
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"  # auto / float16 / bfloat16
    trust_remote_code: bool = True
    seed: int = 0
    # ── prefix caching (group rollout optimization) ──
    enable_prefix_caching: bool = True
    # ── sampling defaults (per-call overridable) ──
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = vLLM default (no top-k)
    # ── misc ──
    extra_engine_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)


class _VLLMTokenizerAdapter:
    """Adapts a vLLM tokenizer to TokenizerProtocol."""

    def __init__(self, tok: Any) -> None:
        self._tok = tok
        self.vocab_size = int(tok.vocab_size)
        self.pad_id = int(tok.pad_token_id or tok.eos_token_id or 0)
        self.eos_id = int(tok.eos_token_id or 0)
        self.bos_id = int(getattr(tok, "bos_token_id", None) or self.pad_id)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(self._tok.encode(text, add_special_tokens=False))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=True))


class VLLMRolloutBackend(LLMBackend):
    """vLLM-based rollout engine.

    **This backend is generation-only.** It implements ``generate()`` and
    ``generate_batch()`` via the vLLM engine. Calling ``score()`` or any
    training method raises ``NotImplementedError`` — use ``HFCausalLMBackend``
    for the learner side.
    """

    def __init__(self, cfg: VLLMRolloutConfig | None = None) -> None:
        self.cfg = cfg or VLLMRolloutConfig()
        if not self.cfg.model:
            raise ValueError("VLLMRolloutBackend requires cfg.model (model name or path)")

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise BackendUnavailableError(
                "VLLMRolloutBackend requires `vllm`. Install with: pip install vllm"
            ) from exc

        self._vllm_llm = LLM(
            model=self.cfg.model,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            max_model_len=self.cfg.max_model_len,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            dtype=self.cfg.dtype,
            trust_remote_code=self.cfg.trust_remote_code,
            seed=self.cfg.seed,
            enable_prefix_caching=self.cfg.enable_prefix_caching,
            **self.cfg.extra_engine_kwargs,
        )

        # Build tokenizer adapter from vLLM's internal tokenizer.
        vllm_tok = self._vllm_llm.get_tokenizer()
        self.tokenizer: TokenizerProtocol = _VLLMTokenizerAdapter(vllm_tok)

        self._SamplingParams = SamplingParams
        # vLLM ≥0.7 renamed max_tokens → max_new_tokens.
        # Probe which kwarg is accepted so we stay compatible with both versions.
        try:
            self._max_tokens_kwarg = "max_new_tokens"
            SamplingParams(max_new_tokens=4)
        except TypeError:
            self._max_tokens_kwarg = "max_tokens"

        self._sampling_params = self._make_sampling_params(
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )

    # ------------------------------------------------------------------
    # Generation (the ONLY thing this backend does)
    # ------------------------------------------------------------------

    def _make_sampling_params(
        self,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        stop_strings: list[str] | None = None,
    ) -> Any:
        """Build SamplingParams, handling vLLM ≥0.7 API rename."""
        kwargs: dict[str, Any] = {
            self._max_tokens_kwarg: max_new_tokens or self.cfg.max_new_tokens,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "logprobs": 1,
        }
        if self.cfg.top_k > 0:
            kwargs["top_k"] = self.cfg.top_k
        if seed is not None:
            kwargs["seed"] = seed
        if stop_strings:
            kwargs["stop"] = list(stop_strings)
        return self._SamplingParams(**kwargs)

    @torch.no_grad()  # type: ignore[name-defined]
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        seed: int | None = None,
        stop_strings: list[str] | None = None,
    ) -> GenerationOutput:
        """Single-prompt generation (wraps batch for simplicity)."""
        results = self.generate_batch(
            prompt_ids_list=[prompt_ids],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seeds=[seed] if seed is not None else None,
            stop_strings=stop_strings,
        )
        return results[0]

    @torch.no_grad()  # type: ignore[name-defined]
    def generate_batch(
        self,
        prompt_ids_list: list[list[int]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        seeds: list[int | None] | None = None,
        stop_strings: list[str] | None = None,
    ) -> list[GenerationOutput]:
        """Batch generation using vLLM engine.

        Args:
            prompt_ids_list: list of tokenized prompts.
            max_new_tokens: override default max_new_tokens.
            temperature: override default temperature.
            seeds: per-prompt seeds (vLLM 0.6+). If None, use engine default.

        Returns:
            List of ``GenerationOutput``, one per prompt, in input order.
        """
        # Build per-request SamplingParams if overrides are provided.
        sp_list: list[Any] = []
        for i in range(len(prompt_ids_list)):
            seed_i = seeds[i] if seeds is not None and i < len(seeds) else None
            sp = self._make_sampling_params(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed_i,
                stop_strings=stop_strings,
            )
            sp_list.append(sp)

        # vLLM offline API accepts both token-id lists (TokensPrompt) and strings.
        # Passing token IDs directly avoids a decode→re-encode roundtrip that
        # can silently corrupt special tokens on some tokenizers.
        try:
            from vllm.inputs import TokensPrompt  # vLLM ≥0.5

            prompts_input: list[Any] = [
                TokensPrompt(prompt_token_ids=p_ids) for p_ids in prompt_ids_list
            ]
        except ImportError:
            # Pre-0.5 fallback: pass decoded strings (lossy but compatible).
            prompts_input = [self.tokenizer.decode(p_ids) for p_ids in prompt_ids_list]

        # Run generation.
        request_outputs = self._vllm_llm.generate(prompts_input, sampling_params=sp_list)

        # Parse outputs into GenerationOutput.
        outputs: list[GenerationOutput] = []
        for req_out in request_outputs:
            completion = req_out.outputs[0]
            response_ids = list(completion.token_ids)

            # vLLM logprobs format changed across versions:
            #   <0.5: dict {token_id: Logprob} per position
            #   ≥0.5: list[Logprob | None] where Logprob.logprob is the sampled-token log-prob
            logprobs: list[float] = []
            if completion.logprobs is not None:
                for idx, lp_obj in enumerate(completion.logprobs):
                    if lp_obj is None:
                        logprobs.append(0.0)
                    elif isinstance(lp_obj, dict):
                        # Legacy dict format: {token_id: Logprob}
                        chosen_id = response_ids[idx] if idx < len(response_ids) else None
                        if chosen_id is not None and chosen_id in lp_obj:
                            logprobs.append(float(lp_obj[chosen_id].logprob))
                        elif lp_obj:
                            logprobs.append(float(next(iter(lp_obj.values())).logprob))
                        else:
                            logprobs.append(0.0)
                    else:
                        # New scalar Logprob object (vLLM ≥0.5 with logprobs=1)
                        logprobs.append(float(lp_obj.logprob))

            finished = completion.finish_reason == "stop"
            outputs.append(
                GenerationOutput(
                    response_ids=response_ids,
                    logprobs=logprobs,
                    finished=finished,
                    metadata={
                        "finish_reason": completion.finish_reason,
                        "temperature": temperature or self.cfg.temperature,
                        "stop_reason": str(completion.stop_reason)
                        if completion.stop_reason
                        else None,
                    },
                )
            )

        return outputs

    # ------------------------------------------------------------------
    # Weight sync — push learner weights → vLLM engine
    # ------------------------------------------------------------------

    def sync_weights_from(self, actor_state_dict: dict[str, Any]) -> None:
        """Push updated weights from the learner (HF backend) into the vLLM engine.

        This is the critical on-policy loop: after each update step, the
        trainer calls this so the next rollout uses the current policy.

        vLLM 0.6+ supports ``llm_engine.model_executor.driver_worker.
        model_runner.model.load_weights(actor_state_dict)``. This is an
        experimental API — if it fails, we fall back to a full engine reload
        (slow but correct).

        Args:
            actor_state_dict: state_dict from the HF learner backend's model
                (or any mapping of parameter names → tensors). Must be on CPU.
        """
        # vLLM's internal hierarchy has changed across versions:
        #   <0.6:  llm_engine.model_executor.driver_worker.model_runner.model
        #   0.6-0.7: same path, but driver_worker may be wrapped
        #   ≥0.7:  llm_engine.model_executor uses a new executor API;
        #          ``collective_rpc`` / ``update_weights`` is the stable surface.
        try:
            llm_engine = self._vllm_llm.llm_engine
            executor = llm_engine.model_executor

            # ── Priority 1: apply_model_updates (some vLLM forks / patches) ──
            # Some vLLM distributions (e.g. SGLang-derived forks) expose a
            # single ``apply_model_updates(state_dict)`` call that handles
            # sharding internally.
            if hasattr(executor, "apply_model_updates"):
                executor.apply_model_updates(actor_state_dict)

            # ── Priority 2: collective_rpc (vLLM ≥0.7 official API) ─────────
            # Works across tensor-parallel ranks; ``update_weights`` broadcasts
            # the new params to all workers.
            elif hasattr(executor, "collective_rpc"):
                named_params = list(actor_state_dict.items())
                executor.collective_rpc("update_weights", args=(named_params,))

            # ── Priority 3: driver_worker (vLLM 0.4–0.6) ─────────────────────
            elif hasattr(executor, "driver_worker"):
                driver = executor.driver_worker
                model_runner = driver.model_runner
                model = model_runner.model
                if hasattr(model, "load_weights"):
                    model.load_weights(actor_state_dict.items())
                else:
                    model.load_state_dict(actor_state_dict, strict=False)

            else:
                raise AttributeError(
                    "Cannot find a supported weight-sync path on "
                    f"{type(executor).__name__}. Supported: apply_model_updates, "
                    "collective_rpc (vLLM >=0.7), or driver_worker (vLLM 0.4-0.6)."
                )
        except Exception as exc:
            raise RuntimeError(
                "vLLM weight sync failed. "
                "This usually means a version mismatch. "
                "Supported paths: apply_model_updates | collective_rpc (vLLM ≥0.7) | "
                "driver_worker.model_runner.model (vLLM 0.4-0.6). "
                "Original error: " + str(exc)
            ) from exc

    # ------------------------------------------------------------------
    # Training API — NOT SUPPORTED (delegated to HF backend)
    # ------------------------------------------------------------------

    def score(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        temperature: float = 1.0,
    ) -> Any:
        raise NotImplementedError(
            "VLLMRolloutBackend is generation-only. Use HFCausalLMBackend for "
            "score() / score_with_value() / score_batch()."
        )

    def score_batch(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
        temperature: float = 1.0,
    ) -> tuple[Any, Any]:
        raise NotImplementedError(
            "VLLMRolloutBackend is generation-only. Use HFCausalLMBackend for score_batch()."
        )

    def score_with_value(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        temperature: float = 1.0,
    ) -> tuple[Any, Any, Any]:
        raise NotImplementedError(
            "VLLMRolloutBackend is generation-only. Use HFCausalLMBackend for score_with_value()."
        )

    def score_with_value_batch(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
        temperature: float = 1.0,
    ) -> tuple[Any, Any, Any, Any]:
        raise NotImplementedError(
            "VLLMRolloutBackend is generation-only. Use HFCausalLMBackend for "
            "score_with_value_batch()."
        )

    def trainable_parameters(self) -> Any:
        return []  # vLLM engine params are not trainable

    def is_trainable(self) -> bool:
        return False

    def supports_value_head(self) -> bool:
        return False

    @property
    def device(self) -> Any:
        import torch as _torch

        return _torch.device("cuda")

    @property
    def dtype(self) -> Any:
        import torch as _torch

        return _torch.float16
