# ADR 0001 — Backend protocol over class inheritance

Date: 2026-05-07
Status: Accepted

## Context

The RL loop must call policy methods on many different LLMs:
- A self-contained tiny transformer (~30K params, CPU MVP)
- HF `AutoModelForCausalLM` (GPT-2 / SmolLM2 / Qwen)
- Future: vLLM, SGLang, user-shipped black-box APIs

Every backend needs: token sampling (`generate`), differentiable log-prob
scoring (`score`), optional value head (`score_with_value`), and a way to
enumerate trainable params (`trainable_parameters`). A tight inheritance tree
("LLMBase → HFBase → GPT2Backend") would couple our core to HF internals.

## Decision

`LLMBackend` is a **protocol** (not a base class). Any object with the five
methods below is a valid policy:

```python
generate(prompt_ids, max_new_tokens, temperature, seed) -> GenerationOutput
score(prompt_ids, response_ids) -> Tensor[T]
score_with_value(prompt_ids, response_ids) -> (logp[T], ent[T], values[T])
trainable_parameters() -> Iterable[nn.Parameter]
supports_value_head() -> bool
```

## Consequences

**Pros:**
- Zero import between trainer and any specific LLM backend → backends live in
  separate files and can be added without touching the trainer.
- Trivial to write mocks in tests.
- LoRA injection (`TinyCausalLMBackend.enable_lora`) returns new trainable
  params automatically — the trainer sees only `trainable_parameters()`.

**Cons:**
- No enforcement of method signatures at import time; type errors surface only
  at the first call.
- Some duplication between `tiny.py` and `hf.py` (the forward-pass pattern is
  similar).

**Mitigation:** `backends/base.py` declares the protocol + `BackendUnavailableError`;
tests cover the protocol shape (see `test_backends_tiny.py`).
