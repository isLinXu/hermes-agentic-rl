"""TinyCausalLMBackend: tokenizer, generate, score-with-grad."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend


def _fresh(seed: int = 0) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=2, seed=seed))


def test_tokenizer_roundtrip():
    b = _fresh()
    tok = b.tokenizer
    assert tok.vocab_size > 60
    ids = tok.encode("hello world")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    text = tok.decode(ids)
    assert "hello world" in text


def test_generate_returns_sensible_output():
    b = _fresh()
    prompt = b.tokenizer.encode("say hi")
    out = b.generate(prompt, max_new_tokens=5, temperature=1.0, seed=42)
    assert len(out.response_ids) <= 5
    assert len(out.logprobs) == len(out.response_ids)
    # logprobs must be <= 0
    assert all(lp <= 0.0 for lp in out.logprobs)


def test_greedy_is_deterministic():
    b = _fresh()
    prompt = b.tokenizer.encode("abc")
    a = b.generate(prompt, max_new_tokens=6, temperature=0.0)
    c = b.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert a.response_ids == c.response_ids


def test_score_is_differentiable():
    b = _fresh()
    prompt = b.tokenizer.encode("foo")
    response = b.tokenizer.encode("bar")
    logp = b.score(prompt, response)
    assert logp.shape == (len(response),)
    # grads flow
    (logp.sum()).backward()
    grads = [p.grad for p in b.trainable_parameters()]
    assert all(g is not None for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_clone_frozen_has_no_grad():
    b = _fresh()
    ref = b.clone_frozen()
    assert ref.is_trainable() is False
    # forward pass still works (no-grad path)
    out = ref.generate(b.tokenizer.encode("x"), max_new_tokens=3, temperature=0.0)
    assert len(out.response_ids) <= 3
