from __future__ import annotations

import torch

from hermes_agentic_rl.backends.vllm_backend import VLLMRolloutBackend


class _ExecutorWithApply:
    def __init__(self) -> None:
        self.payloads = []

    def apply_model_updates(self, payload):
        self.payloads.append(payload)


class _EngineWithApply:
    def __init__(self) -> None:
        self.model_executor = _ExecutorWithApply()


class _FakeLLM:
    def __init__(self, engine) -> None:
        self.llm_engine = engine


def test_vllm_sync_prefers_apply_model_updates() -> None:
    backend = VLLMRolloutBackend.__new__(VLLMRolloutBackend)
    engine = _EngineWithApply()
    backend._vllm_llm = _FakeLLM(engine)
    state = {"w": torch.ones(1)}

    backend.sync_weights_from(state)

    assert engine.model_executor.payloads == [state]


class _ModelWithLoadWeights:
    def __init__(self) -> None:
        self.items = None

    def load_weights(self, items):
        self.items = list(items)


class _Runner:
    def __init__(self, model) -> None:
        self.model = model


class _Driver:
    def __init__(self, model) -> None:
        self.model_runner = _Runner(model)


class _ExecutorWithDriver:
    def __init__(self, model) -> None:
        self.driver_worker = _Driver(model)


class _EngineWithDriver:
    def __init__(self, model) -> None:
        self.model_executor = _ExecutorWithDriver(model)


def test_vllm_sync_falls_back_to_driver_model_load_weights() -> None:
    backend = VLLMRolloutBackend.__new__(VLLMRolloutBackend)
    model = _ModelWithLoadWeights()
    backend._vllm_llm = _FakeLLM(_EngineWithDriver(model))
    state = {"w": torch.ones(1)}

    backend.sync_weights_from(state)

    assert model.items == list(state.items())
