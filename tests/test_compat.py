"""Tests for API stability markers (_compat module)."""

from __future__ import annotations

import warnings

import pytest

from hermes_agentic_rl._compat import (
    __stable_apis__,
    experimental,
    is_experimental,
    is_stable,
    stable,
)


def test_experimental_function_emits_warning():
    @experimental
    def foo():
        return 42

    with pytest.warns(FutureWarning, match="experimental"):
        result = foo()

    assert result == 42
    assert is_experimental(foo)


def test_experimental_class_emits_warning():
    @experimental
    class Foo:
        def __init__(self, x):
            self.x = x

    with pytest.warns(FutureWarning, match="experimental"):
        obj = Foo(10)

    assert obj.x == 10


def test_stable_decorator_sets_flag():
    @stable
    def bar():
        return "ok"

    assert getattr(bar, "_is_stable", False) is True
    assert bar() == "ok"


def test_is_stable_registry():
    assert is_stable("LLMBackend")
    assert is_stable("GRPOTrainer")
    assert is_stable("Trajectory")
    assert not is_stable("SomeRandomName")


def test_stable_apis_list_nonempty():
    assert len(__stable_apis__) > 0
    assert "GRPOTrainerConfig" in __stable_apis__


def test_experimental_function_no_warning_when_silenced():
    @experimental
    def baz():
        return "data"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert baz() == "data"
