from hermes_agentic_rl.core.registry import Registry


def test_registry_register_and_get():
    registry = Registry()

    class Demo:
        pass

    registry.register("reward", "demo", Demo)
    resolved = registry.get("reward", "demo")

    assert resolved is Demo


def test_registry_duplicate_registration_raises():
    registry = Registry()

    class Demo:
        pass

    registry.register("reward", "demo", Demo)

    try:
        registry.register("reward", "demo", Demo)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected ValueError")
