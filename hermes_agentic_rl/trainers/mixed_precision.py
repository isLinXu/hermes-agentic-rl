"""Mixed-precision training utilities.

Provides:
  - :class:`AMPContext` — torch.amp autocast + GradScaler wrapper, used
    inside the trainer update loop.
  - :class:`GradientAccumulator` — lightweight step-counting state machine
    that decides when to call ``optimizer.step()`` / ``optimizer.zero_grad()``.
  - :func:`get_amp_dtype` — resolves a string ("fp16"/"bf16") to a torch dtype.

These are pure helpers; the trainer's update loop calls them, they don't
own any model/optimizer state.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any

import torch

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


def get_amp_dtype(name: str, device: torch.device | None = None) -> torch.dtype:
    """Resolve a user-facing dtype string to a torch dtype.

    - ``"fp16"`` → ``torch.float16``
    - ``"bf16"`` → ``torch.bfloat16`` (checks hardware support)
    - ``"fp32"`` / ``"float32"`` → ``torch.float32`` (no AMP)
    - ``"auto"`` → best available: bf16 > fp16 > fp32

    Raises ``ValueError`` if ``bf16`` is requested but unsupported.
    """
    name = name.lower().strip()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        if device is not None and device.type == "cpu":
            raise ValueError("bfloat16 AMP is not supported on CPU")
        # Check actual hardware support.
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError(
                "bfloat16 AMP requested but not supported on this GPU. "
                "Use fp16 or upgrade to Ampere+ (A100, RTX 3090, H100, etc.)."
            )
        return torch.bfloat16
    if name in ("fp32", "float32", "none", ""):
        return torch.float32
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    raise ValueError(f"Unknown AMP dtype: {name!r}. Expected fp16 / bf16 / fp32 / auto.")


# ---------------------------------------------------------------------------
# AMPContext
# ---------------------------------------------------------------------------


class AMPContext:
    """torch.amp autocast + GradScaler with a minimal API.

    Usage in trainer update loop::

        amp = AMPContext(dtype=torch.bfloat16, enabled=True)
        for mini_batch in batches:
            optimizer.zero_grad()
            with amp.autocast_ctx():
                loss, stats = algo.compute_loss(...)
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm)
            amp.step(optimizer)
            amp.update()
    """

    def __init__(
        self,
        dtype: torch.dtype | str = torch.float32,
        *,
        enabled: bool = True,
        init_scale: float = 2.0**16,
        growth_interval: int = 2000,
    ) -> None:
        self.dtype = get_amp_dtype(dtype) if isinstance(dtype, str) else dtype
        self.enabled = bool(enabled) and self.dtype != torch.float32
        self._scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
            init_scale=init_scale,
            growth_interval=growth_interval,
            enabled=self.enabled,
        )

    @contextlib.contextmanager
    def autocast_ctx(self) -> Generator[None, None, None]:
        """Autocast context manager for the forward pass."""
        if self.enabled:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(  # type: ignore[attr-defined]
                device_type=device_type, dtype=self.dtype
            ):
                yield
        else:
            yield

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return self._scaler.scale(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self._scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self._scaler.step(optimizer)

    def update(self) -> None:
        self._scaler.update()

    def get_scale(self) -> float:
        return float(self._scaler.get_scale())

    def state_dict(self) -> dict[str, Any]:
        return self._scaler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._scaler.load_state_dict(state)


# ---------------------------------------------------------------------------
# GradientAccumulator
# ---------------------------------------------------------------------------


class GradientAccumulator:
    """Counts micro-steps; triggers ``optimizer.step()`` every ``steps``.

    Typical usage (inside trainer update loop)::

        accum = GradientAccumulator(steps=4)
        for mini_batch in batches:
            loss, stats = algo.compute_loss(...)
            (amp.scale(loss) / accum.steps).backward()
            if accum.step(optimizer, amp):
                amp.update()        # update GradScaler after step
                grad_norm = ...     # clip if needed
    """

    def __init__(self, steps: int = 1) -> None:
        self.steps = max(1, int(steps))
        self._counter = 0
        self._total_updates = 0

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        amp_ctx: AMPContext | None = None,
    ) -> bool:
        """Increment counter. If ``steps`` reached, call ``optimizer.step()``.

        Returns ``True`` iff a real optimizer step was taken (so the caller
        can clip grads, update GradScaler, log metrics).
        """
        self._counter += 1
        if self._counter < self.steps:
            return False

        # Real step.
        if amp_ctx is not None:
            amp_ctx.unscale_(optimizer)
        optimizer.step()
        optimizer.zero_grad()
        if amp_ctx is not None:
            amp_ctx.update()
        self._counter = 0
        self._total_updates += 1
        return True

    def advance(self) -> bool:
        """Record one micro-step and report whether a real optimizer step is due."""
        self._counter += 1
        return self._counter >= self.steps

    def finish_step(self) -> None:
        """Reset the micro-step counter after the trainer applies an update."""
        if self._counter <= 0:
            return
        self._counter = 0
        self._total_updates += 1

    def has_pending(self) -> bool:
        """True when there are accumulated grads waiting to be flushed."""
        return self._counter > 0

    def remaining(self) -> int:
        """How many micro-batches until the next real step."""
        return max(0, self.steps - self._counter)

    @property
    def total_updates(self) -> int:
        return self._total_updates
