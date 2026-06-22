"""OPD hint extractor — runs an in-trainer judge to recover directive hints.

Why this exists (closes the "OPD silently dies" gap)
-----------------------------------------------------
The OPD branch (``algos/opd.py``) and the teacher-logprob filler
(``rewards/opd_teacher.py``) both require ``record.metadata["opd_hint"]`` to
be present.  Until now the *only* producers of that hint were:

  * specific environments that pre-populate ``runtime["rl"]["opd_hint"]``
    (currently just ``letter_counting``), and
  * ``NextStatePRMComponent`` when it is wired into the RewardManager.

In practice most training configs do **neither**, so ``opd_hint`` is absent,
the teacher fill is a no-op, and ``HybridAlgo`` silently degrades to plain
GRPO — exactly the "OPD dummy-fires" failure mode flagged in the
deep-analysis report.

``OPDHintExtractor`` plugs that hole by running a configurable judge *inside
the trainer*, right before the teacher fill: for every record that carries a
*next-state signal* but no hint yet, it asks the judge to extract a concise
corrective hint from ``(response, next_state)`` and stamps it onto the record.

It deliberately mirrors OpenClaw-RL §3.2 hint handling (``[HINT_START]…
[HINT_END]`` delimiters, longest-hint selection, min-length quality filter)
and adds two things OpenClaw-RL does not have:

  * an **axis-aware** selection mode (prefer the hint that targets a
    capability axis the run cares about), and
  * a **de-templating** quality filter that rejects low-information hints
    such as "be more helpful".

The extractor is *opt-in* and *fail-soft*: any judge error on a record is
swallowed (logged via warning) and the record is left hint-less, so a flaky
judge can never crash a training run.
"""

from __future__ import annotations

import asyncio
import re
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from hermes_agentic_rl.algos.base import RolloutRecord
from hermes_agentic_rl.algos.opd import extract_hint_text

# A judge function maps (response, next_state) -> hint | None. It may be sync
# or async; ``OPDHintExtractor`` normalises both.
JudgeFn = Callable[[str, str], "str | None | Awaitable[str | None]"]

# Keys we look at, in priority order, to find the next-state signal on a
# record's metadata. Envs/rewards may stash it under any of these.
_NEXT_STATE_KEYS = ("next_state", "next_state_signal", "feedback", "tool_output")

# Low-information hint patterns that should be rejected even if long enough.
# These add no token-level directional signal and only inject noise into OPD.
_LOW_INFO_PATTERNS = (
    r"^be (more|less) \w+\.?$",
    r"^try (again|harder)\.?$",
    r"^(good|nice|great) (job|work)\.?$",
    r"^improve( your)? (answer|response)\.?$",
    r"^(do|answer) better\.?$",
)
_LOW_INFO_RE = re.compile("|".join(_LOW_INFO_PATTERNS), re.IGNORECASE)


SelectMode = Literal["longest", "first", "axis_aware"]

# Capability axes and their associated keywords. Used by the ``axis_aware``
# selection mode to prefer hints that target the training run's focus dimension.
# Keys mirror the FactoredAlgo head names so they can be looked up from cfg.
_AXIS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tool_use": ("tool", "function", "call", "api", "invoke", "execute"),
    "reasoning": ("reason", "step", "logic", "deduc", "infer", "think"),
    "retrieval": ("retriev", "search", "lookup", "fetch", "query", "find"),
    "safety": ("safe", "harm", "refus", "restrict", "danger", "avoid"),
    "instruction": ("follow", "format", "instruct", "comply", "require"),
    "math": ("calculat", "arithm", "equat", "solv", "numeric", "formula"),
    "code": ("code", "function", "method", "variable", "syntax", "compil"),
}


def _axis_score(hint: str, axis: str) -> int:
    """Count keyword matches between hint and capability axis."""
    lower = hint.lower()
    keywords = _AXIS_KEYWORDS.get(axis, ())
    return sum(1 for kw in keywords if kw in lower)


@dataclass(slots=True)
class OPDHintExtractorConfig:
    """Configuration for :class:`OPDHintExtractor`.

    enabled: master switch. When False, ``extract`` is a no-op.
    min_hint_chars: reject hints shorter than this (OpenClaw-RL uses 10).
    overwrite_existing: when False (default), records that already carry an
        ``opd_hint`` (e.g. from the env / NextStatePRM) are left untouched —
        the in-trainer judge only *fills the gaps*. Set True to always re-judge.
    select: hint selection policy when a judge can emit multiple candidates.
        ``longest`` mirrors OpenClaw-RL; ``first`` takes the first passing hint;
        ``axis_aware`` prefers hints that match ``target_axis`` keywords (hermes
        differentiator — enables curriculum-aware distillation).
    target_axis: capability axis name for ``axis_aware`` selection. Must be one
        of: ``tool_use``, ``reasoning``, ``retrieval``, ``safety``,
        ``instruction``, ``math``, ``code``. Ignored for other select modes.
    reject_low_info: drop generic, low-information hints ("be more helpful").
    max_records: hard cap on judge calls per iteration (cost guard). <=0 = no
        cap. Records beyond the cap are skipped (left hint-less).
    """

    enabled: bool = True
    min_hint_chars: int = 10
    overwrite_existing: bool = False
    select: SelectMode = "longest"
    target_axis: str = "reasoning"  # used when select == "axis_aware"
    reject_low_info: bool = True
    max_records: int = 0


@dataclass(slots=True)
class HintExtractStats:
    """Diagnostics returned by :meth:`OPDHintExtractor.extract`."""

    n_records: int = 0
    n_with_next_state: int = 0
    n_judged: int = 0
    n_hints_added: int = 0
    n_rejected_quality: int = 0
    n_errors: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out = {
            "opd_hint_n_records": float(self.n_records),
            "opd_hint_n_with_next_state": float(self.n_with_next_state),
            "opd_hint_n_judged": float(self.n_judged),
            "opd_hint_n_added": float(self.n_hints_added),
            "opd_hint_n_rejected_quality": float(self.n_rejected_quality),
            "opd_hint_n_errors": float(self.n_errors),
            # The headline number: fraction of records that ended up with a
            # usable OPD hint. This is the "OPD effective rate" the report
            # wants to drive from ~0 to >40%.
            "opd_hint_effective_rate": (
                float(self.n_hints_added) / float(self.n_records) if self.n_records else 0.0
            ),
        }
        out.update(self.extra)
        return out


class OPDHintExtractor:
    """Run a judge over rollout records to recover OPD directive hints.

    Usage (inside the trainer, after rewards, before the teacher fill)::

        extractor = OPDHintExtractor(judge_fn, OPDHintExtractorConfig())
        stats = extractor.extract(batch_records)

    The judge function may be sync or async and is expected to return either a
    raw string (optionally wrapped in ``[HINT_START]…[HINT_END]``) or None.
    """

    def __init__(
        self,
        judge_fn: JudgeFn,
        cfg: OPDHintExtractorConfig | None = None,
    ) -> None:
        self._judge = judge_fn
        self.cfg = cfg or OPDHintExtractorConfig()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def extract(self, records: list[RolloutRecord]) -> HintExtractStats:
        """Synchronous entry point used by the (sync) trainer update step."""
        stats = HintExtractStats(n_records=len(records))
        if not self.cfg.enabled or not records:
            return stats
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._extract_async(records, stats))
        # Already inside a running loop (e.g. train_async): run on a worker
        # thread with its own loop to avoid "loop already running" errors.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(lambda: asyncio.run(self._extract_async(records, stats)))
            return fut.result()

    async def _extract_async(
        self, records: list[RolloutRecord], stats: HintExtractStats
    ) -> HintExtractStats:
        budget = self.cfg.max_records
        for rec in records:
            if not self.cfg.overwrite_existing:
                existing = rec.metadata.get("opd_hint")
                if isinstance(existing, str) and existing.strip():
                    continue
            next_state = _record_next_state(rec)
            if not next_state:
                continue
            stats.n_with_next_state += 1
            if budget > 0 and stats.n_judged >= budget:
                continue
            response = _record_response_text(rec)
            stats.n_judged += 1
            try:
                raw = await _maybe_await(self._judge(response, next_state))
            except Exception as exc:  # fail-soft: never crash training
                stats.n_errors += 1
                warnings.warn(
                    f"OPDHintExtractor judge raised {type(exc).__name__}: {exc}; "
                    "record left hint-less.",
                    stacklevel=2,
                )
                continue
            hint, n_rejected = self._clean_hint_with_count(raw)
            stats.n_rejected_quality += n_rejected
            if hint is None:
                continue
            rec.metadata["opd_hint"] = hint
            rec.metadata.setdefault("opd_hint_source", "in_trainer_judge")
            stats.n_hints_added += 1
        return stats

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _clean_hint_with_count(self, raw: Any) -> tuple[str | None, int]:
        """Parse judge output, apply selection policy, return (hint, n_rejected).

        The judge may return:
          - a single string (possibly ``[HINT_START]…[HINT_END]`` wrapped),
          - a ``|``-separated list of candidates (multi-hint judges), or
          - None / "NO_HINT" to signal no hint available.

        The selection policy (``cfg.select``) determines which candidate wins:
          * ``first``:      first candidate that passes quality.
          * ``longest``:    longest candidate that passes quality (OpenClaw-RL).
          * ``axis_aware``: candidate that best matches ``cfg.target_axis``
                            keywords; ties broken by length (hermes extension).

        Returns (winning_hint_or_None, n_candidates_rejected_for_quality).
        """
        if raw is None:
            return None, 0
        text = str(raw).strip()
        if not text or text.upper() == "NO_HINT":
            return None, 0

        # Try to extract all [HINT_START]…[HINT_END] spans first (multi-hint
        # judges may emit several delimited blocks in one response).
        import re as _re

        spans = _re.findall(r"\[HINT_START\](.*?)\[HINT_END\]", text, flags=_re.DOTALL)
        if spans:
            candidates = [s.strip() for s in spans if s.strip()]
        else:
            # Pipe-delimited or single hint fallback.
            raw_stripped = extract_hint_text(text) or text
            candidates = [c.strip() for c in raw_stripped.split("|") if c.strip()]

        if not candidates:
            return None, 0

        # Filter by quality; count rejects for observability.
        passing = []
        n_rejected = 0
        for c in candidates:
            if self._passes_quality(c):
                passing.append(c)
            else:
                n_rejected += 1

        if not passing:
            return None, n_rejected

        if self.cfg.select == "first":
            return passing[0], n_rejected
        elif self.cfg.select == "axis_aware":
            scored = [(_axis_score(c, self.cfg.target_axis), len(c), c) for c in passing]
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            return scored[0][2], n_rejected
        else:  # "longest" — OpenClaw-RL default
            return max(passing, key=len), n_rejected

    def _clean_hint(self, raw: Any) -> str | None:
        """Legacy single-return wrapper for backward compatibility."""
        hint, _ = self._clean_hint_with_count(raw)
        return hint

    def _passes_quality(self, hint: str) -> bool:
        if len(hint) < self.cfg.min_hint_chars:
            return False
        return not (self.cfg.reject_low_info and _LOW_INFO_RE.match(hint.strip()))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


def _record_next_state(rec: RolloutRecord) -> str | None:
    meta = rec.metadata or {}
    for key in _NEXT_STATE_KEYS:
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # Some pipelines nest the runtime block on the record.
    runtime = meta.get("runtime")
    if isinstance(runtime, dict):
        for key in _NEXT_STATE_KEYS:
            val = runtime.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return None


def _record_response_text(rec: RolloutRecord) -> str:
    meta = rec.metadata or {}
    out = meta.get("final_output")
    if isinstance(out, str):
        return out
    return ""


def build_opd_hint_extractor(
    cfg: dict[str, Any] | None,
) -> OPDHintExtractor | None:
    """Construct an :class:`OPDHintExtractor` from a YAML ``opd.hint_extractor``
    block, or return None when extraction is disabled / unconfigured.

    Supported shapes::

        opd:
          hint_extractor:
            type: rule              # rule | llm_judge | letter_counting | disabled
            min_hint_chars: 10
            select: longest         # longest | first | axis_aware
            # for type: llm_judge
            model: gpt-4o-mini
            base_url: https://api.openai.com/v1
            api_key_env: OPENAI_API_KEY
    """
    if not isinstance(cfg, dict) or not cfg:
        return None
    kind = str(cfg.get("type", "disabled")).strip().lower()
    if kind in {"", "disabled", "none", "off"}:
        return None

    extractor_cfg = OPDHintExtractorConfig(
        enabled=True,
        min_hint_chars=int(cfg.get("min_hint_chars", 10)),
        overwrite_existing=bool(cfg.get("overwrite_existing", False)),
        select=str(cfg.get("select", "longest")),  # type: ignore[arg-type]
        target_axis=str(cfg.get("target_axis", "reasoning")),
        reject_low_info=bool(cfg.get("reject_low_info", True)),
        max_records=int(cfg.get("max_records", 0)),
    )

    judge_fn: JudgeFn
    if kind in {"rule", "letter_counting"}:
        from hermes_agentic_rl.rewards.letter_counting_judge import (
            letter_counting_opd_judge_fn,
        )

        judge_fn = letter_counting_opd_judge_fn
    elif kind in {"llm_judge", "llm", "openai"}:
        from hermes_agentic_rl.rewards.llm_judge import (
            OpenAICompatibleJudgeConfig,
            OpenAICompatibleOPDJudge,
        )

        jcfg = OpenAICompatibleJudgeConfig(
            model=str(cfg.get("model", "gpt-4o-mini")),
            base_url=str(cfg.get("base_url", "https://api.openai.com/v1")),
            api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
            api_key_envs=tuple(cfg.get("api_key_envs", ()) or ()),
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 256)),
            timeout=float(cfg.get("timeout", 30.0)),
            retries=int(cfg.get("retries", 2)),
        )
        judge = OpenAICompatibleOPDJudge(jcfg)
        judge_fn = judge.extract_hint
    else:
        raise ValueError(f"unknown opd.hint_extractor.type: {kind!r}")

    # Optional content-addressed cache wrapper (cost control for LLM judges).
    cache_cfg = cfg.get("cache")
    if cache_cfg:
        from hermes_agentic_rl.rewards.judge_cache import JudgeCache, cached_judge

        max_entries = (
            int(cache_cfg.get("max_entries", 4096)) if isinstance(cache_cfg, dict) else 4096
        )
        judge_fn = cached_judge(  # type: ignore[assignment]
            judge_fn, JudgeCache(max_entries=max_entries), judge_id=f"opd_hint:{kind}"
        )

    return OPDHintExtractor(judge_fn, extractor_cfg)
