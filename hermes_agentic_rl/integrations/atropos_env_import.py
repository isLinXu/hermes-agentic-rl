"""Run **any atropos environment** inside a hermes trainer.

Architecture (zero HTTP, zero vLLM)::

    ┌─ hermes GRPOTrainer / PPOTrainer ─────────────────────────────┐
    │  env = AtroposEnvAdapter(atropos_env)                         │
    │  policy = TinyCausalLM / HFCausalLM (+ optional LoRA)         │
    │  reward_manager = RewardManager([AtroposRewardComponent(env)])│
    └───────────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌──── AtroposEnvAdapter(BaseEnv, hermes) ────┐
          │                                             │
          │   atropos_env.server = ServerManager([      │
          │     HermesAPIServer(backend, tokenizer)     │
          │   ])                                        │
          │                                             │
          │   tokenizer: *shared* HF tokenizer          │
          │                                             │
          │   get_next_item:    atropos_env.get_next_item()          │
          │   format_prompt:    atropos_env.tokenizer.apply_chat_template │
          │   (optional) score: atropos_env.score(group_data)        │
          └──────────────────────────────────────────────────────────┘

Key design choices:

- **No HTTP**: HermesAPIServer inherits atropos ``APIServer`` and overrides
  the three ``_*_wrapper`` methods. atropos ``managed_server()`` dispatches
  based on ``isinstance(server, OpenAIServer)`` — since we inherit from
  the raw ``APIServer`` (not ``OpenAIServer``), it goes down the "real"
  ``ManagedServer`` path and happily consumes our token/logprob streams.

- **Shared tokenizer**: the atropos env constructs a HF tokenizer from
  ``config.tokenizer_name``. We wrap that same instance in a
  hermes-compatible ``TokenizerProtocol`` adapter so token IDs are
  identical across rollout-time and train-time contexts.

- **No bypassing of atropos's own training path**: we do *not* call
  ``atropos_env.collect_trajectories``. The hermes trainer runs its own
  rollouts (via ``PolicyAgentLoop`` or ``MultiTurnAgentLoop``) so that
  the old_logprobs it records are exactly the on-policy logprobs under
  the learner's current policy — correct for PPO/GRPO.

- **Reward via env.score (when available)**: if the atropos env exposes
  an async ``score`` method matching the common signature (a list of
  ``[{"messages": [...]}, ...]`` dicts), ``AtroposRewardComponent`` uses
  it. Otherwise the user provides a bespoke reward.

- **Graceful degradation**: missing ``atroposlib`` raises
  ``AtroposUnavailableError`` from any public constructor; the import of
  this module itself never fails.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass
from typing import Any

from hermes_agentic_rl.backends.base import LLMBackend, TokenizerProtocol
from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv

# ---------------------------------------------------------------------------
# Optional-import guard
# ---------------------------------------------------------------------------


class AtroposUnavailableError(RuntimeError):
    """Raised when `atroposlib` is needed but not installed / importable."""


def _require_atropos(full: bool = False) -> Any:
    """Return atroposlib modules, or raise AtroposUnavailableError.

    Lazy + layered imports so we can load ``HermesAPIServer`` (which only
    needs ``server_baseline``) even when ``atroposlib.envs.base`` is
    unimportable due to its heavy deps (jsonlines/wandb/datasets). Passing
    ``full=True`` also requires the full ``BaseEnv`` — this is only needed
    by ``AtroposEnvAdapter`` when constructing an env from a class.
    """
    try:
        from atroposlib.envs.server_handling.server_baseline import (
            APIServer,
            APIServerConfig,
            ReasoningConfig,
        )
    except Exception as exc:  # pragma: no cover — depends on env
        raise AtroposUnavailableError(
            "atroposlib is not importable. Install it with:\n"
            "    pip install -e ./atropos\n"
            "(or `pip install atroposlib`)."
        ) from exc
    out: dict[str, Any] = {
        "APIServer": APIServer,
        "APIServerConfig": APIServerConfig,
        "ReasoningConfig": ReasoningConfig,
    }
    if full:
        try:
            from atroposlib.envs.base import BaseEnv as AtroposBaseEnv
        except Exception as exc:  # pragma: no cover
            raise AtroposUnavailableError(
                "atroposlib.envs.base is not importable — usually because of\n"
                "missing heavy deps (jsonlines/wandb/datasets). Install:\n"
                "    pip install jsonlines wandb datasets\n"
                "or install atroposlib's extras: `pip install -e ./atropos`."
            ) from exc
        out["AtroposBaseEnv"] = AtroposBaseEnv
    return out


# ---------------------------------------------------------------------------
# In-process APIServer backed by a hermes LLMBackend
# ---------------------------------------------------------------------------


def _build_hermes_api_server_cls() -> type:
    """Dynamically build HermesAPIServer *as a subclass of atropos APIServer*.

    We can't reference ``APIServer`` at module-load time (atropos is
    optional), so we build the subclass lazily on first call.
    """
    mods = _require_atropos()
    APIServer = mods["APIServer"]
    APIServerConfig = mods["APIServerConfig"]
    ReasoningConfig = mods["ReasoningConfig"]

    # pull in OpenAI types (atroposlib depends on openai, so these will exist)
    from openai.types.chat.chat_completion import (
        ChatCompletion,
        ChatCompletionMessage,
    )
    from openai.types.chat.chat_completion import (
        Choice as ChatChoice,
    )
    from openai.types.completion import Completion, CompletionChoice
    from openai.types.completion_usage import CompletionUsage

    class HermesAPIServer(APIServer):  # type: ignore[misc,valid-type]
        """Drop-in atropos APIServer that delegates generation to a hermes backend.

        Serves three atropos entry points:
          - ``_chat_completion_wrapper``    → OpenAI ChatCompletion
          - ``_completion_wrapper``         → OpenAI Completion
          - ``_tokens_and_logprobs_completion_wrapper`` →
                (prompt_tokens, output_tokens, output_logprobs, finish_reasons)

        The backend can be swapped at runtime by assigning to ``self.backend``
        (the atropos env keeps the *same* APIServer instance across rollouts,
        which is exactly what we want so weight updates on the hermes side
        are picked up here with no plumbing).
        """

        def __init__(
            self,
            backend: LLMBackend,
            hf_tokenizer: Any,
            *,
            model_name: str = "hermes-local",
            max_new_tokens: int = 128,
            default_temperature: float = 1.0,
            config: Any | None = None,
            reasoning_config: Any | None = None,
        ) -> None:
            cfg = config or APIServerConfig(
                model_name=model_name,
                server_type="vllm",  # important: NOT "openai" so managed_server works
                base_url=None,
                api_key="",
                health_check=False,
                num_max_requests_at_once=32,
                num_requests_for_eval=8,
                timeout=120,
            )
            super().__init__(cfg, reasoning_config=reasoning_config or ReasoningConfig())
            self.backend = backend
            self.hf_tokenizer = hf_tokenizer
            self.max_new_tokens = max_new_tokens
            self.default_temperature = default_temperature
            self.server_healthy = True
            self.initialized = True

        # -- helpers --------------------------------------------------------

        def _messages_to_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
            """Render messages → token IDs via the shared HF chat template.

            Uses ``add_generation_prompt=True`` so the ids end right before
            where the assistant reply should start — the same convention
            atropos's own managed_server uses.
            """
            try:
                return list(
                    self.hf_tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                )
            except Exception:
                # Fallback: concat with role tags
                rendered = "\n".join(
                    f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}" for m in messages
                )
                ids = list(self.hf_tokenizer.encode(rendered, add_special_tokens=False))
                return ids

        async def _generate_group(
            self,
            prompt_ids: list[int],
            n: int,
            max_tokens: int | None,
            temperature: float | None,
            stop_ids: list[int] | None = None,
            seed: int | None = None,
            prompt_text: str | None = None,
        ) -> list[dict[str, Any]]:
            """Run n independent rollouts and collect (text, tokens, logprobs, finish).

            Two modes, auto-detected:

            1) *Direct*: the backend shares the HF tokenizer's vocab (e.g.
               ``HFCausalLMBackend`` pointed at the same model) — we feed
               the HF prompt_ids straight in. Tokens reported back are
               the backend's raw token ids. High-fidelity path for real
               training.

            2) *Text-level bridge*: the backend has a different vocab
               (e.g. ``TinyCausalLMBackend`` with a char tokenizer). We
               decode the HF prompt_ids to text, re-encode with the
               backend's tokenizer, run generate, decode the response to
               text, then re-encode with the HF tokenizer so the tokens
               atropos stores are still in HF vocab. Logprobs are
               stretched/padded per-HF-token with placeholder values
               (1.0 for masked positions, averaged backend logprob for
               new positions — atropos's downstream trainer typically
               ignores or renormalizes these). Fidelity is low but the
               pipeline runs end-to-end on CPU with no HF model needed.
            """
            max_new = int(max_tokens or self.max_new_tokens)
            temp = float(temperature if temperature is not None else self.default_temperature)

            # Detect vocab alignment by checking a couple of prompt ids against
            # the backend tokenizer's vocab_size (cheap & deterministic).
            backend_vocab = int(getattr(self.backend.tokenizer, "vocab_size", 0))
            hf_vocab = int(self.hf_tokenizer.vocab_size)
            # Conservative: require exact vocab_size match AND id in range.
            direct = (
                backend_vocab == hf_vocab
                and all(0 <= i < backend_vocab for i in prompt_ids[:32])
            )

            if prompt_text is None:
                # Decode the HF prompt to text so we can re-encode for the backend.
                try:
                    prompt_text = self.hf_tokenizer.decode(
                        list(prompt_ids), skip_special_tokens=False
                    )
                except Exception:
                    prompt_text = ""

            def _one(seed_i: int | None) -> dict[str, Any]:
                if direct:
                    ids_in = list(prompt_ids)
                else:
                    ids_in = [self.backend.tokenizer.bos_id] + list(
                        self.backend.tokenizer.encode(prompt_text)
                    )
                gen = self.backend.generate(
                    prompt_ids=ids_in,
                    max_new_tokens=max_new,
                    temperature=temp,
                    seed=seed_i,
                )
                text = self.backend.tokenizer.decode(gen.response_ids)
                finish = "stop" if gen.finished else "length"

                if direct:
                    return {
                        "text": text,
                        "tokens": list(gen.response_ids),
                        "logprobs": list(gen.logprobs),
                        "finish": finish,
                    }
                # Bridge: re-encode in HF vocab for atropos consumers.
                hf_tokens = list(
                    self.hf_tokenizer.encode(text, add_special_tokens=False)
                )
                if not hf_tokens:
                    hf_tokens = [self.hf_tokenizer.eos_token_id or 0]
                # Evenly spread the backend's total logprob across HF tokens.
                total_lp = sum(gen.logprobs) if gen.logprobs else 0.0
                n_tok = len(hf_tokens)
                per = total_lp / max(1, n_tok)
                hf_logprobs = [per] * n_tok
                return {
                    "text": text,
                    "tokens": hf_tokens,
                    "logprobs": hf_logprobs,
                    "finish": finish,
                }

            # Run in a worker thread pool so we don't block the event loop for
            # n×compute even if the backend is torch-on-cpu.
            loop = asyncio.get_running_loop()
            seeds = [None if seed is None else seed + i for i in range(n)]
            tasks = [loop.run_in_executor(None, _one, s) for s in seeds]
            results = await asyncio.gather(*tasks)
            return results

        # -- atropos APIServer overrides -----------------------------------

        async def _chat_completion_wrapper(self, **kwargs):  # type: ignore[override]
            messages = kwargs.get("messages") or []
            n = int(kwargs.get("n", 1))
            max_tokens = kwargs.get("max_tokens")
            temperature = kwargs.get("temperature")
            seed = kwargs.get("seed")
            prompt_ids = self._messages_to_prompt_ids(messages)
            # Cheap textual fallback for the bridge path.
            prompt_text = "\n".join(
                f"[{m.get('role','user')}] {m.get('content','')}" for m in messages
            )
            gens = await self._generate_group(
                prompt_ids, n, max_tokens, temperature, seed=seed, prompt_text=prompt_text
            )

            choices = []
            for i, g in enumerate(gens):
                choices.append(
                    ChatChoice(
                        index=i,
                        finish_reason=g["finish"],
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=g["text"],
                        ),
                        logprobs=None,
                    )
                )
            return ChatCompletion(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                object="chat.completion",
                created=int(time.time()),
                model=self.config.model_name,
                choices=choices,
                usage=CompletionUsage(
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=sum(len(g["tokens"]) for g in gens),
                    total_tokens=len(prompt_ids) + sum(len(g["tokens"]) for g in gens),
                ),
            )

        async def _completion_wrapper(self, **kwargs):  # type: ignore[override]
            prompt = kwargs.get("prompt")
            n = int(kwargs.get("n", 1))
            max_tokens = kwargs.get("max_tokens")
            temperature = kwargs.get("temperature")
            seed = kwargs.get("seed")
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
                prompt_ids = list(prompt)  # already tokens
                prompt_text = None
            else:
                prompt_text = str(prompt)
                prompt_ids = list(
                    self.hf_tokenizer.encode(prompt_text, add_special_tokens=False)
                )
            gens = await self._generate_group(
                prompt_ids, n, max_tokens, temperature, seed=seed, prompt_text=prompt_text
            )

            choices = []
            for i, g in enumerate(gens):
                choices.append(
                    CompletionChoice(
                        text=g["text"],
                        index=i,
                        finish_reason=g["finish"],
                        logprobs=None,
                    )
                )
            return Completion(
                id=f"cmpl-{uuid.uuid4().hex[:12]}",
                object="text_completion",
                created=int(time.time()),
                model=self.config.model_name,
                choices=choices,
                usage=CompletionUsage(
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=sum(len(g["tokens"]) for g in gens),
                    total_tokens=len(prompt_ids) + sum(len(g["tokens"]) for g in gens),
                ),
            )

        async def _tokens_and_logprobs_completion_wrapper(self, **kwargs):  # type: ignore[override]
            """Return (prompt_tokens, output_tokens[n], output_logprobs[n], finish[n])."""
            prompt = kwargs.get("prompt")
            n = int(kwargs.get("n", 1))
            max_tokens = kwargs.get("max_tokens")
            temperature = kwargs.get("temperature")
            seed = kwargs.get("seed")
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
                prompt_ids = list(prompt)
                prompt_text = None
            else:
                prompt_text = str(prompt)
                prompt_ids = list(
                    self.hf_tokenizer.encode(prompt_text, add_special_tokens=False)
                )
            gens = await self._generate_group(
                prompt_ids, n, max_tokens, temperature, seed=seed, prompt_text=prompt_text
            )
            out_tokens = [g["tokens"] for g in gens]
            out_logprobs = [g["logprobs"] for g in gens]
            finish_reasons = [g["finish"] for g in gens]
            return prompt_ids, out_tokens, out_logprobs, finish_reasons

        async def check_server_status_task(self, chat_completion: bool = True) -> None:
            """No-op: we're in-process, we're always 'healthy'."""
            self.server_healthy = True

    return HermesAPIServer


# Lazy singleton cache so repeated imports don't rebuild the class each time
_HERMES_API_SERVER_CLS: type | None = None


def HermesAPIServer(*args: Any, **kwargs: Any) -> Any:
    """Public constructor (lazy-builds the subclass on first call)."""
    global _HERMES_API_SERVER_CLS
    if _HERMES_API_SERVER_CLS is None:
        _HERMES_API_SERVER_CLS = _build_hermes_api_server_cls()
    return _HERMES_API_SERVER_CLS(*args, **kwargs)


# ---------------------------------------------------------------------------
# Shared tokenizer adapter (HF tokenizer → hermes TokenizerProtocol)
# ---------------------------------------------------------------------------


class _HFTokenizerToHermes(TokenizerProtocol):
    """Minimal adapter so a HF tokenizer satisfies hermes's protocol."""

    def __init__(self, hf_tok: Any) -> None:
        self._tok = hf_tok
        self.vocab_size = int(getattr(hf_tok, "vocab_size", len(hf_tok)))
        pad = hf_tok.pad_token_id
        eos = hf_tok.eos_token_id
        bos = hf_tok.bos_token_id
        self.pad_id = int(pad if pad is not None else (eos if eos is not None else 0))
        self.eos_id = int(eos) if eos is not None else self.pad_id
        self.bos_id = int(bos) if bos is not None else self.pad_id

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(self._tok.encode(text, add_special_tokens=False))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=True))


# ---------------------------------------------------------------------------
# The env adapter itself
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AtroposEnvAdapterConfig:
    max_new_tokens: int = 128
    default_temperature: float = 1.0
    # If the atropos env takes a long time at setup (dataset download, etc.),
    # tests can inject a pre-built tokenizer to skip `AutoTokenizer.from_pretrained`.
    override_tokenizer: Any | None = None


class AtroposEnvAdapter(BaseEnv):
    """Wrap any ``atroposlib.envs.base.BaseEnv`` as a hermes ``BaseEnv``.

    Usage::

        from atroposlib.envs.base import APIServerConfig, BaseEnvConfig
        from my_atropos_env import MyEnv

        env_cfg = MyEnv.env_config_cls(tokenizer_name="Qwen/Qwen2.5-0.5B", ...)
        adapter = AtroposEnvAdapter(
            env_cls=MyEnv,
            env_config=env_cfg,
            backend=my_hermes_backend,
        )
        # Now `adapter` is a hermes BaseEnv; plug into GRPOTrainer:
        trainer = GRPOTrainer(my_hermes_backend, adapter, reward_manager, cfg=...)
    """

    def __init__(
        self,
        *,
        env_cls: type | None = None,
        env_instance: Any | None = None,
        env_config: Any | None = None,
        backend: LLMBackend,
        adapter_config: AtroposEnvAdapterConfig | None = None,
    ) -> None:
        if (env_cls is None) == (env_instance is None):
            raise ValueError("pass exactly one of env_cls or env_instance")
        # Only the full import is needed when constructing from a class;
        # if env_instance is provided, the caller already has a BaseEnv
        # and we only need the server_baseline types.
        mods = _require_atropos(full=(env_instance is None))
        self._atropos_mods = mods
        self.adapter_config = adapter_config or AtroposEnvAdapterConfig()
        self.backend = backend

        # Resolve tokenizer up-front
        hf_tok = self.adapter_config.override_tokenizer
        if hf_tok is None:
            from transformers import AutoTokenizer

            if env_config is None and env_cls is not None:
                # Use whatever the env's default config says
                env_config, _sc = env_cls.config_init()
            tok_name = getattr(env_config, "tokenizer_name", None) or "gpt2"
            hf_tok = AutoTokenizer.from_pretrained(tok_name)
            if hf_tok.pad_token_id is None:
                hf_tok.pad_token = hf_tok.eos_token or "<|pad|>"
        self.hf_tokenizer = hf_tok

        # Build (or adopt) the atropos env
        if env_instance is not None:
            self.atropos_env = env_instance
        else:
            # We must control the server_configs so that atropos's
            # __init__ doesn't blow up trying to hit a real HTTP endpoint.
            api_cfg = mods["APIServerConfig"](
                model_name="hermes-local",
                server_type="vllm",
                base_url=None,
                api_key="",
                health_check=False,
            )
            self.atropos_env = env_cls(
                config=env_config,
                server_configs=[api_cfg],
                slurm=False,
                testing=True,
            )
            # Replace auto-constructed tokenizer with the one we already have
            # (this keeps token IDs aligned between adapter and env).
            self.atropos_env.tokenizer = self.hf_tokenizer

        # Hot-swap the server manager's servers with OUR in-process APIServer
        api_cls_cfg = mods["APIServerConfig"](
            model_name="hermes-local",
            server_type="vllm",
            base_url=None,
            api_key="",
            health_check=False,
        )
        self._api_server = HermesAPIServer(
            backend=backend,
            hf_tokenizer=self.hf_tokenizer,
            model_name="hermes-local",
            max_new_tokens=self.adapter_config.max_new_tokens,
            default_temperature=self.adapter_config.default_temperature,
            config=api_cls_cfg,
        )
        self.atropos_env.server.servers = [self._api_server]

        # hermes BaseEnv state
        self._setup_done = False
        self._last_item_text: str | None = None
        self._last_item_raw: Any | None = None

    # Tokenizer for hermes trainer to see
    @property
    def tokenizer(self) -> TokenizerProtocol:
        return _HFTokenizerToHermes(self.hf_tokenizer)

    # --- hermes BaseEnv API -----------------------------------------------

    async def setup(self) -> None:
        if self._setup_done:
            return
        # Some atropos envs do dataset loading in setup(); call it if present.
        setup = getattr(self.atropos_env, "setup", None)
        if callable(setup):
            res = setup()
            if inspect.isawaitable(res):
                await res
        self._setup_done = True

    async def get_next_item(self) -> dict[str, Any]:
        """Adapt atropos raw item → hermes-style dict."""
        raw = await self.atropos_env.get_next_item()
        self._last_item_raw = raw
        # Extract a textual "instruction" for hermes's format_prompt().
        # atropos items come in many shapes — we try the common ones.
        text = _extract_prompt_text(raw)
        task_id = f"atropos-{uuid.uuid4().hex[:8]}"
        self._last_item_text = text
        return {
            "task_id": task_id,
            "instruction": text,
            "raw": raw,
        }

    def format_prompt(self, item: dict[str, Any]) -> str:
        return str(item.get("instruction", ""))

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        """Adapter-provided reward using the atropos env's ``score`` hook.

        Hermes ``BaseEnv`` requires this to be implemented. In typical
        usage you'd pass a ``RewardManager`` with an
        ``AtroposRewardComponent`` to the trainer instead — both paths
        produce the same scalar, but the RewardManager form composes
        cleanly with additional reward components.
        """
        del tool_context
        text = trajectory.final_output or ""
        score = await self.score_rollout(item, text)
        if score is None:
            return [
                RewardResult(
                    name="atropos::missing_score",
                    score=0.0,
                    reason="atropos env has no score() hook or scoring failed",
                    weight=1.0,
                )
            ]
        return [
            RewardResult(
                name=f"atropos::{getattr(self.atropos_env, 'name', 'env')}",
                score=float(score),
                reason=f"atropos_score={score:.4f}",
                weight=1.0,
            )
        ]

    # Optional hook: allow the adapter to be used as a reward source
    async def score_rollout(
        self,
        item: dict[str, Any],
        response_text: str,
    ) -> float | None:
        """Return a scalar reward using the atropos env's ``score`` hook.

        atropos envs typically expose ``async def score(self,
        rollout_group_data: List) -> Optional[ScoredDataGroup]``. We
        package our single response into the format the env expects.
        If the env doesn't expose score(), returns None.
        """
        score_fn = getattr(self.atropos_env, "score", None)
        if not callable(score_fn):
            return None
        raw = item.get("raw", self._last_item_raw)
        # atropos usually calls score(rollout_group_data) where rollout_group_data is
        # a list of tuples: (messages_list, answer). We feed a one-element group.
        messages = [{"role": "user", "content": item.get("instruction", "")}]
        messages.append({"role": "assistant", "content": response_text})
        candidate = [(messages, _extract_gold_answer(raw))]
        try:
            result = score_fn(candidate)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            return None
        # ScoredDataGroup or list thereof
        if isinstance(result, dict):
            scores = result.get("scores")
            if scores:
                return float(scores[0])
        if isinstance(result, list) and result:
            inner = result[0]
            if isinstance(inner, dict):
                s = inner.get("scores") or []
                if s:
                    return float(s[0])
        return None


# ---------------------------------------------------------------------------
# Reward component driven by an atropos env's score()
# ---------------------------------------------------------------------------


class AtroposRewardComponent:
    """Reward component that defers to ``AtroposEnvAdapter.score_rollout``.

    Plug in::

        reward_manager = RewardManager([AtroposRewardComponent(adapter)])
    """

    def __init__(self, adapter: AtroposEnvAdapter, weight: float = 1.0) -> None:
        self.name = f"atropos::{getattr(adapter.atropos_env, 'name', 'env')}"
        self.adapter = adapter
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del tool_context
        text = trajectory.final_output or ""
        score = await self.adapter.score_rollout(item, text)
        if score is None:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="atropos env has no score() hook or scoring failed",
                weight=self.weight,
            )
        return RewardResult(
            name=self.name,
            score=float(score),
            reason=f"atropos_score={score:.4f}",
            weight=self.weight,
        )


# ---------------------------------------------------------------------------
# Shape-tolerant extractors
# ---------------------------------------------------------------------------


def _extract_prompt_text(raw: Any) -> str:
    """Best-effort: get a human-readable prompt from an atropos item.

    Supported shapes (covers gsm8k / math / tool-use / dataset_environment):
      - tuple/list (prompt, ..., ...)
      - dict with common keys: question / prompt / instruction / messages
      - pydantic-like object with .question / .prompt
      - string
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (tuple, list)) and raw:
        head = raw[0]
        if isinstance(head, str):
            return head
        if isinstance(head, (tuple, list)) and head and isinstance(head[-1], dict):
            last = head[-1]
            if "content" in last:
                return str(last["content"])
        if isinstance(head, dict):
            return _extract_prompt_text(head)
    if isinstance(raw, dict):
        for key in ("question", "prompt", "instruction", "problem", "text"):
            v = raw.get(key)
            if isinstance(v, str):
                return v
        msgs = raw.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict) and "content" in last:
                return str(last["content"])
    for attr in ("question", "prompt", "instruction"):
        if hasattr(raw, attr):
            v = getattr(raw, attr)
            if isinstance(v, str):
                return v
    return str(raw)


def _extract_gold_answer(raw: Any) -> Any:
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        return raw[1]
    if isinstance(raw, dict):
        for key in ("answer", "gold", "target", "ground_truth", "label"):
            if key in raw:
                return raw[key]
    for attr in ("answer", "gold", "target"):
        if hasattr(raw, attr):
            return getattr(raw, attr)
    return None
