"""Optuna hyperparameter search integration for hermes-agentic-rl.

Why this exists
---------------
RL training is notoriously sensitive to hyperparameters (clip_eps, kl_coef,
entropy_coef, lr, group_size, etc.). Manual grid search is expensive and
often misses interactions between parameters. This module wraps Optuna's
TPE (Tree-structured Parzen Estimator) sampler to efficiently search the
GRPOTrainerConfig hyperparameter space.

Design
------
* **Declarative search space**: defined in YAML or programmatically via
  ``SearchSpace``. Each dimension specifies a type (float/int/categorical),
  range, and optionally a log scale flag.
* **Objective function**: runs a short training trial (reduced n_iters) and
  returns a scalar metric (e.g., mean reward, KL convergence) as the
  optimization target.
* **Pruning**: Optuna's median pruner kills bad trials early, saving compute.
* **Persistence**: trials are stored in a SQLite database for resume and
  analysis.

Usage (programmatic)::

    from hermes_agentic_rl.tuning.hparam_search import (
        HyperparameterSearch,
        SearchSpace,
        SearchDimension,
    )

    space = SearchSpace(
        dimensions=[
            SearchDimension(name="lr", type="float", low=1e-6, high=1e-3, log=True),
            SearchDimension(name="clip_eps", type="float", low=0.1, high=0.3),
            SearchDimension(name="kl_coef", type="float", low=0.001, high=0.1, log=True),
            SearchDimension(name="group_size", type="int", low=4, high=16),
        ],
    )

    search = HyperparameterSearch(
        space=space,
        base_config=my_grpo_config,
        n_trials=50,
        metric_key="reward_mean",
        direction="maximize",
    )
    best_params, best_value = search.run()

Usage (YAML)::

    hparam_search:
      n_trials: 50
      metric_key: reward_mean
      direction: maximize
      storage: sqlite:///hparam_search.db
      dimensions:
        - name: lr
          type: float
          low: 1.0e-6
          high: 1.0e-3
          log: true
        - name: clip_eps
          type: float
          low: 0.1
          high: 0.3
        - name: group_size
          type: int
          low: 4
          high: 16
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Optuna is an optional dependency.
try:
    import optuna
    from optuna.trial import TrialState

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore[assignment]


@dataclass(slots=True)
class SearchDimension:
    """A single dimension in the hyperparameter search space.

    Attributes
    ----------
    name : str
        The GRPOTrainerConfig field name (e.g., ``"lr"``, ``"clip_eps"``).
    type : "float" | "int" | "categorical"
        The parameter type.
    low : float | int | None
        Lower bound (float/int types). Inclusive.
    high : float | int | None
        Upper bound (float/int types). Inclusive for int, exclusive for float.
    log : bool
        Whether to sample on a log scale (float type only).
    choices : list | None
        Categorical choices (categorical type only).
    step : float | int | None
        Discretization step (float/int types). None = continuous.
    """

    name: str
    type: Literal["float", "int", "categorical"] = "float"
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None
    step: float | int | None = None

    def __post_init__(self) -> None:
        if self.type in ("float", "int"):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"SearchDimension {self.name!r}: type={self.type} "
                    "requires low and high"
                )
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError(
                    f"SearchDimension {self.name!r}: type=categorical "
                    "requires non-empty choices"
                )

    def sample(self, trial: Any) -> Any:
        """Sample a value for this dimension from an Optuna trial."""
        if self.type == "float":
            return trial.suggest_float(
                self.name,
                float(self.low),  # type: ignore[arg-type]
                float(self.high),  # type: ignore[arg-type]
                log=self.log,
                step=self.step if self.step is not None else None,
            )
        if self.type == "int":
            return trial.suggest_int(
                self.name,
                int(self.low),  # type: ignore[arg-type]
                int(self.high),  # type: ignore[arg-type]
                log=self.log,
                step=int(self.step) if self.step is not None else None,
            )
        return trial.suggest_categorical(self.name, self.choices)  # type: ignore[arg-type]


@dataclass(slots=True)
class SearchSpace:
    """A collection of search dimensions."""

    dimensions: list[SearchDimension] = field(default_factory=list)

    def sample_all(self, trial: Any) -> dict[str, Any]:
        """Sample all dimensions and return a {name: value} dict."""
        return {d.name: d.sample(trial) for d in self.dimensions}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SearchSpace:
        """Build a SearchSpace from a YAML ``hparam_search.dimensions`` list."""
        dims: list[SearchDimension] = []
        for d_cfg in config.get("dimensions", []):
            dims.append(
                SearchDimension(
                    name=str(d_cfg["name"]),
                    type=str(d_cfg.get("type", "float")),  # type: ignore[arg-type]
                    low=d_cfg.get("low"),
                    high=d_cfg.get("high"),
                    log=bool(d_cfg.get("log", False)),
                    choices=d_cfg.get("choices"),
                    step=d_cfg.get("step"),
                )
            )
        return cls(dimensions=dims)


class HyperparameterSearch:
    """Runs an Optuna hyperparameter search over GRPOTrainerConfig fields.

    Parameters
    ----------
    space : SearchSpace
        The search space defining which config fields to optimize.
    base_config : GRPOTrainerConfig
        The base config; each trial clones this and overrides sampled fields.
    n_trials : int
        Number of Optuna trials to run.
    metric_key : str
        The training stat key to optimize (e.g., ``"reward_mean"``).
    direction : "maximize" | "minimize"
        Optimization direction.
    trial_iters : int
        Number of training iterations per trial (should be small for speed).
    storage : str | None
        Optuna storage URL (e.g., ``"sqlite:///search.db"``). None = in-memory.
    pruner_median : bool
        Whether to use Optuna's median pruner.
    train_fn : callable | None
        Optional custom training function ``(config) -> dict[str, float]``.
        If None, uses the default ``_default_train_fn``.
    """

    def __init__(
        self,
        space: SearchSpace,
        base_config: Any,
        n_trials: int = 50,
        metric_key: str = "reward_mean",
        direction: Literal["maximize", "minimize"] = "maximize",
        trial_iters: int = 10,
        storage: str | None = None,
        pruner_median: bool = True,
        train_fn: Any | None = None,
    ) -> None:
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter search. "
                "Install with: pip install optuna"
            )
        self.space = space
        self.base_config = base_config
        self.n_trials = n_trials
        self.metric_key = metric_key
        self.direction = direction
        self.trial_iters = trial_iters
        self.storage = storage
        self.pruner_median = pruner_median
        self._train_fn = train_fn or self._default_train_fn
        self._study: Any = None

    def _default_train_fn(self, config: Any) -> dict[str, float]:
        """Default training function: runs a short GRPO training loop.

        Override with ``train_fn`` parameter for custom environments/backends.
        Returns a dict of metric values.
        """
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
        from hermes_agentic_rl.core.reward_manager import RewardManager
        from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv, build_sim_tool_dataset
        from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer

        backend = TinyCausalLMBackend(
            TinyBackendConfig(dim=32, n_layers=1, max_len=128)
        )
        env = SimToolEnv(build_sim_tool_dataset(n=8, seed=42))
        rm = RewardManager(rewards=[])

        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            trainer = GRPOTrainer(
                policy=backend,
                env=env,
                reward_manager=rm,
                cfg=config,
            )
        trainer.train()

        # Extract the mean of the metric across all iterations.
        stats_list = trainer.stats.records
        values = [
            float(r.get(self.metric_key, 0.0))
            for r in stats_list
            if isinstance(r, dict)
        ]
        mean_val = sum(values) / max(1, len(values))
        return {self.metric_key: mean_val}

    def _objective(self, trial: Any) -> float:
        """Optuna objective function."""
        # Clone the base config and override sampled fields.
        config = copy.deepcopy(self.base_config)
        sampled = self.space.sample_all(trial)

        # Override trial_iters to keep trials short.
        if hasattr(config, "n_iters"):
            config.n_iters = self.trial_iters

        for name, value in sampled.items():
            if hasattr(config, name):
                setattr(config, name, value)
            else:
                logger.warning(
                    "SearchDimension %r does not match any config field — ignored",
                    name,
                )

        # Run training and extract the metric.
        try:
            metrics = self._train_fn(config)
            return float(metrics.get(self.metric_key, 0.0))
        except Exception as e:
            logger.warning("Trial failed: %s — returning -inf", e)
            return float("-inf") if self.direction == "maximize" else float("inf")

    def run(self) -> tuple[dict[str, Any], float]:
        """Run the hyperparameter search.

        Returns
        -------
        tuple
            (best_params, best_value)
        """
        pruner = (
            optuna.pruners.MedianPruner()  # type: ignore[union-attr]
            if self.pruner_median
            else optuna.pruners.NopPruner()  # type: ignore[union-attr]
        )
        self._study = optuna.create_study(  # type: ignore[union-attr]
            direction=self.direction,
            storage=self.storage,
            pruner=pruner,
            study_name="hermes_hparam_search",
            load_if_exists=self.storage is not None,
        )
        self._study.optimize(self._objective, n_trials=self.n_trials)

        best = self._study.best_trial
        logger.info(
            "Best trial #%d: %s = %.6f, params=%s",
            best.number,
            self.metric_key,
            best.value,
            best.params,
        )
        return best.params, best.value

    @property
    def study(self) -> Any:
        """Access the underlying Optuna study for analysis."""
        return self._study

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the search results."""
        if self._study is None:
            return {"status": "not_run"}
        completed = [
            t for t in self._study.trials
            if t.state == TrialState.COMPLETE  # type: ignore[union-attr]
        ]
        return {
            "n_trials": len(self._study.trials),
            "n_completed": len(completed),
            "best_value": self._study.best_value,
            "best_params": self._study.best_params,
            "metric_key": self.metric_key,
            "direction": self.direction,
        }


# ── Factory ──


def build_search_from_config(
    config: dict[str, Any],
    base_config: Any,
) -> HyperparameterSearch | None:
    """Build a HyperparameterSearch from a YAML ``hparam_search`` config block.

    Returns None if the ``hparam_search`` block is absent.
    """
    search_cfg = config.get("hparam_search", {})
    if not search_cfg:
        return None

    if not _OPTUNA_AVAILABLE:
        logger.warning(
            "hparam_search config present but Optuna is not installed — skipping"
        )
        return None

    space = SearchSpace.from_config(search_cfg)
    if not space.dimensions:
        logger.warning("hparam_search has no dimensions — skipping")
        return None

    return HyperparameterSearch(
        space=space,
        base_config=base_config,
        n_trials=int(search_cfg.get("n_trials", 30)),
        metric_key=str(search_cfg.get("metric_key", "reward_mean")),
        direction=str(search_cfg.get("direction", "maximize")),  # type: ignore[arg-type]
        trial_iters=int(search_cfg.get("trial_iters", 10)),
        storage=search_cfg.get("storage"),
        pruner_median=bool(search_cfg.get("pruner_median", True)),
        train_fn=search_cfg.get("train_fn"),
    )
