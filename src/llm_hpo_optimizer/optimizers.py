"""
Optimizer implementations.

Both :class:`RandomSearchOptimizer` and :class:`OptunaOptimizer` expose
identical ``suggest() / report()`` interfaces so they can be swapped
transparently inside :class:`~llm_hpo_optimizer.runner.ExperimentRunner`.
"""
from __future__ import annotations

import random
from typing import Optional

from llm_hpo_optimizer.space import (
    Categorical,
    LogUniformFloat,
    LogUniformInt,
    SearchSpace,
    UniformFloat,
    UniformInt,
)


class RandomSearchOptimizer:
    """Pure random search: each call to :meth:`suggest` draws an independent
    sample from the search space.  No feedback from :meth:`report` is used.

    Parameters
    ----------
    search_space:
        The :class:`~llm_hpo_optimizer.space.SearchSpace` to sample from.
    seed:
        Optional integer seed for reproducibility.

    Examples
    --------
    ::

        from llm_hpo_optimizer import SearchSpace, Categorical, RandomSearchOptimizer
        space = SearchSpace({"lr": Categorical([1e-3, 1e-4])})
        opt = RandomSearchOptimizer(space, seed=42)
        config = opt.suggest()
        opt.report(config, objective_value=0.3)
    """

    def __init__(self, search_space: SearchSpace, seed: Optional[int] = None) -> None:
        self.search_space = search_space
        self.rng = random.Random(seed)

    def suggest(self) -> dict:
        """Return a randomly-sampled, constraint-valid configuration dict."""
        return self.search_space.sample(rng=self.rng)

    def report(self, config: dict, objective_value: float) -> None:
        """No-op for random search -- feedback is intentionally ignored."""
        pass


class OptunaOptimizer:
    """Bayesian optimisation (TPE) backed by `Optuna <https://optuna.org>`_.

    Requires ``pip install llm-hpo-optimizer[optuna]`` (i.e. ``optuna>=3.0``).

    The ``n_head`` / ``n_embd`` pair is suggested as a *single joint
    categorical* over all valid (head, embed) combinations.  Suggesting them
    independently caused TPE's per-parameter models to collapse onto
    mutually-incompatible values during testing, which a retry budget could
    not reliably fix.

    Parameters
    ----------
    search_space:
        The :class:`~llm_hpo_optimizer.space.SearchSpace` to optimise over.
    seed:
        Optional integer seed passed to Optuna's TPESampler.
    direction:
        ``"minimize"`` (default) or ``"maximize"``.
    n_startup_trials:
        Number of random trials before TPE starts fitting its model.

    Raises
    ------
    ImportError
        If ``optuna`` is not installed.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        seed: Optional[int] = None,
        direction: str = "minimize",
        n_startup_trials: int = 5,
    ) -> None:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "OptunaOptimizer requires optuna. "
                "Install it with: pip install llm-hpo-optimizer[optuna]"
            ) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        self.search_space = search_space
        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials),
        )
        self._current_trial = None

    def suggest(self) -> dict:
        """Ask Optuna for the next configuration to evaluate.

        Returns
        -------
        dict
            A constraint-valid configuration dict.

        Raises
        ------
        TypeError
            If the search space contains an unsupported spec type.
        RuntimeError
            If Optuna produces an invalid (constraint-violating) config.
        """
        trial = self.study.ask()
        config: dict = {}
        params = self.search_space.parameters
        has_joint = "n_head" in params and "n_embd" in params

        if has_joint:
            heads = params["n_head"].choices
            embds = params["n_embd"].choices
            pairs = [(h, e) for h in heads for e in embds if e % h == 0]
            if not pairs:
                raise RuntimeError(
                    "No valid (n_head, n_embd) pairs found -- check that at least one "
                    "n_embd value is divisible by at least one n_head value."
                )
            label = trial.suggest_categorical(
                "n_head_n_embd_pair", [f"{h}_{e}" for h, e in pairs]
            )
            h_str, e_str = label.split("_")
            config["n_head"], config["n_embd"] = int(h_str), int(e_str)

        for name, spec in params.items():
            if has_joint and name in ("n_head", "n_embd"):
                continue
            if isinstance(spec, Categorical):
                config[name] = trial.suggest_categorical(name, spec.choices)
            elif isinstance(spec, LogUniformFloat):
                config[name] = trial.suggest_float(name, spec.low, spec.high, log=True)
            elif isinstance(spec, UniformFloat):
                config[name] = trial.suggest_float(name, spec.low, spec.high)
            elif isinstance(spec, LogUniformInt):
                config[name] = trial.suggest_int(name, spec.low, spec.high, log=True)
            elif isinstance(spec, UniformInt):
                config[name] = trial.suggest_int(name, spec.low, spec.high)
            else:
                raise TypeError(f"Unsupported parameter spec type: {type(spec).__name__}")

        if not self.search_space.is_valid(config):
            raise RuntimeError(f"Optuna produced an invalid config: {config}")

        self._current_trial = trial
        return config

    def report(self, config: dict, objective_value: float) -> None:
        """Tell Optuna the objective value for the last suggested config.

        Parameters
        ----------
        config:
            The configuration dict returned by the preceding :meth:`suggest`
            call (not currently used, kept for API symmetry).
        objective_value:
            The scalar value to optimise (e.g. validation loss).

        Raises
        ------
        RuntimeError
            If called before :meth:`suggest`.
        """
        if self._current_trial is None:
            raise RuntimeError(
                "OptunaOptimizer.report() called before suggest(). "
                "Always call suggest() first to obtain a configuration."
            )
        self.study.tell(self._current_trial, objective_value)
