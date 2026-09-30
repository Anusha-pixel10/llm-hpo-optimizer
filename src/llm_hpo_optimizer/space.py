"""
Search-space primitives.

All parameter specs expose a single ``sample(rng)`` method that accepts a
:class:`random.Random` instance and returns a concrete value.  :class:`SearchSpace`
compose them into a joint distribution with optional constraints.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Categorical:
    """A categorical (discrete, unordered) hyperparameter.

    Parameters
    ----------
    choices:
        Non-empty list of candidate values. May be any type.

    Examples
    --------
    >>> spec = Categorical([4, 8, 16])
    >>> spec.sample(random.Random(0))
    8
    """
    choices: list

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("Categorical.choices must not be empty.")

    def sample(self, rng: random.Random) -> Any:
        """Return a uniformly-sampled element from *choices*."""
        return rng.choice(self.choices)


@dataclass
class UniformFloat:
    """A continuous hyperparameter sampled uniformly from [low, high].

    Parameters
    ----------
    low:
        Lower bound (inclusive).
    high:
        Upper bound (inclusive).

    Examples
    --------
    >>> spec = UniformFloat(0.0, 1.0)
    >>> 0.0 <= spec.sample(random.Random(0)) <= 1.0
    True
    """
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"UniformFloat: low ({self.low}) must be less than high ({self.high}).")

    def sample(self, rng: random.Random) -> float:
        """Return a uniformly-sampled float in [low, high]."""
        return rng.uniform(self.low, self.high)


@dataclass
class LogUniformFloat:
    """A continuous hyperparameter sampled uniformly in *log-space*.

    Use this for parameters such as learning rate where the meaningful scale
    is multiplicative rather than additive.

    Parameters
    ----------
    low:
        Lower bound (exclusive of zero, must be > 0).
    high:
        Upper bound.

    Examples
    --------
    >>> spec = LogUniformFloat(1e-5, 1e-2)
    >>> 1e-5 <= spec.sample(random.Random(0)) <= 1e-2
    True
    """
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0:
            raise ValueError(f"LogUniformFloat: low ({self.low}) must be > 0.")
        if self.low >= self.high:
            raise ValueError(f"LogUniformFloat: low ({self.low}) must be less than high ({self.high}).")

    def sample(self, rng: random.Random) -> float:
        """Return a log-uniformly-sampled float in [low, high]."""
        log_low, log_high = math.log(self.low), math.log(self.high)
        return math.exp(rng.uniform(log_low, log_high))


@dataclass
class UniformInt:
    """A discrete integer hyperparameter sampled uniformly from [low, high].

    Prefer this over ``Categorical`` for ordered integer ranges (e.g. number of
    layers, batch size) because Bayesian optimizers (Optuna) can exploit the
    ordering to guide search more efficiently.

    Parameters
    ----------
    low:
        Lower bound (inclusive).
    high:
        Upper bound (inclusive).

    Examples
    --------
    >>> spec = UniformInt(1, 8)
    >>> 1 <= spec.sample(random.Random(0)) <= 8
    True
    """
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"UniformInt: low ({self.low}) must be less than high ({self.high}).")

    def sample(self, rng: random.Random) -> int:
        """Return a uniformly-sampled integer in [low, high]."""
        return rng.randint(self.low, self.high)


@dataclass
class LogUniformInt:
    """A discrete integer hyperparameter sampled log-uniformly from [low, high].

    Useful for parameters like hidden size or number of attention heads where
    powers-of-two or geometric spacing is more natural than a linear grid.

    Parameters
    ----------
    low:
        Lower bound (inclusive, must be >= 1).
    high:
        Upper bound (inclusive).

    Examples
    --------
    >>> spec = LogUniformInt(4, 512)
    >>> 4 <= spec.sample(random.Random(0)) <= 512
    True
    """
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low < 1:
            raise ValueError(f"LogUniformInt: low ({self.low}) must be >= 1.")
        if self.low >= self.high:
            raise ValueError(f"LogUniformInt: low ({self.low}) must be less than high ({self.high}).")

    def sample(self, rng: random.Random) -> int:
        """Return a log-uniformly-sampled integer in [low, high]."""
        log_low, log_high = math.log(self.low), math.log(self.high)
        return max(self.low, min(self.high, round(math.exp(rng.uniform(log_low, log_high)))))


class SearchSpace:
    """A joint distribution over a named set of hyperparameters.

    Parameters
    ----------
    parameters:
        Mapping of parameter name to a spec object
        (:class:`Categorical`, :class:`UniformFloat`, :class:`LogUniformFloat`,
        :class:`UniformInt`, or :class:`LogUniformInt`).
    constraints:
        Optional list of callables ``(config: dict) -> bool``.  A sampled
        configuration is only returned when *all* constraints are satisfied.
        If the names ``n_embd`` and ``n_head`` are both present, the
        constraint ``n_embd % n_head == 0`` is added automatically.
    max_attempts:
        Maximum number of rejection-sampling attempts before raising
        :class:`RuntimeError`.

    Examples
    --------
    >>> space = SearchSpace(
    ...     {"lr": LogUniformFloat(1e-4, 1e-2), "batch": Categorical([16, 32])}
    ... )
    >>> cfg = space.sample(random.Random(0))
    >>> set(cfg.keys()) == {"lr", "batch"}
    True
    """

    def __init__(
        self,
        parameters: dict,
        constraints: Optional[List] = None,
        max_attempts: int = 100,
    ) -> None:
        if not parameters:
            raise ValueError("SearchSpace.parameters must not be empty.")
        self.parameters = parameters
        self.constraints: List = list(constraints) if constraints else []
        self.max_attempts = max_attempts
        # Automatic constraint: n_embd must be divisible by n_head.
        if "n_embd" in parameters and "n_head" in parameters:
            self.constraints.append(lambda cfg: cfg["n_embd"] % cfg["n_head"] == 0)

    def _sample_once(self, rng: random.Random) -> dict:
        return {name: spec.sample(rng) for name, spec in self.parameters.items()}

    def is_valid(self, config: dict) -> bool:
        """Return True when *config* satisfies all constraints."""
        return all(c(config) for c in self.constraints)

    def sample(self, rng: Optional[random.Random] = None) -> dict:
        """Sample a valid configuration dict using rejection sampling.

        Parameters
        ----------
        rng:
            A :class:`random.Random` instance for reproducibility.  When
            ``None``, an ephemeral ``Random()`` instance is created.

        Raises
        ------
        RuntimeError
            If no valid configuration is found after *max_attempts* draws.
        """
        rng = rng or random.Random()
        for _ in range(self.max_attempts):
            candidate = self._sample_once(rng)
            if self.is_valid(candidate):
                return candidate
        raise RuntimeError(
            f"No valid config found after {self.max_attempts} attempts -- "
            "check your search space leaves valid combinations."
        )
