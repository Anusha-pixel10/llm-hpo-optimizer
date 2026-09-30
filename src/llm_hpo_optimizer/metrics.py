"""
Multi-objective scoring and Pareto-front extraction.

All functions operate on lists of :class:`ExperimentRecord` and never
modify the adapter or optimizer -- they are pure data-transformation
utilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ExperimentRecord:
    """Persistent record of a single trial produced by
    :class:`~llm_hpo_optimizer.runner.ExperimentRunner`.

    Attributes
    ----------
    experiment_id:   Zero-based trial index (0 is always the baseline).
    is_baseline:     ``True`` for the fixed-hyperparameter baseline trial.
    seed:            Seed actually used by the adapter.
    hyperparameters: Configuration dict suggested by the optimizer.
    metrics:         Dict of metric names to float values.
    resources:       Dict of resource-usage names to float values.
    device:          Device string reported by the adapter.
    failed:          ``True`` when the adapter marked the trial as failed.
    score:           Weighted multi-objective score (set by
                     :func:`compute_scores`; ``None`` before scoring).
    """

    experiment_id: int
    is_baseline: bool
    seed: int
    hyperparameters: dict
    metrics: dict
    resources: dict
    device: str
    failed: bool = False
    score: Optional[float] = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy as a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Objective registry
# ---------------------------------------------------------------------------

#: Maps objective name -> (record attribute, metric key).
SCORE_OBJECTIVES: Dict[str, Tuple[str, str]] = {
    "val_loss":        ("metrics",   "best_val_loss"),
    "peak_memory_mb":  ("resources", "peak_memory_mb"),
    "parameter_count": ("resources", "parameter_count"),
    "training_time":   ("metrics",   "training_time_seconds"),
}

#: Default per-objective weights used by :func:`compute_scores`.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "val_loss":        0.50,
    "peak_memory_mb":  0.20,
    "parameter_count": 0.15,
    "training_time":   0.15,
}


def _extract(record: ExperimentRecord, objective: str) -> float:
    """Extract a scalar metric value from *record* by *objective* name."""
    if objective not in SCORE_OBJECTIVES:
        raise KeyError(
            f"Unknown objective {objective!r}. "
            f"Valid objectives: {list(SCORE_OBJECTIVES.keys())}"
        )
    section, key = SCORE_OBJECTIVES[objective]
    return getattr(record, section)[key]


def compute_scores(
    records: List[ExperimentRecord],
    weights: Optional[Dict[str, float]] = None,
) -> None:
    """Assign a weighted min-max normalised score to every non-baseline,
    non-failed record **in place**.

    Normalisation bounds are computed from the pool of scoreable records
    (baseline and failed trials are excluded from both scoring and bound
    estimation).  When the pool contains only one record, or all records
    share the same value for an objective, that objective contributes 0 to
    the score.

    Parameters
    ----------
    records:
        List of :class:`ExperimentRecord` objects, as returned by
        :meth:`~llm_hpo_optimizer.runner.ExperimentRunner.run`.
    weights:
        Dict mapping objective name to a non-negative float weight.  Must
        use keys from :data:`SCORE_OBJECTIVES`.  Defaults to
        :data:`DEFAULT_WEIGHTS`.
    """
    weights = weights or DEFAULT_WEIGHTS
    unknown = set(weights) - set(SCORE_OBJECTIVES)
    if unknown:
        raise KeyError(
            f"Unknown objectives in weights: {unknown}. "
            f"Valid objectives: {list(SCORE_OBJECTIVES.keys())}"
        )
    scoreable = [r for r in records if not r.is_baseline and not r.failed]
    if not scoreable:
        return
    bounds = {
        obj: (
            min(_extract(r, obj) for r in scoreable),
            max(_extract(r, obj) for r in scoreable),
        )
        for obj in weights
    }
    for r in scoreable:
        r.score = sum(
            w * (
                0.0
                if bounds[obj][1] == bounds[obj][0]
                else (
                    (_extract(r, obj) - bounds[obj][0])
                    / (bounds[obj][1] - bounds[obj][0])
                )
            )
            for obj, w in weights.items()
        )


def pareto_front(
    records: List[ExperimentRecord],
    objectives: Optional[List[str]] = None,
) -> List[ExperimentRecord]:
    """Return the subset of *records* that lie on the Pareto front.

    A record *a* dominates *b* when *a* is at least as good on every
    objective and strictly better on at least one.

    Baseline and failed trials are always excluded.

    Parameters
    ----------
    records:
        Full list of experiment records.
    objectives:
        Objectives to use for dominance testing.  Defaults to
        ``list(DEFAULT_WEIGHTS.keys())``.  All values must be keys of
        :data:`SCORE_OBJECTIVES`.

    Returns
    -------
    List[ExperimentRecord]
        Non-dominated records; may be empty.
    """
    objectives = objectives or list(DEFAULT_WEIGHTS.keys())
    candidates = [r for r in records if not r.is_baseline and not r.failed]

    def dominates(a: ExperimentRecord, b: ExperimentRecord) -> bool:
        av = [_extract(a, o) for o in objectives]
        bv = [_extract(b, o) for o in objectives]
        return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))

    return [c for c in candidates if not any(dominates(o, c) for o in candidates if o is not c)]
