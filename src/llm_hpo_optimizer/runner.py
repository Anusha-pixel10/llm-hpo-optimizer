"""
Experiment runner -- orchestrates the full optimization loop.

The runner is agnostic to both the optimizer strategy and the model being
trained: it just calls ``optimizer.suggest()`` / ``optimizer.report()``
and ``adapter.run_trial()``.
"""
from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional, Protocol, runtime_checkable

from llm_hpo_optimizer.adapter import TrainingAdapter
from llm_hpo_optimizer.metrics import (
    DEFAULT_WEIGHTS,
    ExperimentRecord,
    compute_scores,
    pareto_front,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Structural type for optimizer objects accepted by :class:`ExperimentRunner`.

    Any object implementing ``suggest() -> dict`` and
    ``report(config, objective_value)`` satisfies this protocol.
    """

    def suggest(self) -> dict:
        """Return the next hyperparameter configuration to evaluate."""
        ...

    def report(self, config: dict, objective_value: float) -> None:
        """Report the objective value for the last suggested config."""
        ...


class ExperimentRunner:
    """Runs a baseline trial followed by *optimization_iterations* suggested
    trials and collects :class:`~llm_hpo_optimizer.metrics.ExperimentRecord`
    objects.

    Parameters
    ----------
    optimizer:
        Any object implementing the :class:`OptimizerProtocol` (i.e. has
        ``suggest() -> dict`` and ``report(config, objective_value)``).
        Both :class:`~llm_hpo_optimizer.optimizers.RandomSearchOptimizer` and
        :class:`~llm_hpo_optimizer.optimizers.OptunaOptimizer` satisfy this.
    adapter:
        A :class:`~llm_hpo_optimizer.adapter.TrainingAdapter` implementation.
    baseline_hyperparams:
        Fixed configuration dict used for trial 0 (the baseline).
    optimization_iterations:
        Number of optimizer-suggested trials to run (default 10).
    seed:
        Base random seed.  Trial *i* uses seed ``seed + i``.
    score_weights:
        Objective weights forwarded to
        :func:`~llm_hpo_optimizer.metrics.compute_scores`.
    method_name:
        Human-readable label stored in log messages.
    verbose:
        If ``True``, print a one-line progress summary after every trial to
        *stderr*.  Defaults to ``False``.

    Examples
    --------
    ::

        runner = ExperimentRunner(
            optimizer=opt,
            adapter=my_adapter,
            baseline_hyperparams={"lr": 1e-3},
            optimization_iterations=5,
            seed=42,
            verbose=True,
        )
        records = runner.run()
    """

    def __init__(
        self,
        optimizer: OptimizerProtocol,
        adapter: TrainingAdapter,
        baseline_hyperparams: dict,
        optimization_iterations: int = 10,
        seed: int = 1337,
        score_weights: Optional[Dict[str, float]] = None,
        method_name: str = "unknown",
        verbose: bool = False,
    ) -> None:
        if optimization_iterations < 1:
            raise ValueError("optimization_iterations must be >= 1.")
        if seed is None:
            raise ValueError("seed must be an integer, not None.")
        self.optimizer = optimizer
        self.adapter = adapter
        self.baseline_hyperparams = dict(baseline_hyperparams)  # defensive copy
        self.optimization_iterations = optimization_iterations
        self.seed = seed
        self.score_weights = score_weights or DEFAULT_WEIGHTS
        self.method_name = method_name
        self.verbose = verbose
        self.records: List[ExperimentRecord] = []
        self.pareto: List[ExperimentRecord] = []

    def _run_trial(
        self,
        experiment_id: int,
        hyperparams: dict,
        is_baseline: bool,
        trial_seed: int,
    ) -> ExperimentRecord:
        label = "baseline" if is_baseline else self.method_name
        logger.info(
            "Trial %d/%d (%s): %s",
            experiment_id,
            self.optimization_iterations,
            label,
            hyperparams,
        )
        result = self.adapter.run_trial(hyperparams, seed=trial_seed)
        if result.failed:
            logger.warning(
                "Trial %d failed: %s", experiment_id, result.error_message or "(no details)"
            )
        record = ExperimentRecord(
            experiment_id=experiment_id,
            is_baseline=is_baseline,
            seed=result.seed,
            hyperparameters=hyperparams,
            metrics={
                "train_loss": result.train_loss,
                "val_loss": result.val_loss,
                "best_val_loss": result.best_val_loss,
                "best_val_iter": result.best_val_iter,
                "training_iterations": result.training_iterations,
                "training_time_seconds": result.training_time_seconds,
                "samples_seen": result.samples_seen,
                "samples_per_second": result.samples_per_second,
            },
            resources={
                "parameter_count": result.parameter_count,
                "model_size_mb": result.model_size_mb,
                "peak_memory_mb": result.peak_memory_mb,
            },
            device=result.device,
            failed=result.failed,
        )
        self.records.append(record)

        if self.verbose:
            total = self.optimization_iterations + 1  # +1 for baseline
            status = "FAILED" if result.failed else f"val_loss={result.best_val_loss:.4f}"
            time_s = f"{result.training_time_seconds:.1f}s"
            print(
                f"  Trial {experiment_id:>3}/{total} [{label:>8}] | {status} | time={time_s}",
                file=sys.stderr,
                flush=True,
            )

        return record

    def run(self) -> List[ExperimentRecord]:
        """Execute the full sweep: one baseline + *optimization_iterations* trials.

        Returns
        -------
        List[ExperimentRecord]
            All records in order (baseline first).  Pareto-front records are
            also available via :attr:`pareto`.
        """
        if self.verbose:
            total = self.optimization_iterations + 1
            print(
                f"Starting sweep: 1 baseline + {self.optimization_iterations} trials "
                f"({total} total) | method={self.method_name}",
                file=sys.stderr,
                flush=True,
            )

        logger.info("Starting baseline trial.")
        self._run_trial(0, self.baseline_hyperparams, is_baseline=True, trial_seed=self.seed)

        for i in range(1, self.optimization_iterations + 1):
            config = self.optimizer.suggest()
            record = self._run_trial(i, config, is_baseline=False, trial_seed=self.seed + i)
            self.optimizer.report(config, record.metrics["best_val_loss"])

        compute_scores(self.records, weights=self.score_weights)
        self.pareto = pareto_front(self.records)

        if self.verbose:
            print(
                f"Sweep complete. "
                f"{sum(1 for r in self.records if not r.is_baseline and not r.failed)} scoreable | "
                f"{len(self.pareto)} on Pareto front.",
                file=sys.stderr,
                flush=True,
            )

        logger.info(
            "Sweep complete. %d scoreable trials; %d on the Pareto front.",
            sum(1 for r in self.records if not r.is_baseline and not r.failed),
            len(self.pareto),
        )
        return self.records
