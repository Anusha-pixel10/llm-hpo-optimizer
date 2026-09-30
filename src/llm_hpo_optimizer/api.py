"""
Top-level convenience function.

This module exposes :func:`optimize`, a single entry point that wires
together a search space, an optimizer, a training adapter, and the
experiment runner.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Optional

from llm_hpo_optimizer.adapter import TrainingAdapter
from llm_hpo_optimizer.metrics import DEFAULT_WEIGHTS
from llm_hpo_optimizer.optimizers import OptunaOptimizer, RandomSearchOptimizer
from llm_hpo_optimizer.runner import ExperimentRunner
from llm_hpo_optimizer.space import SearchSpace


def optimize(
    dataset_dir: Optional[str] = None,
    method: str = "random",
    optimization_iterations: int = 15,
    training_iterations_per_trial: int = 1000,
    eval_interval: int = 100,
    eval_iters: int = 50,
    block_size: int = 64,
    seed: int = 1337,
    adapter: Optional[TrainingAdapter] = None,
    search_space: Optional[SearchSpace] = None,
    baseline_hyperparams: Optional[dict] = None,
    score_weights: Optional[dict] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Run a full hyperparameter sweep and return the results.

    Default behavior (no *adapter* given): uses the built-in
    :class:`~llm_hpo_optimizer.nanogpt_adapter.NanoGPTAdapter` against
    *dataset_dir* -- requires ``torch``, ``numpy``, and nanoGPT's
    ``model.py`` on your path.

    To use a different model: pass your own *adapter*, *search_space*, and
    *baseline_hyperparams*.  Nothing else about this function changes.

    Parameters
    ----------
    dataset_dir:
        Path to the dataset directory (required when *adapter* is ``None``).
    method:
        ``"random"`` (default) or ``"optuna"`` (Bayesian TPE).
    optimization_iterations:
        Number of optimizer-suggested trials (default 15).
    training_iterations_per_trial:
        Training iterations passed to the adapter per trial (default 1000).
    eval_interval:
        Validation evaluation interval (nanoGPT adapter only).
    eval_iters:
        Validation batches per evaluation (nanoGPT adapter only).
    block_size:
        Context length in tokens (nanoGPT adapter only; default 64).
    seed:
        Base random seed.
    adapter:
        Custom :class:`~llm_hpo_optimizer.adapter.TrainingAdapter` instance.
        When provided, *search_space* and *baseline_hyperparams* are required.
    search_space:
        Custom :class:`~llm_hpo_optimizer.space.SearchSpace`.
    baseline_hyperparams:
        Fixed config dict for the baseline trial.
    score_weights:
        Per-objective weights forwarded to
        :func:`~llm_hpo_optimizer.metrics.compute_scores`.
    output_path:
        If given, write results as JSON to this file path.  A warning is
        emitted if the file already exists so results are not silently
        overwritten.
    verbose:
        If ``True``, print a one-line progress summary to *stderr* after
        every trial.  Defaults to ``False``.

    Returns
    -------
    dict
        Keys: ``best_balanced_config``, ``best_balanced_score``,
        ``best_loss_config``, ``best_loss_value``, ``baseline_config``,
        ``pareto_front`` (list of dicts), ``all_records`` (list of dicts).

    Raises
    ------
    ValueError
        On invalid argument combinations.
    """
    if adapter is None:
        if dataset_dir is None:
            raise ValueError(
                "Provide dataset_dir (to use the built-in NanoGPT adapter) "
                "or your own adapter= and search_space=."
            )
        # Lazy import so torch/numpy errors are deferred.
        from llm_hpo_optimizer.nanogpt_adapter import (
            NanoGPTAdapter,
            DEFAULT_NANOGPT_SEARCH_SPACE,
            DEFAULT_NANOGPT_BASELINE,
        )

        adapter = NanoGPTAdapter(
            dataset_dir=dataset_dir,
            max_iters=training_iterations_per_trial,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            block_size=block_size,
        )
        search_space = search_space or DEFAULT_NANOGPT_SEARCH_SPACE
        baseline_hyperparams = baseline_hyperparams or DEFAULT_NANOGPT_BASELINE
    else:
        # Custom adapter path — validate both required arguments separately.
        if search_space is None:
            raise ValueError(
                "When passing a custom adapter=, also pass search_space=."
            )
        if baseline_hyperparams is None:
            raise ValueError(
                "When passing a custom adapter=, also pass baseline_hyperparams=."
            )

    if method == "random":
        opt = RandomSearchOptimizer(search_space, seed=seed)
    elif method == "optuna":
        opt = OptunaOptimizer(search_space, seed=seed)
    else:
        raise ValueError(f"Unknown method {method!r} -- use 'random' or 'optuna'.")

    runner = ExperimentRunner(
        opt,
        adapter,
        baseline_hyperparams,
        optimization_iterations=optimization_iterations,
        seed=seed,
        score_weights=score_weights,
        method_name=method,
        verbose=verbose,
    )
    runner.run()

    non_baseline = [r for r in runner.records if not r.is_baseline and not r.failed]
    baseline = next((r for r in runner.records if r.is_baseline), None)
    best_loss = min(non_baseline, key=lambda r: r.metrics["best_val_loss"]) if non_baseline else None
    best_balanced = min(non_baseline, key=lambda r: r.score) if non_baseline else None

    result = {
        "best_balanced_config": best_balanced.hyperparameters if best_balanced else None,
        "best_balanced_score": best_balanced.score if best_balanced else None,
        "best_loss_config": best_loss.hyperparameters if best_loss else None,
        "best_loss_value": best_loss.metrics["best_val_loss"] if best_loss else None,
        "baseline_config": baseline.hyperparameters if baseline else None,
        "pareto_front": [r.to_dict() for r in runner.pareto],
        "all_records": [r.to_dict() for r in runner.records],
    }
    if output_path:
        if os.path.exists(output_path):
            warnings.warn(
                f"output_path {output_path!r} already exists and will be overwritten. "
                "Use a unique path to preserve previous results.",
                UserWarning,
                stacklevel=2,
            )
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result
