"""
llm_hpo_optimizer - General-purpose, multi-objective hyperparameter optimizer.

Core usage (no extra dependencies needed)::

    from llm_hpo_optimizer import (
        SearchSpace, Categorical, UniformFloat, LogUniformFloat,
        UniformInt, LogUniformInt,
        TrainingAdapter, TrainingResult,
        RandomSearchOptimizer, ExperimentRunner,
        compute_scores, pareto_front,
    )

For Bayesian/TPE search (requires ``pip install llm-hpo-optimizer[optuna]``)::

    from llm_hpo_optimizer import OptunaOptimizer

For the built-in nanoGPT adapter (requires ``pip install llm-hpo-optimizer[nanogpt]``
and nanoGPT's ``model.py`` on your Python path)::

    from llm_hpo_optimizer import NanoGPTAdapter, optimize
"""

from llm_hpo_optimizer._version import __version__
from llm_hpo_optimizer.space import (
    Categorical,
    UniformFloat,
    LogUniformFloat,
    UniformInt,
    LogUniformInt,
    SearchSpace,
)
from llm_hpo_optimizer.adapter import TrainingAdapter, TrainingResult
from llm_hpo_optimizer.optimizers import RandomSearchOptimizer, OptunaOptimizer
from llm_hpo_optimizer.metrics import (
    ExperimentRecord,
    SCORE_OBJECTIVES,
    DEFAULT_WEIGHTS,
    compute_scores,
    pareto_front,
)
from llm_hpo_optimizer.runner import ExperimentRunner, OptimizerProtocol
from llm_hpo_optimizer.nanogpt_adapter import (
    NanoGPTAdapter,
    DEFAULT_NANOGPT_SEARCH_SPACE,
    DEFAULT_NANOGPT_BASELINE,
)
from llm_hpo_optimizer.api import optimize

__all__ = [
    "__version__",
    # Search space
    "Categorical",
    "UniformFloat",
    "LogUniformFloat",
    "UniformInt",
    "LogUniformInt",
    "SearchSpace",
    # Adapter interface
    "TrainingAdapter",
    "TrainingResult",
    # Optimizers
    "RandomSearchOptimizer",
    "OptunaOptimizer",
    "OptimizerProtocol",
    # Metrics
    "ExperimentRecord",
    "SCORE_OBJECTIVES",
    "DEFAULT_WEIGHTS",
    "compute_scores",
    "pareto_front",
    # Runner
    "ExperimentRunner",
    # Built-in nanoGPT adapter
    "NanoGPTAdapter",
    "DEFAULT_NANOGPT_SEARCH_SPACE",
    "DEFAULT_NANOGPT_BASELINE",
    # Top-level convenience function
    "optimize",
]
