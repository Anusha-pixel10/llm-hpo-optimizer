"""
Adapter interface -- the single extension point of the library.

Implement :class:`TrainingAdapter` once per model type and pass an instance
to :class:`~llm_hpo_optimizer.runner.ExperimentRunner` or
:func:`~llm_hpo_optimizer.api.optimize`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class TrainingResult:
    """All metrics and resource figures produced by a single trial.

    Attributes
    ----------
    train_loss:          Final training loss.
    val_loss:            Final validation loss.
    best_val_loss:       Lowest validation loss seen during training.
    best_val_iter:       Iteration at which the lowest validation loss occurred.
    training_iterations: Total number of training iterations completed.
    parameter_count:     Number of model parameters.
    model_size_mb:       Model size in megabytes (parameter_count × 4 bytes).
    peak_memory_mb:      Peak GPU memory usage in megabytes (0 on CPU).
    training_time_seconds: Wall-clock time for the full trial.
    samples_seen:        Total tokens / samples processed.
    samples_per_second:  Throughput.
    device:              Device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    seed:                The random seed actually used for this trial.
    failed:              ``True`` when the trial raised a recoverable error.
    error_message:       Human-readable error details when *failed* is ``True``.
    """

    train_loss: float
    val_loss: float
    best_val_loss: float
    best_val_iter: int
    training_iterations: int
    parameter_count: int
    model_size_mb: float
    peak_memory_mb: float
    training_time_seconds: float
    samples_seen: int
    samples_per_second: float
    device: str
    seed: int
    failed: bool = False
    error_message: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy as a plain dict."""
        return asdict(self)


class TrainingAdapter(ABC):
    """Abstract base for all model adapters.

    Implement :meth:`run_trial` to train your model, measure its resource
    usage, and return a populated :class:`TrainingResult`.  The rest of the
    library is completely agnostic to how training is performed.

    Example
    -------
    ::

        class MyAdapter(TrainingAdapter):
            def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
                # build model, train, evaluate ...
                return TrainingResult(
                    train_loss=0.5, val_loss=0.6, ...
                )
    """

    @abstractmethod
    def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
        """Train the model with *hyperparams* and return metrics.

        Parameters
        ----------
        hyperparams:
            A configuration dict produced by the optimizer's ``suggest()``
            method.  Keys and value types match those defined in the
            :class:`~llm_hpo_optimizer.space.SearchSpace`.
        seed:
            Integer random seed.  The adapter is responsible for seeding
            all randomness (model weights, data shuffling, etc.).

        Returns
        -------
        TrainingResult
            A fully populated result.  Set ``failed=True`` and
            ``error_message=...`` instead of raising an exception for
            recoverable failures (e.g. OOM).
        """
        raise NotImplementedError
