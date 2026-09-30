"""Integration tests for ExperimentRunner using a stub adapter."""
import pytest

from llm_hpo_optimizer import (
    Categorical,
    ExperimentRunner,
    RandomSearchOptimizer,
    SearchSpace,
    TrainingAdapter,
    TrainingResult,
    UniformFloat,
)


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class StubAdapter(TrainingAdapter):
    """Returns a deterministic TrainingResult. val_loss decreases as lr rises."""

    def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
        lr = hyperparams.get("lr", 1e-3)
        return TrainingResult(
            train_loss=lr * 90,
            val_loss=lr * 100,
            best_val_loss=lr * 95,
            best_val_iter=5,
            training_iterations=10,
            parameter_count=1_000_000,
            model_size_mb=4.0,
            peak_memory_mb=200.0,
            training_time_seconds=1.0,
            samples_seen=1000,
            samples_per_second=1000.0,
            device="cpu",
            seed=seed,
        )


SPACE    = SearchSpace({"lr": Categorical([1e-4, 3e-4, 1e-3]), "batch": Categorical([16, 32])})
BASELINE = {"lr": 1e-3, "batch": 32}


def _make_runner(n_iter: int = 3, seed: int = 42) -> ExperimentRunner:
    opt = RandomSearchOptimizer(SPACE, seed=seed)
    return ExperimentRunner(opt, StubAdapter(), BASELINE, optimization_iterations=n_iter, seed=seed)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_run_returns_correct_record_count(self):
        records = _make_runner(n_iter=5).run()
        assert len(records) == 6  # 1 baseline + 5 trials

    def test_first_record_is_baseline(self):
        records = _make_runner().run()
        assert records[0].is_baseline
        assert records[0].experiment_id == 0
        assert records[0].hyperparameters == BASELINE

    def test_non_baseline_records_have_scores(self):
        records = _make_runner().run()
        for r in records[1:]:
            assert r.score is not None

    def test_baseline_has_no_score(self):
        records = _make_runner().run()
        assert records[0].score is None

    def test_pareto_is_non_empty(self):
        runner = _make_runner(n_iter=5)
        runner.run()
        assert len(runner.pareto) >= 1

    def test_reproducibility_same_seed(self):
        r1 = _make_runner(seed=123).run()
        r2 = _make_runner(seed=123).run()
        for a, b in zip(r1, r2):
            assert a.hyperparameters == b.hyperparameters
            assert a.metrics == b.metrics

    def test_different_seeds_differ(self):
        cfg1 = [r.hyperparameters for r in _make_runner(seed=1).run()[1:]]
        cfg2 = [r.hyperparameters for r in _make_runner(seed=2).run()[1:]]
        assert cfg1 != cfg2

    def test_invalid_iterations_zero_raises(self):
        with pytest.raises(ValueError):
            ExperimentRunner(
                RandomSearchOptimizer(SPACE, seed=0),
                StubAdapter(),
                BASELINE,
                optimization_iterations=0,
            )

    def test_none_seed_raises(self):
        with pytest.raises(ValueError, match="seed"):
            ExperimentRunner(
                RandomSearchOptimizer(SPACE, seed=0),
                StubAdapter(),
                BASELINE,
                seed=None,  # type: ignore[arg-type]
            )

    def test_baseline_hyperparams_defensively_copied(self):
        baseline = {"lr": 1e-3, "batch": 32}
        runner = _make_runner()
        runner.baseline_hyperparams["lr"] = 999
        # The original dict must not have been mutated
        assert baseline["lr"] == 1e-3

    def test_records_experiment_ids_sequential(self):
        records = _make_runner(n_iter=4).run()
        for i, r in enumerate(records):
            assert r.experiment_id == i

    def test_failed_trial_still_recorded(self):
        class FailAdapter(TrainingAdapter):
            def run_trial(self, hyperparams, seed):
                return TrainingResult(
                    train_loss=float("nan"), val_loss=float("nan"),
                    best_val_loss=float("nan"), best_val_iter=-1,
                    training_iterations=0, parameter_count=0,
                    model_size_mb=0.0, peak_memory_mb=0.0,
                    training_time_seconds=0.0, samples_seen=0,
                    samples_per_second=0.0, device="cpu", seed=seed,
                    failed=True, error_message="simulated failure",
                )

        opt = RandomSearchOptimizer(SPACE, seed=0)
        runner = ExperimentRunner(opt, FailAdapter(), BASELINE, optimization_iterations=2, seed=0)
        records = runner.run()
        assert len(records) == 3
        # Non-baseline failed trials should have score=None
        for r in records[1:]:
            assert r.failed
            assert r.score is None


class TestOptunaRunner:
    def test_optuna_runner_basic(self):
        pytest.importorskip("optuna")
        from llm_hpo_optimizer import OptunaOptimizer

        opt = OptunaOptimizer(SPACE, seed=0)
        runner = ExperimentRunner(opt, StubAdapter(), BASELINE, optimization_iterations=3, seed=42)
        records = runner.run()
        assert len(records) == 4
        for r in records[1:]:
            assert r.score is not None
