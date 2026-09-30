"""Tests for the top-level optimize() convenience function and public import API."""
import json

import pytest

from llm_hpo_optimizer import (
    Categorical,
    LogUniformFloat,
    SearchSpace,
    TrainingAdapter,
    TrainingResult,
    UniformFloat,
    optimize,
)


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class DummyAdapter(TrainingAdapter):
    def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
        lr = hyperparams.get("lr", 1e-3)
        return TrainingResult(
            train_loss=lr * 90,
            val_loss=lr * 100,
            best_val_loss=lr * 95,
            best_val_iter=5,
            training_iterations=10,
            parameter_count=100_000,
            model_size_mb=0.4,
            peak_memory_mb=50.0,
            training_time_seconds=0.01,
            samples_seen=100,
            samples_per_second=10_000.0,
            device="cpu",
            seed=seed,
        )


SPACE    = SearchSpace({"lr": Categorical([1e-4, 3e-4, 1e-3])})
BASELINE = {"lr": 1e-3}


def _run(**kwargs) -> dict:
    defaults = dict(
        method="random",
        adapter=DummyAdapter(),
        search_space=SPACE,
        baseline_hyperparams=BASELINE,
        optimization_iterations=3,
        seed=0,
    )
    defaults.update(kwargs)
    return optimize(**defaults)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestOptimizeResultStructure:
    def test_returns_expected_keys(self):
        result = _run()
        expected = {
            "best_balanced_config", "best_balanced_score",
            "best_loss_config", "best_loss_value",
            "baseline_config", "pareto_front", "all_records",
        }
        assert set(result.keys()) == expected

    def test_all_records_count(self):
        result = _run(optimization_iterations=5)
        assert len(result["all_records"]) == 6  # 1 baseline + 5

    def test_best_loss_config_is_dict_with_lr(self):
        result = _run()
        assert isinstance(result["best_loss_config"], dict)
        assert "lr" in result["best_loss_config"]

    def test_baseline_config_matches_input(self):
        result = _run()
        assert result["baseline_config"] == BASELINE

    def test_best_balanced_score_is_float(self):
        result = _run()
        assert isinstance(result["best_balanced_score"], float)

    def test_pareto_front_is_list(self):
        result = _run()
        assert isinstance(result["pareto_front"], list)
        assert len(result["pareto_front"]) >= 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestOptimizeErrors:
    def test_no_adapter_no_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset_dir"):
            optimize(method="random")

    def test_adapter_without_search_space_raises(self):
        with pytest.raises(ValueError, match="search_space"):
            optimize(adapter=DummyAdapter(), search_space=None,
                     baseline_hyperparams=BASELINE)

    def test_adapter_without_baseline_raises(self):
        with pytest.raises(ValueError, match="baseline_hyperparams"):
            optimize(adapter=DummyAdapter(), search_space=SPACE,
                     baseline_hyperparams=None)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            _run(method="bogus")


# ---------------------------------------------------------------------------
# output_path
# ---------------------------------------------------------------------------


class TestOptimizeOutputPath:
    def test_json_file_written(self, tmp_path):
        out = tmp_path / "results.json"
        _run(output_path=str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert "all_records" in data
        assert "best_loss_config" in data

    def test_json_is_valid(self, tmp_path):
        out = tmp_path / "results.json"
        _run(output_path=str(out))
        data = json.loads(out.read_text())
        assert isinstance(data["all_records"], list)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestOptimizeReproducibility:
    def test_same_seed_produces_same_results(self):
        r1 = _run(seed=77)
        r2 = _run(seed=77)
        assert r1["best_loss_config"] == r2["best_loss_config"]
        assert r1["all_records"] == r2["all_records"]

    def test_different_seeds_produce_different_results(self):
        r1 = _run(seed=1, optimization_iterations=6)
        r2 = _run(seed=2, optimization_iterations=6)
        # Baseline is always the same; suggested trials should differ
        non_baseline_1 = [r["hyperparameters"] for r in r1["all_records"][1:]]
        non_baseline_2 = [r["hyperparameters"] for r in r2["all_records"][1:]]
        assert non_baseline_1 != non_baseline_2


# ---------------------------------------------------------------------------
# Optuna method
# ---------------------------------------------------------------------------


class TestOptimizeOptuna:
    def test_optuna_method_works(self):
        pytest.importorskip("optuna")
        result = _run(method="optuna", optimization_iterations=4)
        assert len(result["all_records"]) == 5

    def test_optuna_result_has_all_keys(self):
        pytest.importorskip("optuna")
        result = _run(method="optuna")
        assert "best_loss_config" in result


# ---------------------------------------------------------------------------
# Public import API
# ---------------------------------------------------------------------------


class TestPublicImportAPI:
    def test_version_attribute(self):
        import llm_hpo_optimizer
        assert hasattr(llm_hpo_optimizer, "__version__")
        assert isinstance(llm_hpo_optimizer.__version__, str)
        assert len(llm_hpo_optimizer.__version__) > 0

    def test_all_public_names_importable(self):
        from llm_hpo_optimizer import (
            Categorical,
            DEFAULT_NANOGPT_BASELINE,
            DEFAULT_NANOGPT_SEARCH_SPACE,
            DEFAULT_WEIGHTS,
            ExperimentRecord,
            ExperimentRunner,
            LogUniformFloat,
            LogUniformInt,
            NanoGPTAdapter,
            OptimizerProtocol,
            OptunaOptimizer,
            RandomSearchOptimizer,
            SCORE_OBJECTIVES,
            SearchSpace,
            TrainingAdapter,
            TrainingResult,
            UniformFloat,
            UniformInt,
            compute_scores,
            optimize,
            pareto_front,
        )
        # All names resolve without error; spot-check a few
        assert callable(optimize)
        assert callable(compute_scores)
        assert callable(pareto_front)
        assert isinstance(DEFAULT_WEIGHTS, dict)
        assert isinstance(SCORE_OBJECTIVES, dict)
