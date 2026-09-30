"""Tests for multi-objective scoring (compute_scores) and Pareto front."""
import pytest

from llm_hpo_optimizer import DEFAULT_WEIGHTS, ExperimentRecord, compute_scores, pareto_front


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(exp_id, val_loss, peak_mem, param_count, time_s,
         is_baseline=False, failed=False) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=exp_id,
        is_baseline=is_baseline,
        seed=42,
        hyperparameters={"lr": 1e-3},
        metrics={
            "best_val_loss": val_loss,
            "train_loss": val_loss * 0.9,
            "val_loss": val_loss,
            "best_val_iter": 10,
            "training_iterations": 100,
            "training_time_seconds": time_s,
            "samples_seen": 1000,
            "samples_per_second": 100.0,
        },
        resources={
            "peak_memory_mb": peak_mem,
            "parameter_count": param_count,
            "model_size_mb": param_count * 4 / (1024 ** 2),
        },
        device="cpu",
        failed=failed,
    )


# ---------------------------------------------------------------------------
# ExperimentRecord
# ---------------------------------------------------------------------------


class TestExperimentRecord:
    def test_to_dict_round_trips(self):
        r = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        d = r.to_dict()
        assert d["experiment_id"] == 1
        assert d["failed"] is False
        assert d["score"] is None

    def test_score_initially_none(self):
        r = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        assert r.score is None


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------


class TestComputeScores:
    def test_scores_assigned_to_non_baseline_non_failed(self):
        baseline = _rec(0, 1.0, 100.0, 1_000_000, 10.0, is_baseline=True)
        r1 = _rec(1, 0.5, 200.0, 2_000_000, 20.0)
        r2 = _rec(2, 0.8, 100.0, 1_000_000, 10.0)
        compute_scores([baseline, r1, r2])
        assert baseline.score is None
        assert r1.score is not None
        assert r2.score is not None

    def test_failed_trials_not_scored(self):
        r1 = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        bad = _rec(2, float("nan"), 0.0, 0, 0.0, failed=True)
        compute_scores([r1, bad])
        assert bad.score is None
        assert r1.score is not None

    def test_single_scoreable_record_gets_zero(self):
        r = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        compute_scores([r])
        assert r.score == 0.0

    def test_empty_records_no_error(self):
        compute_scores([])  # must not raise

    def test_only_baseline_no_error(self):
        baseline = _rec(0, 1.0, 100.0, 1_000_000, 10.0, is_baseline=True)
        compute_scores([baseline])
        assert baseline.score is None

    def test_unknown_weight_key_raises(self):
        r = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        with pytest.raises(KeyError, match="Unknown objectives"):
            compute_scores([r], weights={"val_loss": 0.5, "nonexistent": 0.5})

    def test_worst_record_scores_total_weight(self):
        """With two extreme records, worst should equal sum of all weights."""
        r_best = _rec(1, 0.1, 50.0,  500_000,  5.0)
        r_worst = _rec(2, 0.9, 500.0, 5_000_000, 50.0)
        compute_scores([r_best, r_worst])
        total_w = sum(DEFAULT_WEIGHTS.values())
        assert abs(r_best.score - 0.0) < 1e-9
        assert abs(r_worst.score - total_w) < 1e-9

    def test_custom_weights(self):
        r1 = _rec(1, 0.2, 100.0, 1_000_000, 10.0)
        r2 = _rec(2, 0.8, 200.0, 2_000_000, 20.0)
        weights = {"val_loss": 1.0, "peak_memory_mb": 0.0,
                   "parameter_count": 0.0, "training_time": 0.0}
        compute_scores([r1, r2], weights=weights)
        assert r1.score < r2.score  # lower val_loss wins


# ---------------------------------------------------------------------------
# pareto_front
# ---------------------------------------------------------------------------


class TestParetoFront:
    def test_single_record_is_pareto(self):
        r = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        assert pareto_front([r]) == [r]

    def test_dominated_record_excluded(self):
        r_good = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        r_bad  = _rec(2, 0.8, 200.0, 2_000_000, 20.0)  # dominated on every axis
        result = pareto_front([r_good, r_bad])
        assert r_good in result
        assert r_bad not in result

    def test_non_dominated_both_included(self):
        r1 = _rec(1, 0.5, 200.0, 1_000_000, 10.0)  # best loss
        r2 = _rec(2, 0.8, 50.0,  500_000,   5.0)   # best resources
        result = pareto_front([r1, r2])
        assert r1 in result
        assert r2 in result

    def test_baseline_excluded(self):
        baseline = _rec(0, 0.3, 50.0, 500_000, 5.0, is_baseline=True)
        r = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        result = pareto_front([baseline, r])
        assert baseline not in result
        assert r in result

    def test_failed_excluded(self):
        r   = _rec(1, 0.5, 100.0, 1_000_000, 10.0)
        bad = _rec(2, float("nan"), 0.0, 0, 0.0, failed=True)
        result = pareto_front([r, bad])
        assert r in result
        assert bad not in result

    def test_empty_returns_empty(self):
        assert pareto_front([]) == []

    def test_custom_objectives(self):
        r1 = _rec(1, 0.5, 200.0, 1_000_000, 10.0)
        r2 = _rec(2, 0.8, 50.0,  500_000,   5.0)
        # Only care about val_loss: r1 dominates r2 on val_loss only
        result = pareto_front([r1, r2], objectives=["val_loss"])
        assert r1 in result
        assert r2 not in result
