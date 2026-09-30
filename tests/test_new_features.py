"""Tests for new integer parameter types (UniformInt, LogUniformInt)
and OptunaOptimizer report-before-suggest guard."""
import random

import pytest

from llm_hpo_optimizer import (
    Categorical,
    LogUniformInt,
    SearchSpace,
    UniformInt,
)


class TestUniformInt:
    def test_sample_in_range(self):
        spec = UniformInt(1, 10)
        rng = random.Random(0)
        for _ in range(200):
            v = spec.sample(rng)
            assert isinstance(v, int)
            assert 1 <= v <= 10

    def test_invalid_bounds_equal_raises(self):
        with pytest.raises(ValueError):
            UniformInt(5, 5)

    def test_invalid_bounds_low_greater_raises(self):
        with pytest.raises(ValueError):
            UniformInt(10, 1)

    def test_reproducible(self):
        spec = UniformInt(1, 100)
        assert spec.sample(random.Random(7)) == spec.sample(random.Random(7))

    def test_returns_int_type(self):
        spec = UniformInt(1, 100)
        assert type(spec.sample(random.Random(0))) is int


class TestLogUniformInt:
    def test_sample_in_range(self):
        spec = LogUniformInt(4, 512)
        rng = random.Random(0)
        for _ in range(200):
            v = spec.sample(rng)
            assert isinstance(v, int)
            assert 4 <= v <= 512

    def test_low_zero_raises(self):
        with pytest.raises(ValueError):
            LogUniformInt(0, 10)

    def test_low_negative_raises(self):
        with pytest.raises(ValueError):
            LogUniformInt(-1, 10)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            LogUniformInt(10, 5)

    def test_reproducible(self):
        spec = LogUniformInt(1, 256)
        assert spec.sample(random.Random(42)) == spec.sample(random.Random(42))

    def test_returns_int_type(self):
        spec = LogUniformInt(1, 1024)
        assert type(spec.sample(random.Random(0))) is int


class TestSearchSpaceWithIntTypes:
    def test_uniform_int_in_search_space(self):
        space = SearchSpace({"layers": UniformInt(1, 8), "batch": Categorical([16, 32])})
        cfg = space.sample(random.Random(0))
        assert "layers" in cfg
        assert isinstance(cfg["layers"], int)
        assert 1 <= cfg["layers"] <= 8

    def test_log_uniform_int_in_search_space(self):
        space = SearchSpace({"hidden": LogUniformInt(32, 512)})
        cfg = space.sample(random.Random(0))
        assert isinstance(cfg["hidden"], int)
        assert 32 <= cfg["hidden"] <= 512


class TestOptunaReportBeforeSuggestGuard:
    def test_report_before_suggest_raises(self):
        pytest.importorskip("optuna")
        from llm_hpo_optimizer import OptunaOptimizer
        space = SearchSpace({"lr": Categorical([1e-3, 1e-4])})
        opt = OptunaOptimizer(space, seed=0)
        with pytest.raises(RuntimeError, match="suggest"):
            opt.report({}, 0.5)

    def test_optuna_with_uniform_int(self):
        pytest.importorskip("optuna")
        from llm_hpo_optimizer import OptunaOptimizer
        space = SearchSpace({"layers": UniformInt(1, 6), "lr": Categorical([1e-3, 1e-4])})
        opt = OptunaOptimizer(space, seed=0)
        for _ in range(5):
            cfg = opt.suggest()
            assert isinstance(cfg["layers"], int)
            assert 1 <= cfg["layers"] <= 6
            opt.report(cfg, 0.5)

    def test_optuna_with_log_uniform_int(self):
        pytest.importorskip("optuna")
        from llm_hpo_optimizer import OptunaOptimizer
        space = SearchSpace({"hidden": LogUniformInt(32, 256)})
        opt = OptunaOptimizer(space, seed=0)
        for _ in range(5):
            cfg = opt.suggest()
            assert isinstance(cfg["hidden"], int)
            assert 32 <= cfg["hidden"] <= 256
            opt.report(cfg, 0.5)
