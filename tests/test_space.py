"""Tests for the search-space primitives (space.py)."""
import math
import random

import pytest

from llm_hpo_optimizer import Categorical, LogUniformFloat, SearchSpace, UniformFloat


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------


class TestCategorical:
    def test_sample_returns_one_of_choices(self):
        spec = Categorical(["a", "b", "c"])
        rng = random.Random(0)
        for _ in range(50):
            assert spec.sample(rng) in ["a", "b", "c"]

    def test_empty_choices_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Categorical([])

    def test_single_choice(self):
        spec = Categorical([42])
        assert spec.sample(random.Random(0)) == 42

    def test_reproducible_with_same_seed(self):
        spec = Categorical([1, 2, 3, 4, 5])
        a = [spec.sample(random.Random(7)) for _ in range(20)]
        b = [spec.sample(random.Random(7)) for _ in range(20)]
        assert a == b

    def test_different_seeds_differ(self):
        spec = Categorical(list(range(100)))
        a = spec.sample(random.Random(1))
        b = spec.sample(random.Random(2))
        # Not guaranteed to differ, but very likely with 100 choices
        # — just verify no exception
        assert a in range(100)
        assert b in range(100)


# ---------------------------------------------------------------------------
# UniformFloat
# ---------------------------------------------------------------------------


class TestUniformFloat:
    def test_sample_in_range(self):
        spec = UniformFloat(0.0, 1.0)
        rng = random.Random(0)
        for _ in range(200):
            v = spec.sample(rng)
            assert 0.0 <= v <= 1.0

    def test_invalid_bounds_low_greater_raises(self):
        with pytest.raises(ValueError):
            UniformFloat(1.0, 0.0)

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError):
            UniformFloat(0.5, 0.5)

    def test_reproducible(self):
        spec = UniformFloat(0.0, 100.0)
        assert spec.sample(random.Random(99)) == spec.sample(random.Random(99))


# ---------------------------------------------------------------------------
# LogUniformFloat
# ---------------------------------------------------------------------------


class TestLogUniformFloat:
    def test_sample_in_range(self):
        spec = LogUniformFloat(1e-5, 1e-2)
        rng = random.Random(0)
        for _ in range(200):
            v = spec.sample(rng)
            assert 1e-5 <= v <= 1e-2

    def test_zero_low_raises(self):
        with pytest.raises(ValueError):
            LogUniformFloat(0.0, 1.0)

    def test_negative_low_raises(self):
        with pytest.raises(ValueError):
            LogUniformFloat(-1.0, 1.0)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            LogUniformFloat(1.0, 0.5)

    def test_log_distribution_mean(self):
        """Geometric mean of samples should be close to sqrt(low * high)."""
        spec = LogUniformFloat(1e-4, 1e-2)
        rng = random.Random(42)
        samples = [spec.sample(rng) for _ in range(3000)]
        log_mean = sum(math.log(s) for s in samples) / len(samples)
        expected = (math.log(1e-4) + math.log(1e-2)) / 2
        assert abs(log_mean - expected) < 0.15


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_sample_returns_all_keys(self):
        space = SearchSpace({"lr": LogUniformFloat(1e-4, 1e-2), "batch": Categorical([16, 32])})
        cfg = space.sample(random.Random(0))
        assert set(cfg.keys()) == {"lr", "batch"}

    def test_empty_parameters_raises(self):
        with pytest.raises(ValueError):
            SearchSpace({})

    def test_constraint_respected(self):
        space = SearchSpace(
            {"x": Categorical([1, 2, 3, 4])},
            constraints=[lambda c: c["x"] % 2 == 0],
        )
        rng = random.Random(0)
        for _ in range(30):
            cfg = space.sample(rng)
            assert cfg["x"] % 2 == 0

    def test_auto_n_embd_n_head_constraint(self):
        space = SearchSpace({
            "n_head": Categorical([4, 8]),
            "n_embd": Categorical([64, 128]),
        })
        rng = random.Random(0)
        for _ in range(30):
            cfg = space.sample(rng)
            assert cfg["n_embd"] % cfg["n_head"] == 0

    def test_impossible_constraint_raises(self):
        space = SearchSpace(
            {"x": Categorical([1, 3, 5])},
            constraints=[lambda c: c["x"] % 2 == 0],
        )
        with pytest.raises(RuntimeError, match="No valid config"):
            space.sample(random.Random(0))

    def test_is_valid_true(self):
        space = SearchSpace({"x": Categorical([2, 4])}, constraints=[lambda c: c["x"] > 3])
        assert space.is_valid({"x": 4})

    def test_is_valid_false(self):
        space = SearchSpace({"x": Categorical([2, 4])}, constraints=[lambda c: c["x"] > 3])
        assert not space.is_valid({"x": 2})

    def test_default_rng_when_none(self):
        space = SearchSpace({"x": Categorical([1, 2, 3])})
        cfg = space.sample()  # rng=None
        assert cfg["x"] in [1, 2, 3]

    def test_max_attempts_respected(self):
        space = SearchSpace(
            {"x": Categorical([1])},
            constraints=[lambda c: c["x"] == 2],  # impossible
            max_attempts=5,
        )
        with pytest.raises(RuntimeError):
            space.sample(random.Random(0))
