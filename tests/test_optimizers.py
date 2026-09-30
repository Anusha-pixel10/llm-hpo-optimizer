"""Tests for RandomSearchOptimizer and OptunaOptimizer."""
import pytest

from llm_hpo_optimizer import (
    Categorical,
    LogUniformFloat,
    RandomSearchOptimizer,
    SearchSpace,
    UniformFloat,
)


def _simple_space() -> SearchSpace:
    return SearchSpace({
        "lr":      LogUniformFloat(1e-4, 1e-2),
        "batch":   Categorical([16, 32, 64]),
        "dropout": UniformFloat(0.0, 0.5),
    })


# ---------------------------------------------------------------------------
# RandomSearchOptimizer
# ---------------------------------------------------------------------------


class TestRandomSearchOptimizer:
    def test_suggest_returns_valid_keys(self):
        opt = RandomSearchOptimizer(_simple_space(), seed=0)
        cfg = opt.suggest()
        assert set(cfg.keys()) == {"lr", "batch", "dropout"}

    def test_suggest_values_in_range(self):
        opt = RandomSearchOptimizer(_simple_space(), seed=0)
        for _ in range(30):
            cfg = opt.suggest()
            assert cfg["batch"] in [16, 32, 64]
            assert 1e-4 <= cfg["lr"] <= 1e-2
            assert 0.0 <= cfg["dropout"] <= 0.5

    def test_report_is_noop(self):
        opt = RandomSearchOptimizer(_simple_space(), seed=0)
        cfg = opt.suggest()
        opt.report(cfg, 0.5)  # must not raise

    def test_reproducible_with_same_seed(self):
        opt1 = RandomSearchOptimizer(_simple_space(), seed=42)
        opt2 = RandomSearchOptimizer(_simple_space(), seed=42)
        for _ in range(15):
            assert opt1.suggest() == opt2.suggest()

    def test_different_seeds_produce_different_configs(self):
        opt1 = RandomSearchOptimizer(_simple_space(), seed=1)
        opt2 = RandomSearchOptimizer(_simple_space(), seed=2)
        configs_1 = [opt1.suggest() for _ in range(15)]
        configs_2 = [opt2.suggest() for _ in range(15)]
        assert configs_1 != configs_2

    def test_no_seed_does_not_raise(self):
        opt = RandomSearchOptimizer(_simple_space())  # seed=None
        cfg = opt.suggest()
        assert "lr" in cfg


# ---------------------------------------------------------------------------
# OptunaOptimizer
# ---------------------------------------------------------------------------


class TestOptunaOptimizer:
    @pytest.fixture(autouse=True)
    def _require_optuna(self):
        pytest.importorskip("optuna")

    def test_suggest_report_cycle(self):
        from llm_hpo_optimizer import OptunaOptimizer

        opt = OptunaOptimizer(_simple_space(), seed=0)
        for _ in range(5):
            cfg = opt.suggest()
            assert "lr" in cfg and "batch" in cfg and "dropout" in cfg
            opt.report(cfg, 0.5)

    def test_joint_n_head_n_embd(self):
        from llm_hpo_optimizer import OptunaOptimizer

        space = SearchSpace({
            "n_head": Categorical([4, 8]),
            "n_embd": Categorical([64, 128, 256]),
            "dropout": UniformFloat(0.0, 0.3),
        })
        opt = OptunaOptimizer(space, seed=0)
        for _ in range(10):
            cfg = opt.suggest()
            assert cfg["n_embd"] % cfg["n_head"] == 0
            opt.report(cfg, 0.5)

    def test_missing_optuna_friendly_error(self, monkeypatch):
        """If optuna is not importable, a helpful ImportError is raised."""
        import sys
        from unittest.mock import patch

        # Temporarily hide optuna from the import system
        with patch.dict(sys.modules, {"optuna": None}):
            from llm_hpo_optimizer import OptunaOptimizer as _O  # noqa: F401
            with pytest.raises(ImportError, match="optuna"):
                _O(_simple_space(), seed=0)
