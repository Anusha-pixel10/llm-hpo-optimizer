"""Tests for TrainingAdapter ABC and TrainingResult dataclass."""
import pytest

from llm_hpo_optimizer import TrainingAdapter, TrainingResult


def _make_result(**kwargs) -> TrainingResult:
    defaults = dict(
        train_loss=0.5,
        val_loss=0.6,
        best_val_loss=0.55,
        best_val_iter=10,
        training_iterations=100,
        parameter_count=1_000_000,
        model_size_mb=4.0,
        peak_memory_mb=512.0,
        training_time_seconds=30.0,
        samples_seen=100_000,
        samples_per_second=3333.0,
        device="cpu",
        seed=42,
    )
    defaults.update(kwargs)
    return TrainingResult(**defaults)


class TestTrainingResult:
    def test_to_dict_contains_all_fields(self):
        r = _make_result()
        d = r.to_dict()
        assert d["train_loss"] == 0.5
        assert d["device"] == "cpu"
        assert d["failed"] is False
        assert d["error_message"] == ""

    def test_failed_result(self):
        r = _make_result(failed=True, error_message="OOM")
        assert r.failed
        assert r.error_message == "OOM"

    def test_default_not_failed(self):
        r = _make_result()
        assert not r.failed
        assert r.error_message == ""

    def test_to_dict_is_json_serialisable(self):
        import json
        r = _make_result()
        json.dumps(r.to_dict())  # must not raise


class TestTrainingAdapterABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            TrainingAdapter()  # type: ignore[abstract]

    def test_subclass_without_run_trial_raises(self):
        class Incomplete(TrainingAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_valid_subclass_works(self):
        class Stub(TrainingAdapter):
            def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
                return _make_result(seed=seed)

        adapter = Stub()
        result = adapter.run_trial({}, seed=7)
        assert result.seed == 7

    def test_adapter_passes_seed_through(self):
        class Stub(TrainingAdapter):
            def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
                return _make_result(seed=seed)

        adapter = Stub()
        for s in [0, 1, 1337, 99999]:
            assert adapter.run_trial({}, seed=s).seed == s
