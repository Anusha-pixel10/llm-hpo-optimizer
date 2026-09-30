# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-09-30

### Added
- `SearchSpace`, `Categorical`, `UniformFloat`, `LogUniformFloat` primitives.
- `UniformInt`, `LogUniformInt` for ordered integer hyperparameters.
- `RandomSearchOptimizer` — pure random search, zero dependencies.
- `OptunaOptimizer` — Bayesian TPE search via Optuna (optional extra).
- `TrainingAdapter` / `TrainingResult` — pluggable model interface.
- `ExperimentRunner` — orchestrates baseline + sweep trials.
- `compute_scores` — weighted min-max normalised multi-objective scoring.
- `pareto_front` — Pareto-front extraction across 4 objectives.
- `optimize()` — single-function convenience API.
- Built-in `NanoGPTAdapter` with cosine LR decay + linear warmup.
- `verbose` parameter on `optimize()` and `ExperimentRunner` for per-trial progress output.
- `llm-hpo` CLI entry point for nanoGPT sweeps without writing Python.
- `OptimizerProtocol` structural type for custom optimizer implementations.

### Fixed
- `train_loss` / `val_loss` in `NanoGPTAdapter` now initialise to `float("nan")` instead of `None`, preventing silent type violations.
- `OptunaOptimizer.report()` now raises a clear `RuntimeError` when called before `suggest()`.
- Separate, actionable error messages for missing `search_space=` vs missing `baseline_hyperparams=` in `optimize()`.
- `output_path` in `optimize()` now emits a `UserWarning` when the target file already exists.
- `__version__` is now read dynamically via `importlib.metadata` — stays in sync with `pyproject.toml` automatically.
