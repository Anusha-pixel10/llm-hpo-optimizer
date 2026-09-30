# Contributing to llm-hpo-optimizer

Thank you for your interest in contributing! This guide covers how to set up a development environment, run the tests, and submit changes.

---

## 1. Prerequisites

- Python **3.9 – 3.12**
- [Git](https://git-scm.com/)

## 2. Development Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-hpo-optimizer.git
cd llm-hpo-optimizer

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install in editable mode with all dev extras
pip install -e ".[dev]"
```

> The `[dev]` extra installs `pytest`, `pytest-cov`, `ruff`, `optuna`, `build`, and `twine`.

## 3. Running Tests

```bash
# Run the full test suite
pytest

# Run with coverage report
pytest --cov=llm_hpo_optimizer --cov-report=term-missing

# Run a single test file
pytest tests/test_space.py -v
```

All tests should pass before you open a pull request.

## 4. Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and import sorting.

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix safe issues
ruff check --fix src/ tests/
```

## 5. Project Structure

```
src/llm_hpo_optimizer/
├── __init__.py          # Public API surface
├── _version.py          # Version (read from package metadata)
├── adapter.py           # TrainingAdapter ABC + TrainingResult dataclass
├── api.py               # optimize() convenience function
├── cli.py               # llm-hpo command-line interface
├── metrics.py           # compute_scores(), pareto_front(), ExperimentRecord
├── nanogpt_adapter.py   # Built-in nanoGPT adapter (optional: torch + numpy)
├── optimizers.py        # RandomSearchOptimizer, OptunaOptimizer
├── runner.py            # ExperimentRunner, OptimizerProtocol
└── space.py             # Categorical, UniformFloat, LogUniformFloat,
                         # UniformInt, LogUniformInt, SearchSpace
tests/
├── test_adapter.py
├── test_api.py
├── test_metrics.py
├── test_new_features.py
├── test_optimizers.py
├── test_runner.py
└── test_space.py
```

## 6. Adding a New Search Space Type

1. Add a `@dataclass` to [`space.py`](src/llm_hpo_optimizer/space.py) implementing `sample(rng) -> <type>`.
2. Wire it into `OptunaOptimizer.suggest()` in [`optimizers.py`](src/llm_hpo_optimizer/optimizers.py).
3. Export it from [`__init__.py`](src/llm_hpo_optimizer/__init__.py) and add to `__all__`.
4. Add tests in `tests/test_space.py` and `tests/test_optimizers.py`.

## 7. Adding a New Optimizer

Implement `suggest() -> dict` and `report(config, objective_value) -> None` — any object satisfying the `OptimizerProtocol` structural type will work with `ExperimentRunner` without further changes.

## 8. Building and Publishing

```bash
# Build source and wheel distributions
python -m build

# Check the distributions before uploading
twine check dist/*

# Upload to PyPI (maintainers only)
twine upload dist/*
```

## 9. Submitting a Pull Request

1. Fork the repo and create a branch: `git checkout -b feature/my-change`
2. Make your changes and add tests.
3. Ensure `pytest` and `ruff check` both pass.
4. Push and open a PR against `main`.

Please keep PRs focused — one logical change per PR makes review faster.
