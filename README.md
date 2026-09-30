# llm-hpo-optimizer 
Now exists as a Python pip install library at: https://pypi.org/project/llm-hpo-optimizer/0.1.0/

pip install llm-hpo-optimizer==0.1.0

A general-purpose, **multi-objective hyperparameter optimization** library for LLM training.

`llm-hpo-optimizer` provides a clean, pluggable framework for running hyperparameter sweeps against *any* model. The core optimizer needs only the Python standard library. Optional extras unlock Bayesian (TPE) search via [Optuna](https://optuna.org) and a built-in adapter for [nanoGPT](https://github.com/karpathy/nanoGPT).

## Features

- **Model-agnostic**: implement one `TrainingAdapter` class to optimize any model
- **Multi-objective scoring**: weighted combination of validation loss, memory, parameter count, and training time
- **Pareto front extraction**: identifies non-dominated configurations automatically
- **Two search strategies**: pure random search (zero dependencies) or Bayesian TPE via Optuna
- **Reproducible**: all randomness is controlled by a single `seed`
- **Zero-dependency core**: the optimizer engine works with nothing beyond the standard library
- **Typed**: ships with a `py.typed` marker; all public APIs are fully annotated

## Installation

```bash
# Core library — random search, zero extra dependencies
pip install llm-hpo-optimizer

# + Bayesian/TPE search (Optuna)
pip install "llm-hpo-optimizer[optuna]"

# + Built-in nanoGPT adapter
pip install "llm-hpo-optimizer[nanogpt]"
```

## Quick Start

```python
from llm_hpo_optimizer import (
    optimize,
    SearchSpace,
    Categorical,
    UniformFloat,
    LogUniformFloat,
    TrainingAdapter,
    TrainingResult,
)

# 1. Implement your adapter
class MyAdapter(TrainingAdapter):
    def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
        # Train your model here …
        return TrainingResult(
            train_loss=0.5,
            val_loss=0.6,
            best_val_loss=0.55,
            best_val_iter=100,
            training_iterations=500,
            parameter_count=1_000_000,
            model_size_mb=4.0,
            peak_memory_mb=512.0,
            training_time_seconds=30.0,
            samples_seen=100_000,
            samples_per_second=3333.0,
            device="cpu",
            seed=seed,
        )

# 2. Define the search space
search_space = SearchSpace({
    "learning_rate": LogUniformFloat(1e-5, 1e-2),
    "hidden_size":   Categorical([64, 128, 256, 512]),
    "dropout":       UniformFloat(0.0, 0.5),
})

# 3. Run the sweep
result = optimize(
    method="random",          # or "optuna" (requires llm-hpo-optimizer[optuna])
    adapter=MyAdapter(),
    search_space=search_space,
    baseline_hyperparams={"learning_rate": 1e-3, "hidden_size": 128, "dropout": 0.1},
    optimization_iterations=20,
    seed=42,
)

print(result["best_loss_config"])      # best hyperparams by validation loss
print(result["best_balanced_config"]) # best hyperparams by multi-objective score
```

## Parameter / Search Space Definition

| Class | Description | Example |
|---|---|---|
| `Categorical` | Discrete, unordered choices | `Categorical([16, 32, 64])` |
| `UniformFloat` | Continuous uniform distribution | `UniformFloat(0.0, 0.5)` |
| `LogUniformFloat` | Log-scale uniform (for learning rate) | `LogUniformFloat(1e-5, 1e-2)` |

```python
from llm_hpo_optimizer import SearchSpace, Categorical, UniformFloat, LogUniformFloat

space = SearchSpace(
    parameters={
        "learning_rate": LogUniformFloat(1e-5, 1e-2),
        "batch_size":    Categorical([16, 32, 64]),
        "dropout":       UniformFloat(0.0, 0.5),
    },
    constraints=[lambda cfg: cfg["batch_size"] >= 16],  # optional
)
```

> **Note:** If both `n_head` and `n_embd` appear in your search space, the constraint
> `n_embd % n_head == 0` is enforced automatically.

## Objective Function / Adapter

Implement one method:

```python
class MyAdapter(TrainingAdapter):
    def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
        # Use hyperparams to configure and train your model.
        # Use seed to make training reproducible.
        # Catch OOM/runtime errors; return failed=True instead of raising.
        return TrainingResult(train_loss=..., val_loss=..., ...)
```

## Running the Optimization

```python
result = optimize(
    method="random",                  # "random" or "optuna"
    adapter=MyAdapter(),
    search_space=search_space,
    baseline_hyperparams={...},       # config for trial 0 (the baseline)
    optimization_iterations=20,       # optimizer-suggested trials
    seed=42,                          # reproducibility
    output_path="results.json",       # optional — write JSON to disk
)
```

## Accessing Results

```python
# Best hyperparams by validation loss
print(result["best_loss_config"])     # dict
print(result["best_loss_value"])      # float

# Best hyperparams by weighted multi-objective score
print(result["best_balanced_config"]) # dict
print(result["best_balanced_score"])  # float (lower = better)

# The fixed baseline (trial 0)
print(result["baseline_config"])      # dict

# Pareto-optimal configurations
for record in result["pareto_front"]:
    print(record["hyperparameters"], record["metrics"]["best_val_loss"])

# All trials
for record in result["all_records"]:
    print(record["experiment_id"], record["metrics"]["best_val_loss"])
```

## Advanced Usage

### Bayesian Search (Optuna / TPE)

```bash
pip install "llm-hpo-optimizer[optuna]"
```

```python
result = optimize(
    method="optuna",
    adapter=MyAdapter(),
    search_space=search_space,
    baseline_hyperparams=baseline,
    optimization_iterations=30,
    seed=42,
)
```

### Custom Scoring Weights

```python
result = optimize(
    ...,
    score_weights={
        "val_loss":        0.7,
        "peak_memory_mb":  0.1,
        "parameter_count": 0.1,
        "training_time":   0.1,
    },
)
```

### Using the Runner Directly

```python
from llm_hpo_optimizer import RandomSearchOptimizer, ExperimentRunner

opt    = RandomSearchOptimizer(search_space, seed=42)
runner = ExperimentRunner(
    optimizer=opt,
    adapter=MyAdapter(),
    baseline_hyperparams=baseline,
    optimization_iterations=20,
    seed=42,
)
records = runner.run()
print(f"Pareto front: {len(runner.pareto)} configs")
```

### Built-in nanoGPT Adapter

```bash
pip install "llm-hpo-optimizer[nanogpt]"
# nanoGPT's model.py must be on your Python path:
export PYTHONPATH=/path/to/nanoGPT:$PYTHONPATH
```

```python
from llm_hpo_optimizer import optimize

result = optimize(
    dataset_dir="data/shakespeare_char",
    method="optuna",
    optimization_iterations=15,
    training_iterations_per_trial=1000,
)
print(result["best_balanced_config"])
```

## Reproducibility

All randomness is controlled by the `seed` parameter:

```python
result_a = optimize(..., seed=42)
result_b = optimize(..., seed=42)
assert result_a["all_records"] == result_b["all_records"]  # True
```

## Supported Python Versions

Python **3.9, 3.10, 3.11, 3.12**

## License

MIT — see [LICENSE](LICENSE).
