"""
Minimal example: custom adapter + random search.

Run this from the project root::

    python examples/custom_adapter_example.py
"""
import math
import time

from llm_hpo_optimizer import (
    Categorical,
    LogUniformFloat,
    SearchSpace,
    TrainingAdapter,
    TrainingResult,
    UniformFloat,
    optimize,
)


class SyntheticAdapter(TrainingAdapter):
    """Toy adapter with a synthetic loss landscape.

    Optimal configuration is near lr=3e-4, hidden_size=256, dropout=0.1.
    """

    def run_trial(self, hyperparams: dict, seed: int) -> TrainingResult:
        t0 = time.time()
        lr = hyperparams["learning_rate"]
        hidden = hyperparams["hidden_size"]
        dropout = hyperparams["dropout"]

        # Synthetic loss: minimum near lr=3e-4, hidden=256, dropout=0.1
        val_loss = (
            0.3
            + (math.log(lr / 3e-4)) ** 2 * 0.5
            + ((hidden - 256) / 256) ** 2 * 0.2
            + (dropout - 0.1) ** 2 * 0.5
        )
        elapsed = time.time() - t0 + 1e-4  # tiny constant to avoid division by zero

        return TrainingResult(
            train_loss=val_loss * 0.9,
            val_loss=val_loss,
            best_val_loss=val_loss,
            best_val_iter=50,
            training_iterations=100,
            parameter_count=hidden * 4,
            model_size_mb=hidden * 4 * 4 / (1024 ** 2),
            peak_memory_mb=hidden * 0.1,
            training_time_seconds=elapsed,
            samples_seen=10_000,
            samples_per_second=10_000 / elapsed,
            device="cpu",
            seed=seed,
        )


def main() -> None:
    search_space = SearchSpace(
        parameters={
            "learning_rate": LogUniformFloat(1e-5, 1e-2),
            "hidden_size":   Categorical([64, 128, 256, 512]),
            "dropout":       UniformFloat(0.0, 0.5),
        }
    )
    baseline = {"learning_rate": 1e-3, "hidden_size": 128, "dropout": 0.2}

    print("Running hyperparameter optimization (10 trials + 1 baseline)…")
    result = optimize(
        method="random",
        adapter=SyntheticAdapter(),
        search_space=search_space,
        baseline_hyperparams=baseline,
        optimization_iterations=10,
        seed=42,
    )

    print(f"\nBaseline config:           {result['baseline_config']}")
    print(f"Best config (by loss):     {result['best_loss_config']}")
    print(f"  val_loss: {result['best_loss_value']:.4f}")
    print(f"Best config (balanced):    {result['best_balanced_config']}")
    print(f"  score:    {result['best_balanced_score']:.4f}")
    print(f"Pareto front size: {len(result['pareto_front'])}")
    print(f"Total trials:      {len(result['all_records'])}")


if __name__ == "__main__":
    main()
