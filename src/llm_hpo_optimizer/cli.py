"""
Command-line interface for llm-hpo-optimizer.

Usage examples::

    # Random search with a custom adapter (import path)
    llm-hpo --dataset-dir data/shakespeare --method random --iterations 20

    # Bayesian search
    llm-hpo --dataset-dir data/shakespeare --method optuna --iterations 30 --verbose

Run ``llm-hpo --help`` for full usage.
"""
from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-hpo",
        description=(
            "Run a hyperparameter sweep using the built-in nanoGPT adapter.\n"
            "For custom adapters, use the Python API directly."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        metavar="DIR",
        help="Path to directory containing train.bin, val.bin, and optionally meta.pkl.",
    )
    p.add_argument(
        "--method",
        choices=["random", "optuna"],
        default="random",
        help="Search strategy: random or Bayesian TPE (optuna).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=15,
        metavar="N",
        help="Number of optimizer-suggested trials (excluding the baseline).",
    )
    p.add_argument(
        "--training-iters",
        type=int,
        default=1000,
        metavar="N",
        help="Training iterations per trial.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base random seed for reproducibility.",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write results JSON to this file (optional).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trial progress to stderr.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from llm_hpo_optimizer import optimize
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Starting sweep | dataset={args.dataset_dir!r} | "
        f"method={args.method} | iterations={args.iterations}",
        file=sys.stderr,
    )

    try:
        result = optimize(
            dataset_dir=args.dataset_dir,
            method=args.method,
            optimization_iterations=args.iterations,
            training_iterations_per_trial=args.training_iters,
            seed=args.seed,
            output_path=args.output,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Results ===")
    print(f"Best config (by loss):    {result['best_loss_config']}")
    print(f"  val_loss: {result['best_loss_value']:.4f}")
    print(f"Best config (balanced):   {result['best_balanced_config']}")
    print(f"  score:    {result['best_balanced_score']:.4f}")
    print(f"Pareto front size: {len(result['pareto_front'])}")
    print(f"Total trials:      {len(result['all_records'])}")

    if args.output:
        print(f"\nResults written to: {args.output}")
    else:
        print("\nTip: use --output results.json to save full results.")


if __name__ == "__main__":
    main()
