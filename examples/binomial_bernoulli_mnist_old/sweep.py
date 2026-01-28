"""Hyperparameter sweep for conjugation research.

This script systematically explores the hyperparameter space to identify regimes
where we achieve both good clustering performance (accuracy, NMI) and good
conjugation quality (high R squared).

Grid search parameters:
- n_latent: [512, 1024, 2048, 4096]
- conj_reg_weight: [0.1, 0.5, 1.0, 2.0]
- l2_int_weight: [0, 1e-4, 1e-3]

Total combinations: 4 x 4 x 3 = 48 experiments

Usage:
    python -m examples.binomial_bernoulli_mnist.sweep
    python -m examples.binomial_bernoulli_mnist.sweep --n-steps 5000  # shorter runs
    python -m examples.binomial_bernoulli_mnist.sweep --dry-run  # show configs only
"""

import argparse
import itertools
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

# Grid definition
GRID: dict[str, list[int] | list[float]] = {
    "n_latent": [512, 1024, 2048, 4096],
    "conj_reg_weight": [0.1, 0.5, 1.0, 2.0],
    "l2_int_weight": [0.0, 1e-4, 1e-3],
}

# Fixed parameters (can be overridden via command line)
DEFAULT_FIXED: dict[str, float | int | str] = {
    "learning_rate": 1e-4,
    "n_steps": 15000,
    "mode": "conj_reg",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for MNIST")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_FIXED["n_steps"],
        help="Number of training steps per experiment",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_FIXED["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file, skipping completed experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep",
        help="Output directory for results",
    )
    return parser.parse_args()


def _parse_accuracy(line: str, metrics: dict[str, Any]) -> None:
    """Parse accuracy from output line."""
    match = re.search(r"(\d+\.?\d*)%", line)
    if match:
        metrics["accuracy"] = float(match.group(1)) / 100


def _parse_nmi(line: str, metrics: dict[str, Any]) -> None:
    """Parse NMI from output line."""
    match = re.search(r"NMI:\s*([\d.]+)", line)
    if match:
        metrics["nmi"] = float(match.group(1))


def _parse_r2(line: str, metrics: dict[str, Any]) -> None:
    """Parse R squared from output line."""
    match = re.search(r"Final R.*:\s*([-\d.]+)", line)
    if match:
        metrics["r2"] = float(match.group(1))


def _parse_elbo(line: str, metrics: dict[str, Any]) -> None:
    """Parse ELBO from output line."""
    match = re.search(r"Final ELBO:\s*([-\d.]+)", line)
    if match:
        metrics["elbo"] = float(match.group(1))


def _parse_purity(line: str, metrics: dict[str, Any]) -> None:
    """Parse purity from output line."""
    match = re.search(r"Purity:\s*(\d+\.?\d*)%", line)
    if match:
        metrics["purity"] = float(match.group(1)) / 100


def _parse_recon_error(line: str, metrics: dict[str, Any]) -> None:
    """Parse reconstruction error from output line."""
    match = re.search(r"Reconstruction error:\s*([\d.]+)", line)
    if match:
        metrics["recon_error"] = float(match.group(1))


def parse_metrics(output: str) -> dict[str, Any]:
    """Extract final metrics from run output.

    Parses the summary output from run.py to extract:
    - accuracy: Cluster accuracy (Hungarian algorithm)
    - nmi: Normalized Mutual Information
    - r2: Final R squared conjugation quality
    - elbo: Final ELBO value
    - purity: Cluster purity
    - recon_error: Reconstruction error
    """
    metrics: dict[str, Any] = {}

    for line in output.split("\n"):
        if "Accuracy" in line and "%" in line:
            _parse_accuracy(line, metrics)
        elif "NMI:" in line:
            _parse_nmi(line, metrics)
        elif "Final R" in line:
            _parse_r2(line, metrics)
        elif "Final ELBO:" in line:
            _parse_elbo(line, metrics)
        elif "Purity:" in line:
            _parse_purity(line, metrics)
        elif "Reconstruction error:" in line:
            _parse_recon_error(line, metrics)

    return metrics


def run_experiment(config: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    """Run single experiment and return results.

    Args:
        config: Grid search parameters (n_latent, conj_reg_weight, l2_int_weight)
        fixed: Fixed parameters (learning_rate, n_steps, mode)

    Returns:
        Dictionary with config parameters and extracted metrics
    """
    cmd = ["python", "-m", "examples.binomial_bernoulli_mnist.run"]
    for key, val in {**config, **fixed}.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(val)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    metrics = parse_metrics(result.stdout)

    if result.returncode != 0:
        metrics["error"] = True
        metrics["stderr"] = result.stderr[:500] if result.stderr else ""

    return {**config, **metrics}


def config_key(config: dict[str, Any]) -> str:
    """Generate a unique key for a configuration."""
    return (
        f"n{config['n_latent']}_c{config['conj_reg_weight']}_l{config['l2_int_weight']}"
    )


def print_sweep_header(fixed: dict[str, Any], total: int, results_dir: Path) -> None:
    """Print sweep configuration header."""
    print("=" * 70)
    print("HYPERPARAMETER SWEEP")
    print("=" * 70)
    print("Grid parameters:")
    for key, values in GRID.items():
        print(f"  {key}: {values}")
    print("Fixed parameters:")
    for key, val in fixed.items():
        print(f"  {key}: {val}")
    print(f"Total combinations: {total}")
    print(f"Output directory: {results_dir}")
    print("=" * 70)


def print_best_configs(valid_results: list[dict[str, Any]]) -> None:
    """Print best configurations by different metrics."""
    best_r2 = max(valid_results, key=lambda r: r.get("r2", float("-inf")))
    print(f"\nBest R squared: {best_r2.get('r2', 'N/A'):.4f}")
    print(
        f"  Config: n_latent={best_r2['n_latent']}, "
        f"conj_reg_weight={best_r2['conj_reg_weight']}, "
        f"l2_int_weight={best_r2['l2_int_weight']}"
    )

    best_acc = max(valid_results, key=lambda r: r.get("accuracy", float("-inf")))
    print(f"\nBest Accuracy: {best_acc.get('accuracy', 'N/A'):.3f}")
    print(
        f"  Config: n_latent={best_acc['n_latent']}, "
        f"conj_reg_weight={best_acc['conj_reg_weight']}, "
        f"l2_int_weight={best_acc['l2_int_weight']}"
    )

    best_nmi = max(valid_results, key=lambda r: r.get("nmi", float("-inf")))
    print(f"\nBest NMI: {best_nmi.get('nmi', 'N/A'):.4f}")
    print(
        f"  Config: n_latent={best_nmi['n_latent']}, "
        f"conj_reg_weight={best_nmi['conj_reg_weight']}, "
        f"l2_int_weight={best_nmi['l2_int_weight']}"
    )


def print_pareto_optimal(valid_results: list[dict[str, Any]]) -> None:
    """Print Pareto-optimal configurations."""
    print("\n" + "-" * 50)
    print("Pareto-optimal candidates (R squared >= 0.85 AND Accuracy >= 0.45):")
    pareto = [
        r
        for r in valid_results
        if r.get("r2", 0) >= 0.85 and r.get("accuracy", 0) >= 0.45
    ]
    if pareto:
        for r in sorted(pareto, key=lambda x: -x.get("r2", 0)):
            print(
                f"  R2={r.get('r2', 0):.4f}, Acc={r.get('accuracy', 0):.3f}, "
                f"NMI={r.get('nmi', 0):.4f} | "
                f"n_latent={r['n_latent']}, conj_reg={r['conj_reg_weight']}, "
                f"l2={r['l2_int_weight']}"
            )
    else:
        print("  No configurations meet both criteria.")
        print("  Consider relaxing thresholds or extending training.")


def main() -> None:
    """Run hyperparameter sweep."""
    args = parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "sweep_results.json"

    fixed: dict[str, Any] = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "mode": "conj_reg",
    }

    keys = list(GRID.keys())
    combinations = list(itertools.product(*GRID.values()))
    total = len(combinations)

    print_sweep_header(fixed, total, results_dir)

    if args.dry_run:
        print("\nDRY RUN - Configurations to test:")
        for i, values in enumerate(combinations):
            config = dict(zip(keys, values))
            print(f"  [{i + 1:2d}/{total}] {config}")
        print("\nRun without --dry-run to execute experiments.")
        return

    completed_keys: set[str] = set()
    results: list[dict[str, Any]] = []
    if args.resume and results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
            completed_keys = {config_key(r) for r in results}
        print(f"Resuming from {len(results)} completed experiments")

    start_time = datetime.now()

    for i, values in enumerate(combinations):
        config = dict(zip(keys, values))
        key = config_key(config)

        if key in completed_keys:
            print(f"\n[{i + 1}/{total}] SKIP (already completed): {config}")
            continue

        print(f"\n[{i + 1}/{total}] Running: {config}")
        print("-" * 50)

        result = run_experiment(config, fixed)
        results.append(result)
        completed_keys.add(key)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        r2 = result.get("r2", float("nan"))
        acc = result.get("accuracy", float("nan"))
        nmi = result.get("nmi", float("nan"))
        elbo = result.get("elbo", float("nan"))

        if result.get("error"):
            print(f"  ERROR: {result.get('stderr', 'Unknown error')[:100]}")
        else:
            print(f"  R2={r2:.4f}, Acc={acc:.3f}, NMI={nmi:.4f}, ELBO={elbo:.2f}")

        elapsed = (datetime.now() - start_time).total_seconds()
        completed = len(results) - (len(completed_keys) - len(results))
        if completed > 0:
            remaining = total - i - 1
            avg_time = elapsed / completed
            eta_seconds = avg_time * remaining
            eta_hours = eta_seconds / 3600
            print(f"  Elapsed: {elapsed / 60:.1f}min, ETA: {eta_hours:.1f}h remaining")

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Results saved to {results_file}")
    print(f"Total experiments: {len(results)}")

    if results:
        valid_results = [r for r in results if not r.get("error")]

        if valid_results:
            print_best_configs(valid_results)
            print_pareto_optimal(valid_results)


if __name__ == "__main__":
    main()
