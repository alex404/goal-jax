"""Plotting for Variational MNIST example.

Two modes:

- Single run (default): loads ``analysis.json`` from this example's results
  directory and produces the standard training / samples figures.
- Comparison (``--compare PATH_A PATH_B --labels A B``): loads two
  ``analysis.json`` files (e.g. one with ``--latent bernoullis`` and one with
  ``--latent chordal_boltzmann``) and produces a side-by-side figure of ELBO,
  conjugation metrics, cluster metrics, and per-step wallclock.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..shared import (
    apply_style,
    example_paths,
    metric_color,
    model_color,
    plot_image_grid,
)

# Colors for training modes
COLORS = {
    "gradient": model_color(0),  # Blue
    "analytical": model_color(1),  # Red
}

LABELS = {
    "gradient": "Gradient Descent",
    "analytical": "Analytical",
}


def plot_training_elbos(ax: Axes, models: dict[str, Any]) -> None:
    """Plot ELBO training curves for all models."""
    for mode, results in models.items():
        if results["training_elbos"]:
            ax.plot(
                results["training_elbos"],
                color=COLORS.get(mode, "gray"),
                label=LABELS.get(mode, mode),
                alpha=0.7,
            )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("ELBO")
    ax.set_title("Training ELBO")
    ax.legend()



def plot_conjugation_variance(ax: Axes, models: dict[str, Any]) -> None:
    """Plot conjugation error (Var[r]) over time."""
    has_positive = False
    for mode, results in models.items():
        if results["conjugation_vars"]:
            ax.plot(
                results["conjugation_vars"],
                color=COLORS.get(mode, "gray"),
                label=LABELS.get(mode, mode),
                alpha=0.7,
            )
            if any(y > 0 for y in results["conjugation_vars"]):
                has_positive = True
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Var[r]")
    ax.set_title("Conjugation Error (Variance)")
    ax.legend()

    if has_positive:
        ax.set_yscale("log")


def plot_conjugation_std(ax: Axes, models: dict[str, Any]) -> None:
    """Plot conjugation std (Std[r]) over time."""
    for mode, results in models.items():
        if results["conjugation_stds"]:
            ax.plot(
                results["conjugation_stds"],
                color=COLORS.get(mode, "gray"),
                label=LABELS.get(mode, mode),
                alpha=0.7,
            )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Std[r]")
    ax.set_title("Conjugation Error (Std)")
    ax.legend()



def plot_conjugation_r2(ax: Axes, models: dict[str, Any]) -> None:
    """Plot R^2 over time."""
    for mode, results in models.items():
        if results["conjugation_r2s"]:
            ax.plot(
                results["conjugation_r2s"],
                color=COLORS.get(mode, "gray"),
                label=LABELS.get(mode, mode),
                alpha=0.7,
            )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("R^2")
    ax.set_title("Conjugation R^2 (higher = better linear fit)")
    ax.legend()

    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)


def plot_rho_norm(ax: Axes, models: dict[str, Any]) -> None:
    """Plot rho norm over time."""
    for mode, results in models.items():
        if results["rho_norms"]:
            ax.plot(
                results["rho_norms"],
                color=COLORS.get(mode, "gray"),
                label=LABELS.get(mode, mode),
                alpha=0.7,
            )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("||rho||")
    ax.set_title("Rho Norm Over Time")
    ax.legend()



def plot_metrics_comparison(ax: Axes, models: dict[str, Any]) -> None:
    """Plot bar chart comparing purity, NMI, and accuracy."""
    modes = list(models.keys())
    x = np.arange(len(modes))
    width = 0.25

    # Purity bars
    purity_vals = [models[m]["cluster_purity"] * 100 for m in modes]
    bars1 = ax.bar(
        x - width, purity_vals, width, label="Purity (%)", color=metric_color("purity")
    )

    # NMI bars
    nmi_vals = [models[m]["nmi"] * 100 for m in modes]
    bars2 = ax.bar(x, nmi_vals, width, label="NMI (%)", color=metric_color("nmi"))

    # Accuracy bars
    acc_vals = [models[m]["cluster_accuracy"] * 100 for m in modes]
    bars3 = ax.bar(
        x + width,
        acc_vals,
        width,
        label="Accuracy (%)",
        color=metric_color("accuracy"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in modes])
    ax.set_ylabel("Score (%)")
    ax.set_title("Clustering Metrics Comparison")
    ax.set_ylim(0, 80)
    ax.legend()

    # Add value labels
    for bars, vals in [(bars1, purity_vals), (bars2, nmi_vals), (bars3, acc_vals)]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 1,
                    f"{val:.1f}",
                    ha="center",
                    fontsize=6,
                )

    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_axisbelow(True)


def create_samples_and_prototypes_figure(
    models: dict[str, Any],
    original_images: NDArray[np.float64],
    reconstructed_images: NDArray[np.float64],
) -> Figure:
    """Create a figure showing originals, reconstructions, samples, and prototypes."""
    n_models = len(models)
    fig, axes = plt.subplots(1 + n_models, 2, figsize=(10, 4 * (1 + n_models)))

    # Row 0: Original images and reconstructions
    plot_image_grid(
        axes[0, 0],
        original_images,
        "Original Images",
        n_rows=4,
        n_cols=5,
    )
    plot_image_grid(
        axes[0, 1],
        reconstructed_images,
        "Reconstructions (Best Model)",
        n_rows=4,
        n_cols=5,
    )

    # Rows 1+: Each model
    for i, (mode, results) in enumerate(models.items(), start=1):
        label = LABELS.get(mode, mode)
        if results["generated_samples"]:
            plot_image_grid(
                axes[i, 0],
                np.array(results["generated_samples"]),
                f"Generated Samples ({label})",
                n_rows=4,
                n_cols=5,
            )
            plot_image_grid(
                axes[i, 1],
                np.array(results["cluster_prototypes"]),
                f"Cluster Prototypes ({label})",
                n_rows=2,
                n_cols=5,
            )
        else:
            axes[i, 0].text(0.5, 0.5, "Not trained", ha="center", va="center")
            axes[i, 0].set_title(f"Generated Samples ({label})")
            axes[i, 0].axis("off")
            axes[i, 1].text(0.5, 0.5, "Not trained", ha="center", va="center")
            axes[i, 1].set_title(f"Cluster Prototypes ({label})")
            axes[i, 1].axis("off")

    fig.suptitle("Samples and Prototypes (All Models)")

    return fig


def _extract_run_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    """Extract the trained-model record from a saved analysis.json.

    `train.py` saves under ``analysis["models"][mode]`` where ``mode`` is
    ``"gradient"`` or ``"analytical"``. The comparison treats each run as a
    single record; we pick the ``best_model`` entry, falling back to the
    only entry if there is exactly one.
    """
    models = analysis["models"]
    best = analysis.get("best_model")
    if best in models:
        return models[best]
    if len(models) == 1:
        return next(iter(models.values()))
    raise KeyError(
        f"analysis.json has multiple models {list(models)} but no valid best_model"
    )


def _series(metrics: dict[str, Any], key: str) -> list[float]:
    """Return a list of floats for a per-step series; empty list if missing."""
    return list(metrics.get(key, []) or [])


def plot_compare_elbo(ax: Axes, runs: dict[str, dict[str, Any]]) -> None:
    """ELBO curves for each compared run, on shared axes."""
    for i, (label, metrics) in enumerate(runs.items()):
        elbos = _series(metrics, "training_elbos")
        if elbos:
            ax.plot(elbos, color=model_color(i), label=label, alpha=0.85)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("ELBO")
    ax.set_title("Training ELBO")
    ax.legend()


def plot_compare_conjugation(ax: Axes, runs: dict[str, dict[str, Any]]) -> None:
    """Conjugation Var[r] over steps (log y-axis when there's positive data)."""
    has_positive = False
    for i, (label, metrics) in enumerate(runs.items()):
        ys = _series(metrics, "conjugation_vars")
        if ys:
            ax.plot(ys, color=model_color(i), label=label, alpha=0.85)
            if any(y > 0 for y in ys):
                has_positive = True
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Var[r]")
    ax.set_title("Conjugation Var[r]")
    if has_positive:
        ax.set_yscale("log")
    ax.legend()


def plot_compare_r2(ax: Axes, runs: dict[str, dict[str, Any]]) -> None:
    """Conjugation R^2 over (logged) steps."""
    for i, (label, metrics) in enumerate(runs.items()):
        ys = _series(metrics, "conjugation_r2s")
        if ys:
            ax.plot(ys, color=model_color(i), label=label, alpha=0.85)
    ax.set_xlabel("Log step")
    ax.set_ylabel("R^2")
    ax.set_title("Conjugation R^2 (higher = tighter)")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.legend()


def plot_compare_rho_norm(ax: Axes, runs: dict[str, dict[str, Any]]) -> None:
    """||rho|| over steps."""
    for i, (label, metrics) in enumerate(runs.items()):
        ys = _series(metrics, "rho_norms")
        if ys:
            ax.plot(ys, color=model_color(i), label=label, alpha=0.85)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("||rho||")
    ax.set_title("Rho Norm")
    ax.legend()


def plot_compare_cluster_metrics(
    ax: Axes, runs: dict[str, dict[str, Any]]
) -> None:
    """Bar chart: cluster purity / NMI / accuracy per run."""
    labels = list(runs.keys())
    x = np.arange(len(labels))
    width = 0.25

    purity_vals = [100 * float(runs[lbl].get("cluster_purity", 0.0)) for lbl in labels]
    nmi_vals = [100 * float(runs[lbl].get("nmi", 0.0)) for lbl in labels]
    acc_vals = [100 * float(runs[lbl].get("cluster_accuracy", 0.0)) for lbl in labels]

    ax.bar(x - width, purity_vals, width, label="Purity (%)", color=metric_color("purity"))
    ax.bar(x, nmi_vals, width, label="NMI (%)", color=metric_color("nmi"))
    ax.bar(x + width, acc_vals, width, label="Accuracy (%)", color=metric_color("accuracy"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("Cluster Metrics")
    ax.legend()
    ax.set_axisbelow(True)


def plot_compare_wallclock(
    ax: Axes, runs: dict[str, dict[str, Any]]
) -> None:
    """Bar chart: mean warm-step time per run, with JIT compile annotation."""
    labels = list(runs.keys())
    x = np.arange(len(labels))
    warm_means = [
        1000.0 * float(runs[lbl].get("mean_warm_step_seconds", 0.0)) for lbl in labels
    ]
    jit_seconds = [
        float(runs[lbl].get("jit_compile_seconds", 0.0)) for lbl in labels
    ]

    bars = ax.bar(x, warm_means, color=[model_color(i) for i in range(len(labels))])
    for bar, jit in zip(bars, jit_seconds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"JIT: {jit:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean warm step (ms)")
    ax.set_title("Per-step wallclock")


def create_compare_figure(runs: dict[str, dict[str, Any]]) -> Figure:
    """Build the full comparison figure given a label -> run-metrics map."""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    plot_compare_elbo(fig.add_subplot(gs[0, 0]), runs)
    plot_compare_cluster_metrics(fig.add_subplot(gs[0, 1]), runs)
    plot_compare_conjugation(fig.add_subplot(gs[1, 0]), runs)
    plot_compare_r2(fig.add_subplot(gs[1, 1]), runs)
    plot_compare_rho_norm(fig.add_subplot(gs[2, 0]), runs)
    plot_compare_wallclock(fig.add_subplot(gs[2, 1]), runs)
    fig.suptitle("Run comparison", y=0.995)
    return fig


def _print_compare_summary(runs: dict[str, dict[str, Any]]) -> None:
    """Print a small textual summary table to stdout."""
    header = (
        f"{'Run':<24} {'Final ELBO':>12} {'NMI':>8} {'Purity':>8} "
        f"{'Acc':>8} {'Step (ms)':>12} {'JIT (s)':>10}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label, m in runs.items():
        elbos = _series(m, "training_elbos")
        final_elbo = elbos[-1] if elbos else float("nan")
        nmi = float(m.get("nmi", float("nan")))
        purity = float(m.get("cluster_purity", float("nan")))
        accuracy = float(m.get("cluster_accuracy", float("nan")))
        step_ms = 1000.0 * float(m.get("mean_warm_step_seconds", float("nan")))
        jit_s = float(m.get("jit_compile_seconds", float("nan")))
        row = (
            f"{label:<24} {final_elbo:>12.2f} {nmi:>8.3f}"
            + f" {100 * purity:>7.1f}% {100 * accuracy:>7.1f}%"
            + f" {step_ms:>12.2f} {jit_s:>10.2f}"
        )
        print(row)
    print("=" * len(header))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        default=None,
        help="Two or more analysis.json paths to compare side by side",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for the runs given to --compare (same length)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path for the comparison figure (default: ./compare.png)",
    )
    return parser.parse_args()


def _run_compare_mode(args: argparse.Namespace) -> None:
    paths = example_paths(__file__)
    apply_style(paths)

    json_paths: list[Path] = list(args.compare)
    if args.labels and len(args.labels) != len(json_paths):
        raise SystemExit(
            f"--labels has {len(args.labels)} entries but --compare has"
            + f" {len(json_paths)}"
        )
    labels: list[str] = args.labels or [p.stem for p in json_paths]

    runs: dict[str, dict[str, Any]] = {}
    for label, path in zip(labels, json_paths):
        with open(path) as f:
            analysis = json.load(f)
        runs[label] = _extract_run_metrics(analysis)

    fig = create_compare_figure(runs)
    output = args.output or (Path.cwd() / "compare.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Comparison plot saved to {output}")

    _print_compare_summary(runs)


def main():
    args = _parse_args()
    if args.compare:
        _run_compare_mode(args)
        return

    paths = example_paths(__file__)
    apply_style(paths)

    results = paths.load_analysis()

    # Extract data
    models = results["models"]
    original_images = np.array(results["original_images"])
    reconstructed_images = np.array(results["reconstructed_images"])
    best_model = results.get("best_model", "unknown")

    # Create main figure with training metrics (3x2 grid)
    fig1 = plt.figure(figsize=(14, 12))
    gs = fig1.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    # Row 1: ELBO, Metrics comparison
    ax_elbo = fig1.add_subplot(gs[0, 0])
    ax_metrics = fig1.add_subplot(gs[0, 1])

    # Row 2: Conjugation Variance, Conjugation R^2
    ax_var = fig1.add_subplot(gs[1, 0])
    ax_r2 = fig1.add_subplot(gs[1, 1])

    # Row 3: Rho norm, Std[r]
    ax_rho = fig1.add_subplot(gs[2, 0])
    ax_std = fig1.add_subplot(gs[2, 1])

    # Plot training metrics
    plot_training_elbos(ax_elbo, models)
    plot_metrics_comparison(ax_metrics, models)
    plot_conjugation_variance(ax_var, models)
    plot_conjugation_r2(ax_r2, models)
    plot_rho_norm(ax_rho, models)
    plot_conjugation_std(ax_std, models)

    # Get best NMI for title
    best_nmi = max(m["nmi"] for m in models.values())
    best_label = LABELS.get(best_model, best_model)

    fig1.suptitle(
        f"Variational MNIST Training (Best by NMI: {best_label}, NMI={best_nmi:.3f})",
        y=0.99,
    )


    paths.save_plot(fig1)
    print(f"Main plot saved to {paths.plot_path}")

    # Create second figure with samples and prototypes
    fig2 = create_samples_and_prototypes_figure(
        models,
        original_images,
        reconstructed_images,
    )
    samples_path = paths.plot_path.with_stem(paths.plot_path.stem + "_samples")
    fig2.savefig(samples_path, bbox_inches="tight", dpi=150)
    print(f"Samples plot saved to {samples_path}")


if __name__ == "__main__":
    main()
