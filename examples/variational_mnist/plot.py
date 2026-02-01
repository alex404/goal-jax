"""Plotting for Variational MNIST example."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..shared import (
    apply_style,
    example_paths,
    metric_colors,
    model_colors,
    plot_image_grid,
)

# Colors for the three modes
COLORS = {
    "free": model_colors[0],  # Blue
    "fixed": model_colors[1],  # Red
    "regularized": model_colors[2],  # Purple
}

LABELS = {
    "free": "Learnable rho",
    "fixed": "Fixed rho=0",
    "regularized": "Regularized",
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
    ax.grid(True)


def plot_conjugation_variance(ax: Axes, models: dict[str, Any]) -> None:
    """Plot conjugation error (Var[f_tilde]) over time."""
    for mode, results in models.items():
        if results["conjugation_vars"]:
            ax.plot(
                results["conjugation_vars"],
                color=COLORS.get(mode, "gray"),
                label=LABELS.get(mode, mode),
                alpha=0.7,
            )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Var[f_tilde]")
    ax.set_title("Conjugation Error (Variance)")
    ax.legend()
    ax.grid(True)
    ax.set_yscale("log")


def plot_conjugation_std(ax: Axes, models: dict[str, Any]) -> None:
    """Plot conjugation std (Std[f_tilde]) over time."""
    for mode, results in models.items():
        if results["conjugation_stds"]:
            ax.plot(
                results["conjugation_stds"],
                color=COLORS.get(mode, "gray"),
                label=LABELS.get(mode, mode),
                alpha=0.7,
            )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Std[f_tilde]")
    ax.set_title("Conjugation Error (Std)")
    ax.legend()
    ax.grid(True)


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
    ax.grid(True)
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
    ax.grid(True)


def plot_metrics_comparison(ax: Axes, models: dict[str, Any]) -> None:
    """Plot bar chart comparing purity, NMI, and accuracy."""
    modes = list(models.keys())
    x = np.arange(len(modes))
    width = 0.25

    # Purity bars
    purity_vals = [models[m]["cluster_purity"] * 100 for m in modes]
    bars1 = ax.bar(
        x - width, purity_vals, width, label="Purity (%)", color=metric_colors["purity"]
    )

    # NMI bars
    nmi_vals = [models[m]["nmi"] * 100 for m in modes]
    bars2 = ax.bar(x, nmi_vals, width, label="NMI (%)", color=metric_colors["nmi"])

    # Accuracy bars
    acc_vals = [models[m]["cluster_accuracy"] * 100 for m in modes]
    bars3 = ax.bar(
        x + width,
        acc_vals,
        width,
        label="Accuracy (%)",
        color=metric_colors["accuracy"],
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
    ax.grid(True, axis="y")


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

    fig.suptitle("Samples and Prototypes (All Models)", fontsize=12)
    plt.tight_layout()
    return fig


def main():
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

    # Row 3: Rho norm, Std[f_tilde]
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
        fontsize=12,
        y=0.99,
    )

    plt.tight_layout()
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
