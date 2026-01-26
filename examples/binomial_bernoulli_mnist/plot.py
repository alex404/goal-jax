"""Plotting for Binomial-Bernoulli MNIST example."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..shared import apply_style, example_paths, model_colors
from .types import BinomialBernoulliMNISTResults

# MNIST image dimensions
IMG_SIZE = 28
N_TRIALS = 16

# Colors for three models
COLORS = {
    "learnable": model_colors[0],  # Blue
    "fixed": model_colors[1],  # Red
    "conj_reg": model_colors[2],  # Purple
}


def plot_training_comparison(
    ax: Axes,
    elbos_learnable: NDArray[np.float64],
    elbos_fixed: NDArray[np.float64],
    elbos_conj_reg: NDArray[np.float64],
) -> None:
    """Plot ELBO training curves for all three models."""
    ax.plot(elbos_learnable, color=COLORS["learnable"], label="Learnable rho", alpha=0.7)
    ax.plot(elbos_fixed, color=COLORS["fixed"], label="Fixed rho=0", alpha=0.7)
    ax.plot(elbos_conj_reg, color=COLORS["conj_reg"], label="Conj. Regularized", alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("ELBO")
    ax.set_title("Training ELBO")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_conjugation_error(
    ax: Axes,
    conj_errors_learnable: NDArray[np.float64],
    conj_errors_fixed: NDArray[np.float64],
    conj_errors_conj_reg: NDArray[np.float64],
    interval: int,
) -> None:
    """Plot conjugation error (Var[f̃]) over time."""
    steps = np.arange(len(conj_errors_learnable)) * interval

    ax.plot(
        steps,
        conj_errors_learnable,
        color=COLORS["learnable"],
        label="Learnable rho",
        marker="o",
        markersize=3,
    )
    ax.plot(
        steps,
        conj_errors_fixed,
        color=COLORS["fixed"],
        label="Fixed rho=0",
        marker="s",
        markersize=3,
    )
    ax.plot(
        steps,
        conj_errors_conj_reg,
        color=COLORS["conj_reg"],
        label="Conj. Regularized",
        marker="^",
        markersize=3,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Var[f̃]")
    ax.set_title("Conjugation Error Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def plot_rho_norm(
    ax: Axes,
    rho_norms_learnable: NDArray[np.float64],
    rho_norms_fixed: NDArray[np.float64],
    rho_norms_conj_reg: NDArray[np.float64],
    interval: int,
) -> None:
    """Plot rho norm over time."""
    steps = np.arange(len(rho_norms_learnable)) * interval

    ax.plot(
        steps,
        rho_norms_learnable,
        color=COLORS["learnable"],
        label="Learnable rho",
        marker="o",
        markersize=3,
    )
    ax.plot(
        steps,
        rho_norms_fixed,
        color=COLORS["fixed"],
        label="Fixed rho=0",
        marker="s",
        markersize=3,
    )
    ax.plot(
        steps,
        rho_norms_conj_reg,
        color=COLORS["conj_reg"],
        label="Conj. Regularized",
        marker="^",
        markersize=3,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("||rho||")
    ax.set_title("Rho Norm Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_metrics_comparison(
    ax: Axes,
    purity_l: float,
    purity_f: float,
    purity_c: float,
    nmi_l: float,
    nmi_f: float,
    nmi_c: float,
) -> None:
    """Plot bar chart comparing purity and NMI."""
    x = np.array([0, 1, 2])
    width = 0.35

    # Purity bars
    purity_vals = [purity_l * 100, purity_f * 100, purity_c * 100]
    bars1 = ax.bar(x - width / 2, purity_vals, width, label="Purity (%)", color="steelblue")

    # NMI bars (scaled to percentage for visualization)
    nmi_vals = [nmi_l * 100, nmi_f * 100, nmi_c * 100]
    bars2 = ax.bar(x + width / 2, nmi_vals, width, label="NMI (%)", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(["Learnable", "Fixed rho=0", "Conj. Reg."], fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Clustering Metrics Comparison")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=8)

    # Add value labels
    for bar, val in zip(bars1, purity_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}", ha="center", fontsize=7)
    for bar, val in zip(bars2, nmi_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}", ha="center", fontsize=7)

    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.grid(True, alpha=0.3, axis="y")


def plot_cluster_prototypes(
    ax: Axes,
    prototypes: NDArray[np.float64],
    n_clusters: int,
) -> None:
    """Plot mean image per cluster (prototypes)."""
    n_rows = 2
    n_cols = 5
    gap = 2

    grid_h = n_rows * IMG_SIZE + (n_rows - 1) * gap
    grid_w = n_cols * IMG_SIZE + (n_cols - 1) * gap
    grid = np.full((grid_h, grid_w), np.nan)

    for k in range(n_clusters):
        row = k // n_cols
        col = k % n_cols
        y_start = row * (IMG_SIZE + gap)
        x_start = col * (IMG_SIZE + gap)
        img = prototypes[k].reshape(IMG_SIZE, IMG_SIZE)
        grid[y_start : y_start + IMG_SIZE, x_start : x_start + IMG_SIZE] = img

    ax.imshow(grid, cmap="gray", vmin=0, vmax=N_TRIALS)
    ax.set_title("Cluster Prototypes (Best Model)")
    ax.axis("off")


def plot_generated_samples(
    ax: Axes,
    samples: NDArray[np.float64],
    n_rows: int = 4,
    n_cols: int = 5,
) -> None:
    """Plot samples generated from the model."""
    n_images = min(n_rows * n_cols, samples.shape[0])
    gap = 2

    grid_h = n_rows * IMG_SIZE + (n_rows - 1) * gap
    grid_w = n_cols * IMG_SIZE + (n_cols - 1) * gap
    grid = np.full((grid_h, grid_w), np.nan)

    for i in range(n_images):
        row = i // n_cols
        col = i % n_cols
        y_start = row * (IMG_SIZE + gap)
        x_start = col * (IMG_SIZE + gap)
        img = samples[i].reshape(IMG_SIZE, IMG_SIZE)
        grid[y_start : y_start + IMG_SIZE, x_start : x_start + IMG_SIZE] = img

    ax.imshow(grid, cmap="gray", vmin=0, vmax=N_TRIALS)
    ax.set_title("Generated Samples (Best Model)")
    ax.axis("off")


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(BinomialBernoulliMNISTResults, paths.load_analysis())

    # Extract data for all three models
    learnable = results["learnable_rho"]
    fixed = results["fixed_rho"]
    conj_reg = results["conjugate_regularized"]

    elbos_l = np.array(learnable["training_elbos"])
    elbos_f = np.array(fixed["training_elbos"])
    elbos_c = np.array(conj_reg["training_elbos"])

    conj_errors_l = np.array(learnable["conjugation_errors"])
    conj_errors_f = np.array(fixed["conjugation_errors"])
    conj_errors_c = np.array(conj_reg["conjugation_errors"])

    rho_norms_l = np.array(learnable["rho_norms"])
    rho_norms_f = np.array(fixed["rho_norms"])
    rho_norms_c = np.array(conj_reg["rho_norms"])

    prototypes = np.array(results["cluster_prototypes"])
    generated = np.array(results["generated_samples"])

    # Compute interval from data
    n_elbos = len(elbos_l)
    n_conj = len(conj_errors_l)
    interval = n_elbos // (n_conj - 1) if n_conj > 1 else 50

    # Create figure with 3x2 grid
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    # Top row: ELBO comparison, Conjugation error
    ax_elbo = fig.add_subplot(gs[0, 0])
    ax_conj = fig.add_subplot(gs[0, 1])

    # Middle row: Rho norm, Metrics comparison
    ax_rho = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])

    # Bottom row: Generated samples, Cluster prototypes
    ax_gen = fig.add_subplot(gs[2, 0])
    ax_proto = fig.add_subplot(gs[2, 1])

    # Plot training ELBO comparison
    plot_training_comparison(ax_elbo, elbos_l, elbos_f, elbos_c)

    # Plot conjugation error over time
    plot_conjugation_error(ax_conj, conj_errors_l, conj_errors_f, conj_errors_c, interval)

    # Plot rho norm over time
    plot_rho_norm(ax_rho, rho_norms_l, rho_norms_f, rho_norms_c, interval)

    # Plot metrics comparison
    plot_metrics_comparison(
        ax_metrics,
        learnable["cluster_purity"],
        fixed["cluster_purity"],
        conj_reg["cluster_purity"],
        learnable["nmi"],
        fixed["nmi"],
        conj_reg["nmi"],
    )

    # Plot generated samples
    plot_generated_samples(ax_gen, generated, n_rows=4, n_cols=5)

    # Plot cluster prototypes
    plot_cluster_prototypes(ax_proto, prototypes, len(learnable["cluster_counts"]))

    # Add summary text
    best_nmi = max(learnable["nmi"], fixed["nmi"], conj_reg["nmi"])
    if conj_reg["nmi"] == best_nmi:
        best_name = "Conj. Reg."
    elif fixed["nmi"] == best_nmi:
        best_name = "Fixed rho=0"
    else:
        best_name = "Learnable"

    fig.suptitle(
        f"Binomial-Bernoulli MNIST Clustering (Best by NMI: {best_name}, NMI={best_nmi:.3f})",
        fontsize=12,
        y=0.98,
    )

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
