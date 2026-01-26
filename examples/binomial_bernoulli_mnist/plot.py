"""Plotting for Binomial-Bernoulli MNIST example."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..shared import apply_style, example_paths, model_colors
from .types import BinomialBernoulliMNISTResults, ModelResults

# MNIST image dimensions
IMG_SIZE = 28
N_TRIALS = 16

# Colors for three models
COLORS = {
    "learnable": model_colors[0],  # Blue
    "fixed": model_colors[1],  # Red
    "conj_reg": model_colors[2],  # Purple
}

LABELS = {
    "learnable": "Learnable rho",
    "fixed": "Fixed rho=0",
    "conj_reg": "Conj. Regularized",
}


def plot_training_elbos(
    ax: Axes,
    learnable: ModelResults,
    fixed: ModelResults,
    conj_reg: ModelResults,
) -> None:
    """Plot ELBO training curves for all three models."""
    ax.plot(
        learnable["training_elbos"],
        color=COLORS["learnable"],
        label=LABELS["learnable"],
        alpha=0.7,
    )
    ax.plot(
        fixed["training_elbos"], color=COLORS["fixed"], label=LABELS["fixed"], alpha=0.7
    )
    ax.plot(
        conj_reg["training_elbos"],
        color=COLORS["conj_reg"],
        label=LABELS["conj_reg"],
        alpha=0.7,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("ELBO")
    ax.set_title("Training ELBO")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_conjugation_variance(
    ax: Axes,
    learnable: ModelResults,
    fixed: ModelResults,
    conj_reg: ModelResults,
) -> None:
    """Plot conjugation error (Var[f̃]) over time."""
    ax.plot(
        learnable["conjugation_errors"],
        color=COLORS["learnable"],
        label=LABELS["learnable"],
        alpha=0.7,
    )
    ax.plot(
        fixed["conjugation_errors"],
        color=COLORS["fixed"],
        label=LABELS["fixed"],
        alpha=0.7,
    )
    ax.plot(
        conj_reg["conjugation_errors"],
        color=COLORS["conj_reg"],
        label=LABELS["conj_reg"],
        alpha=0.7,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Var[f̃]")
    ax.set_title("Conjugation Error (Variance)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def plot_conjugation_std(
    ax: Axes,
    learnable: ModelResults,
    fixed: ModelResults,
    conj_reg: ModelResults,
) -> None:
    """Plot conjugation std (Std[f̃]) over time."""
    ax.plot(
        learnable["conjugation_stds"],
        color=COLORS["learnable"],
        label=LABELS["learnable"],
        alpha=0.7,
    )
    ax.plot(
        fixed["conjugation_stds"],
        color=COLORS["fixed"],
        label=LABELS["fixed"],
        alpha=0.7,
    )
    ax.plot(
        conj_reg["conjugation_stds"],
        color=COLORS["conj_reg"],
        label=LABELS["conj_reg"],
        alpha=0.7,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Std[f̃]")
    ax.set_title("Conjugation Error (Std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_conjugation_r2(
    ax: Axes,
    learnable: ModelResults,
    fixed: ModelResults,
    conj_reg: ModelResults,
) -> None:
    """Plot R² over time."""
    ax.plot(
        learnable["conjugation_r2s"],
        color=COLORS["learnable"],
        label=LABELS["learnable"],
        alpha=0.7,
    )
    ax.plot(
        fixed["conjugation_r2s"],
        color=COLORS["fixed"],
        label=LABELS["fixed"],
        alpha=0.7,
    )
    ax.plot(
        conj_reg["conjugation_r2s"],
        color=COLORS["conj_reg"],
        label=LABELS["conj_reg"],
        alpha=0.7,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("R²")
    ax.set_title("Conjugation R² (higher = better linear fit)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)


def plot_rho_norm(
    ax: Axes,
    learnable: ModelResults,
    fixed: ModelResults,
    conj_reg: ModelResults,
) -> None:
    """Plot rho norm over time."""
    ax.plot(
        learnable["rho_norms"],
        color=COLORS["learnable"],
        label=LABELS["learnable"],
        alpha=0.7,
    )
    ax.plot(
        fixed["rho_norms"], color=COLORS["fixed"], label=LABELS["fixed"], alpha=0.7
    )
    ax.plot(
        conj_reg["rho_norms"],
        color=COLORS["conj_reg"],
        label=LABELS["conj_reg"],
        alpha=0.7,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("||rho||")
    ax.set_title("Rho Norm Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_metrics_comparison(
    ax: Axes,
    learnable: ModelResults,
    fixed: ModelResults,
    conj_reg: ModelResults,
) -> None:
    """Plot bar chart comparing purity and NMI."""
    x = np.array([0, 1, 2])
    width = 0.35

    # Purity bars
    purity_vals = [
        learnable["cluster_purity"] * 100,
        fixed["cluster_purity"] * 100,
        conj_reg["cluster_purity"] * 100,
    ]
    bars1 = ax.bar(
        x - width / 2, purity_vals, width, label="Purity (%)", color="steelblue"
    )

    # NMI bars (scaled to percentage for visualization)
    nmi_vals = [
        learnable["nmi"] * 100,
        fixed["nmi"] * 100,
        conj_reg["nmi"] * 100,
    ]
    bars2 = ax.bar(x + width / 2, nmi_vals, width, label="NMI (%)", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(["Learnable", "Fixed rho=0", "Conj. Reg."], fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Clustering Metrics Comparison")
    ax.set_ylim(0, 70)
    ax.legend(fontsize=8)

    # Add value labels
    for bar, val in zip(bars1, purity_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}", ha="center", fontsize=7
        )
    for bar, val in zip(bars2, nmi_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}", ha="center", fontsize=7
        )

    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.grid(True, alpha=0.3, axis="y")


def plot_image_grid(
    ax: Axes,
    images: NDArray[np.float64],
    title: str,
    n_rows: int = 2,
    n_cols: int = 5,
) -> None:
    """Plot a grid of images."""
    n_images = min(n_rows * n_cols, images.shape[0])
    gap = 2

    grid_h = n_rows * IMG_SIZE + (n_rows - 1) * gap
    grid_w = n_cols * IMG_SIZE + (n_cols - 1) * gap
    grid = np.full((grid_h, grid_w), np.nan)

    for i in range(n_images):
        row = i // n_cols
        col = i % n_cols
        y_start = row * (IMG_SIZE + gap)
        x_start = col * (IMG_SIZE + gap)
        img = images[i].reshape(IMG_SIZE, IMG_SIZE)
        grid[y_start : y_start + IMG_SIZE, x_start : x_start + IMG_SIZE] = img

    ax.imshow(grid, cmap="gray", vmin=0, vmax=N_TRIALS)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def create_samples_and_prototypes_figure(
    learnable: ModelResults,
    fixed: ModelResults,
    conj_reg: ModelResults,
) -> Figure:
    """Create a figure showing generated samples and prototypes for all three models."""
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Learnable
    plot_image_grid(
        axes[0, 0],
        np.array(learnable["generated_samples"]),
        "Generated Samples (Learnable rho)",
        n_rows=4,
        n_cols=5,
    )
    plot_image_grid(
        axes[0, 1],
        np.array(learnable["cluster_prototypes"]),
        "Cluster Prototypes (Learnable rho)",
        n_rows=2,
        n_cols=5,
    )

    # Fixed
    plot_image_grid(
        axes[1, 0],
        np.array(fixed["generated_samples"]),
        "Generated Samples (Fixed rho=0)",
        n_rows=4,
        n_cols=5,
    )
    plot_image_grid(
        axes[1, 1],
        np.array(fixed["cluster_prototypes"]),
        "Cluster Prototypes (Fixed rho=0)",
        n_rows=2,
        n_cols=5,
    )

    # Conj Reg
    plot_image_grid(
        axes[2, 0],
        np.array(conj_reg["generated_samples"]),
        "Generated Samples (Conj. Regularized)",
        n_rows=4,
        n_cols=5,
    )
    plot_image_grid(
        axes[2, 1],
        np.array(conj_reg["cluster_prototypes"]),
        "Cluster Prototypes (Conj. Regularized)",
        n_rows=2,
        n_cols=5,
    )

    fig.suptitle("Generated Samples and Cluster Prototypes (All Models)", fontsize=12)
    plt.tight_layout()
    return fig


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(BinomialBernoulliMNISTResults, paths.load_analysis())

    # Extract data for all three models
    learnable = results["learnable_rho"]
    fixed = results["fixed_rho"]
    conj_reg = results["conjugate_regularized"]

    # Create main figure with training metrics (4x2 grid)
    fig1 = plt.figure(figsize=(14, 16))
    gs = fig1.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])

    # Row 1: ELBO, Metrics comparison
    ax_elbo = fig1.add_subplot(gs[0, 0])
    ax_metrics = fig1.add_subplot(gs[0, 1])

    # Row 2: Conjugation Variance, Conjugation Std
    ax_var = fig1.add_subplot(gs[1, 0])
    ax_std = fig1.add_subplot(gs[1, 1])

    # Row 3: Conjugation R², Rho norm
    ax_r2 = fig1.add_subplot(gs[2, 0])
    ax_rho = fig1.add_subplot(gs[2, 1])

    # Row 4: Reconstructions
    ax_orig = fig1.add_subplot(gs[3, 0])
    ax_recon = fig1.add_subplot(gs[3, 1])

    # Plot training metrics
    plot_training_elbos(ax_elbo, learnable, fixed, conj_reg)
    plot_metrics_comparison(ax_metrics, learnable, fixed, conj_reg)
    plot_conjugation_variance(ax_var, learnable, fixed, conj_reg)
    plot_conjugation_std(ax_std, learnable, fixed, conj_reg)
    plot_conjugation_r2(ax_r2, learnable, fixed, conj_reg)
    plot_rho_norm(ax_rho, learnable, fixed, conj_reg)

    # Plot reconstructions
    plot_image_grid(
        ax_orig, np.array(results["original_images"]), "Original Images", n_rows=4, n_cols=5
    )
    plot_image_grid(
        ax_recon,
        np.array(results["reconstructed_images"]),
        "Reconstructions (Best Model)",
        n_rows=4,
        n_cols=5,
    )

    # Summary text
    best_nmi = max(learnable["nmi"], fixed["nmi"], conj_reg["nmi"])
    if conj_reg["nmi"] == best_nmi:
        best_name = "Conj. Reg."
    elif fixed["nmi"] == best_nmi:
        best_name = "Fixed rho=0"
    else:
        best_name = "Learnable"

    fig1.suptitle(
        f"Binomial-Bernoulli MNIST Training (Best by NMI: {best_name}, NMI={best_nmi:.3f})",
        fontsize=12,
        y=0.99,
    )

    plt.tight_layout()
    paths.save_plot(fig1)
    print(f"Main plot saved to {paths.plot_path}")

    # Create second figure with samples and prototypes for all models
    fig2 = create_samples_and_prototypes_figure(learnable, fixed, conj_reg)
    samples_path = paths.plot_path.with_stem(paths.plot_path.stem + "_samples")
    fig2.savefig(samples_path, bbox_inches="tight", dpi=150)
    print(f"Samples plot saved to {samples_path}")


if __name__ == "__main__":
    main()
