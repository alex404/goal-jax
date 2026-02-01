"""Plotting for latent trajectory recovery example.

Creates a 2x2 figure showing:
- True latent trajectory (figure-8, colored by position)
- Recovered latent trajectory (colored by reconstruction error)
- Training log-likelihood history
- Reconstruction error along trajectory

Usage:
    python -m examples.dimensionality_reduction.plot
"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import FIGURE_SIZES, apply_style, colors, example_paths
from .types import TrajectoryResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(TrajectoryResults, paths.load_analysis())

    true_latents = np.array(results["true_latents"])
    aligned_latents = np.array(results["aligned_latents"])
    recon_errors = np.array(results["reconstruction_errors"])
    training_lls = np.array(results["training_lls"])

    n_points = results["n_points"]
    obs_dim = results["obs_dim"]
    noise_std = results["noise_std"]

    # Color by position along trajectory
    trajectory_colors = np.linspace(0, 1, n_points)

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES["medium"])

    # Panel 1: True latent trajectory (colored by position)
    ax = axes[0, 0]
    ax.scatter(
        true_latents[:, 0],
        true_latents[:, 1],
        c=trajectory_colors,
        cmap="viridis",
        s=20,
        alpha=0.8,
    )
    ax.plot(true_latents[:, 0], true_latents[:, 1], color="gray", alpha=0.3, linewidth=1)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("True Latent (colored by position)")
    ax.set_aspect("equal")
    ax.grid(True)

    # Panel 2: Recovered latent trajectory (colored by position, like panel 1)
    ax = axes[0, 1]
    ax.scatter(
        aligned_latents[:, 0],
        aligned_latents[:, 1],
        c=trajectory_colors,
        cmap="viridis",
        s=20,
        alpha=0.8,
    )
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("Recovered Latent (Procrustes-aligned)")
    ax.set_aspect("equal")
    ax.grid(True)

    # Compute alignment RMSE for annotation
    alignment_rmse = np.sqrt(np.mean(np.sum((aligned_latents - true_latents) ** 2, axis=1)))
    ax.text(
        0.05,
        0.95,
        f"RMSE: {alignment_rmse:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel 3: Training log-likelihood history
    ax = axes[1, 0]
    ax.plot(training_lls, color=colors["fitted"], linewidth=2)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Training History")
    ax.grid(True)

    # Panel 4: Reconstruction error by trajectory segment (color-coded)
    ax = axes[1, 1]
    n_bins = 8
    bin_size = n_points // n_bins
    bin_errors = []
    bin_colors = []
    cmap = plt.get_cmap("viridis")

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else n_points
        bin_errors.append(np.mean(recon_errors[start_idx:end_idx]))
        # Color based on midpoint of bin
        bin_colors.append(cmap((start_idx + end_idx) / 2 / n_points))

    ax.bar(range(n_bins), bin_errors, color=bin_colors, edgecolor="white", linewidth=1)
    ax.axhline(
        y=np.mean(recon_errors),
        color=colors["ground_truth"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(recon_errors):.2f}",
    )
    ax.set_xlabel("Trajectory Segment (colored as in latent plots)")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Error by Trajectory Segment")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")
    ax.set_xticks([])  # Remove numeric ticks - color tells the story

    fig.suptitle(
        f"Factor Analysis: {obs_dim}D to 2D (noise std={noise_std})",
        fontsize=12,
    )

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
