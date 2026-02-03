"""Plotting for Kalman filter example."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths
from .types import KalmanFilterResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(KalmanFilterResults, paths.load_analysis())

    # Extract data
    lat_dim = results["lat_dim"]
    n_steps = results["n_steps"]
    time = np.arange(n_steps)

    true_latents = np.array(results["true_latents"])  # (n_steps+1, lat_dim)
    filtered_means = np.array(results["filtered_means"])  # (n_steps, lat_dim)
    filtered_stds = np.array(results["filtered_stds"])  # (n_steps, lat_dim)
    smoothed_means = np.array(results["smoothed_means"])  # (n_steps, lat_dim)
    smoothed_stds = np.array(results["smoothed_stds"])  # (n_steps, lat_dim)
    log_likelihoods = np.array(results["log_likelihoods"])

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Top left: EM convergence ---
    ax = axes[0, 0]
    ax.plot(log_likelihoods, color=colors["fitted"], linewidth=2)
    ax.axhline(
        y=results["initial_log_lik"],
        color=colors["initial"],
        linestyle="--",
        label="Initial",
    )
    ax.axhline(
        y=results["final_log_lik"],
        color=colors["ground_truth"],
        linestyle="--",
        label="Final",
    )
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Average Log-Likelihood")
    ax.set_title("EM Learning Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Top right: Observations (first 2 dims) ---
    ax = axes[0, 1]
    observations = np.array(results["observations"])
    ax.scatter(
        observations[:, 0],
        observations[:, 1],
        c=time,
        cmap="viridis",
        alpha=0.7,
        s=30,
    )
    ax.set_xlabel("Observation dim 1")
    ax.set_ylabel("Observation dim 2")
    ax.set_title("Observations (colored by time)")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Time step")
    ax.grid(True, alpha=0.3)

    # --- Bottom left: Filtered vs true latents (dim 0) ---
    ax = axes[1, 0]
    dim = 0  # First latent dimension
    # True latents (z_1 to z_T, skipping z_0)
    ax.plot(
        time,
        true_latents[1:, dim],
        color=colors["ground_truth"],
        linewidth=2,
        label="True",
    )
    # Filtered estimate with uncertainty
    ax.plot(
        time,
        filtered_means[:, dim],
        color=colors["initial"],
        linewidth=2,
        label="Filtered",
    )
    ax.fill_between(
        time,
        filtered_means[:, dim] - 2 * filtered_stds[:, dim],
        filtered_means[:, dim] + 2 * filtered_stds[:, dim],
        color=colors["initial"],
        alpha=0.2,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Latent dim {dim}")
    ax.set_title("Forward Filtering")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom right: Smoothed vs true latents (dim 0) ---
    ax = axes[1, 1]
    # True latents (z_1 to z_T, skipping z_0)
    ax.plot(
        time,
        true_latents[1:, dim],
        color=colors["ground_truth"],
        linewidth=2,
        label="True",
    )
    # Smoothed estimate with uncertainty
    ax.plot(
        time,
        smoothed_means[:, dim],
        color=colors["fitted"],
        linewidth=2,
        label="Smoothed",
    )
    ax.fill_between(
        time,
        smoothed_means[:, dim] - 2 * smoothed_stds[:, dim],
        smoothed_means[:, dim] + 2 * smoothed_stds[:, dim],
        color=colors["fitted"],
        alpha=0.2,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Latent dim {dim}")
    ax.set_title("Forward-Backward Smoothing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add overall title with metrics
    fig.suptitle(
        f"Kalman Filter Example (obs_dim={results['obs_dim']}, lat_dim={lat_dim}, n_sequences={results['n_sequences']})",
        fontsize=12,
        y=1.02,
    )

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
