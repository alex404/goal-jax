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

    observations = np.array(results["observations"])  # (n_steps, obs_dim)
    true_latents = np.array(results["true_latents"])  # (n_steps+1, lat_dim)
    filtered_means = np.array(results["filtered_means"])  # (n_steps, lat_dim)
    filtered_stds = np.array(results["filtered_stds"])  # (n_steps, lat_dim)
    smoothed_means = np.array(results["smoothed_means"])  # (n_steps, lat_dim)
    smoothed_stds = np.array(results["smoothed_stds"])  # (n_steps, lat_dim)
    log_likelihoods = np.array(results["log_likelihoods"])

    # Create figure: 2 columns, 2 rows (top row: EM + observations, bottom: single wide panel)
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # --- Top left: EM convergence ---
    ax = fig.add_subplot(gs[0, 0])
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

    # --- Top right: Observations with true position ---
    ax = fig.add_subplot(gs[0, 1])
    # Observations (first dim)
    ax.scatter(
        time,
        observations[:, 0],
        color=colors["initial"],
        alpha=0.5,
        s=20,
        label="Observations",
        zorder=2,
    )
    # True position (latent dim 0, skip z_0)
    ax.plot(
        time,
        true_latents[1:, 0],
        color=colors["ground_truth"],
        linewidth=2,
        label="True position",
        zorder=3,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Position")
    ax.set_title("Noisy Observations vs True Latent Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom: Filtered and smoothed with uncertainty ---
    ax = fig.add_subplot(gs[1, :])
    dim = 0  # position dimension

    # True latents (z_1 to z_T, skipping z_0)
    ax.plot(
        time,
        true_latents[1:, dim],
        color=colors["ground_truth"],
        linewidth=2,
        label="True",
        zorder=4,
    )

    # Filtered estimate with uncertainty
    ax.plot(
        time,
        filtered_means[:, dim],
        color=colors["initial"],
        linewidth=1.5,
        label="Filtered",
        zorder=3,
    )
    ax.fill_between(
        time,
        filtered_means[:, dim] - 2 * filtered_stds[:, dim],
        filtered_means[:, dim] + 2 * filtered_stds[:, dim],
        color=colors["initial"],
        alpha=0.15,
    )

    # Smoothed estimate with uncertainty
    ax.plot(
        time,
        smoothed_means[:, dim],
        color=colors["fitted"],
        linewidth=1.5,
        label="Smoothed",
        zorder=3,
    )
    ax.fill_between(
        time,
        smoothed_means[:, dim] - 2 * smoothed_stds[:, dim],
        smoothed_means[:, dim] + 2 * smoothed_stds[:, dim],
        color=colors["fitted"],
        alpha=0.15,
    )

    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Latent dim {dim} (position)")
    ax.set_title("Filtering vs Smoothing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Kalman Filter: Damped Oscillator (obs_dim={results['obs_dim']}, lat_dim={lat_dim})",
        fontsize=12,
        y=1.02,
    )

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
