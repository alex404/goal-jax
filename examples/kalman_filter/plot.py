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
    filtered_means = np.array(results["filtered_means"])
    filtered_stds = np.array(results["filtered_stds"])
    smoothed_means = np.array(results["smoothed_means"])
    smoothed_stds = np.array(results["smoothed_stds"])
    learned_filt_means = np.array(results["learned_filtered_means"])
    learned_filt_stds = np.array(results["learned_filtered_stds"])
    learned_smth_means = np.array(results["learned_smoothed_means"])
    learned_smth_stds = np.array(results["learned_smoothed_stds"])
    log_likelihoods = np.array(results["log_likelihoods"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dim = 0  # position dimension

    # --- Top left: True process + observations ---
    ax = axes[0, 0]
    ax.scatter(
        time,
        observations[:, 0],
        color=colors["initial"],
        alpha=0.5,
        s=20,
        label="Observations",
        zorder=2,
    )
    ax.plot(
        time,
        true_latents[1:, dim],
        color=colors["ground_truth"],
        linewidth=2,
        label="True position",
        zorder=3,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Position")
    ax.set_title("True Process and Observations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom left: Oracle filtering/smoothing (true params) ---
    ax = axes[1, 0]
    ax.plot(
        time,
        true_latents[1:, dim],
        color=colors["ground_truth"],
        linewidth=2,
        label="True",
        zorder=4,
    )
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
    ax.set_ylabel("Position")
    ax.set_title("Inference with True Parameters")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Top right: Learning curve ---
    ax = axes[0, 1]
    # x-axis: every 10 steps (since we log every 10 steps, plus initial)
    learn_steps = np.arange(len(log_likelihoods)) * 10
    learn_steps[0] = 0
    ax.plot(learn_steps, log_likelihoods, color=colors["fitted"], linewidth=2)
    ax.axhline(
        y=results["true_log_lik"],
        color=colors["ground_truth"],
        linestyle="--",
        label="True params",
    )
    ax.axhline(
        y=results["initial_log_lik"],
        color=colors["initial"],
        linestyle="--",
        label="Initial",
    )
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Average Log-Likelihood")
    ax.set_title("Gradient Ascent Learning")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom right: Learned filtering/smoothing ---
    ax = axes[1, 1]
    ax.plot(
        time,
        true_latents[1:, dim],
        color=colors["ground_truth"],
        linewidth=2,
        label="True",
        zorder=4,
    )
    ax.plot(
        time,
        learned_filt_means[:, dim],
        color=colors["initial"],
        linewidth=1.5,
        label="Filtered",
        zorder=3,
    )
    ax.fill_between(
        time,
        learned_filt_means[:, dim] - 2 * learned_filt_stds[:, dim],
        learned_filt_means[:, dim] + 2 * learned_filt_stds[:, dim],
        color=colors["initial"],
        alpha=0.15,
    )
    ax.plot(
        time,
        learned_smth_means[:, dim],
        color=colors["fitted"],
        linewidth=1.5,
        label="Smoothed",
        zorder=3,
    )
    ax.fill_between(
        time,
        learned_smth_means[:, dim] - 2 * learned_smth_stds[:, dim],
        learned_smth_means[:, dim] + 2 * learned_smth_stds[:, dim],
        color=colors["fitted"],
        alpha=0.15,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Position")
    ax.set_title("Inference with Learned Parameters")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Kalman Filter: Damped Oscillator (obs_dim={results['obs_dim']}, lat_dim={lat_dim})",
        fontsize=12,
    )

    plt.tight_layout()
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
