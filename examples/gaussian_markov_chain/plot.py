"""Plotting for Gaussian Markov Chain example."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths
from .types import GaussianMarkovChainResults


def main() -> None:
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(GaussianMarkovChainResults, paths.load_analysis())

    true_traj = np.array(results["true_trajectories"])
    learned_traj = np.array(results["learned_trajectories"])
    n_traj, n_states = true_traj.shape

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    time = np.arange(n_states)

    # Left: sample trajectories as 1D time series
    ax = axes[0]
    for i in range(n_traj):
        ax.plot(
            time,
            true_traj[i],
            "-o",
            color=colors["ground_truth"],
            alpha=0.5,
            markersize=3,
            linewidth=1,
            label="True" if i == 0 else None,
        )
        ax.plot(
            time,
            learned_traj[i],
            "-s",
            color=colors["fitted"],
            alpha=0.5,
            markersize=3,
            linewidth=1,
            label="Learned" if i == 0 else None,
        )
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$x_t$")
    ax.set_title("Sample Trajectories")
    ax.legend(fontsize=8)

    # Right: scatter of (x_t, x_{t+1}) consecutive pairs
    ax = axes[1]
    pairs_x = np.array(results["transition_pairs_x"])
    pairs_y = np.array(results["transition_pairs_y"])
    ax.scatter(pairs_x, pairs_y, s=4, alpha=0.3, color=colors["ground_truth"])
    lims = [
        min(pairs_x.min(), pairs_y.min()),
        max(pairs_x.max(), pairs_y.max()),
    ]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8, label=r"$x_{t+1}=x_t$")
    ax.set_xlabel(r"$x_t$")
    ax.set_ylabel(r"$x_{t+1}$")
    ax.set_title("Transition Structure")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Gaussian Markov Chain (dim={results['lat_dim']}, T={results['n_steps']}, LL: true={results['true_log_likelihood']:.2f}, learned={results['learned_log_likelihood']:.2f})",
        fontsize=10,
    )
    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
