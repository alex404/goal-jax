"""Plotting for HMM example.

Visualizes:
- Top row: training log-likelihoods for Gradient and EM (against the oracle log-likelihood under true parameters).
- Middle row: oracle filtered and smoothed state-probability heatmaps with the true state path overlaid.
- Bottom row: learned smoothed state probabilities for Gradient and EM.
"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..shared import (
    apply_style,
    colors,
    example_paths,
    figure_size,
    model_color,
    plot_training_history,
)
from .types import HMMResults


def _heatmap(
    ax: Axes,
    probs: NDArray[np.float64],
    true_states: NDArray[np.int64] | None,
    title: str,
) -> None:
    ax.imshow(probs.T, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    if true_states is not None:
        ax.plot(
            np.arange(len(true_states)),
            true_states,
            color=colors["ground_truth"],
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label="True state",
        )
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("Time step")
    ax.set_ylabel("State")
    ax.set_title(title)


def main() -> None:
    paths = example_paths(__file__)
    apply_style(paths)
    results = cast(HMMResults, paths.load_analysis())

    n_states = results["n_states"]
    n_steps = results["n_steps"]

    observations = np.asarray(results["observations"], dtype=np.int64)
    true_states = np.asarray(results["true_states"], dtype=np.int64)
    filtered = np.asarray(results["filtered_probs"], dtype=np.float64)
    smoothed = np.asarray(results["smoothed_probs"], dtype=np.float64)
    learned_smoothed = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in results["learned_smoothed_probs"].items()
    }

    # true_states has length n_steps + 1 (includes z_0); align to obs length
    aligned_states = true_states[1 : n_steps + 1]

    fig, axes = plt.subplots(
        3, 2, figsize=figure_size("wide"), constrained_layout=True
    )

    # Top row: training history (left) and observation sequence (right)
    plot_training_history(
        axes[0, 0],
        {k: list(v) for k, v in results["log_likelihoods"].items()},
        ylabel="Average log-likelihood",
    )
    axes[0, 0].axhline(
        results["true_log_lik"],
        color=colors["ground_truth"],
        linestyle="--",
        label="True params",
    )
    axes[0, 0].legend()
    axes[0, 0].set_title("Training")

    axes[0, 1].plot(
        observations, marker="o", linestyle="-", markersize=3, color=model_color(0)
    )
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Observation index")
    axes[0, 1].set_title("Observation sequence")

    # Middle row: oracle filtered / smoothed
    _heatmap(axes[1, 0], filtered, aligned_states, "Oracle filtered $p(z_t \\mid x_{1:t})$")
    _heatmap(axes[1, 1], smoothed, aligned_states, "Oracle smoothed $p(z_t \\mid x_{1:T})$")

    # Bottom row: learned smoothed for each method
    method_names = list(learned_smoothed.keys())
    for i, name in enumerate(method_names[:2]):
        _heatmap(
            axes[2, i],
            learned_smoothed[name],
            aligned_states,
            f"Learned smoothed -- {name}",
        )

    title = (
        f"HMM with {n_states} states and {results['n_obs']} observations "
        + f"(N={results['n_sequences']} trajectories of length {n_steps})"
    )
    fig.suptitle(title)
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
