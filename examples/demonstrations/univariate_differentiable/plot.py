"""Plotting script for numerically optimized univariate distributions."""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ...shared import initialize_paths
from .types import DifferentiableUnivariateResults, VonMisesResults


def plot_von_mises_density(ax: Axes, results: VonMisesResults) -> None:
    """Plot von Mises density comparison."""
    # Convert to numpy for plotting
    samples: NDArray[np.float64] = np.array(results["samples"])
    eval_points: NDArray[np.float64] = np.array(results["eval_points"])
    true_density: NDArray[np.float64] = np.array(results["true_density"])
    fitted_density: NDArray[np.float64] = np.array(results["fitted_density"])

    ax.hist(samples, bins=30, density=True, alpha=0.3, label="Samples")
    ax.plot(eval_points, true_density, "b-", label="True Density")
    ax.plot(eval_points, fitted_density, "r--", label="Fitted Density")

    ax.set_title("von Mises Density")
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(-np.pi, np.pi)


def plot_training_history(ax: Axes, results: VonMisesResults) -> None:
    """Plot optimization training history."""
    training_history: NDArray[np.float64] = np.array(results["training_history"])

    ax.plot(training_history, "b-")
    ax.set_title("Training History")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_yscale("log")


def main():
    """Create plots for numerical optimization results."""
    paths = initialize_paths(__file__)

    with open(paths.analysis_path) as f:
        results = cast(DifferentiableUnivariateResults, json.load(f))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_von_mises_density(axes[0], results["von_mises"])
    plot_training_history(axes[1], results["von_mises"])

    plt.tight_layout()
    fig.savefig(paths.results_dir / "univariate_differentiable.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
