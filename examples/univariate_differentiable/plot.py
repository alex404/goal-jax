"""Plotting for numerically optimized univariate distributions."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import apply_style, colors, example_paths
from .types import DifferentiableUnivariateResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(DifferentiableUnivariateResults, paths.load_analysis())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # von Mises density
    vm = results["von_mises"]
    samples = np.array(vm["samples"])
    eval_pts = np.array(vm["eval_points"])

    axes[0, 0].hist(samples, bins=30, density=True, alpha=0.3, label="Samples")
    axes[0, 0].plot(eval_pts, vm["true_density"], color=colors["ground_truth"], label="True")
    axes[0, 0].plot(eval_pts, vm["fitted_density"], color=colors["fitted"], ls="--", label="Fitted")
    axes[0, 0].set_title("von Mises Density")
    axes[0, 0].set_xlabel("Angle (radians)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_xlim(-np.pi, np.pi)
    axes[0, 0].legend()

    # von Mises training
    axes[0, 1].plot(vm["training_history"], color=colors["initial"])
    axes[0, 1].set_title("von Mises Training")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Cross Entropy Loss")
    axes[0, 1].set_yscale("log")

    # COM-Poisson PMF
    cp = results["com_poisson"]
    samples = np.array(cp["samples"])
    eval_pts = np.array(cp["eval_points"])

    unique, counts = np.unique(samples, return_counts=True)
    axes[1, 0].bar(unique, counts / len(samples), alpha=0.3, label="Samples")
    axes[1, 0].plot(eval_pts, cp["true_pmf"], color=colors["ground_truth"], label="True")
    axes[1, 0].plot(eval_pts, cp["fitted_pmf"], color=colors["fitted"], ls="--", label="Fitted")
    axes[1, 0].set_title("COM-Poisson PMF")
    axes[1, 0].set_xlabel("Count")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].legend()

    # COM-Poisson training
    axes[1, 1].plot(cp["training_history"], color=colors["initial"])
    axes[1, 1].set_title("COM-Poisson Training")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Cross Entropy Loss")
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
