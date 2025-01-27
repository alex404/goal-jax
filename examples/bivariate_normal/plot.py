"""Test script for bivariate Normal distributions with different covariance structures."""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..shared import initialize_paths
from .types import BivariateResults


def plot_gaussian_comparison(
    target_name: str,
    model_name: str,
    ax: Axes,
    sample: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    target_zs: NDArray[np.float64],
    model_zs: NDArray[np.float64],
) -> None:
    # Find the overall min and max for both axes
    min_val = min(xs.min(), ys.min())
    max_val = max(xs.max(), ys.max())

    # Plot samples
    ax.scatter(sample[:, 0], sample[:, 1], alpha=0.3, s=10, label="Samples")

    # Plot target contours
    ax.contour(
        np.array(xs),
        np.array(ys),
        np.array(target_zs),
        levels=6,
        colors="b",
        alpha=0.5,
    )
    # Add a blue line to the legend for the target
    ax.plot([], [], color="b", label=target_name)

    ax.contour(
        np.array(xs),
        np.array(ys),
        np.array(model_zs),
        levels=6,
        colors="r",
        alpha=0.5,
    )

    # Add a red line to the legend for the model
    ax.plot([], [], color="r", label=model_name)

    # Set equal limits for both axes
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.legend()
    ax.set_aspect("equal")
    ax.set_title(f"{model_name} vs. {target_name}")


def main():
    paths = initialize_paths(__file__)

    with open(paths.analysis_path) as f:
        results = cast(BivariateResults, json.load(f))

    # Plot and save Normal
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sample = np.array(results["sample"])
    xs = np.array(results["plot_xs"])
    ys = np.array(results["plot_ys"])
    sp_dens = np.array(results["scipy_densities"])
    gt_dens = np.array(results["ground_truth_densities"])
    pd_dens = np.array(results["positive_definite_densities"])
    dia_dens = np.array(results["diagonal_densities"])
    scl_dens = np.array(results["scale_densities"])

    plot_gaussian_comparison(
        "Scipy Ground Truth",
        "Goal Ground-Truth",
        ax[0, 0],
        sample,
        xs,
        ys,
        sp_dens,
        gt_dens,
    )
    plot_gaussian_comparison(
        "Ground Truth",
        "Positive Definite Fit",
        ax[0, 1],
        sample,
        xs,
        ys,
        gt_dens,
        pd_dens,
    )
    plot_gaussian_comparison(
        "Ground Truth",
        "Diagonal Fit",
        ax[1, 0],
        sample,
        xs,
        ys,
        gt_dens,
        dia_dens,
    )
    plot_gaussian_comparison(
        "Ground Truth",
        "Scale Fit",
        ax[1, 1],
        sample,
        xs,
        ys,
        gt_dens,
        scl_dens,
    )
    fig.savefig(paths.results_dir / "gaussian.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
