"""Plotting for Boltzmann HMoG Two Moons example."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..shared import apply_style, colors, example_paths, model_colors
from .types import BoltzmannHMoGMoonsResults


def plot_density_with_data(
    ax: Axes,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    density: NDArray[np.float64],
    data: NDArray[np.float64],
    labels: NDArray[np.int32],
    title: str,
) -> None:
    """Plot density contours with data points colored by moon."""
    # Plot density as filled contours
    ax.contourf(xs, ys, density, levels=20, cmap="Blues", alpha=0.7)
    ax.contour(xs, ys, density, levels=8, colors="darkblue", alpha=0.5, linewidths=0.5)

    # Plot data points colored by ground truth moon
    moon_colors = [colors["fitted"], colors["tertiary"]]
    for moon_id in [0, 1]:
        mask = labels == moon_id
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            c=moon_colors[moon_id],
            s=15,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.3,
            label=f"Moon {moon_id + 1}",
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)


def plot_assignments(
    ax: Axes,
    data: NDArray[np.float64],
    assignments: NDArray[np.float64],
    title: str,
) -> None:
    """Plot data colored by soft cluster assignments."""
    n_components = assignments.shape[1]

    # Use hard assignments for coloring
    hard_assignments = np.argmax(assignments, axis=1)

    # Create colormap
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_components)

    for k in range(n_components):
        mask = hard_assignments == k
        if np.sum(mask) > 0:
            ax.scatter(
                data[mask, 0],
                data[mask, 1],
                c=[cmap(k)],
                s=15,
                alpha=0.7,
                label=f"Cluster {k + 1}",
            )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def plot_binary_activations(
    ax: Axes,
    activations: NDArray[np.float64],
    labels: NDArray[np.int32],
    title: str,
) -> None:
    """Plot binary activation patterns.

    Shows first 2 binary dimensions as x/y position in a separate plot,
    with points colored by ground truth moon to see if binary codes separate moons.
    """
    n_latents = activations.shape[1]

    if n_latents >= 2:
        # Use first two binary activations as coordinates
        moon_colors = [colors["fitted"], colors["tertiary"]]
        for moon_id in [0, 1]:
            mask = labels == moon_id
            ax.scatter(
                activations[mask, 0],
                activations[mask, 1],
                c=moon_colors[moon_id],
                s=20,
                alpha=0.5,
                label=f"Moon {moon_id + 1}",
            )

        ax.set_xlabel("Binary Unit 1 Activation")
        ax.set_ylabel("Binary Unit 2 Activation")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    else:
        # Histogram of single activation
        for moon_id in [0, 1]:
            mask = labels == moon_id
            ax.hist(
                activations[mask, 0],
                bins=20,
                alpha=0.5,
                label=f"Moon {moon_id + 1}",
            )
        ax.set_xlabel("Binary Unit Activation")

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(BoltzmannHMoGMoonsResults, paths.load_analysis())

    # Load data
    data = np.array(results["observations"])
    labels = np.array(results["moon_labels"])
    x_range = np.array(results["plot_range_x"])
    y_range = np.array(results["plot_range_y"])
    xx, yy = np.meshgrid(x_range, y_range)

    # Load densities
    gmm_density = np.array(results["gmm_density"])
    boltz_density = np.array(results["boltzmann_density"])

    # Load training histories
    gmm_lls = np.array(results["gmm_log_likelihoods"])
    boltz_lls = np.array(results["boltzmann_log_likelihoods"])

    # Load assignments
    gmm_assignments = np.array(results["gmm_assignments"])
    boltz_assignments = np.array(results["boltzmann_assignments"])

    # Load binary activations
    boltz_activations = np.array(results["boltzmann_binary_activations"])

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Row 1: Learned densities and training history
    ax_gmm = fig.add_subplot(gs[0, 0])
    ax_boltz = fig.add_subplot(gs[0, 1])
    ax_ll = fig.add_subplot(gs[0, 2])

    plot_density_with_data(
        ax_gmm, xx, yy, gmm_density, data, labels,
        f"GMM ({gmm_assignments.shape[1]} components)"
    )
    plot_density_with_data(
        ax_boltz, xx, yy, boltz_density, data, labels,
        f"Boltzmann HMoG ({boltz_activations.shape[1]} binary units)"
    )

    # Training history
    ax_ll.plot(gmm_lls, label="GMM", color=model_colors[0])
    ax_ll.plot(boltz_lls, label="Boltzmann HMoG", color=model_colors[1])
    ax_ll.set_xlabel("Epoch")
    ax_ll.set_ylabel("Log Likelihood")
    ax_ll.set_title("Training History")
    ax_ll.legend()
    ax_ll.grid(True, alpha=0.3)

    # Row 2: Cluster assignments and binary activations
    ax_gmm_assign = fig.add_subplot(gs[1, 0])
    ax_boltz_assign = fig.add_subplot(gs[1, 1])
    ax_binary = fig.add_subplot(gs[1, 2])

    plot_assignments(ax_gmm_assign, data, gmm_assignments, "GMM Cluster Assignments")
    plot_assignments(
        ax_boltz_assign, data, boltz_assignments, "Boltzmann Cluster Assignments"
    )
    plot_binary_activations(ax_binary, boltz_activations, labels, "Binary Latent Codes")

    plt.suptitle(
        "Boltzmann HMoG vs GMM on Two Moons Dataset",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()
