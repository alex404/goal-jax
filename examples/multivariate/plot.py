"""Two-panel multivariate plot: Dirichlet on a 2-simplex projection, MVN in 2D.

Mirrors the Haskell reference plot at ``/home/alex404/code/goal/workspace/matplotlib/multivariate.py``.
"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import (
    apply_style,
    colors,
    example_paths,
    figure_size,
    scatter_samples,
)
from .types import MultivariateResults


def plot_dirichlet(ax, panel) -> None:
    xs = np.array(panel["xrange"])
    ys = np.array(panel["yrange"])
    true_density = np.array(panel["true_density"])
    learned_density = np.array(panel["learned_density"])
    samples = np.array(panel["samples"])
    log_lik = panel["log_likelihood"]

    scatter_samples(
        ax, samples[:, 0], samples[:, 1], color=colors["ground_truth"], alpha=0.6
    )
    true_contour = ax.contour(xs, ys, true_density, colors=colors["ground_truth"])
    learned_contour = ax.contour(xs, ys, learned_density, colors=colors["fitted"])

    h_true, _ = true_contour.legend_elements()
    h_learned, _ = learned_contour.legend_elements()
    ax.legend(
        [h_true[0], h_learned[0]],
        ["True Density", "Learned Density"],
        loc="lower left",
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Dirichlet(3, 7, 5)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    inset = ax.inset_axes((0.55, 0.55, 0.4, 0.4))
    inset.plot(log_lik, color=colors["ground_truth"])
    inset.set_xlabel("Epoch")
    inset.set_ylabel("Log-Likelihood")


def plot_normal(ax, panel) -> None:
    xs = np.array(panel["xrange"])
    ys = np.array(panel["yrange"])
    true_density = np.array(panel["true_density"])
    source_density = np.array(panel["source_fit_density"])
    natural_density = np.array(panel["natural_fit_density"])
    samples = np.array(panel["samples"])

    scatter_samples(ax, samples[:, 0], samples[:, 1], color=colors["ground_truth"])
    true_c = ax.contour(xs, ys, true_density, colors=colors["ground_truth"])
    source_c = ax.contour(xs, ys, source_density, colors=colors["initial"])
    natural_c = ax.contour(
        xs, ys, natural_density, colors=colors["fitted"], linestyles="dashed"
    )

    h_true, _ = true_c.legend_elements()
    h_source, _ = source_c.legend_elements()
    h_natural, _ = natural_c.legend_elements()
    ax.legend(
        [h_true[0], h_source[0], h_natural[0]],
        ["True Density", "Source Fit", "Natural Fit"],
        loc="upper right",
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Bivariate Normal")


def main() -> None:
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(MultivariateResults, paths.load_analysis())

    fig, axes = plt.subplots(1, 2, figsize=figure_size("wide"))
    plot_dirichlet(axes[0], results["dirichlet"])
    plot_normal(axes[1], results["normal"])
    paths.save_plot(fig)


if __name__ == "__main__":
    main()
