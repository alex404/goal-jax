"""Plot the canonical circuit fit with and without the conjugation penalty.

The first column holds what the fits share: the test images, one per row, at their bump positions
$t$; the test ELBO of each fit during training, against the test log-likelihood of Gaussians fit to
the training set; and the s.d. of the conjugation residual of each level under the model and the
recognition model. The other two columns hold one fit each: samples at their latents $z$, on the
colour scale of the data; the mean of $q(z \\mid x)$ against the bump position, with one s.d.; and the
firing probability of each neuron given $z$ under the generative model, coloured by the
order of the centres of mass of the tuning curves.
"""

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from ..shared import apply_style, colors, example_paths, figure_size, model_color
from .types import Results

LEVEL_STYLES: dict[str, Any] = {
    "X-N level, model": "-",
    "X-N level, recognition": "--",
    "N-Z level, model": ":",
    "N-Z level, recognition": "-.",
}


def edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges around sorted centres: midpoints between neighbours, extended at both ends."""
    mids = (centres[1:] + centres[:-1]) / 2
    return np.concatenate(
        [[2 * centres[0] - mids[0]], mids, [2 * centres[-1] - mids[-1]]]
    )


def image_rows(
    ax: Axes, images: np.ndarray, rows: np.ndarray, vmin: float, vmax: float
) -> QuadMesh:
    """Images as rows of pixels, each row at its value of ``rows``."""
    pixel_edges = np.arange(images.shape[1] + 1) - 0.5
    mesh = ax.pcolormesh(
        pixel_edges, edges(rows), images, cmap="viridis", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("Pixel")
    return mesh


def main() -> None:
    paths = example_paths(__file__)
    res: Results = paths.load_analysis()
    apply_style(paths)
    fig = plt.figure(figsize=figure_size("tall"))
    grid = GridSpec(3, 3, figure=fig)
    fits = res["fits"]

    data = np.asarray(res["data_images"])
    sample_images = [np.asarray(fit["sample_images"]) for fit in fits.values()]
    vmin = float(min(data.min(), *(s.min() for s in sample_images)))
    vmax = float(max(data.max(), *(s.max() for s in sample_images)))

    # Shared panels
    ax = fig.add_subplot(grid[0, 0])
    image_rows(ax, data, np.asarray(res["data_t"]), vmin, vmax)
    ax.set_ylabel("Bump position $t$")
    ax.set_title("Data")

    ax = fig.add_subplot(grid[1, 0])
    for i, (name, fit) in enumerate(fits.items()):
        ax.plot(fit["steps"], fit["elbo"], color=model_color(i), label=name)
    for i, (name, value) in enumerate(res["baselines"].items()):
        ax.axhline(
            value, color=colors["ground_truth"], linestyle=("--", ":")[i], label=name
        )
    elbo_max = max(max(fit["elbo"]) for fit in fits.values())
    ax.set_ylim(min(res["baselines"].values()) - 2, elbo_max + 2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Test log-likelihood (nats)")
    ax.set_title("Training")
    ax.legend()

    ax = fig.add_subplot(grid[2, 0])
    for i, fit in enumerate(fits.values()):
        for name, sds in fit["residual_sds"].items():
            ax.plot(
                fit["steps"], sds, color=model_color(i), linestyle=LEVEL_STYLES[name]
            )
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Residual s.d. (nats)")
    ax.set_title("Conjugation Error")
    ax.legend(
        handles=[
            Line2D([], [], color=colors["ground_truth"], linestyle=style, label=name)
            for name, style in LEVEL_STYLES.items()
        ]
    )

    # One column per fit
    for j, (name, fit) in enumerate(fits.items(), start=1):
        sample_ax = fig.add_subplot(grid[0, j])
        mesh = image_rows(
            sample_ax, sample_images[j - 1], np.asarray(fit["sample_z"]), vmin, vmax
        )
        sample_ax.set_ylabel("$z$")
        sample_ax.set_title(f"Samples, {name}")

        ax = fig.add_subplot(grid[1, j])
        ax.errorbar(
            res["test_t"],
            fit["q_mean"],
            yerr=fit["q_sd"],
            fmt="o",
            markersize=2,
            color=model_color(j - 1),
            alpha=0.6,
            label=f"Spearman {fit['spearman']:.3f}",
        )
        ax.set_xlabel("Bump position $t$")
        ax.set_ylabel(r"Mean of $q(z \mid x)$, $\pm$ 1 s.d.")
        ax.set_title(f"Recognition, {name}")
        ax.legend()

        ax = fig.add_subplot(grid[2, j])
        # Coloured by the order of the centres of mass of the tuning curves
        rates = np.asarray(fit["tuning_rates"])
        tuning_z = np.asarray(fit["tuning_z"])
        centres = rates @ tuning_z / rates.sum(axis=1)
        for i, n in enumerate(np.argsort(centres)):
            ax.plot(
                tuning_z, rates[n], color=plt.get_cmap("viridis")(i / (len(rates) - 1))
            )
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$p(n_i = 1 \mid z)$")
        ax.set_title(f"Tuning Curves, {name}")

        if j == len(fits):
            fig.colorbar(mesh, ax=sample_ax, label="Pixel value")

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
