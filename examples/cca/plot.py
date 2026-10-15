"""Plot the canonical correlation analysis results: the two views, the latent, and what one view predicts of the other."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from numpy.typing import NDArray

from ..shared import (
    apply_style,
    colors,
    example_paths,
    figure_size,
    model_color,
    scatter_samples,
)
from .types import CCAResults, ViewFit

N_SHOWN = 600  # data points drawn in the scatter panels


def ellipse(
    mean: NDArray[np.float64], cov: NDArray[np.float64], n_std: float = 2.0
) -> NDArray[np.float64]:
    """Points of the ``n_std`` standard-deviation ellipse of a 2-D Gaussian, shape ``(100, 2)``."""
    vals, vecs = np.linalg.eigh(cov)
    angles = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return mean + n_std * (circle * np.sqrt(vals)) @ vecs.T


def plot_view(
    ax: Axes,
    data: NDArray[np.float64],
    latents: NDArray[np.float64],
    views: list[tuple[str, ViewFit, str, str]],
    name: str,
    title: str,
) -> PathCollection:
    """Data of one view coloured by the true latent, with each (label, view, color, linestyle): the signal as the segment over two standard deviations of the latent either side of the mean, the noise as its 2-sd ellipse."""
    points = ax.scatter(
        data[:N_SHOWN, 0],
        data[:N_SHOWN, 1],
        c=latents[:N_SHOWN],
        cmap="viridis",
        s=8,
        alpha=0.7,
    )
    for label, view, color, style in views:
        mean, signal = np.asarray(view["mean"]), np.asarray(view["signal"])
        ends = np.stack([mean - 2 * signal, mean + 2 * signal])
        ax.plot(*ends.T, color=color, linestyle=style, linewidth=2.5, label=label)
        noise = ellipse(mean, np.asarray(view["noise_cov"]))
        ax.plot(*noise.T, color=color, linestyle=style)
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel(rf"${name}_1$")
    ax.set_ylabel(rf"${name}_2$")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return points


def main() -> None:
    paths = example_paths(__file__)
    apply_style(paths)
    results = cast(CCAResults, paths.load_analysis())

    obs = np.asarray(results["observations"])
    fst_dim = results["fst_dim"]
    fst, snd = obs[:, :fst_dim], obs[:, fst_dim:]
    latents = np.asarray(results["true_latents"])[:, 0]
    grad_fit = results["gradient_ascent"]
    em_fit = results["expectation_maximization"]
    fits = [(grad_fit, model_color(0)), (em_fit, model_color(1))]

    fig, axes = plt.subplots(2, 3, figsize=figure_size("large"))

    # The two views, coloured by the shared latent
    plot_view(
        axes[0, 0],
        fst,
        latents,
        [("Truth", results["true_fst_view"], colors["ground_truth"], "--")]
        + [(f["method"], f["fst_view"], c, "-") for f, c in fits],
        "x",
        "View x (correlated noise)",
    )
    points = plot_view(
        axes[0, 1],
        snd,
        latents,
        [("Truth", results["true_snd_view"], colors["ground_truth"], "--")]
        + [(f["method"], f["snd_view"], c, "-") for f, c in fits],
        "y",
        "View y (independent noise)",
    )
    fig.colorbar(points, ax=axes[0, 1], label="True latent $z$")

    # Latent recovery
    ax = axes[0, 2]
    for fit, color in fits:
        inferred = np.asarray(fit["inferred_latents"])[:, 0]
        scatter_samples(
            ax,
            latents[:N_SHOWN],
            inferred[:N_SHOWN],
            color=color,
            label=f"{fit['method']}, RMSE {fit['latent_rmse']:.2f}",
        )
    lim = float(np.max(np.abs(latents))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=colors["ground_truth"], linewidth=1)
    ax.set_xlabel("True latent $z$")
    ax.set_ylabel(r"$E[z \mid x, y]$ (aligned)")
    ax.set_title("Latent Recovery")
    ax.legend()

    # Training curves
    ax = axes[1, 0]
    for fit, color in fits:
        lls = fit["log_likelihoods"]
        ax.plot(np.arange(1, len(lls) + 1), lls, color=color, label=fit["method"])
    ax.set_xscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\frac{1}{N}\sum_i \log p(x_i, y_i)$")
    ax.set_title("Training")
    ax.legend()

    # The second view predicted from the first
    ax = axes[1, 1]
    predicted = np.asarray(em_fit["predicted_snd"])
    for k in range(results["snd_dim"]):
        scatter_samples(
            ax,
            snd[:N_SHOWN, k],
            predicted[:N_SHOWN, k],
            color=model_color(2 + k),
            label=rf"$y_{k + 1}$",
        )
    lim = float(np.max(np.abs(snd))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=colors["ground_truth"], linewidth=1)
    ax.set_xlabel(r"Observed $y$")
    ax.set_ylabel(r"$E[y \mid x]$")
    ax.set_title(f"Predicting y from x, RMSE {em_fit['prediction_rmse']:.2f}")
    ax.legend()

    # Cross-view covariance: the structure a two-view model exists to capture
    ax = axes[1, 2]
    data_cross = np.asarray(results["true_cross_covariance"]).ravel()
    for fit, color in fits:
        model_cross = np.asarray(fit["learned_cross_covariance"]).ravel()
        scatter_samples(
            ax,
            data_cross,
            model_cross,
            color=color,
            alpha=0.9,
            s=45,
            label=f"{fit['method']}, {fit['cross_covariance_error']:.1%} error",
        )
    lim = float(np.max(np.abs(data_cross))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=colors["ground_truth"], linewidth=1)
    ax.set_xlabel(r"Data $\mathrm{Cov}(x, y)$")
    ax.set_ylabel(r"Model $\mathrm{Cov}(x, y)$")
    ax.set_title("Cross-View Covariance")
    ax.legend()

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
