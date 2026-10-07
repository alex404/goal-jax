"""Plot the canonical correlation analysis results, one row per training method."""

from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..shared import (
    apply_style,
    colors,
    example_paths,
    figure_size,
    model_color,
    scatter_samples,
)
from .types import CCAFit, CCAResults


def plot_fit(
    axes: list[Axes],
    results: CCAResults,
    fit: CCAFit,
    other: CCAFit,
    color: str,
    step_label: str,
) -> None:
    """Training curve, latent recovery and cross-view covariance of one fit."""
    # Training curve, with the other method's final value for comparison
    ax = axes[0]
    ax.plot(fit["log_likelihoods"], color=color, label=fit["method"])
    ax.axhline(
        other["log_likelihoods"][-1],
        color=colors["ground_truth"],
        linestyle="--",
        linewidth=1,
        label=f"{other['method']}, final",
    )
    ax.set_xlabel(step_label)
    ax.set_ylabel(r"$\frac{1}{N}\sum_i \log p(x_i, y_i)$")
    ax.set_title(f"{fit['method']}: log-likelihood")
    ax.legend()

    # Latent recovery, one series per latent dimension
    ax = axes[1]
    true_latents = jnp.asarray(results["true_latents"])
    inferred = jnp.asarray(fit["inferred_latents"])
    for dim in range(results["lat_dim"]):
        scatter_samples(
            ax,
            true_latents[:, dim],
            inferred[:, dim],
            color=colors["fitted"] if dim == 0 else colors["initial"],
            label=rf"$z_{dim + 1}$",
        )
    lim = float(jnp.max(jnp.abs(true_latents))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=colors["ground_truth"], linewidth=1)
    ax.set_xlabel("True latent")
    ax.set_ylabel("Inferred latent (aligned)")
    ax.set_title(f"Latent recovery, RMSE {fit['latent_rmse']:.3f}")
    ax.legend()

    # Cross-view covariance: the structure a two-view model exists to capture
    ax = axes[2]
    data_cross = jnp.asarray(results["true_cross_covariance"]).ravel()
    model_cross = jnp.asarray(fit["learned_cross_covariance"]).ravel()
    scatter_samples(ax, data_cross, model_cross, color=color, alpha=0.9, s=45)
    lim = float(jnp.max(jnp.abs(data_cross))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=colors["ground_truth"], linewidth=1)
    ax.set_xlabel(r"Data $\mathrm{Cov}(x, y)$")
    ax.set_ylabel(r"Model $\mathrm{Cov}(x, y)$")
    ax.set_title(f"Cross-view covariance, {fit['cross_covariance_error']:.1%} error")


def main() -> None:
    paths = example_paths(__file__)
    apply_style(paths)
    results = cast(CCAResults, paths.load_analysis())

    fig, axes = plt.subplots(2, 3, figsize=figure_size("large"))
    grad_fit = results["gradient_ascent"]
    em_fit = results["expectation_maximization"]
    plot_fit(list(axes[0]), results, grad_fit, em_fit, model_color(0), "Gradient step")
    plot_fit(list(axes[1]), results, em_fit, grad_fit, model_color(1), "EM step")

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
