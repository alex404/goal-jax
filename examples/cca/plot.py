"""Plot the canonical correlation analysis results."""

from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt

from ..shared import (
    apply_style,
    colors,
    example_paths,
    figure_size,
    scatter_samples,
)
from .types import CCAResults


def main() -> None:
    paths = example_paths(__file__)
    apply_style(paths)
    results = cast(CCAResults, paths.load_analysis())

    fig, axes = plt.subplots(1, 3, figsize=figure_size("wide"))

    # Training curve
    ax = axes[0]
    lls = jnp.asarray(results["log_likelihoods"])
    ax.plot(lls, color=colors["fitted"])
    ax.set_xlabel("Gradient step")
    ax.set_ylabel(r"$\frac{1}{N}\sum_i \log p(x_i, y_i)$")
    ax.set_title("Observable log-likelihood")

    # Latent recovery, one series per latent dimension
    ax = axes[1]
    true_latents = jnp.asarray(results["true_latents"])
    inferred = jnp.asarray(results["inferred_latents"])
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
    ax.set_title(f"Latent recovery, RMSE {results['latent_rmse']:.3f}")
    ax.legend()

    # Cross-view covariance: the structure a two-view model exists to capture
    ax = axes[2]
    data_cross = jnp.asarray(results["true_cross_covariance"]).ravel()
    model_cross = jnp.asarray(results["learned_cross_covariance"]).ravel()
    scatter_samples(
        ax, data_cross, model_cross, color=colors["fitted"], alpha=0.9, s=45
    )
    lim = float(jnp.max(jnp.abs(data_cross))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=colors["ground_truth"], linewidth=1)
    ax.set_xlabel(r"Data $\mathrm{Cov}(x, y)$")
    ax.set_ylabel(r"Model $\mathrm{Cov}(x, y)$")
    ax.set_title(
        f"Cross-view covariance, {results['cross_covariance_error']:.1%} error"
    )

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
