"""Test script for bivariate Gaussian distributions with different covariance structures."""

from typing import Type

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from numpy.typing import NDArray

from goal.distributions import Gaussian
from goal.linear import Diagonal, PositiveDefinite, Scale


def create_test_grid(
    mean: Array, cov: Array, n_points: int = 20
) -> tuple[Array, Array, Array]:
    """Create grid of test points for density evaluation."""
    # Compute bounds based on covariance
    std_devs = jnp.sqrt(jnp.diag(cov))
    margin = 3.0  # Number of standard deviations for bounds

    x_min = mean[0] - margin * std_devs[0]
    x_max = mean[0] + margin * std_devs[0]
    y_min = mean[1] - margin * std_devs[1]
    y_max = mean[1] + margin * std_devs[1]

    # Create meshgrid
    x = jnp.linspace(x_min, x_max, n_points)
    y = jnp.linspace(y_min, y_max, n_points)
    xs, ys = jnp.meshgrid(x, y)

    # Stack coordinates for density evaluation
    points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    return points, xs, ys


def compute_gaussian_results(
    key: Array,
    mean: Array,
    covariance: Array,
    sample_size: int,
    covariance_types: tuple[Type[PositiveDefinite], ...] = (
        PositiveDefinite,
        Diagonal,
        Scale,
    ),
) -> tuple[Array, list[tuple[Array, Array, Array, Array]]]:
    """Run bivariate Gaussian computations with different covariance structures.

    Args:
        key: JAX random key
        mean: True mean vector (2,)
        covariance: True covariance matrix (2,2)
        sample_size: Number of samples to draw
        covariance_types: Tuple of covariance structures to test

    Returns:
        samples: Array of samples (sample_size, 2)
        model_results: List of (est_mean, est_cov, X, Z) for each structure
            where Z contains density values on the grid defined by X, Y
    """
    # Create ground truth samples
    samples = jax.random.multivariate_normal(key, mean, covariance, (sample_size,))

    # Create grid for density evaluation
    points, xs, ys = create_test_grid(mean, covariance)

    # Fit models with different covariance structures
    model_results: list[tuple[Array, Array, Array, Array]] = []
    for cov_type in covariance_types:
        # Create manifold with given covariance structure
        manifold = Gaussian(data_dim=2, covariance_shape=cov_type)

        # Estimate parameters
        mean_point = manifold.average_sufficient_statistic(samples)
        natural_point = manifold.to_natural(mean_point)

        # Extract parameters
        est_mean, est_cov = manifold.to_mean_and_covariance(mean_point)

        # Compute densities on grid
        def compute_density(x: Array) -> Array:
            return manifold.density(natural_point, x)

        densities = jax.vmap(compute_density)(points)
        zs = densities.reshape(xs.shape)

        model_results.append((est_mean, est_cov.matrix, xs, zs))

    return samples, model_results


def plot_gaussian_comparison(
    samples: Array,
    model_results: list[tuple[Array, Array, Array, Array]],
    covariance_types: tuple[Type[PositiveDefinite], ...],
    true_mean: Array,
) -> None:
    """Plot samples and density contours for each model.

    Creates a figure with len(covariance_types) + 1 subplots:
    - First shows samples only
    - Others show samples + density contours for one model
    """
    n_plots = len(covariance_types) + 1
    _, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # Convert samples to numpy for plotting
    np_samples: NDArray[np.float64] = np.array(samples)

    # Plot samples only in first panel
    axes[0].scatter(np_samples[:, 0], np_samples[:, 1], alpha=0.5, s=10)
    axes[0].set_title("Samples")

    # Plot each model comparison
    for i, ((est_mean, _, xs, zs), cov_type) in enumerate(
        zip(model_results, covariance_types), 1
    ):
        ax = axes[i]

        # Plot samples
        ax.scatter(np_samples[:, 0], np_samples[:, 1], alpha=0.3, s=10, label="Samples")

        # Plot density contours
        ax.contour(
            np.array(xs), np.array(xs.T), np.array(zs), levels=6, colors="b", alpha=0.5
        )

        # Add means to plot
        ax.plot(true_mean[0], true_mean[1], "r*", markersize=10, label="True Mean")
        ax.plot(est_mean[0], est_mean[1], "b*", markersize=10, label="Est. Mean")

        ax.set_title(f"{cov_type.__name__} Fit")
        ax.legend()

    # Set consistent aspect ratio
    for ax in axes:
        ax.set_aspect("equal")

    plt.tight_layout()


def main():
    """Run bivariate Gaussian tests."""
    key = jax.random.PRNGKey(0)

    # Define ground truth parameters
    true_mean = jnp.array([1.0, -0.5])
    true_cov = jnp.array([[2.0, 1.0], [1.0, 1.0]])

    # Generate samples and fit models
    samples, model_results = compute_gaussian_results(
        key,
        true_mean,
        true_cov,
        sample_size=1000,
        covariance_types=(PositiveDefinite, Diagonal, Scale),
    )

    # Plot results
    plot_gaussian_comparison(
        samples, model_results, (PositiveDefinite, Diagonal, Scale), true_mean
    )
    plt.show()


if __name__ == "__main__":
    main()
