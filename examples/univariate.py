"""Test script for univariate distributions in the exponential family."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from goal.distributions import Categorical, MultivariateGaussian, Poisson
from goal.exponential_family import EF, Natural
from goal.linear import PositiveDefinite
from goal.manifold import Point


def test_gaussian(
    ax, mu: float = 2.0, sigma: float = 1.5
) -> tuple[MultivariateGaussian, Point[Natural, MultivariateGaussian]]:
    """Test univariate Gaussian distribution and plot results."""
    # Create distribution and point
    mu_arr = jnp.array([mu])
    covariance = PositiveDefinite.from_params(jnp.array([sigma**2]), dim=1)

    print("Parameter shapes:")
    print(f"Mean shape: {mu_arr.shape}")
    print(f"Covariance shape: {covariance.params.shape}")

    manifold = MultivariateGaussian(data_dim=1, covariance_shape=PositiveDefinite)
    p_source = manifold.from_mean_and_covariance(mu_arr, covariance)
    # Convert to natural via mean coordinates
    p_natural = manifold.to_natural(Point(p_source.params, manifold))

    print("\nNatural parameters:")
    loc, precision = manifold.split_params(p_natural)
    print(f"Precision weighted mean: {loc}")
    print(f"Precision parameters: {precision.params}")

    # Generate points for plotting
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    x_points = [jnp.array([xi]) for xi in x]  # Each x as a 1D array

    our_densities = jnp.exp(
        jnp.array([manifold.log_density(p_natural, xi) for xi in x_points])
    )
    scipy_densities = stats.norm.pdf(x, loc=mu, scale=sigma)

    ax.plot(x, our_densities, "b-", label="Implementation")
    ax.plot(x, scipy_densities, "r--", label="Scipy")
    ax.set_title("Gaussian Density")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()

    return manifold, p_natural


def test_categorical(ax, probs=None) -> tuple[Categorical, Point[Natural, Categorical]]:
    """Test Categorical distribution and plot results."""
    if probs is None:
        probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])

    manifold = Categorical(n_categories=len(probs))
    p_source = manifold.from_probs(probs)
    # Convert to natural via mean coordinates
    p_natural = manifold.to_natural(Point(p_source.params, manifold))

    print("\nCategorical parameters:")
    print(f"Natural parameters: {p_natural.params}")

    # Plot PMF
    categories = np.arange(len(probs))
    cat_probs = jnp.exp(
        jnp.array([manifold.log_density(p_natural, jnp.array([i])) for i in categories])
    )

    width = 0.35
    ax.bar(
        categories - width / 2,
        np.array(cat_probs),
        width,
        alpha=0.6,
        label="Implementation",
    )
    ax.bar(
        categories + width / 2,
        np.array(probs),
        width,
        alpha=0.6,
        label="True Probabilities",
    )
    ax.set_title("Categorical PMF")
    ax.set_xlabel("Category")
    ax.set_ylabel("Probability")
    ax.legend()

    return manifold, p_natural


def test_poisson(ax, rate: float = 5.0) -> tuple[Poisson, Point[Natural, Poisson]]:
    """Test Poisson distribution and plot results."""
    manifold = Poisson()
    p_source = manifold.from_rate(rate)
    # Convert to natural via mean coordinates
    p_natural = manifold.to_natural(Point(p_source.params, manifold))

    print("\nPoisson parameters:")
    print(f"Natural parameters: {p_natural.params}")

    # Generate points for plotting
    max_k = int(stats.poisson.ppf(0.999, rate))
    k = np.arange(0, max_k + 1)
    k_points = jnp.array([[x] for x in k])

    # Compare with scipy implementation
    our_pmf = jnp.exp(
        jnp.array([manifold.log_density(p_natural, ki) for ki in k_points])
    )
    scipy_pmf = stats.poisson.pmf(k, rate)

    # Plot results
    width = 0.35
    ax.bar(k - width / 2, our_pmf, width, alpha=0.6, label="Implementation")
    ax.bar(k + width / 2, scipy_pmf, width, alpha=0.6, label="Scipy")
    ax.set_title(f"Poisson PMF (λ={rate})")
    ax.set_xlabel("k")
    ax.set_ylabel("Probability")
    ax.legend()

    return manifold, p_natural


def test_coordinate_transforms(
    manifold,
    p_natural: Point[Natural, EF],
    test_point: jnp.ndarray,
    dist_name: str,
):
    """Test coordinate transformations for a given distribution."""
    print(f"\n{dist_name} coordinate transforms:")
    print(f"Test point shape: {test_point.shape}")

    # Transform to mean coordinates and back
    p_mean = manifold.to_mean(p_natural)
    p_natural2 = manifold.to_natural(p_mean)

    # Compare log densities
    log_density_nat = manifold.log_density(p_natural, test_point)
    log_density_nat2 = manifold.log_density(p_natural2, test_point)

    print(f"Natural form log density: {float(log_density_nat):.4f}")
    print(f"Mean->Natural form log density: {float(log_density_nat2):.4f}")


def main():
    """Run all distribution tests and plots."""
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    try:
        print("\n=== Testing Gaussian Distribution ===")
        gaussian_manifold, gaussian_point = test_gaussian(ax1)

        print("\n=== Testing Categorical Distribution ===")
        categorical_manifold, categorical_point = test_categorical(ax2)

        print("\n=== Testing Poisson Distribution ===")
        poisson_manifold, poisson_point = test_poisson(ax3)

        plt.tight_layout()
        plt.show()

        # Test coordinate transforms with proper shapes
        print("\n=== Testing Coordinate Transforms ===")
        test_coordinate_transforms(
            gaussian_manifold, gaussian_point, jnp.array([1.5]), "Gaussian"
        )
        test_coordinate_transforms(
            categorical_manifold, categorical_point, jnp.array([2]), "Categorical"
        )
        test_coordinate_transforms(
            poisson_manifold, poisson_point, jnp.array([3]), "Poisson"
        )

    except Exception as e:
        print(f"\nError during testing: {e!s}")
        raise


if __name__ == "__main__":
    main()
