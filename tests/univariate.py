"""Test script for univariate distributions in the exponential family.

This script provides testing and visualization functionality for univariate
distributions including Gaussian, Categorical, and Poisson distributions.
It tests both density/mass functions and coordinate transformations.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from goal.distributions import Categorical, MultivariateGaussian, Poisson
from goal.exponential_family import ParameterType
from goal.linear import vector_to_symmetric


def test_gaussian(ax, mu: float = 2.0, sigma: float = 1.5) -> MultivariateGaussian:
    """Test univariate Gaussian distribution and plot results."""
    # Create distribution
    mu_arr = jnp.array([mu])
    sigma_arr = jnp.array([[sigma**2]])

    print("Parameter shapes:")
    print(f"Mean shape: {mu_arr.shape}")
    print(f"Covariance shape: {sigma_arr.shape}")

    gaussian = MultivariateGaussian.from_source(
        mean=mu_arr, covariance=sigma_arr, operator_type="posdef"
    )

    nat_params = gaussian.to_natural().params
    print("\nNatural parameters:")
    print(f"Natural parameter shape: {nat_params.shape}")
    print(f"Parameters: {nat_params}")

    dim = gaussian.dim
    eta = nat_params[:dim]
    theta_vec = nat_params[dim:]

    print("\nParameter components:")
    print(f"eta shape: {eta.shape}")
    print(f"theta_vec shape: {theta_vec.shape}")

    if theta_vec.size == 1:
        theta_mat = jnp.array([[theta_vec[0]]])
    else:
        theta_mat = vector_to_symmetric(-2 * theta_vec, dim)

    print(f"Reconstructed precision matrix shape: {theta_mat.shape}")

    # Generate points for plotting
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    x_points = jnp.array([[xi] for xi in x])  # Correct - creates a list first

    our_densities = jnp.exp(jnp.array([gaussian.log_density(xi) for xi in x_points]))
    scipy_densities = stats.norm.pdf(x, loc=mu, scale=sigma)

    ax.plot(x, our_densities, "b-", label="Implementation")
    ax.plot(x, scipy_densities, "r--", label="Scipy")
    ax.set_title("Gaussian Density")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()

    return gaussian


def test_categorical(ax, probs=None) -> Categorical:
    """Test Categorical distribution and plot results.

    Args:
        ax: Matplotlib axis for plotting
        probs: Optional probability vector (defaults to [0.1, 0.2, 0.4, 0.2, 0.1])

    Returns:
        The created Categorical instance
    """
    if probs is None:
        probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])

    # Create distribution in natural coordinates
    categorical = Categorical.from_source(probs[:-1])
    nat_categorical = categorical.to_natural()

    print("\nCategorical parameters:")
    print(f"Natural parameters: {nat_categorical.params}")

    # Plot PMF
    categories = np.arange(len(probs))
    cat_probs = jnp.exp(
        jnp.array([nat_categorical.log_density(jnp.array([i])) for i in categories])
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

    return nat_categorical


def test_poisson(ax, rate: float = 5.0) -> Poisson:
    """Test Poisson distribution and plot results.

    Args:
        ax: Matplotlib axis for plotting
        rate: Rate parameter (lambda)

    Returns:
        The created Poisson instance
    """
    # Create distribution in natural coordinates
    poisson = Poisson.from_source(rate)
    nat_poisson = poisson.to_natural()

    print("\nPoisson parameters:")
    print(f"Natural parameters: {nat_poisson.params}")

    # Generate points for plotting
    max_k = int(stats.poisson.ppf(0.999, rate))
    k = np.arange(0, max_k + 1)
    k_points = jnp.array([[x] for x in k])

    # Compare with scipy implementation
    our_pmf = jnp.exp(jnp.array([nat_poisson.log_density(ki) for ki in k_points]))
    scipy_pmf = stats.poisson.pmf(k, rate)

    # Plot results
    width = 0.35
    ax.bar(k - width / 2, our_pmf, width, alpha=0.6, label="Implementation")
    ax.bar(k + width / 2, scipy_pmf, width, alpha=0.6, label="Scipy")
    ax.set_title(f"Poisson PMF (λ={rate})")
    ax.set_xlabel("k")
    ax.set_ylabel("Probability")
    ax.legend()

    return nat_poisson


def test_coordinate_transforms(dist, test_point: jnp.ndarray, dist_name: str):
    """Test coordinate transformations for a given distribution."""
    print(f"\n{dist_name} coordinate transforms:")
    print(f"Test point shape: {test_point.shape}")

    # Convert to natural coordinates if needed
    if dist._param_type != ParameterType.NATURAL:
        dist = dist.to_natural()

    natural_form = dist
    mean_form = natural_form.to_mean()

    # Convert to float only for printing
    log_density_nat = natural_form.log_density(test_point)
    log_density_mean = mean_form.to_natural().log_density(test_point)

    print(f"Natural form log density: {float(log_density_nat):.4f}")
    print(f"Mean->Natural form log density: {float(log_density_mean):.4f}")


def main():
    """Run all distribution tests and plots."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    try:
        print("\n=== Testing Gaussian Distribution ===")
        gaussian = test_gaussian(ax1)

        print("\n=== Testing Categorical Distribution ===")
        categorical = test_categorical(ax2)

        print("\n=== Testing Poisson Distribution ===")
        poisson = test_poisson(ax3)

        plt.tight_layout()
        plt.show()

        # Test coordinate transforms with proper shapes
        print("\n=== Testing Coordinate Transforms ===")
        test_coordinate_transforms(
            gaussian, jnp.array([1.5]), "Gaussian"
        )  # Changed shape
        test_coordinate_transforms(categorical, jnp.array([2]), "Categorical")
        test_coordinate_transforms(poisson, jnp.array([3]), "Poisson")

    except Exception as e:
        print(f"\nError during testing: {e!s}")
        raise


if __name__ == "__main__":
    main()
