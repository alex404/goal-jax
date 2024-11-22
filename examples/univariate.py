"""Test script for univariate distributions in the exponential family."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from jax import Array
from matplotlib.axes import Axes
from scipy import stats

from goal.distributions import Categorical, MultivariateGaussian, Poisson
from goal.exponential_family import CFEF, Mean, Natural
from goal.linear import PositiveDefinite
from goal.manifold import Point


def test_gaussian(
    ax: Axes, mu: float = 2.0, sigma: float = 1.5
) -> tuple[MultivariateGaussian, Point[Natural, MultivariateGaussian]]:
    """Test univariate Gaussian distribution and plot results."""
    # Create distribution and point
    mu_arr = jnp.array([mu])
    covariance = PositiveDefinite.from_params(jnp.array([sigma**2]), side_length=1)

    manifold = MultivariateGaussian(data_dim=1)
    p_mean = manifold.from_mean_and_covariance(mu_arr, covariance)
    p_natural = manifold.to_natural(p_mean)

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


def test_categorical(
    ax: Axes, probs: Array = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
) -> tuple[Categorical, Point[Natural, Categorical]]:
    """Test Categorical distribution and plot results."""

    manifold = Categorical(n_categories=len(probs))
    p_mean = manifold.from_probs(probs)
    p_natural = manifold.to_natural(p_mean)

    # Plot PMF
    categories = np.arange(len(probs))
    # Make sure to pass category indices as integers
    cat_probs = jnp.exp(
        jnp.array([manifold.log_density(p_natural, i) for i in categories])
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


def test_poisson(
    ax: Axes, rate: float = 5.0
) -> tuple[Poisson, Point[Natural, Poisson]]:
    """Test Poisson distribution and plot results."""
    manifold = Poisson()
    p_mean = Point[Mean, Poisson](jnp.array(rate))
    p_natural = manifold.to_natural(p_mean)

    # Generate points for plotting
    max_k = int(stats.poisson.ppf(0.999, rate))
    k = np.arange(0, max_k + 1)

    # Compute densities
    our_pmf: Array = jnp.exp(
        jnp.array([manifold.log_density(p_natural, float(k[i])) for i in range(len(k))])
    )
    scipy_pmf: npt.NDArray[np.float64] = stats.poisson.pmf(k, rate)

    # Convert to numpy arrays for plotting
    k_np: npt.NDArray[np.int_] = np.asarray(k, dtype=np.int_)
    our_pmf_np: npt.NDArray[np.float64] = np.asarray(our_pmf, dtype=np.float64)
    scipy_pmf_np: npt.NDArray[np.float64] = np.asarray(scipy_pmf, dtype=np.float64)

    # Plot results
    width = 0.35
    ax.bar(k_np - width / 2, our_pmf_np, width, alpha=0.6, label="Implementation")
    ax.bar(k_np + width / 2, scipy_pmf_np, width, alpha=0.6, label="Scipy")
    ax.set_title(f"Poisson PMF (λ={rate})")
    ax.set_xlabel("k")
    ax.set_ylabel("Probability")
    ax.legend()

    return manifold, p_natural


def test_coordinate_transforms(
    ef: CFEF,
    p_natural: Point[Natural, CFEF],
    test_point: jnp.ndarray,
):
    """Test coordinate transformations for a given distribution."""

    # Transform to mean coordinates and back
    p_mean = ef.to_mean(p_natural)
    p_natural2 = ef.to_natural(p_mean)

    print("\nParameter conversion:")

    print(f"Natural parameters: {p_natural.params}")
    print(f"Mean parameters: {p_mean.params}")
    print(f"Natural parameters (recovered): {p_natural2.params}")

    # Compare log densities
    log_density_nat = ef.log_density(p_natural, test_point)
    log_density_nat2 = ef.log_density(p_natural2, test_point)

    # Convert to float for printing to avoid any nan display issues
    print("\nLog densities:")

    print(f"Natural parameter log-density: {float(log_density_nat):.4f}")
    print(f"Recovered natural parameter log-density: {float(log_density_nat2):.4f}")


def main():
    """Run all distribution tests and plots."""
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    print("\nTesting Gaussian Distribution:")

    gaussian_manifold, gaussian_point = test_gaussian(ax1)
    test_coordinate_transforms(gaussian_manifold, gaussian_point, jnp.array([3.0]))

    print("\nTesting Categorical Distribution:")

    categorical_manifold, categorical_point = test_categorical(ax2)
    test_coordinate_transforms(
        categorical_manifold,
        categorical_point,
        jnp.array(2),
    )
    print("\nTesting Poisson Distribution:")

    poisson_manifold, poisson_point = test_poisson(ax3)
    test_coordinate_transforms(
        poisson_manifold,
        poisson_point,
        jnp.array(1),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
