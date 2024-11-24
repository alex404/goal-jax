"""Test script for univariate auistributions in the exponential family."""

from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import stats
from scipy.special import factorial

from goal.distributions import Categorical, Gaussian, Poisson
from goal.exponential_family import Natural
from goal.linear import PositiveDefinite
from goal.manifold import Point


def compute_gaussian_results(
    key: Array,
    manifold: Gaussian,
    mu: Array,
    sigma: Array,
    sample_size: int,
) -> Tuple[Array, Array, Array]:
    """Run Gaussian computations with JAX.

    Returns:
        samples, true_densities, estimated_densities
    """
    # Create ground truth distribution
    eval_points = jnp.array(
        [x for x in jnp.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)]
    )
    covariance = PositiveDefinite(sigma**2, 1)
    p_mean = manifold.from_mean_and_covariance(mu, covariance)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    sample = manifold.sample(key, p_natural, sample_size)
    p_mean_est = manifold.average_sufficient_statistic(sample)
    p_natural_est = manifold.to_natural(p_mean_est)

    # Compute densities using vmap
    def compute_density(p: Point[Natural, Gaussian], x: Array) -> Array:
        return manifold.density(p, x)

    true_densities = jax.vmap(compute_density, in_axes=(None, 0))(
        p_natural, eval_points
    )
    est_densities = jax.vmap(compute_density, in_axes=(None, 0))(
        p_natural_est, eval_points
    )

    return sample, true_densities, est_densities


compute_gaussian_results = jax.jit(
    compute_gaussian_results, static_argnames=["manifold", "sample_size"]
)


def compute_categorical_results(
    manifold: Categorical,
    probs: Array,
    sample_size: int,
    key: Array,
) -> tuple[Array, Array, Array]:
    """Run Categorical computations with JAX.

    Returns:
        samples, true_probs, estimated_probs
    """
    # Create ground truth distribution
    p_mean = manifold.from_probs(probs)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    samples = manifold.sample(key, p_natural, sample_size)
    p_mean_est = manifold.average_sufficient_statistic(samples)
    p_natural_est = manifold.to_natural(p_mean_est)

    # Compute densities using vmap
    categories = jnp.arange(probs.shape[0])

    def compute_prob(p: Point[Natural, Categorical], k: Array) -> Array:
        return manifold.density(p, k)

    true_probs = jax.vmap(compute_prob, in_axes=(None, 0))(p_natural, categories)
    est_probs = jax.vmap(compute_prob, in_axes=(None, 0))(p_natural_est, categories)

    return samples, true_probs, est_probs


compute_categorical_results = jax.jit(
    compute_categorical_results, static_argnames=["manifold", "sample_size"]
)


def compute_poisson_results(
    manifold: Poisson,
    rate: float,
    sample_size: int,
    key: Array,
) -> tuple[Array, Array, Array]:
    """Run Poisson computations with JAX.

    Returns:
        samples, true_pmf, estimated_pmf
    """
    # Create ground truth distribution

    max_k = 20
    eval_points = jnp.arange(0, max_k + 1, dtype=int)
    p_mean = manifold.mean_point(rate)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    samples = manifold.sample(key, p_natural, sample_size)
    p_mean_est = manifold.average_sufficient_statistic(samples)
    p_natural_est = manifold.to_natural(p_mean_est)

    def compute_pmf(p: Point[Natural, Poisson], k: Array) -> Array:
        return manifold.density(p, k)

    true_pmf = jax.vmap(compute_pmf, in_axes=(None, 0))(p_natural, eval_points)
    est_pmf = jax.vmap(compute_pmf, in_axes=(None, 0))(p_natural_est, eval_points)

    return samples, true_pmf, est_pmf


compute_poisson_results = jax.jit(
    compute_poisson_results, static_argnames=["manifold", "sample_size"]
)


def plot_gaussian_results(
    ax: Axes,
    mu: float,
    sigma: float,
    samples: Array,
    true_densities: Array,
    est_densities: Array,
) -> None:
    """Plot Gaussian results using numpy/matplotlib."""
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)

    # Convert everything to numpy for plotting
    np_samples: NDArray[np.float64] = np.array(samples.squeeze())
    np_true_densities: NDArray[np.float64] = np.array(true_densities)
    np_est_densities: NDArray[np.float64] = np.array(est_densities)

    ax.hist(np_samples, bins=50, density=True, alpha=0.3, label="Samples")
    ax.plot(x, np_true_densities, "b-", label="Implementation")
    ax.plot(x, np_est_densities, "g--", label="MLE Estimate")
    ax.plot(
        x,
        stats.norm.pdf(x, loc=mu, scale=sigma),
        "r--",
        label="Theory",
    )

    ax.set_title("Gaussian Density")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()


def plot_categorical_results(
    ax: Axes,
    probs: Array,
    samples: Array,
    true_probs: Array,
    est_probs: Array,
) -> None:
    """Plot Categorical results using numpy/matplotlib."""
    categories = np.arange(len(probs))
    width = 0.25

    # Convert everything to numpy for plotting
    np_samples: NDArray[np.float64] = np.array(samples)
    np_true_probs: NDArray[np.float64] = np.array(true_probs)
    np_est_probs: NDArray[np.float64] = np.array(est_probs)
    np_probs: NDArray[np.float64] = np.array(probs)

    # Sample frequencies
    sample_freqs = np.bincount(np_samples, minlength=len(np_probs)) / len(np_samples)
    ax.bar(categories, sample_freqs, width=1.0, alpha=0.2, label="Samples")

    # Plot bars
    ax.bar(categories - width, np_true_probs, width, alpha=0.6, label="Implementation")
    ax.bar(categories, np_est_probs, width, alpha=0.6, label="MLE Estimate")
    ax.bar(categories + width, np_probs, width, alpha=0.6, label="True Probabilities")

    ax.set_title("Categorical PMF")
    ax.set_xlabel("Category")
    ax.set_ylabel("Probability")
    ax.legend()


def plot_poisson_results(
    ax: Axes,
    rate: float,
    samples: Array,
    true_pmf: Array,
    est_pmf: Array,
) -> None:
    """Plot Poisson results using numpy/matplotlib."""
    max_k = 20
    k = np.arange(0, max_k + 1)

    # Convert everything to numpy for plotting
    np_samples: NDArray[np.float64] = np.array(samples)
    np_true_pmf: NDArray[np.float64] = np.array(true_pmf)
    np_est_pmf: NDArray[np.float64] = np.array(est_pmf)

    ax.hist(np_samples, bins=range(max_k + 2), density=True, alpha=0.3, label="Samples")
    ax.plot(k, np_true_pmf, "b-", label="Implementation", alpha=0.8)
    ax.plot(k, np_est_pmf, "g--", label="MLE Estimate", alpha=0.8)
    ax.plot(
        k,
        rate**k * np.exp(-rate) / factorial(k),
        "r:",
        label="Theory",
        alpha=0.8,
    )

    ax.set_title(f"Poisson PMF (λ={rate})")
    ax.set_xlabel("k")
    ax.set_ylabel("Probability")
    ax.legend()


def main():
    """Run all distribution tests and plots."""
    # Create figure
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Parameters
    sample_size = 10000

    # # Gaussian test
    print("\nTesting Gaussian Distribution:")
    mu, sigma = 2.0, 1.5
    gaussian = Gaussian(data_dim=1)

    sample, true_dens, est_dens = compute_gaussian_results(
        keys[0], gaussian, jnp.atleast_1d(mu), jnp.atleast_1d(sigma), sample_size
    )
    plot_gaussian_results(ax1, mu, sigma, sample, true_dens, est_dens)

    # Categorical test
    print("\nTesting Categorical Distribution:")
    probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    categorical = Categorical(n_categories=len(probs))
    sample, true_probs, est_probs = compute_categorical_results(
        categorical, probs, sample_size, keys[1]
    )
    plot_categorical_results(ax2, probs, sample, true_probs, est_probs)

    # Poisson test
    print("\nTesting Poisson Distribution:")
    rate = 5.0
    poisson = Poisson()
    sample, true_pmf, est_pmf = compute_poisson_results(
        poisson, rate, sample_size, keys[2]
    )
    plot_poisson_results(ax3, rate, sample, true_pmf, est_pmf)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
