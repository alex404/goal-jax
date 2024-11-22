"""Test script for univariate distributions in the exponential family."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.axes import Axes

from goal.distributions import Categorical, MultivariateGaussian, Poisson
from goal.linear import PositiveDefinite


@jax.jit
def compute_gaussian_results(
    manifold: MultivariateGaussian,
    mu: Array,
    sigma: Array,
    eval_points: Array,
    n_samples: int,
) -> tuple[Array, Array, Array]:
    """Run Gaussian computations with JAX.

    Returns:
        samples, true_densities, estimated_densities
    """
    # Create ground truth distribution
    covariance = PositiveDefinite.from_params(sigma**2, side_length=1)
    p_mean = manifold.from_mean_and_covariance(mu, covariance)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    samples = manifold.sample(p_natural, n_samples)
    p_mean_est = manifold.average_sufficient_statistic(samples)
    p_natural_est = manifold.to_natural(p_mean_est)

    # Compute densities using vmap
    compute_density = lambda p, x: manifold.log_density(p, x)
    true_densities = jax.vmap(lambda x: compute_density(p_natural, x))(eval_points)
    est_densities = jax.vmap(lambda x: compute_density(p_natural_est, x))(eval_points)

    return samples, jnp.exp(true_densities), jnp.exp(est_densities)


@jax.jit
def compute_categorical_results(
    manifold: Categorical,
    probs: Array,
    n_samples: int,
) -> tuple[Array, Array, Array]:
    """Run Categorical computations with JAX.

    Returns:
        samples, true_probs, estimated_probs
    """
    # Create ground truth distribution
    p_mean = manifold.from_probs(probs)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    samples = manifold.sample(p_natural, n_samples)
    p_mean_est = manifold.average_sufficient_statistic(samples)
    p_natural_est = manifold.to_natural(p_mean_est)

    # Compute densities using vmap
    categories = jnp.arange(probs.shape[0])
    compute_prob = lambda p, k: jnp.exp(manifold.log_density(p, k))
    true_probs = jax.vmap(lambda k: compute_prob(p_natural, k))(categories)
    est_probs = jax.vmap(lambda k: compute_prob(p_natural_est, k))(categories)

    return samples, true_probs, est_probs


@jax.jit
def compute_poisson_results(
    manifold: Poisson,
    rate: Array,
    eval_points: Array,
    n_samples: int,
) -> tuple[Array, Array, Array]:
    """Run Poisson computations with JAX.

    Returns:
        samples, true_pmf, estimated_pmf
    """
    # Create ground truth distribution
    p_mean = manifold.mean_point(rate)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    samples = manifold.sample(p_natural, n_samples)
    p_mean_est = manifold.average_sufficient_statistic(samples)
    p_natural_est = manifold.to_natural(p_mean_est)

    # Compute PMFs using vmap
    compute_pmf = lambda p, k: manifold.log_density(p, k)
    true_pmf = jax.vmap(lambda k: compute_pmf(p_natural, k))(eval_points)
    est_pmf = jax.vmap(lambda k: compute_pmf(p_natural_est, k))(eval_points)

    return samples, jnp.exp(true_pmf), jnp.exp(est_pmf)


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
    samples = np.array(samples.squeeze())
    true_densities = np.array(true_densities)
    est_densities = np.array(est_densities)

    ax.hist(samples, bins=50, density=True, alpha=0.3, label="Samples")
    ax.plot(x, true_densities, "b-", label="Implementation")
    ax.plot(x, est_densities, "g--", label="MLE Estimate")
    ax.plot(
        x,
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
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
    samples = np.array(samples)
    true_probs = np.array(true_probs)
    est_probs = np.array(est_probs)
    probs = np.array(probs)

    # Sample frequencies
    sample_freqs = np.bincount(samples, minlength=len(probs)) / len(samples)
    ax.bar(categories, sample_freqs, width=1.0, alpha=0.2, label="Samples")

    # Plot bars
    ax.bar(categories - width, true_probs, width, alpha=0.6, label="Implementation")
    ax.bar(categories, est_probs, width, alpha=0.6, label="MLE Estimate")
    ax.bar(categories + width, probs, width, alpha=0.6, label="True Probabilities")

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
    max_k = int(rate * 2 + 4 * np.sqrt(rate))
    k = np.arange(0, max_k + 1)

    # Convert everything to numpy for plotting
    samples = np.array(samples)
    true_pmf = np.array(true_pmf)
    est_pmf = np.array(est_pmf)

    ax.hist(samples, bins=range(max_k + 2), density=True, alpha=0.3, label="Samples")
    ax.plot(k, true_pmf, "b-", label="Implementation", alpha=0.8)
    ax.plot(k, est_pmf, "g--", label="MLE Estimate", alpha=0.8)
    ax.plot(
        k,
        rate**k * np.exp(-rate) / np.math.factorial(k),
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
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Parameters
    n_samples = 1000

    # Gaussian test
    print("\nTesting Gaussian Distribution:")
    mu, sigma = 2.0, 1.5
    x_eval = jnp.array([[x] for x in jnp.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)])
    gaussian = MultivariateGaussian(data_dim=1)
    samples, true_dens, est_dens = compute_gaussian_results(
        gaussian, jnp.array([mu]), jnp.array([sigma**2]), x_eval, n_samples
    )
    plot_gaussian_results(ax1, mu, sigma, samples, true_dens, est_dens)

    # Categorical test
    print("\nTesting Categorical Distribution:")
    probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    categorical = Categorical(n_categories=len(probs))
    samples, true_probs, est_probs = compute_categorical_results(
        categorical, probs, n_samples
    )
    plot_categorical_results(ax2, probs, samples, true_probs, est_probs)

    # Poisson test
    print("\nTesting Poisson Distribution:")
    rate = 5.0
    k_eval = jnp.arange(0, int(rate * 2 + 4 * jnp.sqrt(rate)) + 1, dtype=float)
    poisson = Poisson()
    samples, true_pmf, est_pmf = compute_poisson_results(
        poisson, jnp.array(rate), k_eval, n_samples
    )
    plot_poisson_results(ax3, rate, samples, true_pmf, est_pmf)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
