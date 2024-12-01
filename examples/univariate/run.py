"""Test script for univariate auistributions in the exponential family."""

import json
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from goal.distributions import (
    Categorical,
    FullCovariance,
    FullNormal,
    Poisson,
    full_normal_manifold,
)
from goal.exponential_family import Mean, Natural
from goal.manifold import Euclidean, Point

from .common import (
    CategoricalResults,
    NormalResults,
    PoissonResults,
    UnivariateResults,
    analysis_path,
)


def compute_gaussian_results(
    key: Array,
    manifold: FullNormal,
    mu: Point[Mean, Euclidean],
    sigma: Point[Mean, FullCovariance],
    xs: Array,
    sample_size: int,
) -> Tuple[Array, Array, Array]:
    """Run Normal computations with JAX.

    Returns:
        sample, true_densities, estimated_densities
    """
    # Create ground truth distribution
    p_mean = manifold.from_mean_and_covariance(mu, sigma)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    sample = manifold.sample(key, p_natural, sample_size)
    p_mean_est = manifold.average_sufficient_statistic(sample)
    p_natural_est = manifold.to_natural(p_mean_est)

    # Compute densities using vmap
    def compute_density(p: Point[Natural, FullNormal], x: Array) -> Array:
        return manifold.density(p, x)

    true_dens = jax.vmap(compute_density, in_axes=(None, 0))(p_natural, xs)
    est_dens = jax.vmap(compute_density, in_axes=(None, 0))(p_natural_est, xs)

    return sample, true_dens, est_dens


compute_gaussian_results = jax.jit(
    compute_gaussian_results, static_argnames=["manifold", "sample_size"]
)


def compute_categorical_results(
    key: Array,
    manifold: Categorical,
    probs: Array,
    sample_size: int,
) -> Tuple[Array, Array, Array]:
    """Run Categorical computations with JAX.

    Returns:
        sample, true_probs, estimated_probs
    """
    # Create ground truth distribution
    p_mean = manifold.from_probs(probs)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    sample = manifold.sample(key, p_natural, sample_size)
    p_mean_est = manifold.average_sufficient_statistic(sample)
    p_natural_est = manifold.to_natural(p_mean_est)

    # Compute densities using vmap
    categories = jnp.arange(probs.shape[0])

    def compute_prob(p: Point[Natural, Categorical], k: Array) -> Array:
        return manifold.density(p, k)

    true_probs = jax.vmap(compute_prob, in_axes=(None, 0))(p_natural, categories)
    est_probs = jax.vmap(compute_prob, in_axes=(None, 0))(p_natural_est, categories)

    return sample, true_probs, est_probs


compute_categorical_results = jax.jit(
    compute_categorical_results, static_argnames=["manifold", "sample_size"]
)


def compute_poisson_results(
    key: Array,
    manifold: Poisson,
    rate: float,
    ks: Array,
    sample_size: int,
) -> Tuple[Array, Array, Array]:
    """Run Poisson computations with JAX.

    Returns:
        sample, true_pmf, estimated_pmf
    """
    # Create ground truth distribution

    p_mean = manifold.mean_point(rate)
    p_natural = manifold.to_natural(p_mean)

    # Sample and estimate
    sample = manifold.sample(key, p_natural, sample_size)
    p_mean_est = manifold.average_sufficient_statistic(sample)
    p_natural_est = manifold.to_natural(p_mean_est)

    def compute_pmf(p: Point[Natural, Poisson], k: Array) -> Array:
        return manifold.density(p, k)

    true_pmf = jax.vmap(compute_pmf, in_axes=(None, 0))(p_natural, ks)
    estimated_pmf = jax.vmap(compute_pmf, in_axes=(None, 0))(p_natural_est, ks)

    return sample, true_pmf, estimated_pmf


compute_poisson_results = jax.jit(
    compute_poisson_results, static_argnames=["manifold", "sample_size"]
)


def main():
    """Run all distribution tests and plots."""
    # Create figure
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)

    # Parameters
    sample_size = 100

    # # Normal test
    print("\nTesting Normal Distribution...")

    normal = full_normal_manifold()
    mu0 = 2.0
    sigma = 1.5
    mu: Point[Mean, Euclidean] = Point(jnp.array([mu0]))
    cov: Point[Mean, FullCovariance] = Point(jnp.array([sigma**2]))
    xs = jnp.linspace(mu0 - 4 * sigma, mu0 + 4 * sigma, 200)

    sample, true_dens, est_dens = compute_gaussian_results(
        keys[0],
        normal,
        mu,
        cov,
        xs,
        sample_size,
    )

    gaussian_results = NormalResults(
        mu=mu0,
        sigma=sigma,
        sample=sample.tolist(),
        true_densities=true_dens.tolist(),
        estimated_densities=est_dens.tolist(),
        eval_points=xs.tolist(),
    )
    # Categorical test
    print("\nTesting Categorical Distribution...")

    probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    categorical = Categorical(n_categories=len(probs))

    sample, true_probs, est_probs = compute_categorical_results(
        keys[1], categorical, probs, sample_size
    )

    categorical_results = CategoricalResults(
        probs=probs.tolist(),
        sample=sample.tolist(),
        true_probs=true_probs.tolist(),
        estimated_probs=est_probs.tolist(),
    )

    # Poisson test
    print("\nTesting Poisson Distribution...")

    poisson = Poisson()
    rate = 5.0
    max_k = 20
    ks = jnp.arange(0, max_k + 1, dtype=int)

    sample, true_pmf, est_pmf = compute_poisson_results(
        keys[2], poisson, rate, ks, sample_size
    )

    poisson_results = PoissonResults(
        rate=rate,
        sample=sample.tolist(),
        true_pmf=true_pmf.tolist(),
        estimated_pmf=est_pmf.tolist(),
        eval_points=ks.tolist(),
    )

    results = UnivariateResults(
        gaussian=gaussian_results,
        categorical=categorical_results,
        poisson=poisson_results,
    )

    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
