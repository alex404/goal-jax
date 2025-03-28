"""Test script for univariate auistributions in the exponential family."""

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Mean, Natural, Point, PositiveDefinite
from goal.models import (
    Categorical,
    Euclidean,
    FullCovariance,
    FullNormal,
    Normal,
    Poisson,
)

from ..shared import example_paths, initialize_jax
from .types import (
    CategoricalResults,
    NormalResults,
    PoissonResults,
    UnivariateResults,
)


def compute_gaussian_results(
    key: Array,
    model: FullNormal,
    mu: Point[Mean, Euclidean],
    sigma: Point[Mean, FullCovariance],
    xs: Array,
    sample_size: int,
) -> tuple[Array, Array, Array]:
    """Run Normal computations with JAX.

    Returns:
        sample, true_densities, estimated_densities
    """
    # Create ground truth distribution
    p_mean = model.join_mean_covariance(mu, sigma)
    p_natural = model.to_natural(p_mean)

    # Sample and estimate
    sample = model.sample(key, p_natural, sample_size)
    p_mean_est = model.average_sufficient_statistic(sample)
    p_natural_est = model.to_natural(p_mean_est)

    true_dens = jax.vmap(model.density, in_axes=(None, 0))(p_natural, xs)
    est_dens = jax.vmap(model.density, in_axes=(None, 0))(p_natural_est, xs)

    return sample, true_dens, est_dens


def compute_categorical_results(
    key: Array,
    model: Categorical,
    probs: Array,
    sample_size: int,
) -> tuple[Array, Array, Array]:
    """Run Categorical computations with JAX.

    Returns:
        sample, true_probs, estimated_probs
    """
    # Create ground truth distribution
    p_mean = model.from_probs(probs)
    p_natural = model.to_natural(p_mean)

    # Sample and estimate
    sample = model.sample(key, p_natural, sample_size)
    p_mean_est = model.average_sufficient_statistic(sample)
    p_natural_est = model.to_natural(p_mean_est)

    # Compute densities using vmap
    categories = jnp.arange(probs.shape[0])

    def compute_prob(p: Point[Natural, Categorical], k: Array) -> Array:
        return model.density(p, k)

    true_probs = jax.vmap(compute_prob, in_axes=(None, 0))(p_natural, categories)
    est_probs = jax.vmap(compute_prob, in_axes=(None, 0))(p_natural_est, categories)

    return sample, true_probs, est_probs


def compute_poisson_results(
    key: Array,
    model: Poisson,
    rate: float,
    ks: Array,
    sample_size: int,
) -> tuple[Array, Array, Array]:
    """Run Poisson computations with JAX.

    Returns:
        sample, true_pmf, estimated_pmf
    """
    # Create ground truth distribution

    p_mean = model.mean_point(jnp.atleast_1d(rate))
    p_natural = model.to_natural(p_mean)

    # Sample and estimate
    sample = model.sample(key, p_natural, sample_size)
    p_mean_est = model.average_sufficient_statistic(sample)
    p_natural_est = model.to_natural(p_mean_est)

    def compute_pmf(p: Point[Natural, Poisson], k: Array) -> Array:
        return model.density(p, k)

    true_pmf = jax.vmap(compute_pmf, in_axes=(None, 0))(p_natural, ks)
    estimated_pmf = jax.vmap(compute_pmf, in_axes=(None, 0))(p_natural_est, ks)

    return sample, true_pmf, estimated_pmf


### Main ###


def main():
    initialize_jax()
    paths = example_paths(__file__)

    """Run all distribution tests and plots."""
    # Create figure
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)

    # Parameters
    sample_size = 20

    # # Normal test
    print("\nTesting Normal Distribution...")

    normal = Normal(1, PositiveDefinite)
    mu0 = 2.0
    sigma = 1.5
    mu = normal.loc_man.mean_point(jnp.atleast_1d(mu0))
    cov = normal.cov_man.mean_point(jnp.atleast_1d(sigma**2))
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
        sample=sample.ravel().tolist(),
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
        sample=sample.ravel().tolist(),
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
        sample=sample.ravel().tolist(),
        true_pmf=true_pmf.tolist(),
        estimated_pmf=est_pmf.tolist(),
        eval_points=ks.tolist(),
    )

    results = UnivariateResults(
        gaussian=gaussian_results,
        categorical=categorical_results,
        poisson=poisson_results,
    )

    paths.save_analysis(results)


if __name__ == "__main__":
    main()
