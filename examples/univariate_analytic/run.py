"""Univariate analytic exponential family distributions."""

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import PositiveDefinite
from goal.models import Categorical, Normal, Poisson

from ..shared import example_paths, jax_cli
from .types import CategoricalResults, NormalResults, PoissonResults, UnivariateResults


def fit_normal(
    key: Array, model: Normal, mu: Array, sigma: Array, xs: Array, n_samples: int
) -> tuple[Array, Array, Array]:
    """Fit Normal and return sample, true densities, estimated densities."""
    means = model.join_mean_covariance(mu, sigma)
    natural_params = model.to_natural(means)

    sample = model.sample(key, natural_params, n_samples)
    estimated_mean = model.average_sufficient_statistic(sample)
    estimated_natural = model.to_natural(estimated_mean)

    true_dens = jax.vmap(model.density, in_axes=(None, 0))(natural_params, xs)
    est_dens = jax.vmap(model.density, in_axes=(None, 0))(estimated_natural, xs)

    return sample, true_dens, est_dens


def fit_categorical(
    key: Array, model: Categorical, probs: Array, n_samples: int
) -> tuple[Array, Array, Array]:
    """Fit Categorical and return sample, true probs, estimated probs."""
    means = model.from_probs(probs)
    natural_params = model.to_natural(means)

    sample = model.sample(key, natural_params, n_samples)
    estimated_mean = model.average_sufficient_statistic(sample)
    estimated_natural = model.to_natural(estimated_mean)

    categories = jnp.arange(probs.shape[0])
    true_probs = jax.vmap(model.density, in_axes=(None, 0))(natural_params, categories)
    est_probs = jax.vmap(model.density, in_axes=(None, 0))(
        estimated_natural, categories
    )

    return sample, true_probs, est_probs


def fit_poisson(
    key: Array, model: Poisson, rate: float, ks: Array, n_samples: int
) -> tuple[Array, Array, Array]:
    """Fit Poisson and return sample, true pmf, estimated pmf."""
    means = jnp.atleast_1d(rate)
    natural_params = model.to_natural(means)

    sample = model.sample(key, natural_params, n_samples)
    estimated_mean = model.average_sufficient_statistic(sample)
    estimated_natural = model.to_natural(estimated_mean)

    true_pmf = jax.vmap(model.density, in_axes=(None, 0))(natural_params, ks)
    est_pmf = jax.vmap(model.density, in_axes=(None, 0))(estimated_natural, ks)

    return sample, true_pmf, est_pmf


def main():
    jax_cli()
    paths = example_paths(__file__)

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)
    n_samples = 40

    # Normal
    normal = Normal(1, PositiveDefinite())
    mu, sigma = 2.0, 1.5
    xs = jnp.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    sample, true_dens, est_dens = fit_normal(
        keys[0], normal, jnp.atleast_1d(mu), jnp.atleast_1d(sigma**2), xs, n_samples
    )
    normal_results = NormalResults(
        mu=mu,
        sigma=sigma,
        sample=sample.ravel().tolist(),
        true_densities=true_dens.tolist(),
        estimated_densities=est_dens.tolist(),
        eval_points=xs.tolist(),
    )

    # Categorical
    probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    categorical = Categorical(n_categories=len(probs))
    sample, true_probs, est_probs = fit_categorical(
        keys[1], categorical, probs, n_samples
    )
    categorical_results = CategoricalResults(
        probs=probs.tolist(),
        sample=sample.ravel().tolist(),
        true_probs=true_probs.tolist(),
        estimated_probs=est_probs.tolist(),
    )

    # Poisson
    poisson = Poisson()
    rate = 5.0
    ks = jnp.arange(0, 21, dtype=int)
    sample, true_pmf, est_pmf = fit_poisson(keys[2], poisson, rate, ks, n_samples)
    poisson_results = PoissonResults(
        rate=rate,
        sample=sample.ravel().tolist(),
        true_pmf=true_pmf.tolist(),
        estimated_pmf=est_pmf.tolist(),
        eval_points=ks.tolist(),
    )

    results = UnivariateResults(
        gaussian=normal_results,
        categorical=categorical_results,
        poisson=poisson_results,
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
