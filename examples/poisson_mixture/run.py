"""COM-Poisson mixture model analysis comparing to Factor Analysis."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Optimizer, OptState, PositiveDefinite
from goal.models import (
    CoMPoissonMixture,
    FactorAnalysis,
    Normal,
    com_poisson_mixture,
    poisson_mixture,
)

from ..shared import example_paths, initialize_jax
from .types import CovarianceStatistics, PoissonAnalysisResults

# Model configuration
n_neurons = 10
n_factors = 3
sample_size = 1000
n_em_steps = 200
sgd_factor = 100
n_sgd_steps = n_em_steps * sgd_factor
n_components = 10
learning_rate = 3e-3

psn_mix = poisson_mixture(n_neurons, n_components)
com_mix = com_poisson_mixture(n_neurons, n_components)
fa_man = FactorAnalysis(n_neurons, n_factors)
nor_man = Normal(n_neurons, PositiveDefinite())


def create_ground_truth_fa() -> Array:
    """Create ground truth factor analysis model."""
    loadings = jnp.array([
        [0.9, -0.1, 0.1, 0.4, -0.8, -0.5, -0.2, 0.3, 0.2, -0.6],
        [0.1, 0.8, -0.1, -0.4, 0.3, 0.2, 0.5, -0.4, 0.2, 0.7],
        [0.1, 0.1, 0.7, -0.2, 0.8, 0.3, -0.3, 0.3, -0.6, 0.4],
    ]).T
    means = jnp.array([5 - (k / 3) for k in range(n_neurons)])
    diags = jnp.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3])
    return fa_man.from_loadings(loadings, means, diags)


def compute_statistics(mean: Array, covariance: Array) -> CovarianceStatistics:
    """Compute statistics from mean and covariance."""
    variances = jnp.diag(covariance)
    fano = variances / mean
    std = jnp.sqrt(variances)
    corr = covariance / (std[:, None] @ std[None, :])
    tril_idx = jnp.tril_indices(covariance.shape[0])
    return CovarianceStatistics(
        means=mean.tolist(),
        fano_factors=fano.tolist(),
        covariances=covariance[tril_idx].tolist(),
        correlation_matrix=corr.tolist(),
    )


def normal_statistics(nor_means: Array) -> CovarianceStatistics:
    """Compute statistics from Normal mean parameters."""
    mean, cov = nor_man.split_mean_covariance(nor_means)
    return compute_statistics(mean, nor_man.cov_man.to_matrix(cov))


def fit_fa(key: Array, sample: Array) -> tuple[Array, list[float]]:
    """Fit Factor Analysis via EM."""
    fa = FactorAnalysis(n_neurons, n_factors)
    params = fa.initialize_from_sample(key, sample)

    def em_step(p: Array, _: Any) -> tuple[Array, Array]:
        ll = fa.average_log_observable_density(p, sample)
        return fa.expectation_maximization(p, sample), ll

    final, lls = jax.lax.scan(em_step, params, None, length=n_em_steps)
    return final, lls.tolist()


def fit_poisson(key: Array, sample: Array) -> tuple[Array, list[float]]:
    """Fit Poisson mixture via EM."""
    params = psn_mix.initialize_from_sample(key, sample)

    def em_step(p: Array, _: Any) -> tuple[Array, Array]:
        ll = psn_mix.average_log_observable_density(p, sample)
        return psn_mix.expectation_maximization(p, sample), ll

    final, lls = jax.lax.scan(em_step, params, None, length=n_em_steps)
    return final, lls.tolist()


def fit_com(key: Array, sample: Array) -> tuple[Array, list[float]]:
    """Fit COM-Poisson mixture via gradient descent."""
    params = com_mix.initialize_from_sample(key, sample)
    optimizer: Optimizer[CoMPoissonMixture] = Optimizer.adamw(
        man=com_mix, learning_rate=learning_rate
    )
    opt_state = optimizer.init(params)

    def loss_fn(p: Array) -> Array:
        return -com_mix.average_log_observable_density(p, sample)

    def step(
        state: tuple[OptState, Array], _: Any
    ) -> tuple[tuple[OptState, Array], Array]:
        opt_state, p = state
        loss, grads = jax.value_and_grad(loss_fn)(p)
        opt_state, p = optimizer.update(opt_state, grads, p)
        return (opt_state, p), -loss

    (_, final), lls = jax.lax.scan(step, (opt_state, params), None, length=n_sgd_steps)
    return final, lls[::sgd_factor].tolist()


def main():
    initialize_jax()
    paths = example_paths(__file__)

    key = jax.random.PRNGKey(42)
    key_sample, key_fa, key_psn, key_com = jax.random.split(key, 4)

    # Ground truth
    gt_params = create_ground_truth_fa()
    _, gt_obs = fa_man.observable_distribution(gt_params)
    gt_stats = normal_statistics(nor_man.to_mean(gt_obs))

    # Sample and discretize
    sample = fa_man.observable_sample(key_sample, gt_params, sample_size)
    discrete = jnp.round(jnp.maximum(0, sample))
    sample_stats = normal_statistics(nor_man.average_sufficient_statistic(discrete))

    # Fit models
    fa_final, fa_lls = fit_fa(key_fa, discrete)
    _, fa_obs = fa_man.observable_distribution(fa_final)
    fa_stats = normal_statistics(nor_man.to_mean(fa_obs))

    psn_final, psn_lls = fit_poisson(key_psn, discrete)
    psn_mean, psn_cov = psn_mix.observable_mean_covariance(psn_final)
    psn_stats = compute_statistics(psn_mean, psn_cov)

    com_final, com_lls = fit_com(key_com, discrete)
    com_mean, com_cov = com_mix.observable_mean_covariance(com_final)
    com_stats = compute_statistics(com_mean, com_cov)

    results = PoissonAnalysisResults(
        grt_stats=gt_stats,
        sample_stats=sample_stats,
        fan_stats=fa_stats,
        psn_stats=psn_stats,
        com_stats=com_stats,
        com_lls=com_lls,
        psn_lls=psn_lls,
        fan_lls=fa_lls,
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
