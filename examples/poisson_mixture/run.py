"""COM-Poisson mixture model analysis comparing to Factor Analysis."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import PositiveDefinite
from goal.models import (
    Normal,
    com_poisson_mixture,
    factor_analysis,
    poisson_mixture,
)

from ..shared import example_paths, jax_cli
from .types import CovarianceStatistics, PoissonAnalysisResults


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


def main():
    jax_cli()
    paths = example_paths(__file__)

    # Configuration
    n_neurons = 10
    n_factors = 3
    sample_size = 1000
    n_em_steps = 200
    sgd_factor = 50
    n_sgd_steps = n_em_steps * sgd_factor
    n_components = 10
    learning_rate = 3e-3

    key = jax.random.PRNGKey(42)
    key_sample, key_fa, key_psn, key_com = jax.random.split(key, 4)

    # Models
    psn_mix = poisson_mixture(n_neurons, n_components)
    com_mix = com_poisson_mixture(n_neurons, n_components)
    fa_man = factor_analysis(n_neurons, n_factors)
    nor_man = Normal(n_neurons, PositiveDefinite())

    def normal_statistics(nor_means: Array) -> CovarianceStatistics:
        mean, cov = nor_man.split_mean_covariance(nor_means)
        return compute_statistics(mean, nor_man.cov_man.to_matrix(cov))

    # Ground truth
    loadings = jnp.array(
        [
            [0.9, -0.1, 0.1, 0.4, -0.8, -0.5, -0.2, 0.3, 0.2, -0.6],
            [0.1, 0.8, -0.1, -0.4, 0.3, 0.2, 0.5, -0.4, 0.2, 0.7],
            [0.1, 0.1, 0.7, -0.2, 0.8, 0.3, -0.3, 0.3, -0.6, 0.4],
        ]
    ).T
    means = jnp.array([5 - (k / 3) for k in range(n_neurons)])
    diags = jnp.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3])
    gt_params = fa_man.initialize_from_loadings(loadings, means, diags)
    _, gt_obs = fa_man.observable_distribution(gt_params)
    gt_stats = normal_statistics(nor_man.to_mean(gt_obs))

    # Sample and discretize
    sample = fa_man.observable_sample(key_sample, gt_params, sample_size)
    discrete = jnp.round(jnp.maximum(0, sample))
    sample_stats = normal_statistics(nor_man.average_sufficient_statistic(discrete))

    # Fit Factor Analysis via EM
    fa = factor_analysis(n_neurons, n_factors)
    fa_init = fa.initialize_from_sample(key_fa, discrete)

    def fa_em_step(p: Array, _: Any) -> tuple[Array, Array]:
        ll = fa.average_log_observable_density(p, discrete)
        return fa.expectation_maximization(p, discrete), ll

    fa_final, fa_lls = jax.lax.scan(fa_em_step, fa_init, None, length=n_em_steps)
    _, fa_obs = fa_man.observable_distribution(fa_final)
    fa_stats = normal_statistics(nor_man.to_mean(fa_obs))

    # Fit Poisson mixture via EM
    psn_init = psn_mix.initialize_from_sample(key_psn, discrete)

    def psn_em_step(p: Array, _: Any) -> tuple[Array, Array]:
        ll = psn_mix.average_log_observable_density(p, discrete)
        return psn_mix.expectation_maximization(p, discrete), ll

    psn_final, psn_lls = jax.lax.scan(psn_em_step, psn_init, None, length=n_em_steps)
    psn_mean, psn_cov = psn_mix.observable_mean_covariance(psn_final)
    psn_stats = compute_statistics(psn_mean, psn_cov)

    # Fit COM-Poisson mixture via gradient descent
    com_init = com_mix.initialize_from_sample(key_com, discrete)
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(com_init)

    def com_loss_fn(p: Array) -> Array:
        return -com_mix.average_log_observable_density(p, discrete)

    def com_step(state: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], Array]:
        opt_state, p = state
        loss, grads = jax.value_and_grad(com_loss_fn)(p)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return (opt_state, p), -loss

    (_, com_final), com_lls_full = jax.lax.scan(
        com_step, (opt_state, com_init), None, length=n_sgd_steps
    )
    com_mean, com_cov = com_mix.observable_mean_covariance(com_final)
    com_stats = compute_statistics(com_mean, com_cov)

    results = PoissonAnalysisResults(
        grt_stats=gt_stats,
        sample_stats=sample_stats,
        fan_stats=fa_stats,
        psn_stats=psn_stats,
        com_stats=com_stats,
        com_lls=com_lls_full[::sgd_factor].tolist(),
        psn_lls=psn_lls.tolist(),
        fan_lls=fa_lls.tolist(),
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
