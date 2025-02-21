"""Analysis script for COM-Poisson mixture models.

This script compares three statistical models:
1. A factor analysis model as ground truth
2. A fitted COM-Poisson mixture model
3. Empirical statistics from discretized samples

The script:
1. Creates a ground truth factor analysis model
2. Generates and discretizes samples
3. Fits a COM-Poisson mixture model
4. Computes comparative statistics
"""

import os
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Mean,
    Natural,
    Optimizer,
    OptState,
    Point,
    PositiveDefinite,
)
from goal.models import (
    CoMPoissonMixture,
    FactorAnalysis,
    FullNormal,
    Normal,
    PoissonMixture,
    com_poisson_mixture,
    poisson_mixture,
)

from ..shared import example_paths, initialize_jax
from .types import CovarianceStatistics, PoissonAnalysisResults

# Constants
N_NEURONS = 10  # Observable dimension
N_FACTORS = 3  # Latent dimension
SAMPLE_SIZE = 1000  # Number of samples
N_EM_STEPS = 100  # EM steps
SGD_FACTOR = 100
N_SGD_STEPS = N_EM_STEPS * SGD_FACTOR
N_COMPONENTS = 5  # Number of mixture components
LEARNING_RATE = 3e-3

PSN_MIX_MAN = poisson_mixture(N_NEURONS, N_COMPONENTS)
COM_MIX_MAN = com_poisson_mixture(N_NEURONS, N_COMPONENTS)
FAN_MAN = FactorAnalysis(N_NEURONS, N_FACTORS)
NOR_MAN = Normal(N_NEURONS, PositiveDefinite)


def create_ground_truth_loadings() -> Array:
    """Create ground truth loading matrix from original implementation."""
    return jnp.array(
        [
            [0.9, -0.1, 0.1, 0.4, -0.8, -0.5, -0.2, 0.3, 0.2, -0.6],
            [0.1, 0.8, -0.1, -0.4, 0.3, 0.2, 0.5, -0.4, 0.2, 0.7],
            [0.1, 0.1, 0.7, -0.2, 0.8, 0.3, -0.3, 0.3, -0.6, 0.4],
        ]
    ).T


def create_ground_truth_diagonals() -> tuple[Array, Array]:
    """Create ground truth diagonal parameters from original implementation."""
    means = jnp.array([5 - (k / 3) for k in range(N_NEURONS)])
    diags = jnp.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3])
    return means, diags


def create_ground_truth_fa() -> Point[Natural, FactorAnalysis]:
    """Create ground truth factor analysis model."""
    loadings = create_ground_truth_loadings()
    means, diags = create_ground_truth_diagonals()
    return FAN_MAN.from_loadings(loadings, means, diags)


def compute_sample_statistics(sample: Array) -> CovarianceStatistics:
    nor_means = NOR_MAN.average_sufficient_statistic(sample)
    return compute_normal_statistics(nor_means)


def compute_normal_statistics(
    nor_means: Point[Mean, FullNormal],
) -> CovarianceStatistics:
    mean, cov = NOR_MAN.split_mean_covariance(nor_means)
    return compute_dense_statistics(mean.array, NOR_MAN.cov_man.to_dense(cov))


def compute_dense_statistics(mean: Array, covariance: Array) -> CovarianceStatistics:
    variances = jnp.diag(covariance)
    fano_factors = variances / mean

    # Compute correlation matrix
    std_devs = jnp.sqrt(variances)
    corr_matrix = covariance / (std_devs[:, None] @ std_devs[None, :])

    # Extract lower triangular elements of covariance
    tril_idx = jnp.tril_indices(covariance.shape[0])
    covariances = covariance[tril_idx]

    return CovarianceStatistics(
        means=mean.tolist(),
        fano_factors=fano_factors.tolist(),
        covariances=covariances.tolist(),
        correlation_matrix=corr_matrix.tolist(),
    )


def fit_factor_analysis(
    key_fit: Array,
    sample: Array,
) -> tuple[Point[Natural, FactorAnalysis], list[float]]:
    # Fit Factor Analysis to discrete data
    fa_discrete = FactorAnalysis(N_NEURONS, N_FACTORS)
    fa_params_discrete = fa_discrete.initialize_from_sample(key_fit, sample)

    def em_step(
        params: Point[Natural, FactorAnalysis], _: Any
    ) -> tuple[Point[Natural, FactorAnalysis], Array]:
        ll = fa_discrete.average_log_observable_density(params, sample)
        next_params = fa_discrete.expectation_maximization(params, sample)
        return next_params, ll

    fa_params_final, fa_lls = jax.lax.scan(
        em_step, fa_params_discrete, None, length=N_EM_STEPS
    )

    return fa_params_final, fa_lls.tolist()


def fit_poisson_mixture(
    key: Array,
    sample: Array,
) -> tuple[Point[Natural, PoissonMixture], list[float]]:
    init_params = PSN_MIX_MAN.initialize_from_sample(key, sample)

    def em_step(
        carry: Point[Natural, PoissonMixture], _: Any
    ) -> tuple[Point[Natural, PoissonMixture], Array]:
        params = carry
        ll = PSN_MIX_MAN.average_log_observable_density(params, sample)
        next_params = PSN_MIX_MAN.expectation_maximization(params, sample)
        return next_params, ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=N_EM_STEPS)

    return final_params, lls.tolist()


def fit_com_mixture(
    key: Array,
    sample: Array,
) -> tuple[Point[Natural, CoMPoissonMixture], list[float]]:
    init_params = COM_MIX_MAN.initialize_from_sample(key, sample)

    optimizer: Optimizer[Natural, CoMPoissonMixture] = Optimizer.adam(
        man=COM_MIX_MAN, learning_rate=LEARNING_RATE
    )
    opt_state = optimizer.init(init_params)

    def cross_entropy_loss(params: Point[Natural, CoMPoissonMixture]) -> Array:
        return -COM_MIX_MAN.average_log_observable_density(params, sample)

    def grad_step(
        opt_state_and_params: tuple[OptState, Point[Natural, CoMPoissonMixture]], _: Any
    ) -> tuple[tuple[OptState, Point[Natural, CoMPoissonMixture]], Array]:
        opt_state, params = opt_state_and_params
        loss_val, grads = COM_MIX_MAN.value_and_grad(cross_entropy_loss, params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), -loss_val  # Return log likelihood

    (_, final_params), lls = jax.lax.scan(
        grad_step, (opt_state, init_params), None, length=N_SGD_STEPS
    )

    return final_params, lls.tolist()


def main() -> None:
    """Run COM-Poisson mixture model analysis."""
    initialize_jax(device="gpu")
    paths = example_paths(__file__)

    # Create models and generate data
    key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), byteorder="big"))
    key_sample, key_fit = jax.random.split(key, 2)
    psn_key, com_key, fan_key = jax.random.split(key_fit, 3)

    grt_params = create_ground_truth_fa()
    _, grt_obs_params = FAN_MAN.observable_distribution(grt_params)

    grt_stats = compute_normal_statistics(NOR_MAN.to_mean(grt_obs_params))
    sample = FAN_MAN.observable_sample(key_sample, grt_params, SAMPLE_SIZE)

    # Discretize data and compute empirical statistics
    discrete_sample = jnp.round(jnp.maximum(0, sample))
    discrete_stats = compute_sample_statistics(discrete_sample)

    # Fit Factor Analysis and compute its statistics
    fan_params, fa_lls = fit_factor_analysis(fan_key, discrete_sample)
    _, fan_obs_params = FAN_MAN.observable_distribution(fan_params)
    fan_stats = compute_normal_statistics(NOR_MAN.to_mean(fan_obs_params))

    # Fit Poisson mixture and compute its statistics
    psn_params, psn_lls = fit_poisson_mixture(psn_key, discrete_sample)
    psn_mean, psn_cov = PSN_MIX_MAN.observable_mean_covariance(psn_params)
    psn_stats = compute_dense_statistics(psn_mean, psn_cov)

    # Fit COM mixture and compute its statistics
    com_params, com_lls = fit_com_mixture(com_key, discrete_sample)
    cbm_mean, cbm_cov = COM_MIX_MAN.observable_mean_covariance(com_params)
    cbm_stats = compute_dense_statistics(cbm_mean, cbm_cov)
    # take only every SGD_FACTOR-th log likelihood
    com_lls = com_lls[::SGD_FACTOR]

    # Save results
    results = PoissonAnalysisResults(
        grt_stats=grt_stats,
        sample_stats=discrete_stats,
        fan_stats=fan_stats,
        psn_stats=psn_stats,
        com_stats=cbm_stats,
        com_lls=com_lls,
        psn_lls=psn_lls,
        fan_lls=fa_lls,
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
