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

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Natural,
    Optimizer,
    OptState,
    Point,
)
from goal.models import (
    CoMMixture,
    FactorAnalysis,
)

from ..shared import example_paths, initialize_jax
from .types import ComAnalysisResults, CovarianceStatistics

# Constants
N_NEURONS = 10  # Observable dimension
N_FACTORS = 3  # Latent dimension
SAMPLE_SIZE = 1000  # Number of samples
N_STEPS = 1  # Training steps
N_COMPONENTS = 3  # Number of mixture components
LEARNING_RATE = 0.1


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


def create_factor_analysis() -> tuple[FactorAnalysis, Point[Natural, FactorAnalysis]]:
    """Create ground truth factor analysis model."""
    fa_model = FactorAnalysis(N_NEURONS, N_FACTORS)
    loadings = create_ground_truth_loadings()
    means, diags = create_ground_truth_diagonals()
    fa_params = fa_model.to_natural_from_loadings(loadings, means, diags)
    return fa_model, fa_params


def compute_covariance_statistics(sample: Array) -> CovarianceStatistics:
    """Compute empirical statistics from sample data.

    Computes:
    - Means
    - Fano factors (variance/mean ratios)
    - Covariances (lower triangular)
    - Correlation matrix
    """
    means = jnp.mean(sample, axis=0)
    variances = jnp.var(sample, axis=0)
    fano_factors = variances / means

    # Compute covariance and correlation matrices
    cov_matrix = jnp.cov(sample.T)
    std_devs = jnp.sqrt(jnp.diag(cov_matrix))
    corr_matrix = cov_matrix / (std_devs[:, None] @ std_devs[None, :])

    # Extract lower triangular elements of covariance
    tril_idx = jnp.tril_indices(cov_matrix.shape[0])
    covariances = cov_matrix[tril_idx]

    return CovarianceStatistics(
        means=means.tolist(),
        fano_factors=fano_factors.tolist(),
        covariances=covariances.tolist(),
        correlation_matrix=corr_matrix.tolist(),
    )


def compute_numerical_statistics(
    mean: Array, covariance: Array
) -> CovarianceStatistics:
    """Compute statistics from numerically computed mean and covariance.

    Args:
        mean: Mean vector
        covariance: Covariance matrix

    Returns:
        CovarianceStatistics object
    """
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


def fit_com_mixture(
    key: Array,
    sample: Array,
    n_steps: int = N_STEPS,
    learning_rate: float = LEARNING_RATE,
) -> tuple[CoMMixture, Point[Natural, CoMMixture], list[float]]:
    """Fit COM-Poisson mixture model using cross entropy descent.

    Args:
        key: Random key for initialization
        sample: Discretized sample data
        n_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer

    Returns:
        com_mix: Fitted mixture model
        final_params: Optimized parameters
        lls: Training log likelihoods
    """
    com_mix = CoMMixture(N_NEURONS, N_COMPONENTS)
    init_params = com_mix.initialize(key, shape=0.01)

    optimizer: Optimizer[Natural, CoMMixture] = Optimizer.adam(
        man=com_mix, learning_rate=learning_rate
    )
    opt_state = optimizer.init(init_params)

    def cross_entropy_loss(params: Point[Natural, CoMMixture]) -> Array:
        print(params.array.shape)
        return -com_mix.average_log_observable_density(params, sample)

    def grad_step(
        opt_state_and_params: tuple[OptState, Point[Natural, CoMMixture]], _: Any
    ) -> tuple[tuple[OptState, Point[Natural, CoMMixture]], Array]:
        opt_state, params = opt_state_and_params
        loss_val, grads = com_mix.value_and_grad(cross_entropy_loss, params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), -loss_val  # Return log likelihood

    (_, final_params), lls = jax.lax.scan(
        grad_step, (opt_state, init_params), None, length=n_steps
    )

    return com_mix, final_params, lls.tolist()


fit_com_mixture = jax.jit(fit_com_mixture, static_argnames=["n_steps", "learning_rate"])


def main() -> None:
    """Run COM-Poisson mixture model analysis."""
    initialize_jax(disable_jit=True)
    paths = example_paths(__file__)

    # Create models and generate data
    key = jax.random.PRNGKey(0)
    key_sample, key_fit = jax.random.split(key, 2)

    fa_model, fa_params = create_factor_analysis()
    sample = fa_model.observable_sample(key_sample, fa_params, SAMPLE_SIZE)

    # Compute statistics for original FA model
    continuous_stats = compute_covariance_statistics(sample)

    # Discretize data and compute empirical statistics
    discrete_sample = jnp.round(jnp.maximum(0, sample))
    discrete_stats = compute_covariance_statistics(discrete_sample)

    # Fit COM mixture and compute its statistics
    com_mix, com_params, training_lls = fit_com_mixture(key_fit, discrete_sample)
    # Use numerical approximation instead of sampling
    obs_params, int_params, lat_params = com_mix.split_params(com_params)
    print(f"Learned Obs params: {obs_params}")
    print(f"Learned Int params: {int_params}")
    print(f"Learned Lat params: {lat_params}")
    _, prr_params = com_mix.split_conjugated(com_params)
    prr_means = com_mix.lat_man.to_mean(prr_params)
    print(f"Learned component probabilities: {com_mix.lat_man.to_probs(prr_means)}")
    # comp_params, mix_params = com_mix.split_natural_mixture(com_params)
    # print(f"Learned Comp params: {comp_params}")
    # print(f"Learned Mix params: {mix_params}")
    cbm_mean, cbm_cov = com_mix.numerical_mean_covariance(com_params)
    cbm_stats = compute_numerical_statistics(cbm_mean, cbm_cov)

    # Save results
    results = ComAnalysisResults(
        fa_stats=continuous_stats,
        cbm_stats=cbm_stats,
        sample_stats=discrete_stats,
        training_lls=training_lls,
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
