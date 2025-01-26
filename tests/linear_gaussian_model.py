"""Tests for Linear Gaussian Model parameterizations.

This module tests the core functionality of Linear Gaussian Models across different
covariance representations (Scale, Diagonal, PositiveDefinite). Tests cover:

1. Parameter conversion and recovery
2. Likelihood computation
3. Consistency between different representations
4. Conversion to full normal distributions
"""

import logging

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.geometry import Diagonal, Mean, Point, PositiveDefinite, Scale
from goal.models import Euclidean, LinearGaussianModel, Normal

# Configure JAX
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Set up logging
logger = logging.getLogger(__name__)

# Test Parameters
obs_mean = jnp.array([2.0, -1.0, 0.0])
lat_mean = jnp.array([3.0, -2.0])
obs_cov = jnp.array(
    [
        [4.0, 1.5, 0.0],
        [1.5, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
)
int_cov = jnp.array(
    [
        [-1.0, 0.5],
        [0.0, 0.0],
        [0.5, 0.0],
    ]
)
lat_cov = jnp.array([[3.0, 1.0], [1.0, 2.0]])
sample_size = 1000
relative_tol = 1e-4
absolute_tol = 1e-4


@pytest.fixture
def ground_truth_normal() -> Normal[PositiveDefinite]:
    """Create ground truth normal model."""
    joint_dim = obs_mean.shape[0] + lat_mean.shape[0]
    return Normal(joint_dim, PositiveDefinite)


@pytest.fixture
def ground_truth_params(
    ground_truth_normal: Normal[PositiveDefinite],
) -> Point[Mean, Normal[PositiveDefinite]]:
    """Create ground truth parameters in mean coordinates."""
    # Create joint mean
    joint_mean = jnp.concatenate([obs_mean, lat_mean])

    # Create joint covariance
    joint_cov = jnp.block(
        [
            [obs_cov, int_cov],
            [int_cov.T, lat_cov],
        ]
    )

    mu: Point[Mean, Euclidean] = Point(joint_mean)
    cov = ground_truth_normal.cov_man.from_dense(joint_cov)
    return ground_truth_normal.join_mean_covariance(mu, cov)


@pytest.fixture
def sample_data(
    ground_truth_normal: Normal[PositiveDefinite],
    ground_truth_params: Point[Mean, Normal[PositiveDefinite]],
) -> Array:
    """Generate sample data from ground truth distribution."""
    key = jax.random.PRNGKey(0)
    gt_natural = ground_truth_normal.to_natural(ground_truth_params)
    return ground_truth_normal.sample(key, gt_natural, sample_size)


def scipy_log_likelihood(x: Array, mean: Array, cov: Array) -> Array:
    """Compute average log likelihood using scipy's implementation."""
    return jnp.mean(
        jax.vmap(stats.multivariate_normal.logpdf, in_axes=(0, None, None))(
            x, mean, cov
        )
    )


def test_observable_distributions(
    sample_data: Array, caplog: pytest.LogCaptureFixture
) -> None:
    """Test equivalence between observable_density and the density of observable_distribution."""
    caplog.set_level(logging.INFO)
    logger.info("\nTesting observable distribution consistency")

    # Create model
    obs_dim = obs_mean.shape[0]
    lat_dim = lat_mean.shape[0]
    lgm = LinearGaussianModel(obs_dim, Diagonal, lat_dim)

    # Get model parameters
    means = lgm.average_sufficient_statistic(sample_data)
    params = lgm.to_natural(means)

    # Compare densities
    obs_data = sample_data[:, :obs_dim]  # Extract observable components
    direct_ll = lgm.average_log_observable_density(params, obs_data)

    nor_man, marginal = lgm.observable_distribution(params)
    marginal_ll = nor_man.average_log_density(marginal, obs_data)

    logger.info(f"Direct observable log-density: {direct_ll}")
    logger.info(f"Marginal distribution log-density: {marginal_ll}")

    assert jnp.allclose(direct_ll, marginal_ll, rtol=0.1, atol=0.01), (
        "Log-density differs between direct computation and marginal distribution"
    )


@pytest.mark.parametrize("rep", [Scale, Diagonal, PositiveDefinite])
def test_single_model(
    rep: type[Scale | Diagonal | PositiveDefinite],
    sample_data: Array,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test fitting and evaluation of a single model.

    Args:
        rep: Covariance representation to test
        sample_data: Generated test data
        caplog: Pytest fixture for capturing logs
    """
    caplog.set_level(logging.INFO)
    logger.info(f"\nTesting {rep.__name__} representation")

    # Create models
    obs_dim = obs_mean.shape[0]
    lat_dim = lat_mean.shape[0]
    lgm_man = LinearGaussianModel(obs_dim, rep, lat_dim)
    nor_man = Normal(obs_dim + lat_dim, PositiveDefinite)

    # Fit LGM model
    means = lgm_man.average_sufficient_statistic(sample_data)
    logger.info(f"Mean parameters: {means.array}")

    # Natural -> Mean recovery
    params = lgm_man.to_natural(means)

    # Add validation check and logging
    valid = bool(lgm_man.check_natural_parameters(params))
    logger.info("Natural parameter check for %s: %s", type(lgm_man).__name__, valid)
    assert valid, (
        f"Invalid natural parameters after fitting for {type(lgm_man).__name__}"
    )

    remeans = lgm_man.to_mean(params)
    reparams = lgm_man.to_natural(remeans)
    logger.info(f"Mean parameters: {means.array}")
    logger.info(f"Remean parameters: {remeans.array}")
    logger.info(f"Natural parameters: {params.array}")
    logger.info(f"ReNatural parameters: {reparams.array}")

    assert jnp.allclose(
        remeans.array, means.array, rtol=relative_tol, atol=absolute_tol
    )

    assert jnp.allclose(
        reparams.array, params.array, rtol=relative_tol, atol=absolute_tol
    )

    # Test parameter recovery
    recovered_mean_params = lgm_man.to_mean(params)
    assert jnp.allclose(
        recovered_mean_params.array,
        means.array,
        rtol=relative_tol,
        atol=absolute_tol,
    ), "Failed to recover mean parameters"

    # Convert to normal model
    nor_params = lgm_man.to_normal(params)
    logger.info(f"Normal natural parameters: {nor_params.array}")

    # Compare log-partition functions
    logger.info("\nComparing log partition functions:")
    lgm_lpf = lgm_man.log_partition_function(params)
    nor_lpf = nor_man.log_partition_function(nor_params)
    logger.info(f"LGM log partition function: {lgm_lpf}")
    logger.info(f"Normal log partition function: {nor_lpf}")
    assert jnp.allclose(lgm_lpf, nor_lpf, rtol=relative_tol, atol=absolute_tol)

    # Compare log densities
    lgm_ll = lgm_man.average_log_density(params, sample_data)
    nor_ll = nor_man.average_log_density(nor_params, sample_data)

    logger.info(f"LGM average log-density: {lgm_ll}")
    logger.info(f"Normal average log-density: {nor_ll}")

    assert jnp.allclose(lgm_ll, nor_ll, rtol=relative_tol, atol=absolute_tol), (
        "Log-likelihood differs between LGM and Normal"
    )

    # Compare to scipy
    mean, cov = nor_man.split_mean_covariance(nor_man.to_mean(nor_params))
    scipy_ll = scipy_log_likelihood(
        sample_data, mean.array, nor_man.cov_man.to_dense(cov)
    )
    logger.info(f"Scipy average log-density: {scipy_ll}")

    assert jnp.allclose(lgm_ll, scipy_ll, rtol=relative_tol, atol=absolute_tol), (
        "Log-likelihood differs from scipy implementation"
    )


def test_model_consistency(
    sample_data: Array, caplog: pytest.LogCaptureFixture
) -> None:
    """Test consistency between different model representations.

    This test takes an isotropic fit and expands it to more complex representations,
    verifying that key quantities remain consistent.
    """
    caplog.set_level(logging.INFO)
    logger.info("\nTesting model consistency")

    # Create models
    obs_dim = obs_mean.shape[0]
    lat_dim = lat_mean.shape[0]
    iso_model = LinearGaussianModel(obs_dim, Scale, lat_dim)
    dia_model = LinearGaussianModel(obs_dim, Diagonal, lat_dim)
    pod_model = LinearGaussianModel(obs_dim, PositiveDefinite, lat_dim)

    # Fit isotropic model
    iso_means = iso_model.average_sufficient_statistic(sample_data)
    iso_natural = iso_model.to_natural(iso_means)

    dia_natural = iso_model.transform_observable_rep(dia_model, iso_natural)
    pod_natural = iso_model.transform_observable_rep(pod_model, iso_natural)

    # Test log partition function
    logger.info("\nComparing log partition functions:")
    iso_lpf = iso_model.log_partition_function(iso_natural)
    dia_lpf = dia_model.log_partition_function(dia_natural)
    pod_lpf = pod_model.log_partition_function(pod_natural)

    logger.info(f"Isotropic: {iso_lpf}")
    logger.info(f"Diagonal: {dia_lpf}")
    logger.info(f"PositiveDefinite: {pod_lpf}")

    assert jnp.allclose(iso_lpf, dia_lpf, rtol=relative_tol, atol=absolute_tol)
    assert jnp.allclose(dia_lpf, pod_lpf, rtol=relative_tol, atol=absolute_tol)

    # Test log density
    logger.info("\nComparing log densities:")
    iso_ll = iso_model.average_log_density(iso_natural, sample_data)
    dia_ll = dia_model.average_log_density(dia_natural, sample_data)
    pod_ll = pod_model.average_log_density(pod_natural, sample_data)

    logger.info(f"Isotropic: {iso_ll}")
    logger.info(f"Diagonal: {dia_ll}")
    logger.info(f"PositiveDefinite: {pod_ll}")

    assert jnp.allclose(iso_ll, dia_ll, rtol=relative_tol, atol=absolute_tol)
    assert jnp.allclose(dia_ll, pod_ll, rtol=relative_tol, atol=absolute_tol)

    # Test coordinate conversion
    logger.info("\nTesting coordinate conversion:")

    # Natural -> Mean
    iso_means = iso_model.to_mean(iso_natural)
    dia_means = dia_model.to_mean(dia_natural)
    pod_means = pod_model.to_mean(pod_natural)

    # Compare negative entropy
    iso_ne = iso_model.negative_entropy(iso_means)
    dia_ne = dia_model.negative_entropy(dia_means)
    pod_ne = pod_model.negative_entropy(pod_means)

    logger.info(f"Isotropic negative entropy: {iso_ne}")
    logger.info(f"Diagonal negative entropy: {dia_ne}")
    logger.info(f"PositiveDefinite negative entropy: {pod_ne}")

    assert jnp.allclose(iso_ne, dia_ne, rtol=relative_tol, atol=absolute_tol)
    assert jnp.allclose(dia_ne, pod_ne, rtol=relative_tol, atol=absolute_tol)


if __name__ == "__main__":
    _ = pytest.main()
