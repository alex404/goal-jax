"""Tests for normal distribution parameterizations.

This module tests the core functionality of Normal distributions across different
covariance representations (Scale, Diagonal, PositiveDefinite). Tests cover:

1. Parameter conversion and recovery
2. Likelihood computation
3. Consistency between different representations
"""

import logging

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.geometry import (
    Diagonal,
    Mean,
    Point,
    PositiveDefinite,
    Scale,
)
from goal.models import Euclidean, Normal

# Configure JAX
jax.config.update("jax_platform_name", "cpu")

# Set up logging
logger = logging.getLogger(__name__)

# Test Parameters
sampling_mean = jnp.array([1.0, -0.5])
sampling_covariance = jnp.array([[2.0, 1.0], [1.0, 1.0]])
sample_size = 1000
relative_tol = 1e-5
absolute_tol = 1e-7


@pytest.fixture
def ground_truth_normal() -> Normal[PositiveDefinite]:
    """Create ground truth normal model."""
    return Normal(sampling_mean.shape[0], PositiveDefinite)


@pytest.fixture
def ground_truth_params(
    ground_truth_normal: Normal[PositiveDefinite],
) -> Point[Mean, Normal[PositiveDefinite]]:
    """Create ground truth parameters in mean coordinates."""
    gt_cov = ground_truth_normal.cov_man.from_dense(sampling_covariance)
    mu: Point[Mean, Euclidean] = Point(sampling_mean)
    return ground_truth_normal.join_mean_covariance(mu, gt_cov)


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

    # Create model
    model = Normal(sampling_mean.shape[0], rep)

    # Fit model
    mean_params = model.average_sufficient_statistic(sample_data)
    logger.info(f"Mean parameters: {mean_params.params}")

    # Convert to natural parameters
    natural_params = model.to_natural(mean_params)
    logger.info(f"Natural parameters: {natural_params.params}")

    valid = bool(model.check_natural_parameters(natural_params))

    logger.info("Natural parameter check for %s: %s", type(model).__name__, valid)
    assert (
        valid
    ), f"Invalid natural parameters after initialization for {type(model).__name__}"

    # Compute average log-density
    log_density = model.average_log_density(natural_params, sample_data)
    logger.info(f"Average log-density: {log_density}")

    # Test parameter recovery
    recovered_mean_params = model.to_mean(natural_params)
    assert jnp.allclose(
        recovered_mean_params.params,
        mean_params.params,
        rtol=relative_tol,
        atol=absolute_tol,
    ), "Failed to recover mean parameters"

    # Compare to scipy implementation
    mean, cov = model.split_mean_covariance(mean_params)
    dense_cov = model.cov_man.to_dense(cov)
    scipy_ll = scipy_log_likelihood(sample_data, mean.params, dense_cov)
    logger.info(f"Scipy average log-density: {scipy_ll}")

    assert jnp.allclose(
        log_density, scipy_ll, rtol=relative_tol, atol=absolute_tol
    ), "Log-likelihood differs from scipy implementation"


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
    iso_model = Normal(sampling_mean.shape[0], Scale)
    dia_model = Normal(sampling_mean.shape[0], Diagonal)
    pod_model = Normal(sampling_mean.shape[0], PositiveDefinite)

    # Fit isotropic model
    iso_means = iso_model.average_sufficient_statistic(sample_data)
    iso_natural = iso_model.to_natural(iso_means)
    dia_natural = iso_model.embed_rep(dia_model, iso_natural)
    pod_natural = iso_model.embed_rep(pod_model, iso_natural)

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

    logger.info(f"Isotropic natural parameters: {iso_natural.params}")
    logger.info(f"Diagonal natural parameters: {dia_natural.params}")
    logger.info(f"PositiveDefinite natural parameters: {pod_natural.params}")
    logger.info(
        f"Isotropic precisions: {iso_model.cov_man.to_dense(iso_model.split_location_precision(iso_natural)[1])}"
    )
    logger.info(
        f"Diagonal precisions: {dia_model.cov_man.to_dense(dia_model.split_location_precision(dia_natural)[1])}"
    )
    logger.info(
        f"Positive Definite precisions: {pod_model.cov_man.to_dense(pod_model.split_location_precision(pod_natural)[1])}"
    )
    logger.info(f"Isotropic mean parameters: {iso_means.params}")
    logger.info(f"Diagonal mean parameters: {dia_means.params}")
    logger.info(f"PositiveDefinite mean parameters: {pod_means.params}")

    # Compare negative entropy
    iso_ne = iso_model.negative_entropy(iso_means)
    dia_ne = dia_model.negative_entropy(dia_means)
    pod_ne = pod_model.negative_entropy(pod_means)

    logger.info(f"Isotropic negative entropy: {iso_ne}")
    logger.info(f"Diagonal negative entropy: {dia_ne}")
    logger.info(f"PositiveDefinite negative entropy: {pod_ne}")

    assert jnp.allclose(iso_ne, dia_ne, rtol=relative_tol, atol=absolute_tol)
    assert jnp.allclose(dia_ne, pod_ne, rtol=relative_tol, atol=absolute_tol)

    # Mean -> Natural recovery
    iso_recovered = iso_model.to_natural(iso_means)
    dia_recovered = dia_model.to_natural(dia_means)
    pod_recovered = pod_model.to_natural(pod_means)

    assert jnp.allclose(
        iso_recovered.params, iso_natural.params, rtol=relative_tol, atol=absolute_tol
    )
    assert jnp.allclose(
        dia_recovered.params, dia_natural.params, rtol=relative_tol, atol=absolute_tol
    )
    assert jnp.allclose(
        pod_recovered.params, pod_natural.params, rtol=relative_tol, atol=absolute_tol
    )


if __name__ == "__main__":
    pytest.main()
