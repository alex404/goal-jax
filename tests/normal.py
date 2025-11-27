"""Tests for normal distribution parameterizations.

This module tests the core functionality of Normal distributions across different
covariance representations (Scale(), Diagonal(), PositiveDefinite()). Tests cover:

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
    PositiveDefinite,
    Scale,
)
from goal.models import Covariance, Normal

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
def ground_truth_normal() -> Normal:
    """Create ground truth normal model."""
    return Normal(sampling_mean.shape[0], PositiveDefinite())


@pytest.fixture
def ground_truth_params(
    ground_truth_normal: Normal,
) -> Array:
    """Create ground truth parameters in mean coordinates."""
    gt_cov = ground_truth_normal.cov_man.from_dense(
        sampling_covariance
    )
    mu = sampling_mean
    return ground_truth_normal.join_mean_covariance(mu, gt_cov)


@pytest.fixture
def sample_data(
    ground_truth_normal: Normal,
    ground_truth_params: Array,
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


@pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
def test_single_model(
    rep: Scale | Diagonal | PositiveDefinite,
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
    logger.info(f"\nTesting {type(rep).__name__} representation")

    # Create model
    model = Normal(sampling_mean.shape[0], rep)

    # Fit model
    means = model.average_sufficient_statistic(sample_data)
    logger.info(f"Mean parameters: {means}")

    # Convert to natural parameters
    params = model.to_natural(means)
    logger.info(f"Natural parameters: {params}")

    valid = bool(model.check_natural_parameters(params))

    logger.info("Natural parameter check for %s: %s", type(model).__name__, valid)
    assert valid, (
        f"Invalid natural parameters after initialization for {type(model).__name__}"
    )

    # Compute average log-density
    log_density = model.average_log_density(params, sample_data)
    logger.info(f"Average log-density: {log_density}")

    # Test parameter recovery
    recovered_means = model.to_mean(params)
    assert jnp.allclose(
        recovered_means,
        means,
        rtol=relative_tol,
        atol=absolute_tol,
    ), "Failed to recover mean parameters"

    # Compare to scipy implementation
    mean, cov = model.split_mean_covariance(means)
    dense_cov = model.cov_man.to_dense(cov)
    scipy_ll = scipy_log_likelihood(sample_data, mean, dense_cov)
    logger.info(f"Scipy average log-density: {scipy_ll}")

    assert jnp.allclose(log_density, scipy_ll, rtol=relative_tol, atol=absolute_tol), (
        "Log-likelihood differs from scipy implementation"
    )


def test_whiten(
    ground_truth_normal: Normal,
    ground_truth_params: Array,
) -> None:
    """Test whitening of normal distributions."""
    # Extract ground truth mean and covariance
    gt_mean, gt_cov = ground_truth_normal.split_mean_covariance(ground_truth_params)
    gt_mean_array = gt_mean
    gt_cov_matrix = ground_truth_normal.cov_man.to_dense(gt_cov)

    # Create a test distribution different from ground truth
    test_mean = jnp.array([3.0, -2.0])  # Different from ground truth
    test_cov_matrix = jnp.array([[1.5, 0.3], [0.3, 2.0]])  # Different from ground truth

    test_cov_point = ground_truth_normal.cov_man.from_dense(test_cov_matrix)
    test_dist = ground_truth_normal.join_mean_covariance(
        test_mean, test_cov_point
    )

    # Test 1: Whiten ground truth relative to itself should give standard normal
    norm_gt = ground_truth_normal.whiten(ground_truth_params, ground_truth_params)
    norm_gt_mean, norm_gt_cov = ground_truth_normal.split_mean_covariance(norm_gt)

    assert jnp.allclose(
        norm_gt_mean, jnp.zeros_like(gt_mean_array), rtol=1e-5, atol=1e-7
    ), "Whitening a distribution relative to itself should give zero mean"
    assert jnp.allclose(
        ground_truth_normal.cov_man.to_dense(norm_gt_cov),
        jnp.eye(gt_mean_array.shape[0]),
        rtol=1e-5,  # Add relative tolerance
        atol=1e-7,  # Add absolute tolerance
    ), "Whitening a distribution relative to itself should give identity covariance"

    # Test 2: Whiten test distribution relative to ground truth
    whitened_test = ground_truth_normal.whiten(test_dist, ground_truth_params)
    whitened_mean, whitened_cov = ground_truth_normal.split_mean_covariance(
        whitened_test
    )

    # Compute expected values manually
    chol = jnp.linalg.cholesky(gt_cov_matrix)
    expected_mean = jax.scipy.linalg.solve_triangular(
        chol, test_mean - gt_mean_array, lower=True
    )

    temp = jax.scipy.linalg.solve_triangular(chol, test_cov_matrix, lower=True)
    expected_cov = jax.scipy.linalg.solve_triangular(chol, temp.T, lower=True).T

    # Verify results
    assert jnp.allclose(whitened_mean, expected_mean, rtol=1e-5, atol=1e-7), (
        "Whitened mean doesn't match expected value"
    )
    assert jnp.allclose(
        ground_truth_normal.cov_man.to_dense(whitened_cov),
        expected_cov,
        rtol=1e-5,
        atol=1e-7,
    ), "Whitened covariance doesn't match expected value"

    # Test 3: Verify whitening is invertible - this is a nice property check
    # Create a third distribution to be whitened with respect to test_dist
    third_mean = jnp.array([-1.0, 0.5])
    third_cov_matrix = jnp.array([[0.8, -0.1], [-0.1, 1.2]])

    third_cov_point = ground_truth_normal.cov_man.from_dense(third_cov_matrix)
    third_dist = ground_truth_normal.join_mean_covariance(
        third_mean, third_cov_point
    )

    # Whiten third_dist with respect to test_dist
    whitened_third = ground_truth_normal.whiten(third_dist, test_dist)


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
    iso_model = Normal(sampling_mean.shape[0], Scale())
    dia_model = Normal(sampling_mean.shape[0], Diagonal())
    pod_model = Normal(sampling_mean.shape[0], PositiveDefinite())

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

    logger.info(f"Isotropic natural parameters: {iso_natural}")
    logger.info(f"Diagonal natural parameters: {dia_natural}")
    logger.info(f"PositiveDefinite natural parameters: {pod_natural}")
    logger.info(
        f"Isotropic precisions: {iso_model.cov_man.to_dense(iso_model.split_location_precision(iso_natural)[1])}"
    )
    logger.info(
        f"Diagonal precisions: {dia_model.cov_man.to_dense(dia_model.split_location_precision(dia_natural)[1])}"
    )
    logger.info(
        f"Positive Definite precisions: {pod_model.cov_man.to_dense(pod_model.split_location_precision(pod_natural)[1])}"
    )
    logger.info(f"Isotropic mean parameters: {iso_means}")
    logger.info(f"Diagonal mean parameters: {dia_means}")
    logger.info(f"PositiveDefinite mean parameters: {pod_means}")

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
        iso_recovered, iso_natural, rtol=relative_tol, atol=absolute_tol
    )
    assert jnp.allclose(
        dia_recovered, dia_natural, rtol=relative_tol, atol=absolute_tol
    )
    assert jnp.allclose(
        pod_recovered, pod_natural, rtol=relative_tol, atol=absolute_tol
    )


def test_precision_is_inverse_of_covariance(
    ground_truth_normal: Normal,
    ground_truth_params: Array,
):
    """Test that split_location_precision returns the actual precision matrix Σ^{-1}."""
    # Convert to natural parameters
    natural_params = ground_truth_normal.to_natural(ground_truth_params)

    # Split to get precision
    _, precision = ground_truth_normal.split_location_precision(natural_params)

    # Get covariance from mean parameters
    _, covariance = ground_truth_normal.split_mean_covariance(ground_truth_params)

    # Convert both to dense matrices
    precision_dense = ground_truth_normal.cov_man.to_dense(precision)
    covariance_dense = ground_truth_normal.cov_man.to_dense(covariance)

    # Compute actual inverse
    actual_precision = jnp.linalg.inv(covariance_dense)

    logger.info("\n=== Precision Matrix Test ===")
    logger.info(f"Precision from split_location_precision:\n{precision_dense}")
    logger.info(f"Actual precision (Σ^{{-1}}):\n{actual_precision}")
    logger.info(f"Difference:\n{precision_dense - actual_precision}")

    max_error = jnp.max(jnp.abs(precision_dense - actual_precision))
    logger.info(f"Max absolute difference: {max_error:.2e}")

    assert jnp.allclose(
        precision_dense, actual_precision, rtol=relative_tol, atol=absolute_tol
    ), f"Precision matrix is not the inverse of covariance! Max error: {max_error}"

    # Also verify that Σ^{-1} @ Σ = I
    identity_check = precision_dense @ covariance_dense
    expected_identity = jnp.eye(ground_truth_normal.data_dim)

    logger.info(f"Σ^{{-1}} @ Σ:\n{identity_check}")
    logger.info(f"Expected identity:\n{expected_identity}")

    assert jnp.allclose(
        identity_check, expected_identity, rtol=relative_tol, atol=absolute_tol
    ), "Precision @ Covariance does not equal identity!"


if __name__ == "__main__":
    _ = pytest.main()
