"""Tests for multivariate normal distribution parameterizations.

Tests parameter conversion, likelihood computation, and consistency across
covariance representations (Scale, Diagonal, PositiveDefinite).
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import Normal

jax.config.update("jax_platform_name", "cpu")

# Test parameters
sample_mean = jnp.array([1.0, -0.5])
sample_cov = jnp.array([[2.0, 1.0], [1.0, 1.0]])
sample_size = 1000
rtol, atol = 1e-5, 1e-7


@pytest.fixture
def ground_truth() -> tuple[Normal, Array]:
    """Ground truth normal model and mean parameters."""
    model = Normal(sample_mean.shape[0], PositiveDefinite())
    cov_params = model.cov_man.from_matrix(sample_cov)
    means = model.join_mean_covariance(sample_mean, cov_params)
    return model, means


@pytest.fixture
def samples(ground_truth: tuple[Normal, Array]) -> Array:
    """Sample data from ground truth distribution."""
    model, means = ground_truth
    key = jax.random.PRNGKey(0)
    natural_params = model.to_natural(means)
    return model.sample(key, natural_params, sample_size)


@pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
def test_fit_and_evaluate(rep: Scale | Diagonal | PositiveDefinite, samples: Array):
    """Test fitting model and comparing log-density to scipy."""
    model = Normal(sample_mean.shape[0], rep)

    # Fit model
    means = model.average_sufficient_statistic(samples)
    params = model.to_natural(means)

    assert model.check_natural_parameters(params), "Invalid natural parameters"

    # Test parameter recovery
    recovered = model.to_mean(params)
    assert jnp.allclose(recovered, means, rtol=rtol, atol=atol)

    # Compare to scipy
    log_density = model.average_log_density(params, samples)
    mean, cov = model.split_mean_covariance(means)
    dense_cov = model.cov_man.to_matrix(cov)
    scipy_ll = jnp.mean(
        jax.vmap(stats.multivariate_normal.logpdf, in_axes=(0, None, None))(
            samples, mean, dense_cov
        )
    )
    assert jnp.allclose(log_density, scipy_ll, rtol=rtol, atol=atol)


def test_representation_consistency(samples: Array):
    """Test that embedded representations give same log-density."""
    dim = sample_mean.shape[0]
    iso = Normal(dim, Scale())
    dia = Normal(dim, Diagonal())
    pod = Normal(dim, PositiveDefinite())

    # Fit isotropic and embed to other representations
    iso_means = iso.average_sufficient_statistic(samples)
    iso_natural = iso.to_natural(iso_means)
    dia_natural = iso.embed_rep(dia, iso_natural)
    pod_natural = iso.embed_rep(pod, iso_natural)

    # Log partition functions should match
    iso_lpf = iso.log_partition_function(iso_natural)
    dia_lpf = dia.log_partition_function(dia_natural)
    pod_lpf = pod.log_partition_function(pod_natural)
    assert jnp.allclose(iso_lpf, dia_lpf, rtol=rtol, atol=atol)
    assert jnp.allclose(dia_lpf, pod_lpf, rtol=rtol, atol=atol)

    # Log densities should match
    iso_ll = iso.average_log_density(iso_natural, samples)
    dia_ll = dia.average_log_density(dia_natural, samples)
    pod_ll = pod.average_log_density(pod_natural, samples)
    assert jnp.allclose(iso_ll, dia_ll, rtol=rtol, atol=atol)
    assert jnp.allclose(dia_ll, pod_ll, rtol=rtol, atol=atol)


def test_precision_is_covariance_inverse(ground_truth: tuple[Normal, Array]):
    """Test that precision matrix is the inverse of covariance."""
    model, means = ground_truth
    natural_params = model.to_natural(means)

    _, precision = model.split_location_precision(natural_params)
    _, covariance = model.split_mean_covariance(means)

    prec_dense = model.cov_man.to_matrix(precision)
    cov_dense = model.cov_man.to_matrix(covariance)
    expected_prec = jnp.linalg.inv(cov_dense)

    assert jnp.allclose(prec_dense, expected_prec, rtol=rtol, atol=atol)
    assert jnp.allclose(
        prec_dense @ cov_dense, jnp.eye(model.data_dim), rtol=rtol, atol=atol
    )


def test_whiten(ground_truth: tuple[Normal, Array]):
    """Test whitening transformation."""
    model, means = ground_truth
    gt_mean, gt_cov = model.split_mean_covariance(means)
    gt_cov_mat = model.cov_man.to_matrix(gt_cov)

    # Whitening relative to itself should give standard normal
    whitened = model.whiten(means, means)
    w_mean, w_cov = model.split_mean_covariance(whitened)

    assert jnp.allclose(w_mean, jnp.zeros_like(gt_mean), rtol=rtol, atol=atol)
    assert jnp.allclose(
        model.cov_man.to_matrix(w_cov), jnp.eye(model.data_dim), rtol=rtol, atol=atol
    )

    # Whitening a different distribution
    test_mean = jnp.array([3.0, -2.0])
    test_cov_mat = jnp.array([[1.5, 0.3], [0.3, 2.0]])
    test_params = model.join_mean_covariance(
        test_mean, model.cov_man.from_matrix(test_cov_mat)
    )

    whitened_test = model.whiten(test_params, means)
    wt_mean, wt_cov = model.split_mean_covariance(whitened_test)

    # Verify against manual computation
    chol = jnp.linalg.cholesky(gt_cov_mat)
    expected_mean = jax.scipy.linalg.solve_triangular(
        chol, test_mean - gt_mean, lower=True
    )
    temp = jax.scipy.linalg.solve_triangular(chol, test_cov_mat, lower=True)
    expected_cov = jax.scipy.linalg.solve_triangular(chol, temp.T, lower=True).T

    assert jnp.allclose(wt_mean, expected_mean, rtol=rtol, atol=atol)
    assert jnp.allclose(
        model.cov_man.to_matrix(wt_cov), expected_cov, rtol=rtol, atol=atol
    )
