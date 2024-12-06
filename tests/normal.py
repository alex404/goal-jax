"""Tests for normal distribution parameterizations."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.exponential_family import Mean
from goal.exponential_family.distributions import (
    Euclidean,
    Normal,
)
from goal.manifold import Point, reduce_dual
from goal.transforms import Diagonal, PositiveDefinite, Scale


@pytest.fixture
def sample_data() -> Array:
    """Create sample data for testing."""
    key = jax.random.PRNGKey(0)
    mean = jnp.array([1.0, -0.5])
    cov = jnp.array([[2.0, 1.0], [1.0, 1.0]])

    pd_man = Normal(2)
    gt_cov = pd_man.cov_man.from_dense(cov)
    mu: Point[Mean, Euclidean] = Point(mean)
    gt_mean_point = pd_man.from_mean_and_covariance(mu, gt_cov)
    gt_natural_point = pd_man.to_natural(gt_mean_point)

    return pd_man.sample(key, gt_natural_point, 100)


def validate_normal_coordinates[R: PositiveDefinite](
    man: Normal[R],
    mean_params: Point[Mean, Normal[R]],
    rtol: float = 1e-5,
) -> None:
    """Test coordinate transformations and parameter relationships.

    Args:
        man: Normal manifold with specific representation
        mean_params: Parameters in mean coordinates
        sample: Sample data for density computation
        rtol: Relative tolerance for float comparisons
    """
    # Test mean parameterization
    mean, second_moment = man.split_mean_params(mean_params)
    outer_mean = man.cov_man.outer_product(mean, mean)
    covariance = second_moment - outer_mean
    log_det = man.cov_man.logdet(covariance)

    entropy = 0.5 * (man.data_dim + man.data_dim * jnp.log(2 * jnp.pi) + log_det)
    assert jnp.allclose(man.negative_entropy(mean_params), -entropy, rtol=rtol)

    # Test natural parameterization
    natural_params = man.to_natural(mean_params)
    loc, precision = man.split_natural_params(natural_params)
    if man.cov_man.rep is Scale():
        precision = precision / 2

    recovered_covariance = reduce_dual(man.cov_man.inverse(precision))
    recovered_mean = man.cov_man(recovered_covariance, loc)

    # Test log partition consistency
    log_partition = 0.5 * jnp.dot(
        loc.params, recovered_mean.params
    ) - 0.5 * man.cov_man.logdet(precision)
    assert jnp.allclose(
        man.log_partition_function(natural_params), log_partition, rtol=rtol
    )

    # Test coordinate recovery
    recovered_mean_params = man.to_mean(natural_params)
    assert jnp.allclose(recovered_mean_params.params, mean_params.params, rtol=rtol)


def test_normal_representations(sample_data: Array) -> None:
    """Test consistency across different normal representations."""
    # Setup manifolds
    pd_man = Normal(2, PositiveDefinite)
    dia_man = Normal(2, Diagonal)
    iso_man = Normal(2, Scale)

    # Get scale parameters and convert to other representations
    iso_mean_params = iso_man.average_sufficient_statistic(sample_data)
    mean, iso_covariance = iso_man.to_mean_and_covariance(iso_mean_params)

    dense_covariance = iso_man.cov_man.to_dense(iso_covariance)
    pd_covariance = pd_man.cov_man.from_dense(dense_covariance)
    dia_covariance = dia_man.cov_man.from_dense(dense_covariance)

    pd_mean_params = pd_man.from_mean_and_covariance(mean, pd_covariance)
    dia_mean_params = dia_man.from_mean_and_covariance(mean, dia_covariance)

    # Validate each representation
    validate_normal_coordinates(iso_man, iso_mean_params)
    validate_normal_coordinates(dia_man, dia_mean_params)
    validate_normal_coordinates(pd_man, pd_mean_params)

    # Test consistency of log-likelihoods across representations
    iso_natural = iso_man.to_natural(iso_mean_params)
    dia_natural = dia_man.to_natural(dia_mean_params)
    pd_natural = pd_man.to_natural(pd_mean_params)

    iso_ll = iso_man.average_log_density(iso_natural, sample_data)
    dia_ll = dia_man.average_log_density(dia_natural, sample_data)
    pd_ll = pd_man.average_log_density(pd_natural, sample_data)

    assert jnp.allclose(iso_ll, dia_ll, rtol=1e-5)
    assert jnp.allclose(dia_ll, pd_ll, rtol=1e-5)
