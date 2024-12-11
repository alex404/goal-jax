"""Tests for normal distribution parameterizations."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Diagonal, Mean, Point, PositiveDefinite, Scale, reduce_dual
from goal.models import (
    Euclidean,
    Normal,
)

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def sample_data() -> Array:
    """Create sample data for testing."""
    key = jax.random.PRNGKey(0)
    mean = jnp.array([1.0, -0.5])
    cov = jnp.array([[2.0, 1.0], [1.0, 1.0]])

    pd_man = Normal(2)
    gt_cov = pd_man.cov_man.from_dense(cov)
    mu: Point[Mean, Euclidean] = Point(mean)
    gt_mean_point = pd_man.join_mean_covariance(mu, gt_cov)
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
    mean, second_moment = man.split_mean_second_moment(mean_params)
    outer_mean = man.cov_man.outer_product(mean, mean)
    covariance = second_moment - outer_mean
    log_det = man.cov_man.logdet(covariance)

    entropy = 0.5 * (man.data_dim + man.data_dim * jnp.log(2 * jnp.pi) + log_det)
    assert jnp.allclose(man.negative_entropy(mean_params), -entropy, rtol=rtol)
    print("mean parameters", mean_params)
    print("log_det1", log_det)
    print("covariance", covariance)
    print("negative_entropy", man.negative_entropy(mean_params))

    # Test natural parameterization
    natural_params = man.to_natural(mean_params)
    _, precision = man.split_location_precision(natural_params)

    # if isinstance(man.cov_man.rep, Scale):
    #     precision /= 2.0
    recovered_covariance = reduce_dual(man.cov_man.inverse(precision))

    # assert jnp.allclose(
    #     man.log_partition_function(natural_params), log_partition, rtol=rtol
    # )
    print("natural parameters", natural_params)
    print("split natural", man.split_location_precision(natural_params))
    print("recovered_covariance", recovered_covariance)
    print("log_det2", man.cov_man.logdet(precision))
    print("log_partition", man.log_partition_function(natural_params))
    # print("log_partition", log_partition)

    # Test coordinate recovery
    recovered_mean_params = man.to_mean(natural_params)
    print("recovered_mean_params", recovered_mean_params)
    assert jnp.allclose(recovered_mean_params.params, mean_params.params, rtol=rtol)

    mu, sig0 = man.split_mean_covariance(mean_params)
    sig = man.cov_man.to_dense(sig0)
    print("mu", mu)
    print("sig", sig)


def test_normal_representations(sample_data: Array) -> None:
    """Test consistency across different normal representations."""
    # Setup manifolds
    pd_man = Normal(2, PositiveDefinite)
    dia_man = Normal(2, Diagonal)
    iso_man = Normal(2, Scale)

    # Get scale parameters and convert to other representations
    iso_mean_params = iso_man.average_sufficient_statistic(sample_data)
    mean, iso_covariance = iso_man.split_mean_covariance(iso_mean_params)

    dense_covariance = iso_man.cov_man.to_dense(iso_covariance)
    pd_covariance = pd_man.cov_man.from_dense(dense_covariance)
    dia_covariance = dia_man.cov_man.from_dense(dense_covariance)

    pd_mean_params = pd_man.join_mean_covariance(mean, pd_covariance)
    dia_mean_params = dia_man.join_mean_covariance(mean, dia_covariance)

    # Validate each representation
    print("\nIsotropic")
    validate_normal_coordinates(iso_man, iso_mean_params)
    print("\nDiagonal")
    validate_normal_coordinates(dia_man, dia_mean_params)
    print("\nPositive Definite")
    validate_normal_coordinates(pd_man, pd_mean_params)

    # Test consistency of log-likelihoods across representations
    iso_natural = iso_man.to_natural(iso_mean_params)
    dia_natural = dia_man.to_natural(dia_mean_params)
    pd_natural = pd_man.to_natural(pd_mean_params)

    iso_ll = iso_man.average_log_density(iso_natural, sample_data)
    dia_ll = dia_man.average_log_density(dia_natural, sample_data)
    pd_ll = pd_man.average_log_density(pd_natural, sample_data)
    print("iso_ll", iso_ll)
    print("dia_ll", dia_ll)
    print("pd_ll", pd_ll)

    # assert jnp.allclose(iso_ll, dia_ll, rtol=1e-5)
    # assert jnp.allclose(dia_ll, pd_ll, rtol=1e-5)
