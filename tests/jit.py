"""Test JAX JIT compatibility of dataclasses."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, random

from goal.distributions import Categorical, MultivariateGaussian, Poisson
from goal.exponential_family import Mean
from goal.linear import (
    PositiveDefinite,
)
from goal.manifold import Point


def test_point_creation():
    """Test if Points can be created inside jitted functions."""

    @jax.jit
    def make_point(params: Array) -> Point[Mean, MultivariateGaussian]:
        return Point[Mean, MultivariateGaussian](params)

    params: Array = jnp.array([1.0, 2.0, 3.0])
    try:
        _ = make_point(params)  # type: ignore
        print("✓ Point creation is jit-compatible")
    except Exception as e:
        print("✗ Point creation failed under jit:", str(e))


def test_linear_operators():
    """Test if LinearOperators can be created and used inside jitted functions."""

    side_length = 2

    @jax.jit
    def linear_op_test(
        params: jnp.ndarray, v: jnp.ndarray
    ) -> tuple[Array, Array, Array]:
        op = PositiveDefinite(params, side_length)
        result1 = op.matvec(v)
        result2 = op.logdet()
        inv_op = op.inverse()
        result3 = inv_op.matvec(v)
        return result1, result2, result3

    # Create valid positive definite parameters
    params = jnp.array([2.0, 0.5, 1.0])  # Upper triangular elements
    v = jnp.array([1.0, 1.0])

    try:
        _ = linear_op_test(params, v)  # type: ignore
        print("✓ PositiveDefinite operations are jit-compatible")
    except Exception as e:
        print("✗ PositiveDefinite operations failed under jit:", str(e))


def test_distributions():
    """Test if distribution operations can be jitted."""

    # Test MultivariateGaussian
    @jax.jit
    def gaussian_test(params: jnp.ndarray, x: jnp.ndarray) -> tuple[Any, ...]:
        mvn = MultivariateGaussian(2)  # 2D Gaussian
        p = mvn.natural_point(params)
        result1 = mvn.log_density(p, x)
        result2 = mvn.log_partition_function(p)
        mean_p = mvn.to_mean(p)
        return result1, result2, mean_p

    # Test Categorical
    @jax.jit
    def categorical_test(params: jnp.ndarray, x: jnp.ndarray) -> tuple[Any, ...]:
        dist = Categorical(3)  # 3 categories
        p = dist.natural_point(params)
        result1 = dist.log_density(p, x)
        result2 = dist.log_partition_function(p)
        return result1, result2

    # Test Poisson
    @jax.jit
    def poisson_test(param: jnp.ndarray, x: jnp.ndarray) -> tuple[Any, ...]:
        dist = Poisson()
        p = dist.natural_point(param)
        result1 = dist.log_density(p, x)
        result2 = dist.log_partition_function(p)
        return result1, result2

    try:
        # Test Gaussian
        gauss_params = jnp.array(
            [0.0, 0.0, 1.0, 0.0, 1.0]
        )  # mean and covariance params
        x = jnp.array([1.0, 1.0])
        _ = gaussian_test(gauss_params, x)  # type: ignore
        print("✓ MultivariateGaussian operations are jit-compatible")
    except Exception as e:
        print("✗ MultivariateGaussian operations failed under jit:", str(e))

    try:
        # Test Categorical
        cat_params = jnp.array([0.5, 0.5])  # 2 params for 3 categories
        x = jnp.array(1)
        _ = categorical_test(cat_params, x)  # type: ignore
        print("✓ Categorical operations are jit-compatible")
    except Exception as e:
        print("✗ Categorical operations failed under jit:", str(e))

    try:
        # Test Poisson
        poisson_param = jnp.array([1.0])
        x = jnp.array(2)
        _ = poisson_test(poisson_param, x)  # type: ignore
        print("✓ Poisson operations are jit-compatible")
    except Exception as e:
        print("✗ Poisson operations failed under jit:", str(e))


def test_sampling():
    """Test if sampling operations can be jitted."""

    @jax.jit
    def sample_test(key: jnp.ndarray, params: jnp.ndarray) -> Any:
        dist = MultivariateGaussian(2)
        p = dist.natural_point(params)
        return dist.sample(key, p, n=10)

    try:
        params = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0])
        key = random.PRNGKey(0)
        _ = sample_test(key, params)  # type: ignore
        print("✓ Sampling operations are jit-compatible")
    except Exception as e:
        print("✗ Sampling operations failed under jit:", str(e))


if __name__ == "__main__":
    print("\nTesting JAX JIT compatibility...")
    print("-" * 40)

    test_point_creation()
    test_linear_operators()
    test_distributions()
    test_sampling()
