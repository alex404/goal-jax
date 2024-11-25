"""Test JAX JIT compatibility of dataclasses."""

from typing import Type

import jax
import jax.numpy as jnp
from jax import Array, random

from goal.distributions import Categorical, Gaussian, Poisson
from goal.exponential_family import Mean
from goal.linear import Diagonal, PositiveDefinite, Scale
from goal.manifold import Point


def test_point_creation():
    """Test if Points can be created inside jitted functions."""

    @jax.jit
    def make_point(params: Array) -> Point[Mean, Gaussian]:
        return Point[Mean, Gaussian](params)

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

    # Test Gaussian
    @jax.jit
    def gaussian_test(
        params: jnp.ndarray, x: jnp.ndarray
    ) -> tuple[Array, Array, Point[Mean, Gaussian]]:
        mvn = Gaussian(2)  # 2D Gaussian
        p = mvn.natural_point(params)
        result1 = mvn.log_density(p, x)
        result2 = mvn.log_partition_function(p)
        mean_p = mvn.to_mean(p)
        return result1, result2, mean_p

    # Test Categorical
    @jax.jit
    def categorical_test(params: jnp.ndarray, x: jnp.ndarray) -> tuple[Array, Array]:
        dist = Categorical(3)  # 3 categories
        p = dist.natural_point(params)
        result1 = dist.log_density(p, x)
        result2 = dist.log_partition_function(p)
        return result1, result2

    # Test Poisson
    @jax.jit
    def poisson_test(param: jnp.ndarray, x: jnp.ndarray) -> tuple[Array, Array]:
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
        print("✓ Gaussian operations are jit-compatible")
    except Exception as e:
        print("✗ Gaussian operations failed under jit:", str(e))

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
    def sample_test(key: jnp.ndarray, params: jnp.ndarray) -> Array:
        dist = Gaussian(2)
        p = dist.natural_point(params)
        return dist.sample(key, p, n=10)

    try:
        params = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0])
        key = random.PRNGKey(0)
        _ = sample_test(key, params)  # type: ignore
        print("✓ Sampling operations are jit-compatible")
    except Exception as e:
        print("✗ Sampling operations failed under jit:", str(e))


def test_mvn_conversions():
    """Test multivariate normal coordinate conversions under jit."""

    def mvn_conversion_test(
        mean: Array,
        covariance: Array,
        covariance_shape: Type[PositiveDefinite],
    ) -> tuple[Array, Array]:
        """Test conversion cycle: mean -> natural -> mean."""
        mvn = Gaussian(2, covariance_shape=covariance_shape)

        # Convert covariance matrix to appropriate operator
        cov_op = covariance_shape.from_matrix(covariance)

        # Test mean/covariance -> natural -> mean/covariance conversion
        mean_point = mvn.from_mean_and_covariance(mean, cov_op)
        natural_point = mvn.to_natural(mean_point)
        mean_point_back = mvn.to_mean(natural_point)

        # Extract parameters to compare
        mean_back, cov_back = mvn.to_mean_and_covariance(mean_point_back)

        return mean_back, cov_back.matrix

    mvn_conversion_test = jax.jit(
        mvn_conversion_test, static_argnames="covariance_shape"
    )

    try:
        # Test values that caused issues
        mean = jnp.array([1.0, -0.5])
        covariance = jnp.array([[2.0, 1.0], [1.0, 1.0]])

        # Test each covariance type
        for cov_type in [PositiveDefinite, Diagonal, Scale]:
            mean_back, cov_back = mvn_conversion_test(mean, covariance, cov_type)

            print(f"\n{cov_type.__name__} conversion test:")
            print(f"Original mean: {mean}")
            print(f"Recovered mean: {mean_back}")
            print(f"Original covariance:\n{covariance}")
            print(f"Recovered covariance:\n{cov_back}")

            # Check if values are close
            mean_close = jnp.allclose(mean, mean_back)
            cov_close = jnp.allclose(covariance, cov_back)

            if mean_close and cov_close:
                print(f"✓ {cov_type.__name__} conversions are accurate")
            else:
                print(f"✗ {cov_type.__name__} conversions show discrepancies")
                if not mean_close:
                    print("  Mean values differ")
                if not cov_close:
                    print("  Covariance values differ")

        print("\n✓ MVN conversions are jit-compatible")
    except Exception as e:
        print("\n✗ MVN conversions failed under jit:", str(e))

    def mvn_sampling_test(
        key: Array,
        mean: Array,
        covariance: Array,
        covariance_shape: Type[PositiveDefinite],
    ) -> Array:
        """Test sampling with different covariance types."""
        mvn = Gaussian(2, covariance_shape=covariance_shape)
        cov_op = covariance_shape.from_matrix(covariance)
        mean_point = mvn.from_mean_and_covariance(mean, cov_op)
        natural_point = mvn.to_natural(mean_point)
        return mvn.sample(key, natural_point, n=1000)

    try:
        # Test sampling with each covariance type
        key = random.PRNGKey(0)
        for cov_type in [PositiveDefinite, Diagonal, Scale]:
            sample = mvn_sampling_test(key, mean, covariance, cov_type)
            empirical_mean = jnp.mean(sample, axis=0)
            empirical_cov = jnp.cov(sample.T)

            print(f"\n{cov_type.__name__} sampling test:")
            print(f"Empirical mean: {empirical_mean}")
            print(f"Empirical covariance:\n{empirical_cov}")

            # Check if empirical values are reasonable
            mean_close = jnp.allclose(mean, empirical_mean, atol=0.1)
            cov_close = jnp.allclose(covariance, empirical_cov, atol=0.2)

            if mean_close and cov_close:
                print(f"✓ {cov_type.__name__} sampling looks correct")
            else:
                print(f"✗ {cov_type.__name__} sampling shows discrepancies")

        print("\n✓ MVN sampling is jit-compatible")
    except Exception as e:
        print("\n✗ MVN sampling failed under jit:", str(e))

    mvn_sampling_test = jax.jit(mvn_sampling_test)


if __name__ == "__main__":
    print("\nTesting JAX JIT compatibility...")
    print("-" * 40)

    test_point_creation()
    test_linear_operators()
    test_distributions()
    test_sampling()
    test_mvn_conversions()
