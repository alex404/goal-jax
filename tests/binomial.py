"""Tests for models/base/binomial.py.

Covers Binomial and Binomials distributions. Verifies sufficient statistics,
density normalization, log partition function, mean computation, sampling
bounds, and the Binomial(1) == Bernoulli identity.
"""

import jax
import jax.numpy as jnp
import pytest

from goal.models import Bernoulli, Binomial, Binomials

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


class TestBinomial:
    """Test Binomial distribution properties."""

    @pytest.mark.parametrize("n_trials", [1, 5, 10])
    def test_dimensions(self, n_trials: int) -> None:
        model = Binomial(n_trials=n_trials)
        assert model.dim == 1
        assert model.data_dim == 1

    @pytest.mark.parametrize("n_trials", [1, 5, 10])
    def test_log_partition_is_n_softplus(self, n_trials: int) -> None:
        model = Binomial(n_trials=n_trials)
        for theta in [-2.0, 0.0, 2.0]:
            params = jnp.array([theta])
            expected = n_trials * jax.nn.softplus(theta)
            assert jnp.allclose(
                model.log_partition_function(params), expected, rtol=RTOL, atol=ATOL
            )

    @pytest.mark.parametrize("n_trials", [5, 10])
    def test_density_normalizes(self, n_trials: int) -> None:
        model = Binomial(n_trials=n_trials)
        for theta in [-1.0, 0.0, 1.0]:
            params = jnp.array([theta])
            xs = jnp.arange(n_trials + 1, dtype=jnp.float32)
            densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
            assert jnp.allclose(jnp.sum(densities), 1.0, rtol=1e-3, atol=ATOL)

    @pytest.mark.parametrize("n_trials", [5, 10])
    def test_mean_is_np(self, n_trials: int) -> None:
        model = Binomial(n_trials=n_trials)
        for theta in [-2.0, 0.0, 2.0]:
            params = jnp.array([theta])
            means = model.to_mean(params)
            assert jnp.allclose(
                means[0], n_trials * jax.nn.sigmoid(theta), rtol=RTOL, atol=ATOL
            )

    def test_sampling(self) -> None:
        """Samples within [0, n] with empirical mean matching to_mean."""
        model = Binomial(n_trials=10)
        params = jnp.array([0.5])  # p ~ 0.62
        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert samples.shape == (50_000, 1)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= 10)

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.02)

    @pytest.mark.parametrize("n_trials", [5, 10])
    def test_negative_entropy(self, n_trials: int) -> None:
        """Legendre identity: negative_entropy = -psi + dot(theta, eta)."""
        model = Binomial(n_trials=n_trials)
        for theta in [-1.0, 0.0, 1.0]:
            params = jnp.array([theta])
            means = model.to_mean(params)
            ne = model.negative_entropy(means)
            expected = -model.log_partition_function(params) + jnp.dot(params, means)
            assert jnp.allclose(ne, expected, rtol=RTOL, atol=ATOL)

    def test_binomial_one_equals_bernoulli(self) -> None:
        """Binomial(1) matches Bernoulli in log-partition and means."""
        binomial = Binomial(n_trials=1)
        bernoulli = Bernoulli()
        for theta in [-2.0, 0.0, 2.0]:
            params = jnp.array([theta])
            assert jnp.allclose(
                binomial.log_partition_function(params),
                bernoulli.log_partition_function(params),
                rtol=RTOL,
                atol=ATOL,
            )
            assert jnp.allclose(
                binomial.to_mean(params), bernoulli.to_mean(params), rtol=RTOL, atol=ATOL
            )


class TestBinomials:
    """Test multivariate Binomials (product of independent Binomials)."""

    @pytest.mark.parametrize("n", [2, 3])
    def test_dimensions(self, n: int) -> None:
        model = Binomials(n, n_trials=10)
        assert model.dim == n
        assert model.data_dim == n

    def test_log_partition_is_sum(self) -> None:
        model = Binomials(3, n_trials=10)
        params = jax.random.normal(jax.random.PRNGKey(42), (3,))
        assert jnp.allclose(
            model.log_partition_function(params),
            jnp.sum(10 * jax.nn.softplus(params)),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_sampling(self) -> None:
        """Samples within [0, n] with empirical mean matching to_mean."""
        model = Binomials(3, n_trials=10)
        params = jnp.zeros(3)  # p=0.5, mean=5 for each
        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert samples.shape == (50_000, 3)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= 10)

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.02)
