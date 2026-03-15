"""Tests for models/base/poisson.py.

Covers Poisson, Poissons, and CoMPoisson distributions. Verifies sufficient
statistics, density normalization, parameter conversions, the Poisson-CoMPoisson
relationship, and sampling.
"""

import jax
import jax.numpy as jnp
import pytest

from goal.models import CoMPoisson, Poisson, Poissons

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


class TestPoisson:
    """Test Poisson distribution properties."""

    def test_dimensions(self) -> None:
        model = Poisson()
        assert model.dim == 1
        assert model.data_dim == 1

    def test_sufficient_statistic(self) -> None:
        model = Poisson()
        for x in [0, 1, 5, 10]:
            s = model.sufficient_statistic(jnp.array(float(x)))
            assert jnp.allclose(s, jnp.array([float(x)]))

    def test_natural_param_is_log_rate(self) -> None:
        """Mean of Poisson equals rate = exp(theta)."""
        model = Poisson()
        for rate in [0.5, 1.0, 2.0, 5.0]:
            means = model.to_mean(jnp.array([jnp.log(rate)]))
            assert jnp.allclose(means[0], rate, rtol=RTOL, atol=ATOL)

    def test_density_normalizes(self) -> None:
        model = Poisson()
        params = model.initialize(jax.random.PRNGKey(42))
        xs = jnp.arange(500)
        total = jnp.sum(jax.vmap(model.density, in_axes=(None, 0))(params, xs))
        assert jnp.allclose(total, 1.0, rtol=1e-3)

    def test_mean_natural_round_trip(self) -> None:
        model = Poisson()
        params = model.initialize(jax.random.PRNGKey(42))
        recovered = model.to_natural(model.to_mean(params))
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_log_partition_is_rate(self) -> None:
        """For Poisson: log Z = exp(theta) = rate."""
        model = Poisson()
        for rate in [0.5, 1.0, 5.0]:
            params = jnp.array([jnp.log(rate)])
            assert jnp.allclose(
                model.log_partition_function(params), rate, rtol=RTOL, atol=ATOL
            )

    def test_sampling(self) -> None:
        """Samples are non-negative with empirical mean matching to_mean."""
        model = Poisson()
        params = jnp.array([jnp.log(3.0)])
        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert samples.shape == (50_000, 1)
        assert jnp.all(samples >= 0)

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.02)

    def test_negative_entropy(self) -> None:
        """Legendre identity: negative_entropy = -psi + dot(theta, eta)."""
        model = Poisson()
        for rate in [0.5, 1.0, 3.0]:
            params = jnp.array([jnp.log(rate)])
            means = model.to_mean(params)
            ne = model.negative_entropy(means)
            expected = -model.log_partition_function(params) + jnp.dot(params, means)
            assert jnp.allclose(ne, expected, rtol=RTOL, atol=ATOL)

    def test_initialized_params_valid(self) -> None:
        model = Poisson()
        params = model.initialize(jax.random.PRNGKey(42))
        assert model.check_natural_parameters(params)


class TestPoissons:
    """Test multivariate Poissons (product of independent Poissons)."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_dimensions(self, n: int) -> None:
        model = Poissons(n)
        assert model.dim == n
        assert model.data_dim == n

    def test_log_partition_is_sum(self) -> None:
        model = Poissons(3)
        params = jax.random.normal(jax.random.PRNGKey(42), (3,))
        assert jnp.allclose(
            model.log_partition_function(params),
            jnp.sum(jnp.exp(params)),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_sampling(self) -> None:
        """Samples are non-negative with empirical mean matching to_mean."""
        model = Poissons(3)
        params = jnp.array([0.5, 1.0, 1.5])  # rates ~ 1.6, 2.7, 4.5
        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert samples.shape == (50_000, 3)
        assert jnp.all(samples >= 0)

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.03)


class TestCoMPoisson:
    """Test Conway-Maxwell-Poisson distribution."""

    def test_dimensions(self) -> None:
        model = CoMPoisson()
        assert model.dim == 2
        assert model.data_dim == 1

    def test_sufficient_statistic(self) -> None:
        """Sufficient statistic is (x, log(x!))."""
        model = CoMPoisson()
        for x in [0, 1, 5]:
            s = model.sufficient_statistic(jnp.array(float(x)))
            assert s.shape == (2,)
            assert jnp.allclose(s[0], float(x))
            assert jnp.allclose(
                s[1], jax.scipy.special.gammaln(x + 1), rtol=RTOL, atol=ATOL
            )

    def test_density_normalizes(self) -> None:
        model = CoMPoisson()
        params = model.initialize(jax.random.PRNGKey(42))
        xs = jnp.arange(200)
        total = jnp.sum(jax.vmap(model.density, in_axes=(None, 0))(params, xs))
        assert jnp.allclose(total, 1.0, rtol=1e-2)

    def test_reduces_to_poisson(self) -> None:
        """CoMPoisson with dispersion=-1 matches standard Poisson."""
        model = CoMPoisson()
        poisson = Poisson()
        rate = 3.0
        xs = jnp.arange(20)
        com_densities = jax.vmap(model.density, in_axes=(None, 0))(
            jnp.array([jnp.log(rate), -1.0]), xs
        )
        pois_densities = jax.vmap(poisson.density, in_axes=(None, 0))(
            jnp.array([jnp.log(rate)]), xs
        )
        assert jnp.allclose(com_densities, pois_densities, rtol=1e-3, atol=1e-6)

    def test_to_mean_at_poisson_limit(self) -> None:
        """At dispersion=-1, CoMPoisson to_mean rate component matches Poisson mean."""
        model = CoMPoisson()
        poisson = Poisson()
        for rate in [1.0, 3.0, 5.0]:
            com_params = jnp.array([jnp.log(rate), -1.0])
            com_means = model.to_mean(com_params)
            pois_means = poisson.to_mean(jnp.array([jnp.log(rate)]))
            assert jnp.allclose(com_means[0], pois_means[0], rtol=1e-3, atol=1e-4)

    def test_sampling_nonnegative(self) -> None:
        model = CoMPoisson()
        params = model.initialize(jax.random.PRNGKey(42))
        samples = model.sample(jax.random.PRNGKey(0), params, 50_000)
        assert samples.shape == (50_000, 1)
        assert jnp.all(samples >= 0)
