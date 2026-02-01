"""Tests for Poisson and CoMPoisson distributions.

Tests the exponential family properties of count distributions.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import CoMPoisson, Poisson, Poissons

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-4
ATOL = 1e-6


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestPoissonBasics:
    """Test basic Poisson properties."""

    @pytest.fixture
    def model(self) -> Poisson:
        """Create Poisson model."""
        return Poisson()

    def test_dimensions(self, model: Poisson) -> None:
        """Test Poisson has correct dimensions."""
        assert model.dim == 1
        assert model.data_dim == 1

    def test_sufficient_statistic(self, model: Poisson) -> None:
        """Test sufficient statistic is identity."""
        for x in [0, 1, 5, 10]:
            s = model.sufficient_statistic(jnp.array(float(x)))
            assert jnp.allclose(s, jnp.array([float(x)]))

    def test_natural_param_is_log_rate(self, model: Poisson) -> None:
        """Test natural parameter is log of rate."""
        rates = jnp.array([0.5, 1.0, 2.0, 5.0])
        for rate in rates:
            params = jnp.array([jnp.log(rate)])
            means = model.to_mean(params)
            # Mean of Poisson equals rate
            assert jnp.allclose(means[0], rate, rtol=RTOL, atol=ATOL)


class TestPoissonDensity:
    """Test Poisson density properties."""

    @pytest.fixture
    def model(self) -> Poisson:
        """Create Poisson model."""
        return Poisson()

    def test_density_normalizes(self, model: Poisson, key: Array) -> None:
        """Test that pmf sums to 1 (approximately, over finite range)."""
        params = model.initialize(key)
        # Sum over large enough range
        xs = jnp.arange(500)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
        total = jnp.sum(densities)
        assert jnp.allclose(total, 1.0, rtol=1e-3)

    def test_density_positive(self, model: Poisson, key: Array) -> None:
        """Test densities are positive."""
        params = model.initialize(key)
        xs = jnp.arange(50)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
        assert jnp.all(densities >= 0)


class TestPoissonConversions:
    """Test Poisson parameter conversions."""

    @pytest.fixture
    def model(self) -> Poisson:
        """Create Poisson model."""
        return Poisson()

    def test_mean_natural_round_trip(self, model: Poisson, key: Array) -> None:
        """Test natural -> mean -> natural round trip."""
        for i in range(3):
            params = model.initialize(jax.random.fold_in(key, i))
            means = model.to_mean(params)
            recovered = model.to_natural(means)
            assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_log_partition_is_rate(self, model: Poisson) -> None:
        """Test log partition function equals the rate."""
        rates = jnp.array([0.5, 1.0, 2.0, 5.0])
        for rate in rates:
            params = jnp.array([jnp.log(rate)])
            log_z = model.log_partition_function(params)
            # For Poisson: log Z = exp(theta) = rate
            assert jnp.allclose(log_z, rate, rtol=RTOL, atol=ATOL)


class TestPoissonSampling:
    """Test Poisson sampling."""

    @pytest.fixture
    def model(self) -> Poisson:
        """Create Poisson model."""
        return Poisson()

    def test_sampling_nonnegative(self, model: Poisson, key: Array) -> None:
        """Test samples are non-negative integers."""
        rate = 5.0
        params = jnp.array([jnp.log(rate)])
        n_samples = 2000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, 1)
        assert jnp.all(samples >= 0)

    def test_sampling_mean(self, model: Poisson, key: Array) -> None:
        """Test sample mean approximates rate."""
        rate = 3.0
        params = jnp.array([jnp.log(rate)])
        n_samples = 10000
        samples = model.sample(key, params, n_samples)

        empirical_mean = jnp.mean(samples)
        assert jnp.allclose(empirical_mean, rate, rtol=0.1)


class TestPoissonsBasics:
    """Test multivariate Poissons (product of independent Poissons)."""

    @pytest.fixture(params=[2, 3, 5])
    def model(self, request: pytest.FixtureRequest) -> Poissons:
        """Create Poissons of various sizes."""
        return Poissons(request.param)

    def test_dimensions(self, model: Poissons) -> None:
        """Test Poissons has correct dimensions."""
        n = model.n_neurons
        assert model.dim == n
        assert model.data_dim == n

    def test_log_partition_is_sum(self, model: Poissons, key: Array) -> None:
        """Test log partition is sum of rates."""
        n = model.n_neurons
        params = jax.random.normal(key, (n,))

        log_z = model.log_partition_function(params)
        expected = jnp.sum(jnp.exp(params))
        assert jnp.allclose(log_z, expected, rtol=RTOL, atol=ATOL)

    def test_sampling_bounds(self, model: Poissons, key: Array) -> None:
        """Test samples are non-negative."""
        n = model.n_neurons
        params = jnp.zeros(n)  # rate = 1 for all

        n_samples = 1000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, n)
        assert jnp.all(samples >= 0)


class TestCoMPoissonBasics:
    """Test Conway-Maxwell-Poisson distribution basics."""

    @pytest.fixture
    def model(self) -> CoMPoisson:
        """Create CoMPoisson model."""
        return CoMPoisson()

    def test_dimensions(self, model: CoMPoisson) -> None:
        """Test CoMPoisson has correct dimensions."""
        assert model.dim == 2  # Two parameters: rate and dispersion
        assert model.data_dim == 1

    def test_sufficient_statistic(self, model: CoMPoisson) -> None:
        """Test sufficient statistic is (x, log(x!))."""
        for x in [0, 1, 5]:
            s = model.sufficient_statistic(jnp.array(float(x)))
            assert s.shape == (2,)
            # First component is x
            assert jnp.allclose(s[0], float(x))
            # Second component is log(x!)
            expected_log_fact = jax.scipy.special.gammaln(x + 1)
            assert jnp.allclose(s[1], expected_log_fact, rtol=RTOL, atol=ATOL)


class TestCoMPoissonDensity:
    """Test CoMPoisson density properties."""

    @pytest.fixture
    def model(self) -> CoMPoisson:
        """Create CoMPoisson model."""
        return CoMPoisson()

    def test_density_normalizes(self, model: CoMPoisson, key: Array) -> None:
        """Test that pmf sums to 1 (approximately)."""
        params = model.initialize(key)
        xs = jnp.arange(200)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
        total = jnp.sum(densities)
        assert jnp.allclose(total, 1.0, rtol=1e-2)

    def test_reduces_to_poisson(self, model: CoMPoisson) -> None:
        """Test CoMPoisson reduces to Poisson when dispersion parameter is -1."""
        # Natural params: (log(rate), dispersion)
        # When dispersion = -1, CoMPoisson becomes Poisson
        poisson = Poisson()
        rate = 3.0

        com_params = jnp.array([jnp.log(rate), -1.0])
        pois_params = jnp.array([jnp.log(rate)])

        xs = jnp.arange(20)
        com_densities = jax.vmap(model.density, in_axes=(None, 0))(com_params, xs)
        pois_densities = jax.vmap(poisson.density, in_axes=(None, 0))(pois_params, xs)

        # Should be approximately equal (numerical precision limits)
        assert jnp.allclose(com_densities, pois_densities, rtol=1e-3, atol=1e-6)


class TestCoMPoissonSampling:
    """Test CoMPoisson sampling."""

    @pytest.fixture
    def model(self) -> CoMPoisson:
        """Create CoMPoisson model."""
        return CoMPoisson()

    def test_sampling_nonnegative(self, model: CoMPoisson, key: Array) -> None:
        """Test samples are non-negative."""
        params = model.initialize(key)
        n_samples = 1000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, 1)
        assert jnp.all(samples >= 0)


class TestPoissonValidity:
    """Test parameter validity checking."""

    @pytest.fixture
    def model(self) -> Poisson:
        """Create Poisson model."""
        return Poisson()

    def test_initialized_params_valid(self, model: Poisson, key: Array) -> None:
        """Test initialized parameters pass validity check."""
        for i in range(3):
            params = model.initialize(jax.random.fold_in(key, i))
            assert model.check_natural_parameters(params)
