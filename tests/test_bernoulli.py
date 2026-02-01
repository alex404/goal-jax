"""Tests for Bernoulli and Binomial distributions.

Tests the exponential family properties of binary and count distributions.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import Bernoulli, Bernoullis, Binomial, Binomials

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-4
ATOL = 1e-6


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestBernoulliBasics:
    """Test basic Bernoulli properties."""

    @pytest.fixture
    def model(self) -> Bernoulli:
        """Create Bernoulli model."""
        return Bernoulli()

    def test_dimensions(self, model: Bernoulli) -> None:
        """Test Bernoulli has correct dimensions."""
        assert model.dim == 1
        assert model.data_dim == 1

    def test_sufficient_statistic(self, model: Bernoulli) -> None:
        """Test sufficient statistic is identity."""
        assert jnp.allclose(
            model.sufficient_statistic(jnp.array(0.0)), jnp.array([0.0])
        )
        assert jnp.allclose(
            model.sufficient_statistic(jnp.array(1.0)), jnp.array([1.0])
        )

    def test_log_partition_matches_softplus(self, model: Bernoulli) -> None:
        """Test log partition function matches softplus."""
        theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        for theta in theta_values:
            params = jnp.array([theta])
            log_z = model.log_partition_function(params)
            expected = jax.nn.softplus(theta)
            assert jnp.allclose(log_z, expected, rtol=RTOL, atol=ATOL)


class TestBernoulliDensity:
    """Test Bernoulli density properties."""

    @pytest.fixture
    def model(self) -> Bernoulli:
        """Create Bernoulli model."""
        return Bernoulli()

    def test_density_normalizes(self, model: Bernoulli) -> None:
        """Test p(0) + p(1) = 1."""
        theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        for theta in theta_values:
            params = jnp.array([theta])
            p0 = model.density(params, jnp.array(0.0))
            p1 = model.density(params, jnp.array(1.0))
            assert jnp.allclose(p0 + p1, 1.0, rtol=RTOL, atol=ATOL)

    def test_mean_is_probability(self, model: Bernoulli) -> None:
        """Test that mean parameter equals P(x=1)."""
        theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        for theta in theta_values:
            params = jnp.array([theta])
            means = model.to_mean(params)
            expected_prob = jax.nn.sigmoid(theta)
            assert jnp.allclose(means[0], expected_prob, rtol=RTOL, atol=ATOL)


class TestBernoulliConversions:
    """Test Bernoulli parameter conversions."""

    @pytest.fixture
    def model(self) -> Bernoulli:
        """Create Bernoulli model."""
        return Bernoulli()

    def test_mean_natural_round_trip(self, model: Bernoulli) -> None:
        """Test natural -> mean -> natural round trip."""
        theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        for theta in theta_values:
            params = jnp.array([theta])
            means = model.to_mean(params)
            recovered = model.to_natural(means)
            assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_negative_entropy_at_half(self, model: Bernoulli) -> None:
        """Test negative entropy at p=0.5 equals -log(2)."""
        params = jnp.array([0.0])
        means = model.to_mean(params)
        neg_ent = model.negative_entropy(means)
        expected = -jnp.log(2.0)
        assert jnp.allclose(neg_ent, expected, rtol=RTOL, atol=ATOL)


class TestBernoulliSampling:
    """Test Bernoulli sampling."""

    @pytest.fixture
    def model(self) -> Bernoulli:
        """Create Bernoulli model."""
        return Bernoulli()

    def test_sampling_distribution(self, model: Bernoulli, key: Array) -> None:
        """Test sampling produces correct empirical mean."""
        theta = 1.0  # P(x=1) ~ 0.73
        params = jnp.array([theta])
        n_samples = 10000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, 1)
        assert jnp.all((samples == 0) | (samples == 1))

        empirical_mean = jnp.mean(samples)
        expected_prob = jax.nn.sigmoid(theta)
        assert jnp.allclose(empirical_mean, expected_prob, rtol=0.05, atol=0.02)


class TestBinomialBasics:
    """Test basic Binomial properties."""

    @pytest.fixture(params=[1, 5, 10])
    def model(self, request: pytest.FixtureRequest) -> Binomial:
        """Create Binomial with various n_trials."""
        return Binomial(n_trials=request.param)

    def test_dimensions(self, model: Binomial) -> None:
        """Test Binomial has correct dimensions."""
        assert model.dim == 1
        assert model.data_dim == 1

    def test_sufficient_statistic(self, model: Binomial) -> None:
        """Test sufficient statistic is identity (count)."""
        for x in [0, model.n_trials // 2, model.n_trials]:
            assert jnp.allclose(
                model.sufficient_statistic(jnp.array(float(x))), jnp.array([float(x)])
            )

    def test_log_partition_is_n_softplus(self, model: Binomial) -> None:
        """Test log partition function is n * softplus."""
        theta_values = jnp.array([-2.0, 0.0, 2.0])
        for theta in theta_values:
            params = jnp.array([theta])
            log_z = model.log_partition_function(params)
            expected = model.n_trials * jax.nn.softplus(theta)
            assert jnp.allclose(log_z, expected, rtol=RTOL, atol=ATOL)


class TestBinomialDensity:
    """Test Binomial density properties."""

    @pytest.fixture(params=[1, 5, 10])
    def model(self, request: pytest.FixtureRequest) -> Binomial:
        """Create Binomial with various n_trials."""
        return Binomial(n_trials=request.param)

    def test_density_normalizes(self, model: Binomial) -> None:
        """Test that pmf sums to 1."""
        theta_values = jnp.array([-1.0, 0.0, 1.0])
        for theta in theta_values:
            params = jnp.array([theta])
            xs = jnp.arange(model.n_trials + 1, dtype=jnp.float32)
            densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
            total = jnp.sum(densities)
            assert jnp.allclose(total, 1.0, rtol=1e-3, atol=ATOL)

    def test_mean_is_np(self, model: Binomial) -> None:
        """Test that mean parameter equals n*p."""
        theta_values = jnp.array([-2.0, 0.0, 2.0])
        for theta in theta_values:
            params = jnp.array([theta])
            means = model.to_mean(params)
            p = jax.nn.sigmoid(theta)
            expected_mean = model.n_trials * p
            assert jnp.allclose(means[0], expected_mean, rtol=RTOL, atol=ATOL)


class TestBinomialSampling:
    """Test Binomial sampling."""

    @pytest.fixture(params=[5, 10])
    def model(self, request: pytest.FixtureRequest) -> Binomial:
        """Create Binomial with various n_trials."""
        return Binomial(n_trials=request.param)

    def test_sampling_bounds(self, model: Binomial, key: Array) -> None:
        """Test samples are within [0, n_trials]."""
        theta = 0.0  # p = 0.5
        params = jnp.array([theta])
        n_samples = 2000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, 1)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= model.n_trials)


class TestBinomialBernoulliRelation:
    """Test relationship between Binomial and Bernoulli."""

    def test_binomial_one_equals_bernoulli(self) -> None:
        """Test that Binomial(1) is equivalent to Bernoulli."""
        binomial = Binomial(n_trials=1)
        bernoulli = Bernoulli()

        theta_values = jnp.array([-2.0, 0.0, 2.0])
        for theta in theta_values:
            params = jnp.array([theta])

            # Log partition should match
            assert jnp.allclose(
                binomial.log_partition_function(params),
                bernoulli.log_partition_function(params),
                rtol=RTOL,
                atol=ATOL,
            )

            # Means should match
            assert jnp.allclose(
                binomial.to_mean(params),
                bernoulli.to_mean(params),
                rtol=RTOL,
                atol=ATOL,
            )


class TestBernoullisBasics:
    """Test multivariate Bernoullis (product of independent Bernoullis)."""

    @pytest.fixture(params=[2, 3, 5])
    def model(self, request: pytest.FixtureRequest) -> Bernoullis:
        """Create Bernoullis of various sizes."""
        return Bernoullis(request.param)

    def test_dimensions(self, model: Bernoullis) -> None:
        """Test Bernoullis has correct dimensions."""
        n = model.n_neurons
        assert model.dim == n
        assert model.data_dim == n

    def test_log_partition_is_sum(self, model: Bernoullis, key: Array) -> None:
        """Test log partition is sum of individual softplus values."""
        n = model.n_neurons
        params = jax.random.normal(key, (n,))

        log_z = model.log_partition_function(params)
        expected = jnp.sum(jax.nn.softplus(params))
        assert jnp.allclose(log_z, expected, rtol=RTOL, atol=ATOL)

    def test_mean_natural_round_trip(self, model: Bernoullis, key: Array) -> None:
        """Test natural -> mean -> natural round trip."""
        n = model.n_neurons
        params = jax.random.normal(key, (n,))

        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)


class TestBernoullissampling:
    """Test Bernoullis sampling."""

    @pytest.fixture(params=[2, 3])
    def model(self, request: pytest.FixtureRequest) -> Bernoullis:
        """Create Bernoullis of various sizes."""
        return Bernoullis(request.param)

    def test_sampling_binary_values(self, model: Bernoullis, key: Array) -> None:
        """Test sampling produces binary values with correct marginals."""
        n = model.n_neurons
        params = jnp.zeros(n)  # All p=0.5

        n_samples = 5000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, n)
        assert jnp.all((samples == 0) | (samples == 1))

        # Each component should have empirical mean ~ 0.5
        empirical_means = jnp.mean(samples, axis=0)
        assert jnp.allclose(empirical_means, 0.5 * jnp.ones(n), rtol=0.1, atol=0.05)


class TestBinomialsBasics:
    """Test multivariate Binomials (product of independent Binomials)."""

    @pytest.fixture(params=[2, 3])
    def model(self, request: pytest.FixtureRequest) -> Binomials:
        """Create Binomials of various sizes."""
        return Binomials(request.param, n_trials=10)

    def test_dimensions(self, model: Binomials) -> None:
        """Test Binomials has correct dimensions."""
        n = model.n_neurons
        assert model.dim == n
        assert model.data_dim == n

    def test_log_partition_is_sum(self, model: Binomials, key: Array) -> None:
        """Test log partition is sum of individual n*softplus values."""
        n = model.n_neurons
        n_trials = model.n_trials
        params = jax.random.normal(key, (n,))

        log_z = model.log_partition_function(params)
        expected = jnp.sum(n_trials * jax.nn.softplus(params))
        assert jnp.allclose(log_z, expected, rtol=RTOL, atol=ATOL)

    def test_sampling_bounds(self, model: Binomials, key: Array) -> None:
        """Test samples are within [0, n_trials]."""
        n = model.n_neurons
        params = jnp.zeros(n)  # p=0.5 for all

        n_samples = 2000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, n)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= model.n_trials)
