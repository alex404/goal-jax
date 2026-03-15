"""Tests for models/base/categorical.py.

Covers Categorical, Bernoulli, and Bernoullis — all models defined in the
categorical module. Verifies sufficient statistics, density normalization,
parameter conversions, entropy, and sampling.
"""

import jax
import jax.numpy as jnp
import pytest

from goal.models import Bernoulli, Bernoullis, Categorical

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


# --- Categorical ---


class TestCategorical:
    """Test Categorical distribution properties."""

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_dimensions(self, n: int) -> None:
        model = Categorical(n_categories=n)
        assert model.dim == n - 1
        assert model.data_dim == 1

    def test_sufficient_statistic_one_hot(self) -> None:
        """Category 0 is reference (all zeros); others are one-hot."""
        model = Categorical(n_categories=5)
        assert jnp.allclose(model.sufficient_statistic(jnp.array(0)), jnp.zeros(4))
        for k in range(1, 5):
            s = model.sufficient_statistic(jnp.array(k))
            expected = jnp.zeros(4).at[k - 1].set(1.0)
            assert jnp.allclose(s, expected)

    @pytest.mark.parametrize("n", [3, 5])
    def test_density_normalizes(self, n: int) -> None:
        model = Categorical(n_categories=n)
        params = model.initialize(jax.random.PRNGKey(42))
        categories = jnp.arange(n)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, categories)
        assert jnp.allclose(jnp.sum(densities), 1.0, rtol=RTOL, atol=ATOL)

    def test_uniform_at_zero_params(self) -> None:
        n = 5
        model = Categorical(n_categories=n)
        params = jnp.zeros(n - 1)
        categories = jnp.arange(n)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, categories)
        assert jnp.allclose(densities, jnp.ones(n) / n, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("n", [3, 5])
    def test_mean_natural_round_trip(self, n: int) -> None:
        model = Categorical(n_categories=n)
        params = model.initialize(jax.random.PRNGKey(42))
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_mean_params_are_probabilities(self) -> None:
        model = Categorical(n_categories=5)
        params = model.initialize(jax.random.PRNGKey(42))
        means = model.to_mean(params)
        assert jnp.all(means >= 0)
        assert jnp.all(means <= 1)
        assert jnp.sum(means) < 1.0 + ATOL

    def test_sampling(self) -> None:
        """Samples are valid categories with empirical mean matching to_mean."""
        model = Categorical(n_categories=5)
        params = model.initialize(jax.random.PRNGKey(42))
        samples = model.sample(jax.random.PRNGKey(0), params, 50_000)
        assert samples.shape == (50_000, 1)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < 5)

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.01)

    def test_entropy(self) -> None:
        """Uniform distribution maximizes entropy; value equals -log(n)."""
        model = Categorical(n_categories=4)
        params_uniform = jnp.zeros(3)
        means_uniform = model.to_mean(params_uniform)
        neg_ent_uniform = model.negative_entropy(means_uniform)

        params_biased = jnp.array([2.0, 0.0, -1.0])
        neg_ent_biased = model.negative_entropy(model.to_mean(params_biased))
        assert neg_ent_uniform < neg_ent_biased

        expected = -jnp.log(4.0)
        assert jnp.allclose(neg_ent_uniform, expected, rtol=RTOL, atol=ATOL)

    def test_initialized_params_valid(self) -> None:
        model = Categorical(n_categories=5)
        params = model.initialize(jax.random.PRNGKey(42))
        assert model.check_natural_parameters(params)


# --- Bernoulli ---


class TestBernoulli:
    """Test Bernoulli distribution (binary categorical)."""

    def test_dimensions(self) -> None:
        model = Bernoulli()
        assert model.dim == 1
        assert model.data_dim == 1

    def test_sufficient_statistic(self) -> None:
        model = Bernoulli()
        assert jnp.allclose(model.sufficient_statistic(jnp.array(0.0)), jnp.array([0.0]))
        assert jnp.allclose(model.sufficient_statistic(jnp.array(1.0)), jnp.array([1.0]))

    def test_log_partition_matches_softplus(self) -> None:
        model = Bernoulli()
        for theta in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            params = jnp.array([theta])
            assert jnp.allclose(
                model.log_partition_function(params),
                jax.nn.softplus(theta),
                rtol=RTOL,
                atol=ATOL,
            )

    def test_density_normalizes(self) -> None:
        """p(0) + p(1) = 1 for various parameter values."""
        model = Bernoulli()
        for theta in [-2.0, 0.0, 2.0]:
            params = jnp.array([theta])
            total = model.density(params, jnp.array(0.0)) + model.density(
                params, jnp.array(1.0)
            )
            assert jnp.allclose(total, 1.0, rtol=RTOL, atol=ATOL)

    def test_mean_is_probability(self) -> None:
        """Mean parameter equals sigmoid(theta) = P(x=1)."""
        model = Bernoulli()
        for theta in [-2.0, 0.0, 2.0]:
            params = jnp.array([theta])
            means = model.to_mean(params)
            assert jnp.allclose(means[0], jax.nn.sigmoid(theta), rtol=RTOL, atol=ATOL)

    def test_mean_natural_round_trip(self) -> None:
        model = Bernoulli()
        for theta in [-2.0, 0.0, 2.0]:
            params = jnp.array([theta])
            recovered = model.to_natural(model.to_mean(params))
            assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_negative_entropy_at_half(self) -> None:
        """At p=0.5 (theta=0), negative entropy equals -log(2)."""
        model = Bernoulli()
        means = model.to_mean(jnp.array([0.0]))
        assert jnp.allclose(
            model.negative_entropy(means), -jnp.log(2.0), rtol=RTOL, atol=ATOL
        )

    def test_sampling(self) -> None:
        """Samples are binary with empirical sufficient statistics matching to_mean."""
        model = Bernoulli()
        params = jnp.array([1.0])  # P(x=1) ~ 0.73
        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert samples.shape == (50_000, 1)
        assert jnp.all((samples == 0) | (samples == 1))

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.01)


# --- Bernoullis ---


class TestBernoullis:
    """Test multivariate Bernoullis (product of independent Bernoullis)."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_dimensions(self, n: int) -> None:
        model = Bernoullis(n)
        assert model.dim == n
        assert model.data_dim == n

    def test_log_partition_is_sum(self) -> None:
        """Log partition is sum of individual softplus values."""
        model = Bernoullis(3)
        params = jax.random.normal(jax.random.PRNGKey(42), (3,))
        assert jnp.allclose(
            model.log_partition_function(params),
            jnp.sum(jax.nn.softplus(params)),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_mean_natural_round_trip(self) -> None:
        model = Bernoullis(3)
        params = jax.random.normal(jax.random.PRNGKey(42), (3,))
        recovered = model.to_natural(model.to_mean(params))
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_sampling(self) -> None:
        """Samples are binary with empirical sufficient statistics matching to_mean."""
        model = Bernoullis(3)
        params = jnp.zeros(3)  # All p=0.5
        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert samples.shape == (50_000, 3)
        assert jnp.all((samples == 0) | (samples == 1))

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.01)
