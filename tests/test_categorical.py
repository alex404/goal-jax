"""Tests for Categorical distribution.

Tests the exponential family properties of categorical/multinomial distributions.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import Categorical

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-4
ATOL = 1e-6


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestCategoricalBasics:
    """Test basic Categorical properties."""

    @pytest.fixture(params=[3, 5, 7])
    def model(self, request: pytest.FixtureRequest) -> Categorical:
        """Create Categorical with various number of categories."""
        return Categorical(n_categories=request.param)

    def test_dimensions(self, model: Categorical) -> None:
        """Test Categorical has correct dimensions."""
        # Natural params have n-1 free parameters
        assert model.dim == model.n_categories - 1
        # Data dimension is 1 (category index)
        assert model.data_dim == 1

    def test_sufficient_statistic_shape(self, model: Categorical) -> None:
        """Test sufficient statistic has correct shape."""
        for k in range(model.n_categories):
            s = model.sufficient_statistic(jnp.array(k))
            assert s.shape == (model.n_categories - 1,)

    def test_sufficient_statistic_one_hot(self, model: Categorical) -> None:
        """Test sufficient statistic is one-hot encoding (category 0 is reference)."""
        # Category 0 is the reference category - gives all zeros
        s_ref = model.sufficient_statistic(jnp.array(0))
        assert jnp.allclose(s_ref, jnp.zeros(model.n_categories - 1))

        # Categories 1 to n-1 have 1 at position k-1
        for k in range(1, model.n_categories):
            s = model.sufficient_statistic(jnp.array(k))
            expected = jnp.zeros(model.n_categories - 1).at[k - 1].set(1.0)
            assert jnp.allclose(s, expected)


class TestCategoricalDensity:
    """Test Categorical density properties."""

    @pytest.fixture(params=[3, 5])
    def model(self, request: pytest.FixtureRequest) -> Categorical:
        """Create Categorical with various number of categories."""
        return Categorical(n_categories=request.param)

    def test_density_normalizes(self, model: Categorical, key: Array) -> None:
        """Test that pmf sums to 1."""
        params = model.initialize(key)
        categories = jnp.arange(model.n_categories)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, categories)
        total = jnp.sum(densities)
        assert jnp.allclose(total, 1.0, rtol=RTOL, atol=ATOL)

    def test_uniform_at_zero_params(self, model: Categorical) -> None:
        """Test uniform distribution when all natural parameters are zero."""
        params = jnp.zeros(model.n_categories - 1)
        categories = jnp.arange(model.n_categories)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, categories)

        # All probabilities should be 1/n
        expected = jnp.ones(model.n_categories) / model.n_categories
        assert jnp.allclose(densities, expected, rtol=RTOL, atol=ATOL)


class TestCategoricalConversions:
    """Test Categorical parameter conversions."""

    @pytest.fixture(params=[3, 5])
    def model(self, request: pytest.FixtureRequest) -> Categorical:
        """Create Categorical with various number of categories."""
        return Categorical(n_categories=request.param)

    def test_mean_natural_round_trip(self, model: Categorical, key: Array) -> None:
        """Test natural -> mean -> natural round trip."""
        params = model.initialize(key)
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_mean_params_are_probabilities(self, model: Categorical, key: Array) -> None:
        """Test that mean parameters sum to less than 1 (missing last category)."""
        params = model.initialize(key)
        means = model.to_mean(params)

        # Mean params represent probabilities of first n-1 categories
        assert jnp.all(means >= 0)
        assert jnp.all(means <= 1)
        assert jnp.sum(means) < 1.0 + ATOL  # Sum should be < 1


class TestCategoricalSampling:
    """Test Categorical sampling."""

    @pytest.fixture(params=[3, 5])
    def model(self, request: pytest.FixtureRequest) -> Categorical:
        """Create Categorical with various number of categories."""
        return Categorical(n_categories=request.param)

    def test_sampling_valid_categories(self, model: Categorical, key: Array) -> None:
        """Test samples are valid category indices."""
        params = model.initialize(key)
        n_samples = 2000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, 1)
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < model.n_categories)

    def test_sampling_distribution(self, model: Categorical, key: Array) -> None:
        """Test sampling approximates correct distribution."""
        # Create biased distribution
        n = model.n_categories
        params = jnp.zeros(n - 1).at[0].set(2.0)  # Category 0 more likely

        n_samples = 10000
        samples = model.sample(key, params, n_samples)

        # Compute empirical frequencies
        counts = jnp.zeros(n)
        for k in range(n):
            counts = counts.at[k].set(jnp.sum(samples == k))
        empirical_probs = counts / n_samples

        # Compute expected probabilities
        categories = jnp.arange(n)
        expected_probs = jax.vmap(model.density, in_axes=(None, 0))(params, categories)

        # Check empirical matches expected
        assert jnp.allclose(empirical_probs, expected_probs, rtol=0.1, atol=0.05)


class TestCategoricalEntropy:
    """Test Categorical entropy properties."""

    @pytest.fixture
    def model(self) -> Categorical:
        """Create Categorical model."""
        return Categorical(n_categories=4)

    def test_max_entropy_at_uniform(self, model: Categorical) -> None:
        """Test entropy is maximized at uniform distribution."""
        # Uniform distribution
        params_uniform = jnp.zeros(model.n_categories - 1)
        means_uniform = model.to_mean(params_uniform)
        neg_ent_uniform = model.negative_entropy(means_uniform)

        # Non-uniform distribution
        params_biased = jnp.array([2.0, 0.0, -1.0])
        means_biased = model.to_mean(params_biased)
        neg_ent_biased = model.negative_entropy(means_biased)

        # Uniform should have lower negative entropy (higher entropy)
        assert neg_ent_uniform < neg_ent_biased

    def test_negative_entropy_value_at_uniform(self, model: Categorical) -> None:
        """Test negative entropy equals -log(n) at uniform."""
        params = jnp.zeros(model.n_categories - 1)
        means = model.to_mean(params)
        neg_ent = model.negative_entropy(means)
        expected = -jnp.log(model.n_categories)
        assert jnp.allclose(neg_ent, expected, rtol=RTOL, atol=ATOL)


class TestCategoricalValidity:
    """Test parameter validity checking."""

    @pytest.fixture
    def model(self) -> Categorical:
        """Create Categorical model."""
        return Categorical(n_categories=5)

    def test_initialized_params_valid(self, model: Categorical, key: Array) -> None:
        """Test initialized parameters pass validity check."""
        for i in range(3):
            params = model.initialize(jax.random.fold_in(key, i))
            assert model.check_natural_parameters(params)
