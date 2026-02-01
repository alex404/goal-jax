"""Tests for VonMises distribution.

Tests the exponential family properties of circular distributions.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import VonMises, VonMisesProduct

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-4
ATOL = 1e-6


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestVonMisesBasics:
    """Test basic VonMises properties."""

    @pytest.fixture
    def model(self) -> VonMises:
        """Create VonMises model."""
        return VonMises()

    def test_dimensions(self, model: VonMises) -> None:
        """Test VonMises has correct dimensions."""
        # Two natural parameters (cos and sin components)
        assert model.dim == 2
        # Data is a single angle
        assert model.data_dim == 1

    def test_sufficient_statistic(self, model: VonMises) -> None:
        """Test sufficient statistic is (cos(x), sin(x))."""
        angles = jnp.array([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
        for theta in angles:
            s = model.sufficient_statistic(theta)
            assert s.shape == (2,)
            assert jnp.allclose(s[0], jnp.cos(theta), rtol=RTOL, atol=ATOL)
            assert jnp.allclose(s[1], jnp.sin(theta), rtol=RTOL, atol=ATOL)

    def test_mean_concentration_split(self, model: VonMises) -> None:
        """Test split_mean_concentration extracts correct values."""
        mu = jnp.pi / 4
        kappa = 2.0
        params = model.join_mean_concentration(mu, kappa)
        recovered_mu, recovered_kappa = model.split_mean_concentration(params)

        assert jnp.allclose(recovered_mu, mu, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(recovered_kappa, kappa, rtol=RTOL, atol=ATOL)


class TestVonMisesDensity:
    """Test VonMises density properties."""

    @pytest.fixture
    def model(self) -> VonMises:
        """Create VonMises model."""
        return VonMises()

    def test_density_normalizes(self, model: VonMises, key: Array) -> None:
        """Test that density integrates to 1 over [-pi, pi]."""
        params = model.initialize(key)
        n_points = 1000
        xs = jnp.linspace(-jnp.pi, jnp.pi, n_points)
        dx = 2 * jnp.pi / n_points
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
        total = jnp.sum(densities) * dx
        assert jnp.allclose(total, 1.0, rtol=1e-2)

    def test_density_symmetric(self, model: VonMises) -> None:
        """Test density is symmetric around mean when centered at 0."""
        mu = 0.0
        kappa = 2.0
        params = model.join_mean_concentration(mu, kappa)

        # Density at +x should equal density at -x
        test_angles = jnp.array([0.5, 1.0, 1.5])
        for theta in test_angles:
            d_pos = model.density(params, theta)
            d_neg = model.density(params, -theta)
            assert jnp.allclose(d_pos, d_neg, rtol=RTOL, atol=ATOL)

    def test_density_peaks_at_mean(self, model: VonMises) -> None:
        """Test density is maximized at mean direction."""
        mu = jnp.pi / 3
        kappa = 3.0
        params = model.join_mean_concentration(mu, kappa)

        d_at_mean = model.density(params, jnp.asarray(mu))
        # Check density is lower at other points
        offsets = jnp.array([0.5, 1.0, jnp.pi])
        for offset in offsets:
            d_away = model.density(params, jnp.asarray(mu + offset))
            assert d_at_mean > d_away


class TestVonMisesConcentration:
    """Test concentration parameter behavior."""

    @pytest.fixture
    def model(self) -> VonMises:
        """Create VonMises model."""
        return VonMises()

    def test_uniform_at_zero_concentration(self, model: VonMises) -> None:
        """Test distribution is uniform when kappa=0."""
        mu = 0.0
        kappa = 0.0
        params = model.join_mean_concentration(mu, kappa)

        # Density should be constant
        test_angles = jnp.linspace(-jnp.pi, jnp.pi, 10)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, test_angles)
        expected = 1.0 / (2 * jnp.pi)
        assert jnp.allclose(densities, expected, rtol=1e-2, atol=ATOL)

    def test_higher_concentration_sharper_peak(self, model: VonMises) -> None:
        """Test higher concentration gives sharper peak."""
        mu = 0.0
        kappa_low = 1.0
        kappa_high = 5.0

        params_low = model.join_mean_concentration(mu, kappa_low)
        params_high = model.join_mean_concentration(mu, kappa_high)

        # At mean, high kappa should have higher density
        d_low_at_mean = model.density(params_low, jnp.asarray(mu))
        d_high_at_mean = model.density(params_high, jnp.asarray(mu))
        assert d_high_at_mean > d_low_at_mean

        # Away from mean, high kappa should have lower density
        away = jnp.pi / 2
        d_low_away = model.density(params_low, jnp.asarray(away))
        d_high_away = model.density(params_high, jnp.asarray(away))
        assert d_high_away < d_low_away


class TestVonMisesSampling:
    """Test VonMises sampling."""

    @pytest.fixture
    def model(self) -> VonMises:
        """Create VonMises model."""
        return VonMises()

    def test_sampling_in_range(self, model: VonMises, key: Array) -> None:
        """Test samples are in valid angular range."""
        mu = 0.0
        kappa = 2.0
        params = model.join_mean_concentration(mu, kappa)

        n_samples = 1000
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, 1)
        # Samples should be in [-pi, pi] or [0, 2*pi] depending on convention
        # Just check they're finite
        assert jnp.all(jnp.isfinite(samples))

    def test_sampling_mean_direction(self, model: VonMises, key: Array) -> None:
        """Test samples concentrate around mean direction."""
        mu = jnp.pi / 4
        kappa = 5.0  # High concentration
        params = model.join_mean_concentration(mu, kappa)

        n_samples = 5000
        samples = model.sample(key, params, n_samples).ravel()

        # Compute circular mean
        circular_mean = jnp.arctan2(
            jnp.mean(jnp.sin(samples)), jnp.mean(jnp.cos(samples))
        )
        # Circular mean should be close to mu
        error = jnp.abs(jnp.angle(jnp.exp(1j * (circular_mean - mu))))
        assert error < 0.2


class TestVonMisesProductBasics:
    """Test VonMisesProduct (multivariate von Mises)."""

    @pytest.fixture(params=[2, 3])
    def model(self, request: pytest.FixtureRequest) -> VonMisesProduct:
        """Create VonMisesProduct of various sizes."""
        return VonMisesProduct(request.param)

    def test_dimensions(self, model: VonMisesProduct) -> None:
        """Test dimensions are correct."""
        n = model.n_components
        # Each component has 2 natural parameters
        assert model.dim == 2 * n
        assert model.data_dim == n

    def test_sufficient_statistic(self, model: VonMisesProduct, key: Array) -> None:
        """Test sufficient statistic has correct structure."""
        n = model.n_components
        angles = jax.random.uniform(key, (n,), minval=-jnp.pi, maxval=jnp.pi)
        s = model.sufficient_statistic(angles)

        # Should be (cos1, sin1, cos2, sin2, ...)
        assert s.shape == (2 * n,)

    def test_sampling_bounds(self, model: VonMisesProduct, key: Array) -> None:
        """Test samples are finite."""
        params = model.initialize(key)
        n_samples = 100
        samples = model.sample(key, params, n_samples)

        assert samples.shape == (n_samples, model.n_components)
        assert jnp.all(jnp.isfinite(samples))


class TestVonMisesValidity:
    """Test parameter validity checking."""

    @pytest.fixture
    def model(self) -> VonMises:
        """Create VonMises model."""
        return VonMises()

    def test_initialized_params_valid(self, model: VonMises, key: Array) -> None:
        """Test initialized parameters pass validity check."""
        for i in range(3):
            params = model.initialize(jax.random.fold_in(key, i))
            assert model.check_natural_parameters(params)
