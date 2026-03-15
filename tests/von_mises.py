"""Tests for models/base/von_mises.py.

Covers VonMises and VonMisesProduct. Verifies sufficient statistics, density
normalization and symmetry, concentration behavior, parameter conversions,
and sampling.
"""

import jax
import jax.numpy as jnp
import pytest

from goal.models import VonMises, VonMisesProduct

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


class TestVonMises:
    """Test VonMises distribution properties."""

    def test_dimensions(self) -> None:
        model = VonMises()
        assert model.dim == 2
        assert model.data_dim == 1

    def test_sufficient_statistic(self) -> None:
        """Sufficient statistic is (cos(x), sin(x))."""
        model = VonMises()
        for theta in [0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]:
            s = model.sufficient_statistic(jnp.asarray(theta))
            assert s.shape == (2,)
            assert jnp.allclose(s[0], jnp.cos(theta), rtol=RTOL, atol=ATOL)
            assert jnp.allclose(s[1], jnp.sin(theta), rtol=RTOL, atol=ATOL)

    def test_mean_concentration_split_join(self) -> None:
        model = VonMises()
        mu, kappa = jnp.pi / 4, 2.0
        params = model.join_mean_concentration(mu, kappa)
        recovered_mu, recovered_kappa = model.split_mean_concentration(params)
        assert jnp.allclose(recovered_mu, mu, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(recovered_kappa, kappa, rtol=RTOL, atol=ATOL)

    def test_density_normalizes(self) -> None:
        """Density integrates to 1 over [-pi, pi]."""
        model = VonMises()
        params = model.initialize(jax.random.PRNGKey(42))
        xs = jnp.linspace(-jnp.pi, jnp.pi, 1000)
        dx = 2 * jnp.pi / 1000
        total = jnp.sum(jax.vmap(model.density, in_axes=(None, 0))(params, xs)) * dx
        assert jnp.allclose(total, 1.0, rtol=1e-2)

    def test_density_symmetric(self) -> None:
        """Density centered at 0 is symmetric: f(x) = f(-x)."""
        model = VonMises()
        params = model.join_mean_concentration(0.0, 2.0)
        for theta in [0.5, 1.0, 1.5]:
            assert jnp.allclose(
                model.density(params, jnp.asarray(theta)),
                model.density(params, jnp.asarray(-theta)),
                rtol=RTOL,
                atol=ATOL,
            )

    def test_density_peaks_at_mean(self) -> None:
        model = VonMises()
        mu = jnp.pi / 3
        params = model.join_mean_concentration(mu, 3.0)
        d_at_mean = model.density(params, jnp.asarray(mu))
        for offset in [0.5, 1.0, jnp.pi]:
            assert d_at_mean > model.density(params, jnp.asarray(mu + offset))

    def test_uniform_at_zero_concentration(self) -> None:
        """kappa=0 gives uniform distribution on the circle."""
        model = VonMises()
        params = model.join_mean_concentration(0.0, 0.0)
        xs = jnp.linspace(-jnp.pi, jnp.pi, 10)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
        assert jnp.allclose(densities, 1.0 / (2 * jnp.pi), rtol=1e-2, atol=ATOL)

    def test_higher_concentration_sharper_peak(self) -> None:
        model = VonMises()
        params_low = model.join_mean_concentration(0.0, 1.0)
        params_high = model.join_mean_concentration(0.0, 5.0)
        # At mean: high kappa has higher density
        assert model.density(params_high, jnp.asarray(0.0)) > model.density(
            params_low, jnp.asarray(0.0)
        )
        # Away from mean: high kappa has lower density
        assert model.density(params_high, jnp.asarray(jnp.pi / 2)) < model.density(
            params_low, jnp.asarray(jnp.pi / 2)
        )

    def test_sampling(self) -> None:
        """Empirical sufficient statistics (cos, sin) match to_mean at 50k samples."""
        model = VonMises()
        mu = jnp.pi / 4
        params = model.join_mean_concentration(mu, 5.0)
        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert samples.shape == (50_000, 1)
        assert jnp.all(jnp.isfinite(samples))

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.01)

    def test_initialized_params_valid(self) -> None:
        model = VonMises()
        params = model.initialize(jax.random.PRNGKey(42))
        assert model.check_natural_parameters(params)


class TestVonMisesProduct:
    """Test VonMisesProduct (multivariate von Mises)."""

    @pytest.mark.parametrize("n", [2, 3])
    def test_dimensions(self, n: int) -> None:
        model = VonMisesProduct(n)
        assert model.dim == 2 * n
        assert model.data_dim == n

    def test_sufficient_statistic(self) -> None:
        model = VonMisesProduct(3)
        angles = jax.random.uniform(
            jax.random.PRNGKey(42), (3,), minval=-jnp.pi, maxval=jnp.pi
        )
        s = model.sufficient_statistic(angles)
        assert s.shape == (6,)

    def test_sampling(self) -> None:
        """Empirical sufficient statistics match to_mean at 50k samples."""
        model = VonMisesProduct(2)
        params = model.initialize(jax.random.PRNGKey(42))
        samples = model.sample(jax.random.PRNGKey(0), params, 50_000)
        assert samples.shape == (50_000, 2)
        assert jnp.all(jnp.isfinite(samples))

        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.01)
