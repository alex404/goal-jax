"""Tests for models/base/dirichlet.py.

Covers the Dirichlet exponential family: dimensions, sufficient statistic,
log-partition function, natural-to-mean conversion, density evaluation, and
sampling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import digamma

from goal.models import Dirichlet

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


class TestDirichlet:
    """Test Dirichlet distribution properties."""

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_dimensions(self, k: int) -> None:
        model = Dirichlet(n_categories=k)
        assert model.dim == k
        assert model.data_dim == k

    def test_sufficient_statistic_is_elementwise_log(self) -> None:
        model = Dirichlet(n_categories=4)
        x = jnp.array([0.1, 0.2, 0.3, 0.4])
        assert jnp.allclose(model.sufficient_statistic(x), jnp.log(x))

    def test_log_partition_matches_scipy_gammaln(self) -> None:
        """psi(alpha) = sum(gammaln(alpha_i)) - gammaln(sum(alpha)), checked against scipy."""
        from scipy.special import gammaln

        model = Dirichlet(n_categories=3)
        alpha = jnp.array([3.0, 7.0, 5.0])
        expected = float(gammaln(np.asarray(alpha)).sum() - gammaln(float(alpha.sum())))
        assert jnp.allclose(
            model.log_partition_function(alpha), expected, rtol=RTOL, atol=ATOL
        )

    def test_to_mean_matches_digamma_formula(self) -> None:
        """E[log x_i] = digamma(alpha_i) - digamma(sum alpha)."""
        model = Dirichlet(n_categories=3)
        alpha = jnp.array([3.0, 7.0, 5.0])
        expected = digamma(alpha) - digamma(jnp.sum(alpha))
        assert jnp.allclose(model.to_mean(alpha), expected, rtol=RTOL, atol=ATOL)

    def test_log_density_matches_standard_formula(self) -> None:
        """log p(x) = sum((alpha_i - 1) log x_i) - log B(alpha)."""
        model = Dirichlet(n_categories=3)
        alpha = jnp.array([3.0, 7.0, 5.0])
        x = jnp.array([0.2, 0.5, 0.3])
        log_beta = jnp.sum(jax.lax.lgamma(alpha)) - jax.lax.lgamma(jnp.sum(alpha))
        expected = jnp.sum((alpha - 1) * jnp.log(x)) - log_beta
        assert jnp.allclose(
            model.log_density(alpha, x), expected, rtol=RTOL, atol=ATOL
        )

    def test_uniform_dirichlet_density(self) -> None:
        """Dirichlet(1,...,1) is uniform on the simplex: p(x) = (k-1)!."""
        k = 3
        model = Dirichlet(n_categories=k)
        alpha = jnp.ones(k)
        # Evaluate at a few simplex points — density should be constant and equal to (k-1)!
        xs = jnp.array([[0.3, 0.3, 0.4], [0.1, 0.8, 0.1], [0.25, 0.25, 0.5]])
        densities = jax.vmap(model.density, in_axes=(None, 0))(alpha, xs)
        expected = jnp.exp(jax.lax.lgamma(jnp.array(k, dtype=jnp.float64)))  # (k-1)!
        assert jnp.allclose(densities, expected, rtol=RTOL, atol=ATOL)

    def test_sampling_shape_and_simplex(self) -> None:
        model = Dirichlet(n_categories=3)
        alpha = jnp.array([3.0, 7.0, 5.0])
        samples = model.sample(jax.random.PRNGKey(0), alpha, n=100)
        assert samples.shape == (100, 3)
        assert jnp.all(samples > 0)
        assert jnp.allclose(samples.sum(axis=1), 1.0, atol=1e-5)

    def test_sampling_mean_matches_to_mean(self) -> None:
        """Empirical E[log x] across 50k samples should match the analytic mean."""
        model = Dirichlet(n_categories=3)
        alpha = jnp.array([3.0, 7.0, 5.0])
        samples = model.sample(jax.random.PRNGKey(42), alpha, n=50_000)
        empirical = model.average_sufficient_statistic(samples)
        theoretical = model.to_mean(alpha)
        assert jnp.allclose(empirical, theoretical, atol=0.02)

    def test_density_normalizes_by_monte_carlo(self) -> None:
        """Integral of the density over the simplex equals 1 (Monte-Carlo estimate).

        Uses importance sampling with the uniform Dirichlet(1,...,1) as proposal.
        """
        k = 3
        model = Dirichlet(n_categories=k)
        alpha = jnp.array([2.0, 3.0, 4.0])
        uniform_alpha = jnp.ones(k)
        n = 200_000
        # Samples from Dirichlet(1,...,1) are uniform on the simplex;
        # uniform proposal density is (k-1)! so importance weights are p(x)/q(x).
        samples = model.sample(jax.random.PRNGKey(0), uniform_alpha, n=n)
        p_densities = jax.vmap(model.density, in_axes=(None, 0))(alpha, samples)
        q_density = jnp.exp(
            jax.lax.lgamma(jnp.array(float(k)))  # (k-1)!
        )
        estimate = jnp.mean(p_densities / q_density)
        assert jnp.allclose(estimate, 1.0, atol=0.02)

    def test_initialize_produces_positive_params(self) -> None:
        model = Dirichlet(n_categories=4)
        params = model.initialize(jax.random.PRNGKey(0))
        assert params.shape == (4,)
        assert jnp.all(params > 0)
        assert model.check_natural_parameters(params)
