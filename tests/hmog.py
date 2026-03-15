"""Tests for models/graphical/hmog.py.

Covers AnalyticHMoG and DifferentiableHMoG. Verifies dimensions, parameter
conversions, observable density, posterior assignments, sampling, and EM.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Diagonal, Scale
from goal.models import AnalyticHMoG, DifferentiableHMoG, analytic_hmog, differentiable_hmog

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-4


class TestAnalyticHMoG:
    """Test AnalyticHMoG properties and operations."""

    @pytest.fixture(params=[(6, 2, 3), (8, 3, 4)])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[AnalyticHMoG[Diagonal], Array]:
        obs_dim, lat_dim, n_components = request.param
        model = analytic_hmog(obs_dim=obs_dim, obs_rep=Diagonal(), lat_dim=lat_dim, n_components=n_components)
        params = model.initialize(jax.random.PRNGKey(42), location=0.0, shape=0.5)
        return model, params

    def test_dimensions(
        self, model_and_params: tuple[AnalyticHMoG[Diagonal], Array]
    ) -> None:
        model, params = model_and_params
        assert params.shape[0] == model.dim
        assert model.obs_man.dim == model.lwr_hrm.obs_man.dim
        assert model.pst_upr_hrm.obs_man.dim == model.lwr_hrm.lat_man.dim

    def test_to_mean_to_natural_round_trip(
        self, model_and_params: tuple[AnalyticHMoG[Diagonal], Array]
    ) -> None:
        model, params = model_and_params
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_log_observable_density(
        self, model_and_params: tuple[AnalyticHMoG[Diagonal], Array]
    ) -> None:
        model, params = model_and_params
        xs = jax.random.normal(jax.random.PRNGKey(7), (50, model.obs_man.data_dim))
        avg_ll = model.average_log_observable_density(params, xs)
        assert jnp.isfinite(avg_ll)

    def test_posterior_soft_assignments(
        self, model_and_params: tuple[AnalyticHMoG[Diagonal], Array]
    ) -> None:
        model, params = model_and_params
        x = jax.random.normal(jax.random.PRNGKey(7), (model.obs_man.data_dim,))
        soft = model.posterior_soft_assignments(params, x)
        n_components = model.pst_upr_hrm.lat_man.n_categories
        assert soft.shape == (n_components,)
        assert jnp.allclose(jnp.sum(soft), 1.0, atol=1e-6)
        assert jnp.all(soft >= 0)

    def test_sampling(
        self, model_and_params: tuple[AnalyticHMoG[Diagonal], Array]
    ) -> None:
        model, params = model_and_params
        samples = model.sample(jax.random.PRNGKey(0), params, 1000)
        obs_dim = model.obs_man.data_dim
        lat_dim = model.lwr_hrm.lat_dim
        # Joint sample: obs + lat + categorical index
        assert samples.shape == (1000, obs_dim + lat_dim + 1)
        assert jnp.all(jnp.isfinite(samples[:, :obs_dim]))
        # Categorical indices should be valid
        cat_indices = samples[:, -1]
        n_components = model.pst_upr_hrm.lat_man.n_categories
        assert jnp.all(cat_indices >= 0)
        assert jnp.all(cat_indices < n_components)

    def test_expectation_maximization(
        self, model_and_params: tuple[AnalyticHMoG[Diagonal], Array]
    ) -> None:
        model, params = model_and_params
        xs = jax.random.normal(jax.random.PRNGKey(7), (200, model.obs_man.data_dim))
        ll_before = model.average_log_observable_density(params, xs)
        em_params = model.expectation_maximization(params, xs)
        assert jnp.all(jnp.isfinite(em_params))
        ll_after = model.average_log_observable_density(em_params, xs)
        assert jnp.isfinite(ll_after)
        assert ll_after >= ll_before - 1e-4  # EM should not decrease likelihood


class TestDifferentiableHMoG:
    """Test DifferentiableHMoG properties and operations."""

    @pytest.fixture(params=[(6, 2, 3), (8, 3, 4)])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[DifferentiableHMoG[Diagonal, Scale], Array]:
        obs_dim, lat_dim, n_components = request.param
        model = differentiable_hmog(
            obs_dim=obs_dim, obs_rep=Diagonal(), lat_dim=lat_dim, pst_rep=Scale(), n_components=n_components
        )
        params = model.initialize(jax.random.PRNGKey(42), location=0.0, shape=0.5)
        return model, params

    def test_dimensions(
        self, model_and_params: tuple[DifferentiableHMoG[Diagonal, Scale], Array]
    ) -> None:
        model, params = model_and_params
        assert params.shape[0] == model.dim
        assert model.obs_man.dim == model.lwr_hrm.obs_man.dim

    def test_pst_prr_differ(
        self, model_and_params: tuple[DifferentiableHMoG[Diagonal, Scale], Array]
    ) -> None:
        """Posterior and prior upper harmoniums have different dimensions (Scale != PD)."""
        model, _ = model_and_params
        assert model.pst_upr_hrm.dim != model.prr_upr_hrm.dim

    def test_log_observable_density(
        self, model_and_params: tuple[DifferentiableHMoG[Diagonal, Scale], Array]
    ) -> None:
        model, params = model_and_params
        xs = jax.random.normal(jax.random.PRNGKey(7), (50, model.obs_man.data_dim))
        avg_ll = model.average_log_observable_density(params, xs)
        assert jnp.isfinite(avg_ll)

    def test_posterior_soft_assignments(
        self, model_and_params: tuple[DifferentiableHMoG[Diagonal, Scale], Array]
    ) -> None:
        model, params = model_and_params
        x = jax.random.normal(jax.random.PRNGKey(7), (model.obs_man.data_dim,))
        soft = model.posterior_soft_assignments(params, x)
        n_components = model.pst_upr_hrm.lat_man.n_categories
        assert soft.shape == (n_components,)
        assert jnp.allclose(jnp.sum(soft), 1.0, atol=1e-6)
        assert jnp.all(soft >= 0)

    def test_sampling(
        self, model_and_params: tuple[DifferentiableHMoG[Diagonal, Scale], Array]
    ) -> None:
        model, params = model_and_params
        samples = model.sample(jax.random.PRNGKey(0), params, 1000)
        obs_dim = model.obs_man.data_dim
        lat_dim = model.lwr_hrm.lat_dim
        assert samples.shape == (1000, obs_dim + lat_dim + 1)
        assert jnp.all(jnp.isfinite(samples[:, :obs_dim]))
        cat_indices = samples[:, -1]
        n_components = model.pst_upr_hrm.lat_man.n_categories
        assert jnp.all(cat_indices >= 0)
        assert jnp.all(cat_indices < n_components)
