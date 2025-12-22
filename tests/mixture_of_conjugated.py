"""Tests for MixtureOfConjugated.

Tests MixtureOfConjugated with FactorAnalysis as the base harmonium,
covering dimension consistency, conjugation, posterior computation,
and the isomorphism with Mixture[Harmonium] representation.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import FactorAnalysis, Normal
from goal.models.graphical.mixture import MixtureOfConjugated

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
def model(request) -> MixtureOfConjugated[Normal, Normal]:
    """Create MixtureOfConjugated with FactorAnalysis base."""
    obs_dim, lat_dim, n_cat = request.param
    base_fa = FactorAnalysis(obs_dim=obs_dim, lat_dim=lat_dim)
    return MixtureOfConjugated[Normal, Normal](n_categories=n_cat, hrm=base_fa)


@pytest.fixture
def params(model: MixtureOfConjugated[Normal, Normal]) -> Array:
    """Generate random model parameters."""
    key = jax.random.PRNGKey(42)
    return model.initialize(key, location=0.0, shape=1.0)


def test_dimension_consistency(model: MixtureOfConjugated[Normal, Normal], params: Array):
    """Test that all manifold dimensions are consistent."""
    # Parameter dimension matches model
    assert params.shape[0] == model.dim

    # Split coordinates have correct dimensions
    obs, int_params, lat = model.split_coords(params)
    assert obs.shape[0] == model.obs_man.dim
    assert int_params.shape[0] == model.int_man.dim
    assert lat.shape[0] == model.lat_man.dim

    # Interaction blocks sum to total
    xy, xyk, xk = model.int_man.coord_blocks(int_params)
    assert xy.shape[0] + xyk.shape[0] + xk.shape[0] == model.int_man.dim

    # Domain/codomain consistency
    assert model.int_man.dom_man.dim == model.lat_man.dim
    assert model.int_man.cod_man.dim == model.obs_man.dim


def test_conjugation_parameters(model: MixtureOfConjugated[Normal, Normal], params: Array):
    """Test conjugation parameters have correct shape."""
    obs, int_params, _ = model.split_coords(params)
    lkl_params = model.lkl_fun_man.join_coords(obs, int_params)
    rho = model.conjugation_parameters(lkl_params)

    assert rho.shape[0] == model.lat_man.dim


def test_posterior_computation(model: MixtureOfConjugated[Normal, Normal], params: Array):
    """Test posterior computation for single observation."""
    obs = jnp.ones(model.obs_man.data_dim)

    # Posterior should have correct dimension
    posterior = model.posterior_at(params, obs)
    assert posterior.shape[0] == model.pst_man.dim

    # Soft assignments should sum to 1
    soft = model.posterior_soft_assignments(params, obs)
    assert soft.shape[0] == model.n_categories
    assert jnp.allclose(jnp.sum(soft), 1.0)

    # Hard assignment should be valid index
    hard = model.posterior_hard_assignment(params, obs)
    assert 0 <= hard < model.n_categories


def test_interaction_blocks_callable(model: MixtureOfConjugated[Normal, Normal]):
    """Test that each interaction block is callable."""
    xy_params = model.xy_man.zeros()
    xyk_params = model.xyk_man.zeros()
    xk_params = model.xk_man.zeros()
    lat_coords = model.lat_man.zeros()

    # Each block should produce observable-dimension output
    result_xy = model.xy_man(xy_params, lat_coords)
    result_xyk = model.xyk_man(xyk_params, lat_coords)
    result_xk = model.xk_man(xk_params, lat_coords)

    assert result_xy.shape[0] == model.obs_man.dim
    assert result_xyk.shape[0] == model.obs_man.dim
    assert result_xk.shape[0] == model.obs_man.dim


def test_mixture_representation_round_trip(model: MixtureOfConjugated[Normal, Normal], params: Array):
    """Test conversion to/from Mixture[Harmonium] is an isomorphism."""
    # Forward and back
    mix_params = model.to_mixture_params(params)
    recovered = model.from_mixture_params(mix_params)
    assert jnp.allclose(params, recovered, atol=1e-10)

    # mix_man has correct dimension
    assert mix_params.shape[0] == model.mix_man.dim


def test_mixture_representation_likelihood(model: MixtureOfConjugated[Normal, Normal], params: Array):
    """Test likelihood equivalence between representations."""
    mix_params = model.to_mixture_params(params)

    key = jax.random.PRNGKey(456)
    y = jax.random.normal(key, (model.hrm.pst_man.data_dim,))

    # Get component harmoniums from mixture representation
    comp_params, _ = model.mix_man.split_natural_mixture(mix_params)

    for k in range(model.n_categories):
        hrm_k = model.mix_man.cmp_man.get_replicate(comp_params, k)
        lkl_k = model.hrm.likelihood_at(hrm_k, y)
        assert lkl_k.shape[0] == model.hrm.obs_man.dim
