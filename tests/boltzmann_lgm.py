"""Tests for Boltzmann Linear Gaussian Model.

Tests the mathematical properties of Boltzmann-LGMs: conjugation equation,
observable density via marginalization, and posterior normalization.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from pytest import FixtureRequest

from goal.geometry import PositiveDefinite
from goal.models import BoltzmannLGM

jax.config.update("jax_platform_name", "cpu")

rtol, atol = 1e-4, 1e-6


@pytest.fixture(params=[(2, 3), (2, 4), (3, 5)])
def model(request: FixtureRequest) -> BoltzmannLGM:
    """Create Boltzmann-LGM model for different dimension combinations."""
    obs_dim, lat_dim = request.param
    return BoltzmannLGM(obs_dim=obs_dim, obs_rep=PositiveDefinite(), lat_dim=lat_dim)


@pytest.fixture
def params(model: BoltzmannLGM) -> Array:
    """Generate random model parameters."""
    key = jax.random.PRNGKey(123)
    return model.initialize(key, location=0.0, shape=0.5)


def test_conjugation_equation(model: BoltzmannLGM, params: Array):
    """Test conjugation: ψ(θ_X + θ_{XZ}·s_Z(z)) = ρ·s_Z(z) + ψ_X(θ_X)."""
    obs_params, int_params, _ = model.split_coords(params)
    lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
    rho = model.conjugation_parameters(lkl_params)

    states = model.lat_man.states
    n_test = min(10, len(states))
    step = max(1, len(states) // n_test)

    for idx in range(0, len(states), step):
        state = states[idx]
        s_z = model.lat_man.sufficient_statistic(state)

        # LHS: ψ(θ_X + θ_{XZ}·s_Z(z))
        conditional_obs = model.lkl_fun_man(lkl_params, state)
        lhs = model.obs_man.log_partition_function(conditional_obs)

        # RHS: ρ·s_Z(z) + ψ_X(θ_X)
        rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

        assert jnp.abs(lhs - rhs) < atol, f"Conjugation violated at state {idx}"


def test_observable_density_marginalization(model: BoltzmannLGM, params: Array):
    """Test p(x) = Σ_z p(x|z)p(z)."""
    key = jax.random.PRNGKey(456)
    obs = jax.random.normal(key, (model.obs_dim,))

    density_model = model.observable_density(params, obs)

    lkl_params, prior_params = model.split_conjugated(params)
    states = model.lat_man.states

    def joint_density(state: Array) -> Array:
        conditional_obs = model.lkl_fun_man(lkl_params, state)
        p_x_given_z = model.obs_man.density(conditional_obs, obs)
        p_z = model.lat_man.density(prior_params, state)
        return p_x_given_z * p_z

    density_marginal = jnp.sum(jax.vmap(joint_density)(states))

    assert jnp.allclose(density_model, density_marginal, rtol=rtol, atol=atol)


def test_posterior_normalization(model: BoltzmannLGM, params: Array):
    """Test that p(z|x) sums to 1."""
    key = jax.random.PRNGKey(789)
    obs = jax.random.normal(key, (model.obs_dim,))

    posterior_params = model.posterior_at(params, obs)
    states = model.lat_man.states

    densities = jax.vmap(lambda s: model.lat_man.density(posterior_params, s))(states)
    total = jnp.sum(densities)

    assert jnp.allclose(total, 1.0, rtol=rtol, atol=atol)


def test_log_density_formula(model: BoltzmannLGM, params: Array):
    """Test log p(x) = θ·s(x) + ψ(post) - ψ(prior) - ψ_X(θ_X) + log μ(x)."""
    key = jax.random.PRNGKey(321)
    obs = jax.random.normal(key, (model.obs_dim,))

    log_density = model.log_observable_density(params, obs)

    obs_params, _, _ = model.split_coords(params)
    obs_stats = model.obs_man.sufficient_statistic(obs)
    posterior_params = model.posterior_at(params, obs)
    prior_params = model.prior(params)

    manual = (
        jnp.dot(obs_params, obs_stats)
        + model.lat_man.log_partition_function(posterior_params)
        - model.prr_man.log_partition_function(prior_params)
        - model.obs_man.log_partition_function(obs_params)
        + model.obs_man.log_base_measure(obs)
    )

    assert jnp.allclose(log_density, manual, rtol=rtol, atol=atol)


def test_conjugation_parameters_shape(model: BoltzmannLGM, params: Array):
    """Test conjugation parameters have correct dimension."""
    obs_params, int_params, _ = model.split_coords(params)
    lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
    rho = model.conjugation_parameters(lkl_params)

    assert rho.shape[0] == model.prr_man.dim
