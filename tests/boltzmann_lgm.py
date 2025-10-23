"""Tests for Boltzmann Linear Gaussian Model.

This module tests the critical mathematical properties of Boltzmann-LGMs:

1. Conjugation equation: ψ(θ_X + θ_{XZ} · s_Z(z)) = ρ · s_Z(z) + ψ_X(θ_X)
2. Observable density via marginalization
3. Posterior normalization
"""

import logging

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from pytest import FixtureRequest

from goal.geometry import (
    Natural,
    Point,
    PositiveDefinite,
)
from goal.models import (
    BoltzmannDifferentiableLinearGaussianModel,
)

# Type aliases
Model = BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]
ModelParams = Point[Natural, BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]]

# Configure JAX
jax.config.update("jax_platform_name", "cpu")

# Set up logging
logger = logging.getLogger(__name__)

# Test Parameters
relative_tol = 1e-4  # Slightly relaxed for complex computations
absolute_tol = 1e-6
n_test_states = 10  # Number of random states to test


@pytest.fixture(params=[(2, 3), (2, 4), (3, 5)])
def model_dims(request: FixtureRequest) -> tuple[int, int]:
    """Parametrize tests over different (obs_dim, lat_dim) combinations."""
    return request.param


@pytest.fixture
def model(model_dims: tuple[int, int]) -> Model:
    """Create Boltzmann-LGM model."""
    obs_dim, lat_dim = model_dims
    return BoltzmannDifferentiableLinearGaussianModel(
        obs_dim=obs_dim, obs_rep=PositiveDefinite, lat_dim=lat_dim
    )


@pytest.fixture
def random_params(
    model: Model,
) -> ModelParams:
    """Generate random model parameters."""
    key = jax.random.PRNGKey(123)
    # Initialize with small random values to ensure numerical stability
    return model.initialize(key, location=0.0, shape=0.5)


def test_conjugation_equation(
    model: Model,
    random_params: ModelParams,
):
    """Test the fundamental conjugation equation for Boltzmann-LGM.

    For a conjugated harmonium, the following must hold for all latent states z:

    ψ(θ_X + θ_{XZ} · s_Z(z)) = ρ · s_Z(z) + ψ_X(θ_X)

    where:
    - ψ is the log partition function
    - θ_X are observable natural parameters
    - θ_{XZ} are interaction parameters
    - s_Z(z) is the sufficient statistic of latent state z
    - ρ are the conjugation parameters

    This is the defining property of conjugated harmoniums and must be
    exactly satisfied for the model to be mathematically correct.
    """
    # Split parameters
    obs_params, int_params, _ = model.split_params(random_params)
    lkl_params = model.lkl_man.join_params(obs_params, int_params)

    # Compute conjugation parameters
    rho = model.conjugation_parameters(lkl_params)

    # Get all latent states
    lat_man = model.pst_lat_man
    states = lat_man.states  # All binary states

    # Test equation for subset of states
    n_states_to_test = min(n_test_states, len(states))
    test_indices = jnp.linspace(0, len(states) - 1, n_states_to_test, dtype=int)

    errors = []
    for idx in test_indices:
        state = states[idx]

        # Compute sufficient statistic of latent state
        s_z = lat_man.sufficient_statistic(state)

        # LHS: ψ(θ_X + θ_{XZ} · s_Z(z))
        # This is the log partition function of the conditional p(x|z)
        z_loc = model.pst_lat_man.loc_man.point(state)
        conditional_obs = model.lkl_man(lkl_params, z_loc)
        lhs = model.obs_man.log_partition_function(conditional_obs)

        # RHS: ρ · s_Z(z) + ψ_X(θ_X)
        rhs_term1 = model.prr_lat_man.dot(rho, s_z)
        rhs_term2 = model.obs_man.log_partition_function(obs_params)
        rhs = rhs_term1 + rhs_term2

        error = jnp.abs(lhs - rhs)
        errors.append(float(error))

        logger.info(f"State {idx}: LHS={lhs:.8f}, RHS={rhs:.8f}, error={error:.2e}")

    max_error = max(errors)
    mean_error = sum(errors) / len(errors)

    logger.info(
        f"Conjugation equation: max_error={max_error:.2e}, mean_error={mean_error:.2e}"
    )

    assert all(e < absolute_tol for e in errors), (
        f"Conjugation equation violated! Max error: {max_error:.2e}, mean error: {mean_error:.2e}"
    )


def test_observable_density_via_marginalization(
    model: Model,
    random_params: ModelParams,
):
    """Test that observable density equals sum over latent states.

    p(x) should equal sum_z p(x, z) = sum_z p(x|z) * p(z)
    """
    # Generate a random observation
    key = jax.random.PRNGKey(456)
    obs = jax.random.normal(key, (model.obs_dim,))

    # Compute observable density using model's method
    density_model = model.observable_density(random_params, obs)

    # Compute via marginalization
    lkl_params, prior_params = model.split_conjugated(random_params)
    lat_man = model.pst_lat_man
    states = lat_man.states

    def joint_density(state: Array) -> Array:
        # p(x|z)
        z_loc = model.pst_lat_man.loc_man.point(state)
        conditional_obs = model.lkl_man(lkl_params, z_loc)
        p_x_given_z = model.obs_man.density(conditional_obs, obs)

        # p(z)
        p_z = lat_man.density(prior_params, state)

        return p_x_given_z * p_z

    joint_densities = jax.vmap(joint_density)(states)
    density_marginal = jnp.sum(joint_densities)

    logger.info(
        f"Observable density: model={density_model:.8f}, marginal={density_marginal:.8f}"
    )

    rel_error = jnp.abs(density_model - density_marginal) / (density_marginal + 1e-10)

    assert jnp.allclose(
        density_model, density_marginal, rtol=relative_tol, atol=absolute_tol
    ), (
        f"Observable density mismatch: {density_model:.8f} vs {density_marginal:.8f} (rel_error={rel_error:.2e})"
    )


def test_posterior_normalization(
    model: Model,
    random_params: ModelParams,
):
    """Test that posterior p(z|x) sums to 1 over all latent states."""
    # Generate a random observation
    key = jax.random.PRNGKey(789)
    obs = jax.random.normal(key, (model.obs_dim,))

    # Get all latent states
    lat_man = model.pst_lat_man
    states = lat_man.states

    # Compute posterior parameters
    posterior_params = model.posterior_at(random_params, obs)

    # Compute posterior densities for all states
    def posterior_density(state: Array) -> Array:
        return lat_man.density(posterior_params, state)

    posterior_densities = jax.vmap(posterior_density)(states)
    total_prob = jnp.sum(posterior_densities)

    logger.info(f"Posterior normalization: sum={total_prob:.10f}")

    assert jnp.allclose(total_prob, 1.0, rtol=relative_tol, atol=absolute_tol), (
        f"Posterior doesn't sum to 1: {total_prob}"
    )


def test_log_observable_density_formula(
    model: Model,
    random_params: ModelParams,
):
    """Test that log_observable_density follows harmonium formula.

    log p(x) = θ_X · s_X(x) + ψ_Z(posterior) - ψ_Z(prior) - ψ_X(θ_X) + log μ_X(x)
    """
    # Generate a random observation
    key = jax.random.PRNGKey(321)
    obs = jax.random.normal(key, (model.obs_dim,))

    # Compute using model's method
    log_density = model.log_observable_density(random_params, obs)

    # Compute manually following the formula
    obs_params, _, _ = model.split_params(random_params)

    # θ_X · s_X(x)
    obs_stats = model.obs_man.sufficient_statistic(obs)
    term1 = model.obs_man.dot(obs_params, obs_stats)

    # ψ_Z(posterior)
    posterior_params = model.posterior_at(random_params, obs)
    term2 = model.pst_lat_man.log_partition_function(posterior_params)

    # ψ_Z(prior)
    prior_params = model.prior(random_params)
    term3 = model.prr_lat_man.log_partition_function(prior_params)

    # ψ_X(θ_X)
    term4 = model.obs_man.log_partition_function(obs_params)

    # log μ_X(x)
    term5 = model.obs_man.log_base_measure(obs)

    manual_log_density = term1 + term2 - term3 - term4 + term5

    logger.info(
        f"Log density: model={log_density:.8f}, manual={manual_log_density:.8f}"
    )
    logger.info(
        f"  Terms: θ·s={term1:.4f}, ψ_post={term2:.4f}, ψ_prior={term3:.4f}, ψ_X={term4:.4f}, log_μ={term5:.4f}"
    )

    assert jnp.allclose(
        log_density, manual_log_density, rtol=relative_tol, atol=absolute_tol
    ), f"Log density formula mismatch: {log_density:.8f} vs {manual_log_density:.8f}"


def test_conjugation_parameters_shape(
    model: Model,
    random_params: ModelParams,
):
    """Test that conjugation parameters have correct dimension."""
    obs_params, int_params, _ = model.split_params(random_params)
    lkl_params = model.lkl_man.join_params(obs_params, int_params)

    rho = model.conjugation_parameters(lkl_params)

    expected_dim = model.prr_lat_man.dim

    assert rho.array.shape[0] == expected_dim, (
        f"Conjugation parameters dimension mismatch: {rho.array.shape[0]} vs {expected_dim}"
    )
