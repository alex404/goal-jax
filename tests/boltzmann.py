"""Tests for Boltzmann distribution.

This module tests the core functionality of Boltzmann machines.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Symmetric
from goal.models import Boltzmann

# Configure JAX
jax.config.update("jax_platform_name", "cpu")

# Test Parameters
relative_tol = 1e-5
absolute_tol = 1e-7


@pytest.fixture(params=[3, 4, 5])
def n_neurons(request: pytest.FixtureRequest) -> int:
    """Parametrize tests over different numbers of neurons."""
    return request.param


@pytest.fixture
def boltzmann(n_neurons: int) -> Boltzmann:
    """Create Boltzmann model."""
    return Boltzmann(n_neurons)


@pytest.fixture
def random_params(boltzmann: Boltzmann) -> Array:
    """Generate random Boltzmann parameters."""
    key = jax.random.PRNGKey(42)
    params_array = jax.random.uniform(key, (boltzmann.dim,), minval=-2.0, maxval=2.0)
    return params_array


def test_sufficient_statistic(boltzmann: Boltzmann):
    """Test sufficient statistic is upper triangular outer product x ⊗ x."""
    state = jax.random.bernoulli(jax.random.PRNGKey(0), 0.5, (boltzmann.n_neurons,))
    suff_stat = boltzmann.sufficient_statistic(state)
    expected = Symmetric().from_matrix(jnp.outer(state, state))

    assert jnp.allclose(suff_stat, expected, rtol=relative_tol, atol=absolute_tol)


def test_log_partition_function(boltzmann: Boltzmann, random_params: Array):
    """Test log partition function equals log(∑ exp(θ·s(x))) over all states."""
    log_Z = boltzmann.log_partition_function(random_params)

    def energy(state: Array) -> Array:
        return jnp.dot(random_params, boltzmann.sufficient_statistic(state))

    energies = jax.vmap(energy)(boltzmann.states)
    log_Z_direct = jax.scipy.special.logsumexp(energies)

    assert jnp.allclose(log_Z, log_Z_direct, rtol=relative_tol, atol=absolute_tol)


def test_density_normalization(boltzmann: Boltzmann, random_params: Array):
    """Test densities sum to 1 over all binary states."""
    densities = jax.vmap(lambda s: boltzmann.density(random_params, s))(
        boltzmann.states
    )
    total_prob = jnp.sum(densities)

    assert jnp.allclose(total_prob, 1.0, rtol=relative_tol, atol=absolute_tol)


def test_split_join_round_trip(boltzmann: Boltzmann, random_params: Array):
    """Test that split_location_precision then join_location_precision recovers parameters."""
    loc, prec = boltzmann.split_location_precision(random_params)
    recovered = boltzmann.join_location_precision(loc, prec)

    assert jnp.allclose(random_params, recovered, rtol=relative_tol, atol=absolute_tol)


def test_unit_conditional_prob(boltzmann: Boltzmann, random_params: Array):
    """Test unit_conditional_prob computes correct conditional probability.

    Verifies P(x_i=1|x_{-i}) = p(x with x_i=1) / (p(x with x_i=0) + p(x with x_i=1))
    """
    # Use shp_man (CouplingMatrix) which has unit_conditional_prob
    coupling = boltzmann.shp_man

    # Test with a few random states
    key = jax.random.PRNGKey(123)
    for _ in range(5):
        key, state_key, unit_key = jax.random.split(key, 3)
        state = jax.random.bernoulli(state_key, 0.5, (boltzmann.n_neurons,)).astype(
            jnp.float32
        )
        unit_idx = jax.random.randint(unit_key, (), 0, boltzmann.n_neurons)

        # Compute using unit_conditional_prob
        prob_fast = coupling.unit_conditional_prob(state, unit_idx, random_params)

        # Compute directly using density ratio
        state_0 = state.at[unit_idx].set(0.0)
        state_1 = state.at[unit_idx].set(1.0)

        # Use unnormalized probabilities (densities without partition function)
        energy_0 = jnp.dot(random_params, coupling.sufficient_statistic(state_0))
        energy_1 = jnp.dot(random_params, coupling.sufficient_statistic(state_1))

        # P(x_i=1|x_{-i}) = exp(E_1) / (exp(E_0) + exp(E_1)) = sigmoid(E_1 - E_0)
        prob_direct = jax.nn.sigmoid(energy_1 - energy_0)

        assert jnp.allclose(prob_fast, prob_direct, rtol=relative_tol, atol=absolute_tol)
