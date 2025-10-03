"""Tests for Boltzmann distribution.

This module tests the core functionality of Boltzmann machines.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Natural, Point, Symmetric
from goal.models import Boltzmann

# Configure JAX
jax.config.update("jax_platform_name", "cpu")

# Test Parameters
relative_tol = 1e-5
absolute_tol = 1e-7


@pytest.fixture(params=[3, 4, 5])
def n_neurons(request) -> int:
    """Parametrize tests over different numbers of neurons."""
    return request.param


@pytest.fixture
def boltzmann(n_neurons: int) -> Boltzmann:
    """Create Boltzmann model."""
    return Boltzmann(n_neurons)


@pytest.fixture
def random_params(boltzmann: Boltzmann) -> Point[Natural, Boltzmann]:
    """Generate random Boltzmann parameters."""
    key = jax.random.PRNGKey(42)
    params_array = jax.random.uniform(key, (boltzmann.dim,), minval=-2.0, maxval=2.0)
    return boltzmann.natural_point(params_array)


def test_sufficient_statistic(boltzmann: Boltzmann):
    """Test sufficient statistic is upper triangular outer product x ⊗ x."""
    state = jax.random.bernoulli(jax.random.PRNGKey(0), 0.5, (boltzmann.n_neurons,))
    suff_stat = boltzmann.sufficient_statistic(state)
    expected = Symmetric.from_dense(jnp.outer(state, state))

    assert jnp.allclose(suff_stat.array, expected, rtol=relative_tol, atol=absolute_tol)


def test_log_partition_function(
    boltzmann: Boltzmann, random_params: Point[Natural, Boltzmann]
):
    """Test log partition function equals log(∑ exp(θ·s(x))) over all states."""
    log_Z = boltzmann.log_partition_function(random_params)

    def energy(state: Array) -> Array:
        return boltzmann.dot(random_params, boltzmann.sufficient_statistic(state))

    energies = jax.vmap(energy)(boltzmann.states)
    log_Z_direct = jax.scipy.special.logsumexp(energies)

    assert jnp.allclose(log_Z, log_Z_direct, rtol=relative_tol, atol=absolute_tol)


def test_density_normalization(
    boltzmann: Boltzmann, random_params: Point[Natural, Boltzmann]
):
    """Test densities sum to 1 over all binary states."""
    densities = jax.vmap(lambda s: boltzmann.density(random_params, s))(
        boltzmann.states
    )
    total_prob = jnp.sum(densities)

    assert jnp.allclose(total_prob, 1.0, rtol=relative_tol, atol=absolute_tol)


def test_split_join_round_trip(boltzmann: Boltzmann, random_params: Point[Natural, Boltzmann]):
    """Test that split_location_precision then join_location_precision recovers parameters."""
    loc, prec = boltzmann.split_location_precision(random_params)
    recovered = boltzmann.join_location_precision(loc, prec)

    assert jnp.allclose(random_params.array, recovered.array, rtol=relative_tol, atol=absolute_tol)
