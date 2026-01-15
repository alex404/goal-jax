"""Tests for Boltzmann Hierarchical Mixture of Gaussians.

Tests the mathematical properties of BoltzmannHMoG models including:
- Dimension consistency across all manifolds
- Conjugation equation verification
- Posterior computation and normalization
- Observable density via marginalization
- Equivalence of 1-component HMoG to BoltzmannLGM
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from pytest import FixtureRequest

from goal.geometry import PositiveDefinite
from goal.models import BoltzmannLGM, SymmetricBoltzmannHMoG, symmetric_boltzmann_hmog

jax.config.update("jax_platform_name", "cpu")

rtol, atol = 1e-4, 1e-6


@pytest.fixture(
    params=[
        (3, 2, 2),  # (obs_dim, boltz_dim, n_components)
        (4, 3, 3),
        (5, 2, 4),
    ]
)
def model(request: FixtureRequest) -> SymmetricBoltzmannHMoG:
    """Create SymmetricBoltzmannHMoG for different configurations."""
    obs_dim, boltz_dim, n_components = request.param
    return symmetric_boltzmann_hmog(
        obs_dim=obs_dim,
        obs_rep=PositiveDefinite(),
        boltz_dim=boltz_dim,
        n_components=n_components,
    )


@pytest.fixture
def params(model: SymmetricBoltzmannHMoG) -> Array:
    """Generate random model parameters."""
    key = jax.random.PRNGKey(123)
    return model.initialize(key, location=0.0, shape=0.5)


def test_dimension_consistency(model: SymmetricBoltzmannHMoG, params: Array):
    """Test that all manifold dimensions are consistent."""
    # Parameter dimension matches model
    assert params.shape[0] == model.dim

    # Split coordinates have correct dimensions
    obs, int_params, lat = model.split_coords(params)
    assert obs.shape[0] == model.obs_man.dim
    assert int_params.shape[0] == model.int_man.dim
    assert lat.shape[0] == model.lat_man.dim

    # Lower harmonium dimensions
    assert model.lwr_hrm.obs_man.dim == model.obs_man.dim
    assert model.lwr_hrm.lat_man.dim == model.upr_hrm.obs_man.dim

    # Upper harmonium dimensions
    assert model.upr_hrm.lat_man.n_categories == model.upr_hrm.n_categories


def test_posterior_computation(model: SymmetricBoltzmannHMoG, params: Array):
    """Test posterior computation returns correct dimensions."""
    key = jax.random.PRNGKey(456)
    obs = jax.random.normal(key, (model.obs_man.data_dim,))

    # Posterior should have correct dimension
    posterior = model.posterior_at(params, obs)
    assert posterior.shape[0] == model.pst_man.dim

    # Categorical posterior
    cat_posterior = model.posterior_categorical(params, obs)
    assert cat_posterior.shape[0] == model.upr_hrm.lat_man.dim


def test_posterior_soft_assignments_sum_to_one(
    model: SymmetricBoltzmannHMoG, params: Array
):
    """Test that soft assignments sum to 1."""
    key = jax.random.PRNGKey(789)
    obs = jax.random.normal(key, (model.obs_man.data_dim,))

    soft = model.posterior_soft_assignments(params, obs)
    assert soft.shape[0] == model.upr_hrm.n_categories
    assert jnp.allclose(jnp.sum(soft), 1.0, rtol=rtol, atol=atol)


def test_posterior_hard_assignment_valid(model: SymmetricBoltzmannHMoG, params: Array):
    """Test that hard assignment is a valid component index."""
    key = jax.random.PRNGKey(321)
    obs = jax.random.normal(key, (model.obs_man.data_dim,))

    hard = model.posterior_hard_assignment(params, obs)
    assert 0 <= hard < model.upr_hrm.n_categories


def test_posterior_boltzmann_shape(model: SymmetricBoltzmannHMoG, params: Array):
    """Test that posterior Boltzmann has correct dimension."""
    key = jax.random.PRNGKey(654)
    obs = jax.random.normal(key, (model.obs_man.data_dim,))

    boltz_params = model.posterior_boltzmann(params, obs)
    assert boltz_params.shape[0] == model.upr_hrm.obs_man.dim


def test_conjugation_parameters_shape(model: SymmetricBoltzmannHMoG, params: Array):
    """Test conjugation parameters have correct dimension."""
    obs_params, int_params, _ = model.split_coords(params)
    lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
    rho = model.conjugation_parameters(lkl_params)

    assert rho.shape[0] == model.prr_man.dim


def test_prior_computation(model: SymmetricBoltzmannHMoG, params: Array):
    """Test prior computation returns correct dimension."""
    prior = model.prior(params)
    assert prior.shape[0] == model.prr_man.dim


def test_join_conjugated_round_trip(model: SymmetricBoltzmannHMoG, params: Array):
    """Test that join_conjugated and split_conjugated are inverses."""
    lkl_params, prior_params = model.split_conjugated(params)
    recovered = model.join_conjugated(lkl_params, prior_params)

    assert jnp.allclose(params, recovered, rtol=rtol, atol=atol)


def test_log_partition_function(model: SymmetricBoltzmannHMoG, params: Array):
    """Test log partition function is finite."""
    log_z = model.log_partition_function(params)
    assert jnp.isfinite(log_z)


def test_observable_density(model: SymmetricBoltzmannHMoG, params: Array):
    """Test observable density is finite and positive."""
    key = jax.random.PRNGKey(222)
    obs = jax.random.normal(key, (model.obs_man.data_dim,))

    log_density = model.log_observable_density(params, obs)
    assert jnp.isfinite(log_density)

    density = model.observable_density(params, obs)
    assert density > 0
    assert jnp.allclose(density, jnp.exp(log_density), rtol=rtol, atol=atol)


# === 1-Component HMoG Equivalence to BoltzmannLGM ===


@pytest.fixture(
    params=[
        (2, 3),  # (obs_dim, boltz_dim)
        (3, 2),
        (4, 4),
    ]
)
def lgm_hmog_params(
    request: FixtureRequest,
) -> tuple[BoltzmannLGM, SymmetricBoltzmannHMoG, Array, Array]:
    """Create matched BoltzmannLGM and 1-component HMoG with equivalent parameters.

    The initialize function uses random keys differently for hierarchical vs flat models,
    so we construct equivalent parameters manually by using LGM params and mapping them
    to the HMoG structure.
    """
    obs_dim, boltz_dim = request.param
    obs_rep = PositiveDefinite()

    lgm = BoltzmannLGM(obs_dim=obs_dim, obs_rep=obs_rep, lat_dim=boltz_dim)
    hmog = symmetric_boltzmann_hmog(
        obs_dim=obs_dim, obs_rep=obs_rep, boltz_dim=boltz_dim, n_components=1
    )

    # Initialize LGM params
    key = jax.random.PRNGKey(111)
    lgm_params = lgm.initialize(key, location=0.0, shape=1.0)

    # Construct equivalent HMoG params by mapping LGM structure to HMoG structure
    # LGM: (obs, int, boltzmann)
    # HMoG: (obs, int, upper_harmonium) where upper_harmonium = (boltzmann, [], [])
    lgm_obs, lgm_int, lgm_lat = lgm.split_coords(lgm_params)

    # For 1-component HMoG, the upper harmonium lat_params is just the Boltzmann params
    # (since Categorical(1) has dim 0 and interaction is empty)
    hmog_lat = hmog.upr_hrm.join_coords(lgm_lat, jnp.array([]), jnp.array([]))
    hmog_params = hmog.join_coords(lgm_obs, lgm_int, hmog_lat)

    return lgm, hmog, lgm_params, hmog_params


def test_one_component_log_partition_equivalence(
    lgm_hmog_params: tuple[BoltzmannLGM, SymmetricBoltzmannHMoG, Array, Array],
):
    """Test that 1-component HMoG has same log partition as BoltzmannLGM."""
    lgm, hmog, lgm_params, hmog_params = lgm_hmog_params

    lgm_log_z = lgm.log_partition_function(lgm_params)
    hmog_log_z = hmog.log_partition_function(hmog_params)

    assert jnp.allclose(lgm_log_z, hmog_log_z, rtol=rtol, atol=atol)


def test_one_component_density_equivalence(
    lgm_hmog_params: tuple[BoltzmannLGM, SymmetricBoltzmannHMoG, Array, Array],
):
    """Test that 1-component HMoG produces same density as BoltzmannLGM."""
    lgm, hmog, lgm_params, hmog_params = lgm_hmog_params

    # Test on multiple observations
    key = jax.random.PRNGKey(222)
    obs = jax.random.normal(key, (10, lgm.obs_man.data_dim))

    lgm_densities = jax.vmap(lgm.log_observable_density, in_axes=(None, 0))(
        lgm_params, obs
    )
    hmog_densities = jax.vmap(hmog.log_observable_density, in_axes=(None, 0))(
        hmog_params, obs
    )

    assert jnp.allclose(lgm_densities, hmog_densities, rtol=rtol, atol=atol)


def test_one_component_posterior_equivalence(
    lgm_hmog_params: tuple[BoltzmannLGM, SymmetricBoltzmannHMoG, Array, Array],
):
    """Test that 1-component HMoG posterior over Boltzmann matches BoltzmannLGM."""
    lgm, hmog, lgm_params, hmog_params = lgm_hmog_params

    key = jax.random.PRNGKey(333)
    obs = jax.random.normal(key, (lgm.obs_man.data_dim,))

    # LGM posterior is directly over Boltzmann
    lgm_posterior = lgm.posterior_at(lgm_params, obs)

    # HMoG posterior over Boltzmann (marginalizing out the trivial categorical)
    hmog_boltz_posterior = hmog.posterior_boltzmann(hmog_params, obs)

    assert jnp.allclose(lgm_posterior, hmog_boltz_posterior, rtol=rtol, atol=atol)


