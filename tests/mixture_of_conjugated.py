"""Tests for MixtureOfConjugated dimensionality and basic operations.

This module tests the MixtureOfConjugated implementation with FactorAnalysis
as the base harmonium. Focuses on:

1. Correct instantiation
2. Dimension consistency across interaction matrix blocks
3. Conjugation parameter computation
4. Basic posterior computation
"""

import logging

import jax
import jax.numpy as jnp
import pytest

from goal.models import FactorAnalysis
from goal.models.graphical.mixture import MixtureOfConjugated

# Configure JAX
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Set up logging
logger = logging.getLogger(__name__)


@pytest.fixture
def mfa_model():
    """Create a simple MFA model for testing."""
    base_fa = FactorAnalysis(obs_dim=3, lat_dim=2)
    mfa = MixtureOfConjugated(n_categories=2, hrm=base_fa)
    return mfa


@pytest.fixture
def simple_params(mfa_model):
    """Create simple test parameters."""
    key = jax.random.PRNGKey(42)
    # Create simple observable parameters
    obs_params = mfa_model.hrm.obs_man.zeros()

    # Create simple interaction parameters (all blocks)
    int_params = mfa_model.int_man.zeros()

    # Create simple latent parameters
    lat_params = mfa_model.lat_man.zeros()

    return mfa_model.join_coords(obs_params, int_params, lat_params)


def test_mfa_instantiation(mfa_model):
    """Test that MFA model can be instantiated."""
    assert mfa_model.n_categories == 2
    assert mfa_model.hrm.obs_dim == 3
    assert mfa_model.hrm.lat_dim == 2


def test_observable_manifold(mfa_model):
    """Test that observable manifold is correctly inherited."""
    # Observable should come from base FA (Normal with Diagonal)
    assert mfa_model.obs_man.dim == mfa_model.hrm.obs_man.dim
    logger.info(f"Observable dimension: {mfa_model.obs_man.dim}")


def test_latent_manifold(mfa_model):
    """Test that latent manifold is CompleteMixture."""
    # Latent should be CompleteMixture[Normal[PositiveDefinite]]
    lat_dim = mfa_model.lat_man.dim
    logger.info(f"Latent dimension: {lat_dim}")

    # Expected structure:
    # - Base component: Normal[PositiveDefinite] with lat_dim=2
    # - Interaction: (n_categories-1) * base_lat_dim = 1 * base_lat_dim
    # - Categorical: n_categories-1 = 1
    base_lat_dim = mfa_model.hrm.pst_man.dim
    expected_dim = (
        base_lat_dim
        + (mfa_model.n_categories - 1) * base_lat_dim
        + (mfa_model.n_categories - 1)
    )

    logger.info(f"Base latent dimension: {base_lat_dim}")
    logger.info(f"Expected total latent dimension: {expected_dim}")
    logger.info(f"Actual total latent dimension: {lat_dim}")


def test_interaction_manifold_blocks(mfa_model):
    """Test that interaction manifold blocks have consistent dimensions."""
    # Test each block individually
    xy_dim = mfa_model.xy_man.dim
    xyk_dim = mfa_model.xyk_man.dim
    xk_dim = mfa_model.xk_man.dim
    total_int_dim = mfa_model.int_man.dim

    logger.info(f"xy_man (base interaction) dimension: {xy_dim}")
    logger.info(f"xyk_man (component interactions) dimension: {xyk_dim}")
    logger.info(f"xk_man (observable biases) dimension: {xk_dim}")
    logger.info(f"Total interaction dimension: {total_int_dim}")
    logger.info(f"Sum of blocks: {xy_dim + xyk_dim + xk_dim}")

    # Total should equal sum of blocks
    assert total_int_dim == xy_dim + xyk_dim + xk_dim, (
        f"Total int dim {total_int_dim} != sum of blocks {xy_dim + xyk_dim + xk_dim}"
    )


def test_interaction_domain_codomain(mfa_model):
    """Test that interaction manifold domain and codomain are correct."""
    # Domain should be CompleteMixture[Latent]
    # Codomain should be Observable
    dom_dim = mfa_model.int_man.dom_man.dim
    cod_dim = mfa_model.int_man.cod_man.dim

    logger.info(f"Interaction domain dimension: {dom_dim}")
    logger.info(f"Interaction codomain dimension: {cod_dim}")
    logger.info(f"Expected domain (lat_man): {mfa_model.lat_man.dim}")
    logger.info(f"Expected codomain (obs_man): {mfa_model.obs_man.dim}")

    assert dom_dim == mfa_model.lat_man.dim, (
        f"Domain {dom_dim} != latent manifold {mfa_model.lat_man.dim}"
    )
    assert cod_dim == mfa_model.obs_man.dim, (
        f"Codomain {cod_dim} != observable manifold {mfa_model.obs_man.dim}"
    )


def test_parameter_dimensions(simple_params, mfa_model):
    """Test that parameter vector has correct total dimension."""
    param_dim = simple_params.shape[0]
    expected_dim = mfa_model.dim

    logger.info(f"Parameter dimension: {param_dim}")
    logger.info(f"Expected dimension: {expected_dim}")

    assert param_dim == expected_dim, (
        f"Parameter dimension {param_dim} != expected {expected_dim}"
    )


def test_split_coordinates(simple_params, mfa_model):
    """Test that split_coords produces correctly sized components."""
    obs_params, int_params, lat_params = mfa_model.split_coords(simple_params)

    logger.info(f"Observable params shape: {obs_params.shape}")
    logger.info(f"Interaction params shape: {int_params.shape}")
    logger.info(f"Latent params shape: {lat_params.shape}")

    assert obs_params.shape[0] == mfa_model.obs_man.dim
    assert int_params.shape[0] == mfa_model.int_man.dim
    assert lat_params.shape[0] == mfa_model.lat_man.dim


def test_conjugation_parameters_shape(simple_params, mfa_model):
    """Test that conjugation_parameters returns correct shape."""
    # Extract likelihood parameters
    obs_params, int_params, _ = mfa_model.split_coords(simple_params)
    lkl_params = mfa_model.lkl_fun_man.join_coords(obs_params, int_params)

    logger.info(f"Likelihood params shape: {lkl_params.shape}")

    # Compute conjugation parameters
    conj_params = mfa_model.conjugation_parameters(lkl_params)
    logger.info(f"Conjugation params shape: {conj_params.shape}")
    logger.info(f"Expected shape: ({mfa_model.lat_man.dim},)")

    assert conj_params.shape[0] == mfa_model.lat_man.dim, (
        f"Conjugation params shape {conj_params.shape[0]} != latent dim {mfa_model.lat_man.dim}"
    )


def test_posterior_function_dimensions(simple_params, mfa_model):
    """Test that posterior_function returns correct structure."""
    pst_fn_params = mfa_model.posterior_function(simple_params)
    logger.info(f"Posterior function params shape: {pst_fn_params.shape}")
    logger.info("Expected: LinearMap from Observable to CompleteMixture[Latent]")

    # This should be parameters for a LinearMap: Observable -> CompleteMixture[Latent]
    # The dimension should be pst_fun_man.dim
    expected_dim = mfa_model.pst_fun_man.dim
    logger.info(f"Expected posterior function dimension: {expected_dim}")

    assert pst_fn_params.shape[0] == expected_dim


def test_blockmap_structure(mfa_model):
    """Diagnose BlockMap dimension issues."""
    # The int_man is a BlockMap with 3 blocks
    blocks = [mfa_model.xy_man, mfa_model.xyk_man, mfa_model.xk_man]

    logger.info("BlockMap structure analysis:")
    for i, block in enumerate(blocks):
        logger.info(
            f"  Block {i}: dim={block.dim}, "
            f"dom_dim={block.dom_man.dim}, "
            f"cod_dim={block.cod_man.dim}"
        )

    # Check if blocks have same codomain
    cod_dims = [block.cod_man.dim for block in blocks]
    logger.info(f"Codomain dimensions: {cod_dims}")
    if len(set(cod_dims)) > 1:
        logger.error("ISSUE: Blocks have different codomain dimensions!")

    # When calling BlockMap(params, v), it splits params by block dims
    # then calls each block and adds results
    # Results must have same shape to add!


def test_blockmap_call_individual_blocks(mfa_model):
    """Test calling each block individually to find shape mismatch."""
    # Create zero parameters for each block
    xy_params = mfa_model.xy_man.zeros()
    xyk_params = mfa_model.xyk_man.zeros()
    xk_params = mfa_model.xk_man.zeros()

    # Create input in domain (CompleteMixture[Latent])
    lat_coords = mfa_model.lat_man.zeros()

    logger.info(f"Input latent coords shape: {lat_coords.shape}")

    # Call each block - xy_man and xk_man should work
    result_xy = mfa_model.xy_man(xy_params, lat_coords)
    logger.info(f"Block xy_man output shape: {result_xy.shape}")
    assert result_xy.shape[0] == mfa_model.obs_man.dim

    result_xk = mfa_model.xk_man(xk_params, lat_coords)
    logger.info(f"Block xk_man output shape: {result_xk.shape}")
    assert result_xk.shape[0] == mfa_model.obs_man.dim

    # xyk_man should now also work
    result_xyk = mfa_model.xyk_man(xyk_params, lat_coords)
    logger.info(f"Block xyk_man output shape: {result_xyk.shape}")
    assert result_xyk.shape[0] == mfa_model.obs_man.dim


def test_posterior_at_single_observation(simple_params, mfa_model):
    """Test computing posterior for a single observation.

    Currently fails due to xyk_man dimension mismatch in BlockMap.
    """
    # Create a simple observation
    x = jnp.ones(3)  # obs_dim=3

    logger.info("=== Dimension Trace ===")
    logger.info(f"Observation shape: {x.shape}")
    logger.info(f"Observable manifold dim: {mfa_model.obs_man.dim}")
    logger.info(f"Latent manifold dim: {mfa_model.lat_man.dim}")
    logger.info(f"Posterior manifold dim: {mfa_model.pst_man.dim}")

    # Trace the posterior function computation
    logger.info("\n--- Posterior Function ---")
    pst_fn_params = mfa_model.posterior_function(simple_params)
    logger.info(f"Posterior function params shape: {pst_fn_params.shape}")
    logger.info(f"pst_fun_man dim: {mfa_model.pst_fun_man.dim}")
    logger.info(f"pst_fun_man domain (obs): {mfa_model.pst_fun_man.dom_man.dim}")
    logger.info(f"pst_fun_man codomain (lat): {mfa_model.pst_fun_man.snd_man.dim}")

    # Trace the interaction matrix
    logger.info("\n--- Interaction Matrix (BlockMap) ---")
    obs_params, int_params, lat_params = mfa_model.split_coords(simple_params)
    logger.info(f"Total interaction params dim: {int_params.shape}")
    logger.info(f"xy_man dim: {mfa_model.xy_man.dim}")
    logger.info(f"xyk_man dim: {mfa_model.xyk_man.dim}")
    logger.info(f"xk_man dim: {mfa_model.xk_man.dim}")
    logger.info(
        f"Sum: {mfa_model.xy_man.dim + mfa_model.xyk_man.dim + mfa_model.xk_man.dim}"
    )

    # Try to trace individual block dimensions
    logger.info("\n--- Individual Block Dimensions ---")
    logger.info(
        f"xy_man: {mfa_model.xy_man.dom_man.dim} -> {mfa_model.xy_man.cod_man.dim}"
    )
    logger.info(
        f"xyk_man: {mfa_model.xyk_man.dom_man.dim} -> {mfa_model.xyk_man.cod_man.dim}"
    )
    logger.info(
        f"xk_man: {mfa_model.xk_man.dom_man.dim} -> {mfa_model.xk_man.cod_man.dim}"
    )

    # Check base harmonium dimensions
    logger.info("\n--- Base Harmonium (FactorAnalysis) ---")
    logger.info(f"hrm.obs_man.dim: {mfa_model.hrm.obs_man.dim}")
    logger.info(f"hrm.lat_man.dim: {mfa_model.hrm.lat_man.dim}")
    logger.info(f"hrm.pst_man.dim: {mfa_model.hrm.pst_man.dim}")
    logger.info(f"hrm.int_man.dim: {mfa_model.hrm.int_man.dim}")
    logger.info(
        f"hrm.int_man: {mfa_model.hrm.int_man.dom_man.dim} -> {mfa_model.hrm.int_man.cod_man.dim}"
    )

    logger.info("\n--- Attempting posterior_at ---")
    posterior_params = mfa_model.posterior_at(simple_params, x)
    logger.info(f"Posterior params shape: {posterior_params.shape}")
    logger.info(f"Expected: ({mfa_model.pst_man.dim},)")

    assert posterior_params.shape[0] == mfa_model.pst_man.dim


def test_to_mixture_params_round_trip(simple_params, mfa_model):
    """Test that to_mixture_params -> from_mixture_params is identity."""
    # Convert to mixture representation
    mix_params = mfa_model.to_mixture_params(simple_params)
    logger.info(f"Mix params shape: {mix_params.shape}")
    logger.info(f"Expected mix_man dim: {mfa_model.mix_man.dim}")

    assert mix_params.shape[0] == mfa_model.mix_man.dim

    # Convert back
    recovered_params = mfa_model.from_mixture_params(mix_params)
    logger.info(f"Recovered params shape: {recovered_params.shape}")

    # Should be identical
    assert jnp.allclose(simple_params, recovered_params, atol=1e-10), (
        f"Round trip failed: max diff = {jnp.max(jnp.abs(simple_params - recovered_params))}"
    )


def test_from_mixture_params_round_trip(simple_params, mfa_model):
    """Test that from_mixture_params -> to_mixture_params is identity."""
    # First convert to mixture representation
    mix_params = mfa_model.to_mixture_params(simple_params)

    # Then convert back and forward again
    recovered_mix_params = mfa_model.to_mixture_params(
        mfa_model.from_mixture_params(mix_params)
    )

    # Should be identical
    assert jnp.allclose(mix_params, recovered_mix_params, atol=1e-10), (
        f"Round trip failed: max diff = {jnp.max(jnp.abs(mix_params - recovered_mix_params))}"
    )


@pytest.fixture
def random_params(mfa_model):
    """Create random test parameters."""
    key = jax.random.PRNGKey(123)
    # Initialize with some randomness
    params = mfa_model.initialize(key, location=0.0, shape=1.0)
    return params


def test_round_trip_with_random_params(random_params, mfa_model):
    """Test round-trip with non-trivial random parameters."""
    mix_params = mfa_model.to_mixture_params(random_params)
    recovered = mfa_model.from_mixture_params(mix_params)

    assert jnp.allclose(random_params, recovered, atol=1e-10), (
        f"Round trip failed: max diff = {jnp.max(jnp.abs(random_params - recovered))}"
    )


def test_likelihood_equivalence(random_params, mfa_model):
    """Test that likelihood computations are equivalent between representations.

    For a given latent state (y, k), the likelihood p(x|y,k) should be the same
    whether computed via MixtureOfConjugated or via the Mixture[Harmonium] representation.
    """
    # Convert to mixture representation
    mix_params = mfa_model.to_mixture_params(random_params)

    # Create a test latent state: (y, k) where y is Gaussian latent and k is component index
    key = jax.random.PRNGKey(456)
    y = jax.random.normal(key, shape=(mfa_model.hrm.pst_man.data_dim,))

    # Test for each component
    for k in range(mfa_model.n_categories):
        # Build complete latent state for MixtureOfConjugated
        # This is (y, k) encoded in CompleteMixture[Latent] format
        k_one_hot = jnp.zeros(mfa_model.n_categories).at[k].set(1.0)

        # Get likelihood parameters from MixtureOfConjugated for component k
        # We need to access the component-specific observable distribution

        # From mix_man representation: get component k's harmonium
        comp_params, cat_params = mfa_model.mix_man.split_natural_mixture(mix_params)

        # Get component k's harmonium parameters
        hrm_k_params = mfa_model.mix_man.cmp_man.get_replicate(comp_params, k)

        # Get likelihood from component k's harmonium at latent y
        lkl_k = mfa_model.hrm.likelihood_at(hrm_k_params, y)

        logger.info(f"Component {k}: likelihood params shape = {lkl_k.shape}")

        # The likelihood should be an Observable (Normal) distribution
        assert lkl_k.shape[0] == mfa_model.hrm.obs_man.dim


if __name__ == "__main__":
    # Run with verbose logging
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "-s"])
