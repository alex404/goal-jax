"""Tests for Linear Gaussian Model parameterizations.

This module tests the core functionality of Linear Gaussian Models across different
covariance representations (Scale(), Diagonal(), PositiveDefinite()). Tests cover:

1. Parameter conversion and recovery
2. Likelihood computation
3. Consistency between different representations
4. Conversion to full normal distributions
"""

import logging

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import FactorAnalysis, Normal, NormalAnalyticLGM

# Configure JAX
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Set up logging
logger = logging.getLogger(__name__)

# Test Parameters
obs_mean = jnp.array([2.0, -1.0, 0.0])
lat_mean = jnp.array([3.0, -2.0])
obs_cov = jnp.array(
    [
        [4.0, 1.5, 0.0],
        [1.5, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
)
int_cov = jnp.array(
    [
        [-1.0, 0.5],
        [0.0, 0.0],
        [0.5, 0.0],
    ]
)
lat_cov = jnp.array([[3.0, 1.0], [1.0, 2.0]])
sample_size = 1000
relative_tol = 1e-4
absolute_tol = 1e-4


@pytest.fixture
def ground_truth_normal() -> Normal:
    """Create ground truth normal model."""
    joint_dim = obs_mean.shape[0] + lat_mean.shape[0]
    return Normal(joint_dim, PositiveDefinite())


@pytest.fixture
def ground_truth_params(
    ground_truth_normal: Normal,
) -> Array:
    """Create ground truth parameters in mean coordinates."""
    # Create joint mean
    joint_mean = jnp.concatenate([obs_mean, lat_mean])

    # Create joint covariance
    joint_cov = jnp.block(
        [
            [obs_cov, int_cov],
            [int_cov.T, lat_cov],
        ]
    )

    mu = joint_mean
    cov = ground_truth_normal.cov_man.from_dense(joint_cov)
    return ground_truth_normal.join_mean_covariance(mu, cov)


@pytest.fixture
def sample_data(
    ground_truth_normal: Normal,
    ground_truth_params: Array,
) -> Array:
    """Generate sample data from ground truth distribution."""
    key = jax.random.PRNGKey(0)
    gt_natural = ground_truth_normal.to_natural(ground_truth_params)
    return ground_truth_normal.sample(key, gt_natural, sample_size)


def scipy_log_likelihood(x: Array, mean: Array, cov: Array) -> Array:
    """Compute average log likelihood using scipy's implementation."""
    return jnp.mean(
        jax.vmap(stats.multivariate_normal.logpdf, in_axes=(0, None, None))(
            x, mean, cov
        )
    )


def test_factor_analysis_loadings(caplog: pytest.LogCaptureFixture) -> None:
    """Test that factor analysis model correctly handles loadings parameterization."""
    caplog.set_level(logging.INFO)
    logger.info("\nTesting factor analysis loadings parameterization")

    # Create simple ground truth parameters
    obs_dim = 3
    lat_dim = 2
    loadings = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    means = jnp.array([1.0, 2.0, 3.0])
    diags = jnp.array([0.1, 0.1, 0.1])  # Small diagonal noise

    # Create factor analysis model
    fa_model = FactorAnalysis(obs_dim, lat_dim)

    # Convert to natural parameters
    fa_params = fa_model.from_loadings(loadings, means, diags)

    # Get the equivalent normal distribution
    nor_man, nor_params = fa_model.observable_distribution(fa_params)

    # Convert to mean parameters to check the result
    nor_means = nor_man.to_mean(nor_params)
    mean, cov = nor_man.split_mean_covariance(nor_means)

    # Check that we get expected results

    # 1. Mean should match input means
    assert jnp.allclose(mean, means, rtol=1e-4, atol=1e-4), "Means don't match"

    # 2. Covariance should be LLᵀ + D
    expected_cov = loadings @ loadings.T + jnp.diag(diags)
    actual_cov = nor_man.cov_man.to_dense(cov)
    assert jnp.allclose(actual_cov, expected_cov, rtol=1e-4, atol=1e-4), (
        "Covariance doesn't match LLᵀ + D"
    )

    # 3. Log density at mean should be finite
    ll = nor_man.log_density(nor_params, means)
    assert jnp.isfinite(ll), "Log density at mean is not finite"

    logger.info(f"Mean: {mean}")
    logger.info(f"Expected covariance:\n{expected_cov}")
    logger.info(f"Actual covariance:\n{actual_cov}")
    logger.info(f"Log density at mean: {ll}")


def test_observable_distributions(
    sample_data: Array, caplog: pytest.LogCaptureFixture
) -> None:
    """Test equivalence between observable_density and the density of observable_distribution."""
    caplog.set_level(logging.INFO)
    logger.info("\nTesting observable distribution consistency")

    # Create model
    obs_dim = obs_mean.shape[0]
    lat_dim = lat_mean.shape[0]
    lgm = NormalAnalyticLGM(obs_dim, Diagonal(), lat_dim)

    # Get model parameters
    means = lgm.average_sufficient_statistic(sample_data)
    params = lgm.to_natural(means)

    # Compare densities
    obs_data = sample_data[:, :obs_dim]  # Extract observable components
    direct_ll = lgm.average_log_observable_density(params, obs_data.astype(jnp.float32))

    nor_man, marginal = lgm.observable_distribution(params)
    marginal_ll = nor_man.average_log_density(marginal, obs_data)

    logger.info(f"Direct observable log-density: {direct_ll}")
    logger.info(f"Marginal distribution log-density: {marginal_ll}")

    assert jnp.allclose(direct_ll, marginal_ll, rtol=relative_tol, atol=absolute_tol), (
        "Log-density differs between direct computation and marginal distribution"
    )


@pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
def test_single_model(
    rep: Scale | Diagonal | PositiveDefinite,
    sample_data: Array,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test fitting and evaluation of a single model.

    Args:
        rep: Covariance representation to test
        sample_data: Generated test data
        caplog: Pytest fixture for capturing logs
    """
    caplog.set_level(logging.INFO)
    logger.info(f"\nTesting {type(rep).__name__} representation")

    # Create models
    obs_dim = obs_mean.shape[0]
    lat_dim = lat_mean.shape[0]
    lgm_man = NormalAnalyticLGM(obs_dim, rep, lat_dim)
    nor_man = Normal(obs_dim + lat_dim, PositiveDefinite())

    # Fit LGM model
    means = lgm_man.average_sufficient_statistic(sample_data)
    logger.info(f"Mean parameters: {means}")

    # Natural -> Mean recovery
    params = lgm_man.to_natural(means)

    # Add validation check and logging
    valid = bool(lgm_man.check_natural_parameters(params))
    logger.info("Natural parameter check for %s: %s", type(lgm_man).__name__, valid)
    assert valid, (
        f"Invalid natural parameters after fitting for {type(lgm_man).__name__}"
    )

    remeans = lgm_man.to_mean(params)
    reparams = lgm_man.to_natural(remeans)
    logger.info(f"Mean parameters: {means}")
    logger.info(f"Remean parameters: {remeans}")
    logger.info(f"Natural parameters: {params}")
    logger.info(f"ReNatural parameters: {reparams}")

    assert jnp.allclose(remeans, means, rtol=relative_tol, atol=absolute_tol)

    assert jnp.allclose(reparams, params, rtol=relative_tol, atol=absolute_tol)

    # Test parameter recovery
    recovered_mean_params = lgm_man.to_mean(params)
    assert jnp.allclose(
        recovered_mean_params,
        means,
        rtol=relative_tol,
        atol=absolute_tol,
    ), "Failed to recover mean parameters"

    # Convert to normal model
    nor_params = lgm_man.to_normal(params)
    logger.info(f"Normal natural parameters: {nor_params}")

    # Compare log-partition functions
    logger.info("\nComparing log partition functions:")
    lgm_lpf = lgm_man.log_partition_function(params)
    nor_lpf = nor_man.log_partition_function(nor_params)
    logger.info(f"LGM log partition function: {lgm_lpf}")
    logger.info(f"Normal log partition function: {nor_lpf}")
    assert jnp.allclose(lgm_lpf, nor_lpf, rtol=relative_tol, atol=absolute_tol)

    # Compare log densities
    lgm_ll = lgm_man.average_log_density(params, sample_data)
    nor_ll = nor_man.average_log_density(nor_params, sample_data)

    logger.info(f"LGM average log-density: {lgm_ll}")
    logger.info(f"Normal average log-density: {nor_ll}")

    assert jnp.allclose(lgm_ll, nor_ll, rtol=relative_tol, atol=absolute_tol), (
        "Log-likelihood differs between LGM and Normal"
    )

    # Compare to scipy
    mean, cov = nor_man.split_mean_covariance(nor_man.to_mean(nor_params))
    scipy_ll = scipy_log_likelihood(sample_data, mean, nor_man.cov_man.to_dense(cov))
    logger.info(f"Scipy average log-density: {scipy_ll}")

    assert jnp.allclose(lgm_ll, scipy_ll, rtol=relative_tol, atol=absolute_tol), (
        "Log-likelihood differs from scipy implementation"
    )


def test_model_consistency(
    sample_data: Array, caplog: pytest.LogCaptureFixture
) -> None:
    """Test consistency between different model representations.

    This test takes an isotropic fit and expands it to more complex representations,
    verifying that key quantities remain consistent.
    """
    caplog.set_level(logging.INFO)
    logger.info("\nTesting model consistency")

    # Create models
    obs_dim = obs_mean.shape[0]
    lat_dim = lat_mean.shape[0]
    iso_model = NormalAnalyticLGM(obs_dim, Scale(), lat_dim)
    dia_model = NormalAnalyticLGM(obs_dim, Diagonal(), lat_dim)
    pod_model = NormalAnalyticLGM(obs_dim, PositiveDefinite(), lat_dim)

    # Fit isotropic model
    iso_means = iso_model.average_sufficient_statistic(sample_data)
    iso_natural = iso_model.to_natural(iso_means)

    obs_params, int_params, lat_params = iso_model.split_coords(iso_natural)
    trn_obs_params = iso_model.obs_man.embed_rep(dia_model.obs_man, obs_params)
    dia_natural = dia_model.join_coords(trn_obs_params, int_params, lat_params)

    obs_params, int_params, lat_params = iso_model.split_coords(iso_natural)
    trn_obs_params = iso_model.obs_man.embed_rep(pod_model.obs_man, obs_params)
    pod_natural = pod_model.join_coords(trn_obs_params, int_params, lat_params)

    # Test log partition function
    logger.info("\nComparing log partition functions:")
    iso_lpf = iso_model.log_partition_function(iso_natural)
    dia_lpf = dia_model.log_partition_function(dia_natural)
    pod_lpf = pod_model.log_partition_function(pod_natural)

    logger.info(f"Isotropic: {iso_lpf}")
    logger.info(f"Diagonal: {dia_lpf}")
    logger.info(f"PositiveDefinite: {pod_lpf}")

    assert jnp.allclose(iso_lpf, dia_lpf, rtol=relative_tol, atol=absolute_tol)
    assert jnp.allclose(dia_lpf, pod_lpf, rtol=relative_tol, atol=absolute_tol)

    # Test log density
    logger.info("\nComparing log densities:")
    iso_ll = iso_model.average_log_density(iso_natural, sample_data)
    dia_ll = dia_model.average_log_density(dia_natural, sample_data)
    pod_ll = pod_model.average_log_density(pod_natural, sample_data)

    logger.info(f"Isotropic: {iso_ll}")
    logger.info(f"Diagonal: {dia_ll}")
    logger.info(f"PositiveDefinite: {pod_ll}")

    assert jnp.allclose(iso_ll, dia_ll, rtol=relative_tol, atol=absolute_tol)
    assert jnp.allclose(dia_ll, pod_ll, rtol=relative_tol, atol=absolute_tol)

    # Test coordinate conversion
    logger.info("\nTesting coordinate conversion:")

    # Natural -> Mean
    iso_means = iso_model.to_mean(iso_natural)
    dia_means = dia_model.to_mean(dia_natural)
    pod_means = pod_model.to_mean(pod_natural)

    # Compare negative entropy
    iso_ne = iso_model.negative_entropy(iso_means)
    dia_ne = dia_model.negative_entropy(dia_means)
    pod_ne = pod_model.negative_entropy(pod_means)

    logger.info(f"Isotropic negative entropy: {iso_ne}")
    logger.info(f"Diagonal negative entropy: {dia_ne}")
    logger.info(f"PositiveDefinite negative entropy: {pod_ne}")

    assert jnp.allclose(iso_ne, dia_ne, rtol=relative_tol, atol=absolute_tol)
    assert jnp.allclose(dia_ne, pod_ne, rtol=relative_tol, atol=absolute_tol)


def test_normal_lgm_conjugation_equation():
    """Test the conjugation equation for Normal LGM.

    For a conjugated harmonium, the following must hold for all latent states z:

    ψ(θ_X + θ_{XZ} · s_Z(z)) = rho · s_Z(z) + ψ_X(θ_X)

    This is a fundamental property that must be satisfied.
    """
    # Create model
    obs_dim, lat_dim = 3, 2
    model = NormalAnalyticLGM(obs_dim, PositiveDefinite(), lat_dim)

    # Initialize with random parameters
    key = jax.random.PRNGKey(42)
    params = model.initialize(key, location=0.0, shape=1.0)

    # Split parameters
    obs_params, int_params, _ = model.split_coords(params)
    lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)

    # Compute conjugation parameters
    rho = model.conjugation_parameters(lkl_params)

    # Test equation for several random latent vectors
    n_test = 5
    errors = []

    for i in range(n_test):
        # Generate random latent vector
        key_i = jax.random.fold_in(key, i)
        z = jax.random.normal(key_i, (lat_dim,))

        # Compute sufficient statistic of latent state
        s_z = model.pst_man.sufficient_statistic(z)

        # LHS: ψ(θ_X + θ_{XZ} · s_Z(z))
        # This is the log partition function of the conditional p(x|z)
        conditional_obs = model.lkl_fun_man(lkl_params, z)
        lhs = model.obs_man.log_partition_function(conditional_obs)

        # RHS: rho · s_Z(z) + ψ_X(θ_X)
        rhs_term1 = jnp.dot(rho, s_z)
        rhs_term2 = model.obs_man.log_partition_function(obs_params)
        rhs = rhs_term1 + rhs_term2

        error = float(jnp.abs(lhs - rhs))
        errors.append(error)

        logger.info(f"z={i}: LHS={lhs:.8f}, RHS={rhs:.8f}, error={error:.2e}")

    max_error = max(errors)
    mean_error = sum(errors) / len(errors)

    logger.info(
        f"Normal LGM conjugation: max_error={max_error:.2e}, mean_error={mean_error:.2e}"
    )

    # Should be very close to zero
    assert max_error < 1e-5, (
        f"Conjugation equation violated! Max error: {max_error:.2e}"
    )


if __name__ == "__main__":
    _ = pytest.main()
