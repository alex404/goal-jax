"""Tests for Linear Gaussian Model parameterizations.

Tests FactorAnalysis and NormalAnalyticLGM across covariance representations.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import FactorAnalysis, Normal, NormalAnalyticLGM

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Test parameters for joint distribution
obs_mean = jnp.array([2.0, -1.0, 0.0])
lat_mean = jnp.array([3.0, -2.0])
obs_cov = jnp.array([[4.0, 1.5, 0.0], [1.5, 3.0, 1.0], [0.0, 1.0, 2.0]])
int_cov = jnp.array([[-1.0, 0.5], [0.0, 0.0], [0.5, 0.0]])
lat_cov = jnp.array([[3.0, 1.0], [1.0, 2.0]])
sample_size = 1000
rtol, atol = 1e-4, 1e-4


@pytest.fixture
def joint_normal() -> tuple[Normal, Array]:
    """Ground truth joint normal model and mean parameters."""
    joint_dim = obs_mean.shape[0] + lat_mean.shape[0]
    model = Normal(joint_dim, PositiveDefinite())
    joint_mean = jnp.concatenate([obs_mean, lat_mean])
    joint_cov = jnp.block([[obs_cov, int_cov], [int_cov.T, lat_cov]])
    cov_params = model.cov_man.from_matrix(joint_cov)
    means = model.join_mean_covariance(joint_mean, cov_params)
    return model, means


@pytest.fixture
def samples(joint_normal: tuple[Normal, Array]) -> Array:
    """Sample data from ground truth distribution."""
    model, means = joint_normal
    key = jax.random.PRNGKey(0)
    natural_params = model.to_natural(means)
    return model.sample(key, natural_params, sample_size)


def test_factor_analysis_from_loadings():
    """Test factor analysis loadings parameterization."""
    loadings = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    means = jnp.array([1.0, 2.0, 3.0])
    diags = jnp.array([0.1, 0.1, 0.1])

    fa = FactorAnalysis(obs_dim=3, lat_dim=2)
    params = fa.from_loadings(loadings, means, diags)

    nor_man, nor_params = fa.observable_distribution(params)
    mean, cov = nor_man.split_mean_covariance(nor_man.to_mean(nor_params))
    cov_mat = nor_man.cov_man.to_matrix(cov)

    expected_cov = loadings @ loadings.T + jnp.diag(diags)

    assert jnp.allclose(mean, means, rtol=rtol, atol=atol)
    assert jnp.allclose(cov_mat, expected_cov, rtol=rtol, atol=atol)
    assert jnp.isfinite(nor_man.log_density(nor_params, means))


def test_observable_distribution_consistency(samples: Array):
    """Test observable_density matches density of observable_distribution."""
    obs_dim, lat_dim = obs_mean.shape[0], lat_mean.shape[0]
    lgm = NormalAnalyticLGM(obs_dim, Diagonal(), lat_dim)

    means = lgm.average_sufficient_statistic(samples)
    params = lgm.to_natural(means)

    obs_data = samples[:, :obs_dim]
    direct_ll = lgm.average_log_observable_density(params, obs_data)

    nor_man, marginal = lgm.observable_distribution(params)
    marginal_ll = nor_man.average_log_density(marginal, obs_data)

    assert jnp.allclose(direct_ll, marginal_ll, rtol=rtol, atol=atol)


@pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
def test_lgm_matches_normal(
    rep: Scale | Diagonal | PositiveDefinite,
    samples: Array,
    joint_normal: tuple[Normal, Array],
):
    """Test LGM log-density matches equivalent Normal distribution."""
    obs_dim, lat_dim = obs_mean.shape[0], lat_mean.shape[0]
    lgm = NormalAnalyticLGM(obs_dim, rep, lat_dim)
    nor_man, _ = joint_normal

    # Fit and convert
    means = lgm.average_sufficient_statistic(samples)
    params = lgm.to_natural(means)
    nor_params = lgm.to_normal(params)

    assert lgm.check_natural_parameters(params)

    # Parameter recovery
    recovered = lgm.to_mean(params)
    assert jnp.allclose(recovered, means, rtol=rtol, atol=atol)

    # Log partition functions match
    lgm_lpf = lgm.log_partition_function(params)
    nor_lpf = nor_man.log_partition_function(nor_params)
    assert jnp.allclose(lgm_lpf, nor_lpf, rtol=rtol, atol=atol)

    # Log densities match
    lgm_ll = lgm.average_log_density(params, samples)
    nor_ll = nor_man.average_log_density(nor_params, samples)
    assert jnp.allclose(lgm_ll, nor_ll, rtol=rtol, atol=atol)

    # Compare to scipy
    mean, cov = nor_man.split_mean_covariance(nor_man.to_mean(nor_params))
    scipy_ll = jnp.mean(
        jax.vmap(stats.multivariate_normal.logpdf, in_axes=(0, None, None))(
            samples, mean, nor_man.cov_man.to_matrix(cov)
        )
    )
    assert jnp.allclose(lgm_ll, scipy_ll, rtol=rtol, atol=atol)


def test_representation_consistency(samples: Array):
    """Test embedded representations give same results."""
    obs_dim, lat_dim = obs_mean.shape[0], lat_mean.shape[0]
    iso = NormalAnalyticLGM(obs_dim, Scale(), lat_dim)
    dia = NormalAnalyticLGM(obs_dim, Diagonal(), lat_dim)
    pod = NormalAnalyticLGM(obs_dim, PositiveDefinite(), lat_dim)

    iso_means = iso.average_sufficient_statistic(samples)
    iso_natural = iso.to_natural(iso_means)

    # Embed to other representations
    obs_params, int_params, lat_params = iso.split_coords(iso_natural)
    dia_obs = iso.obs_man.embed_rep(dia.obs_man, obs_params)
    dia_natural = dia.join_coords(dia_obs, int_params, lat_params)
    pod_obs = iso.obs_man.embed_rep(pod.obs_man, obs_params)
    pod_natural = pod.join_coords(pod_obs, int_params, lat_params)

    # Log partition functions match
    iso_lpf = iso.log_partition_function(iso_natural)
    dia_lpf = dia.log_partition_function(dia_natural)
    pod_lpf = pod.log_partition_function(pod_natural)
    assert jnp.allclose(iso_lpf, dia_lpf, rtol=rtol, atol=atol)
    assert jnp.allclose(dia_lpf, pod_lpf, rtol=rtol, atol=atol)

    # Log densities match
    iso_ll = iso.average_log_density(iso_natural, samples)
    dia_ll = dia.average_log_density(dia_natural, samples)
    pod_ll = pod.average_log_density(pod_natural, samples)
    assert jnp.allclose(iso_ll, dia_ll, rtol=rtol, atol=atol)
    assert jnp.allclose(dia_ll, pod_ll, rtol=rtol, atol=atol)


def test_conjugation_equation():
    """Test conjugation equation: ψ(θ_X + θ_{XZ}·s_Z(z)) = ρ·s_Z(z) + ψ_X(θ_X)."""
    obs_dim, lat_dim = 3, 2
    model = NormalAnalyticLGM(obs_dim, PositiveDefinite(), lat_dim)

    key = jax.random.PRNGKey(42)
    params = model.initialize(key, location=0.0, shape=1.0)

    obs_params, int_params, _ = model.split_coords(params)
    lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
    rho = model.conjugation_parameters(lkl_params)

    # Test for several random latent vectors
    for i in range(5):
        z = jax.random.normal(jax.random.fold_in(key, i), (lat_dim,))
        s_z = model.pst_man.sufficient_statistic(z)

        # LHS: ψ(θ_X + θ_{XZ}·s_Z(z))
        conditional_obs = model.lkl_fun_man(lkl_params, z)
        lhs = model.obs_man.log_partition_function(conditional_obs)

        # RHS: ρ·s_Z(z) + ψ_X(θ_X)
        rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

        assert jnp.abs(lhs - rhs) < 1e-5, f"Conjugation violated: {lhs} vs {rhs}"
