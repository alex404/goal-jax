"""Tests for models/harmonium/lgm.py.

Covers FactorAnalysis, NormalAnalyticLGM, and BoltzmannLGM. Verifies fitting,
log-density equivalence with joint Normal, observable distribution consistency,
representation embedding, conjugation equations, and BoltzmannLGM density/posterior.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import (
    BoltzmannLGM,
    FullNormal,
    Normal,
    NormalAnalyticLGM,
    NormalCovarianceEmbedding,
    factor_analysis,
    principal_component_analysis,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-4

OBS_MEAN = jnp.array([2.0, -1.0, 0.0])
LAT_MEAN = jnp.array([3.0, -2.0])
OBS_COV = jnp.array([[4.0, 1.5, 0.0], [1.5, 3.0, 1.0], [0.0, 1.0, 2.0]])
INT_COV = jnp.array([[-1.0, 0.5], [0.0, 0.0], [0.5, 0.0]])
LAT_COV = jnp.array([[3.0, 1.0], [1.0, 2.0]])
SAMPLE_SIZE = 1000


@pytest.fixture
def joint_normal() -> tuple[FullNormal, Array]:
    """Ground truth joint normal model and mean parameters."""
    joint_dim = OBS_MEAN.shape[0] + LAT_MEAN.shape[0]
    model = Normal(joint_dim, PositiveDefinite())
    joint_mean = jnp.concatenate([OBS_MEAN, LAT_MEAN])
    joint_cov = jnp.block([[OBS_COV, INT_COV], [INT_COV.T, LAT_COV]])
    cov_params = model.cov_man.from_matrix(joint_cov)
    means = model.join_mean_covariance(joint_mean, cov_params)
    return model, means


@pytest.fixture
def samples(joint_normal: tuple[FullNormal, Array]) -> Array:
    """Sample data from ground truth distribution."""
    model, means = joint_normal
    natural_params = model.to_natural(means)
    return model.sample(jax.random.PRNGKey(0), natural_params, SAMPLE_SIZE)


class TestFactorAnalysis:
    """Test FactorAnalysis properties."""

    @pytest.mark.parametrize("obs_dim,lat_dim", [(3, 2), (4, 2)])
    def test_dimensions(self, obs_dim: int, lat_dim: int) -> None:
        fa = factor_analysis(obs_dim=obs_dim, lat_dim=lat_dim)
        assert fa.obs_dim == fa.obs_man.data_dim
        assert fa.lat_dim == fa.lat_man.data_dim

    def test_from_loadings(self) -> None:
        """Factor analysis loadings parameterization recovers correct observable."""
        loadings = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        means = jnp.array([1.0, 2.0, 3.0])
        diags = jnp.array([0.1, 0.1, 0.1])

        fa = factor_analysis(obs_dim=3, lat_dim=2)
        params = fa.initialize_from_loadings(loadings, means, diags)

        nor_man, nor_params = fa.observable_distribution(params)
        mean, cov = nor_man.split_mean_covariance(nor_man.to_mean(nor_params))
        cov_mat = nor_man.cov_man.to_matrix(cov)

        assert jnp.allclose(mean, means, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(cov_mat, loadings @ loadings.T + jnp.diag(diags), rtol=RTOL, atol=ATOL)


class TestNormalAnalyticLGM:
    """Test NormalAnalyticLGM fitting, density, and conjugation."""

    @pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
    def test_lgm_matches_normal(
        self,
        rep: Scale | Diagonal | PositiveDefinite,
        samples: Array,
        joint_normal: tuple[FullNormal, Array],
    ) -> None:
        """LGM log-density matches equivalent Normal distribution and scipy."""
        obs_dim, lat_dim = OBS_MEAN.shape[0], LAT_MEAN.shape[0]
        lgm = NormalAnalyticLGM(obs_dim, rep, lat_dim)
        nor_man, _ = joint_normal

        means = lgm.average_sufficient_statistic(samples)
        params = lgm.to_natural(means)
        nor_params = lgm.to_normal(params)

        assert lgm.check_natural_parameters(params)
        assert jnp.allclose(lgm.to_mean(params), means, rtol=RTOL, atol=ATOL)

        # Log partition and density match Normal
        assert jnp.allclose(
            lgm.log_partition_function(params),
            nor_man.log_partition_function(nor_params),
            rtol=RTOL,
            atol=ATOL,
        )
        lgm_ll = lgm.average_log_density(params, samples)
        assert jnp.allclose(
            lgm_ll, nor_man.average_log_density(nor_params, samples), rtol=RTOL, atol=ATOL
        )

        # Match scipy
        mean, cov = nor_man.split_mean_covariance(nor_man.to_mean(nor_params))
        scipy_ll = jnp.mean(
            jax.vmap(stats.multivariate_normal.logpdf, in_axes=(0, None, None))(
                samples, mean, nor_man.cov_man.to_matrix(cov)
            )
        )
        assert jnp.allclose(lgm_ll, scipy_ll, rtol=RTOL, atol=ATOL)

    def test_observable_distribution_consistency(self, samples: Array) -> None:
        """observable_density matches density of observable_distribution."""
        obs_dim, lat_dim = OBS_MEAN.shape[0], LAT_MEAN.shape[0]
        lgm = NormalAnalyticLGM(obs_dim, Diagonal(), lat_dim)

        means = lgm.average_sufficient_statistic(samples)
        params = lgm.to_natural(means)

        obs_data = samples[:, :obs_dim]
        direct_ll = lgm.average_log_observable_density(params, obs_data)
        nor_man, marginal = lgm.observable_distribution(params)
        marginal_ll = nor_man.average_log_density(marginal, obs_data)
        assert jnp.allclose(direct_ll, marginal_ll, rtol=RTOL, atol=ATOL)

    def test_representation_consistency(self, samples: Array) -> None:
        """Embedded representations give same log-density."""
        obs_dim, lat_dim = OBS_MEAN.shape[0], LAT_MEAN.shape[0]
        iso = NormalAnalyticLGM(obs_dim, Scale(), lat_dim)
        dia = NormalAnalyticLGM(obs_dim, Diagonal(), lat_dim)
        pod = NormalAnalyticLGM(obs_dim, PositiveDefinite(), lat_dim)

        iso_means = iso.average_sufficient_statistic(samples)
        iso_natural = iso.to_natural(iso_means)

        obs_params, int_params, lat_params = iso.split_coords(iso_natural)
        dia_natural = dia.join_coords(iso.obs_man.embed_rep(dia.obs_man, obs_params), int_params, lat_params)
        pod_natural = pod.join_coords(iso.obs_man.embed_rep(pod.obs_man, obs_params), int_params, lat_params)

        iso_lpf = iso.log_partition_function(iso_natural)
        dia_lpf = dia.log_partition_function(dia_natural)
        pod_lpf = pod.log_partition_function(pod_natural)
        assert jnp.allclose(iso_lpf, dia_lpf, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(dia_lpf, pod_lpf, rtol=RTOL, atol=ATOL)

    def test_conjugation_equation(self) -> None:
        """Verify \\psi(\\theta_X + \\theta_{XZ} s_Z(z)) = \\rho s_Z(z) + \\psi_X(\\theta_X)."""
        obs_dim, lat_dim = 3, 2
        model = NormalAnalyticLGM(obs_dim, PositiveDefinite(), lat_dim)
        key = jax.random.PRNGKey(42)
        params = model.initialize(key, location=0.0, shape=1.0)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        for i in range(3):
            z = jax.random.normal(jax.random.fold_in(key, i), (lat_dim,))
            s_z = model.pst_man.sufficient_statistic(z)
            lhs = model.obs_man.log_partition_function(model.lkl_fun_man(lkl_params, z))
            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)
            assert jnp.abs(lhs - rhs) < 1e-5


class TestBoltzmannLGM:
    """Test BoltzmannLGM conjugation, density, and posterior."""

    @pytest.mark.parametrize("obs_dim,lat_dim", [(2, 3), (2, 4)])
    def test_conjugation_equation(self, obs_dim: int, lat_dim: int) -> None:
        """Conjugation equation holds for all Boltzmann states."""
        model = BoltzmannLGM(obs_dim=obs_dim, obs_rep=PositiveDefinite(), lat_dim=lat_dim)
        key = jax.random.PRNGKey(42)
        params = model.initialize(key, location=0.0, shape=0.5)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)
        assert rho.shape[0] == model.prr_man.dim

        states = model.lat_man.states
        n_test = min(8, len(states))
        step = max(1, len(states) // n_test)
        for idx in range(0, len(states), step):
            state = states[idx]
            s_z = model.lat_man.sufficient_statistic(state)
            lhs = model.obs_man.log_partition_function(model.lkl_fun_man(lkl_params, state))
            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)
            assert jnp.abs(lhs - rhs) < ATOL

    def test_observable_density_marginalization(self) -> None:
        """p(x) = sum_z p(x|z)p(z) by direct marginalization."""
        model = BoltzmannLGM(obs_dim=2, obs_rep=PositiveDefinite(), lat_dim=3)
        key = jax.random.PRNGKey(42)
        params = model.initialize(key, location=0.0, shape=0.5)
        obs = jax.random.normal(key, (2,))

        density_model = model.observable_density(params, obs)
        lkl_params, prior_params = model.split_conjugated(params)

        def joint_density(state: Array) -> Array:
            conditional_obs = model.lkl_fun_man(lkl_params, state)
            return model.obs_man.density(conditional_obs, obs) * model.lat_man.density(
                prior_params, state
            )

        density_marginal = jnp.sum(jax.vmap(joint_density)(model.lat_man.states))
        assert jnp.allclose(density_model, density_marginal, rtol=RTOL, atol=ATOL)

    def test_posterior_normalization(self) -> None:
        """p(z|x) sums to 1 over all states."""
        model = BoltzmannLGM(obs_dim=2, obs_rep=PositiveDefinite(), lat_dim=3)
        key = jax.random.PRNGKey(42)
        params = model.initialize(key, location=0.0, shape=0.5)
        obs = jax.random.normal(key, (2,))

        posterior_params = model.posterior_at(params, obs)
        densities = jax.vmap(lambda s: model.lat_man.density(posterior_params, s))(
            model.lat_man.states
        )
        assert jnp.allclose(jnp.sum(densities), 1.0, rtol=RTOL, atol=ATOL)

    def test_log_density_formula(self) -> None:
        """Log-density decomposes as theta*s(x) + psi(post) - psi(prior) - psi_X(theta_X) + log mu(x)."""
        model = BoltzmannLGM(obs_dim=2, obs_rep=PositiveDefinite(), lat_dim=3)
        key = jax.random.PRNGKey(42)
        params = model.initialize(key, location=0.0, shape=0.5)
        obs = jax.random.normal(key, (2,))

        log_density = model.log_observable_density(params, obs)
        obs_params, _, _ = model.split_coords(params)

        manual = (
            jnp.dot(obs_params, model.obs_man.sufficient_statistic(obs))
            + model.lat_man.log_partition_function(model.posterior_at(params, obs))
            - model.prr_man.log_partition_function(model.prior(params))
            - model.obs_man.log_partition_function(obs_params)
            + model.obs_man.log_base_measure(obs)
        )
        assert jnp.allclose(log_density, manual, rtol=RTOL, atol=ATOL)


class TestFactorAnalysisSampling:
    """Test ancestral sampling from FactorAnalysis."""

    def test_sampling(self) -> None:
        """Ancestral samples have correct shape and observable marginal matches."""
        obs_dim, lat_dim = 4, 2
        fa = factor_analysis(obs_dim=obs_dim, lat_dim=lat_dim)
        params = fa.initialize(jax.random.PRNGKey(42), location=0.0, shape=0.5)
        samples = fa.sample(jax.random.PRNGKey(0), params, 50_000)
        assert samples.shape == (50_000, obs_dim + lat_dim)

        nor_man, obs_nat = fa.observable_distribution(params)
        obs_mean = nor_man.split_mean_covariance(nor_man.to_mean(obs_nat))[0]
        empirical_obs_mean = jnp.mean(samples[:, :obs_dim], axis=0)
        assert jnp.allclose(empirical_obs_mean, obs_mean, atol=0.05)


class TestPCA:
    """Test PrincipalComponentAnalysis factory and properties."""

    def test_pca_dimensions(self) -> None:
        pca = principal_component_analysis(obs_dim=4, lat_dim=2)
        assert pca.obs_dim == 4
        assert pca.lat_dim == 2
        assert pca.obs_man.data_dim == 4
        assert pca.lat_man.data_dim == 2

    def test_pca_matches_lgm(self) -> None:
        """PCA log-density matches NormalAnalyticLGM with Scale rep."""
        pca = principal_component_analysis(obs_dim=4, lat_dim=2)
        lgm = NormalAnalyticLGM(4, Scale(), 2)
        params = pca.initialize(jax.random.PRNGKey(42), location=0.0, shape=0.5)
        xs = jax.random.normal(jax.random.PRNGKey(7), (100, 6))
        pca_ll = pca.average_log_density(params, xs)
        lgm_ll = lgm.average_log_density(params, xs)
        assert jnp.allclose(pca_ll, lgm_ll, rtol=RTOL, atol=ATOL)


class TestNormalCovarianceEmbedding:
    """Test NormalCovarianceEmbedding between Diagonal and FullNormal."""

    def test_embed_project_round_trip(self) -> None:
        """Embed DiagonalNormal into FullNormal, project back, recover original (mean coords)."""
        dim = 3
        sub = Normal(dim, Diagonal())
        amb = Normal(dim, PositiveDefinite())
        emb = NormalCovarianceEmbedding(sub, amb)

        diag_cov = jnp.array([1.5, 2.0, 0.8])
        means = sub.join_mean_covariance(jnp.array([1.0, -0.5, 0.3]), diag_cov)
        params = sub.to_natural(means)

        amb_params = emb.embed(params)
        amb_means = amb.to_mean(amb_params)
        recovered_means = emb.project(amb_means)
        assert jnp.allclose(means, recovered_means, rtol=RTOL, atol=ATOL)

    def test_embed_preserves_density(self) -> None:
        """Embedded params give same log-density as original for the same data."""
        dim = 3
        sub = Normal(dim, Diagonal())
        amb = Normal(dim, PositiveDefinite())
        emb = NormalCovarianceEmbedding(sub, amb)

        diag_cov = jnp.array([1.5, 2.0, 0.8])
        means = sub.join_mean_covariance(jnp.array([1.0, -0.5, 0.3]), diag_cov)
        params = sub.to_natural(means)
        amb_params = emb.embed(params)

        xs = jax.random.normal(jax.random.PRNGKey(7), (100, dim))
        sub_ll = sub.average_log_density(params, xs)
        amb_ll = amb.average_log_density(amb_params, xs)
        assert jnp.allclose(sub_ll, amb_ll, rtol=RTOL, atol=ATOL)
