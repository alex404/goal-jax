"""Tests for Linear Gaussian Models.

Tests FactorAnalysis, NormalAnalyticLGM, and BoltzmannLGM.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import BoltzmannLGM, FactorAnalysis, Normal, NormalAnalyticLGM

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Tolerances
RTOL = 1e-4
ATOL = 1e-4

# Test parameters for joint distribution
OBS_MEAN = jnp.array([2.0, -1.0, 0.0])
LAT_MEAN = jnp.array([3.0, -2.0])
OBS_COV = jnp.array([[4.0, 1.5, 0.0], [1.5, 3.0, 1.0], [0.0, 1.0, 2.0]])
INT_COV = jnp.array([[-1.0, 0.5], [0.0, 0.0], [0.5, 0.0]])
LAT_COV = jnp.array([[3.0, 1.0], [1.0, 2.0]])
SAMPLE_SIZE = 1000


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def joint_normal() -> tuple[Normal, Array]:
    """Ground truth joint normal model and mean parameters."""
    joint_dim = OBS_MEAN.shape[0] + LAT_MEAN.shape[0]
    model = Normal(joint_dim, PositiveDefinite())
    joint_mean = jnp.concatenate([OBS_MEAN, LAT_MEAN])
    joint_cov = jnp.block([[OBS_COV, INT_COV], [INT_COV.T, LAT_COV]])
    cov_params = model.cov_man.from_matrix(joint_cov)
    means = model.join_mean_covariance(joint_mean, cov_params)
    return model, means


@pytest.fixture
def samples(joint_normal: tuple[Normal, Array]) -> Array:
    """Sample data from ground truth distribution."""
    model, means = joint_normal
    key = jax.random.PRNGKey(0)
    natural_params = model.to_natural(means)
    return model.sample(key, natural_params, SAMPLE_SIZE)


class TestFactorAnalysisBasics:
    """Test basic FactorAnalysis properties."""

    @pytest.fixture(params=[(3, 2), (4, 2)])
    def model(self, request: pytest.FixtureRequest) -> FactorAnalysis:
        """Create FactorAnalysis model."""
        obs_dim, lat_dim = request.param
        return FactorAnalysis(obs_dim=obs_dim, lat_dim=lat_dim)

    def test_dimensions(self, model: FactorAnalysis) -> None:
        """Test dimensions are correct."""
        assert model.obs_dim == model.obs_man.data_dim
        assert model.lat_dim == model.lat_man.data_dim

    def test_from_loadings(self) -> None:
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

        assert jnp.allclose(mean, means, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(cov_mat, expected_cov, rtol=RTOL, atol=ATOL)


class TestNormalAnalyticLGMBasics:
    """Test NormalAnalyticLGM basic properties."""

    @pytest.fixture(params=[Scale(), Diagonal(), PositiveDefinite()])
    def model(
        self, request: pytest.FixtureRequest
    ) -> NormalAnalyticLGM:
        """Create LGM with various representations."""
        obs_dim, lat_dim = OBS_MEAN.shape[0], LAT_MEAN.shape[0]
        return NormalAnalyticLGM(obs_dim, request.param, lat_dim)

    def test_dimensions(self, model: NormalAnalyticLGM) -> None:
        """Test dimensions are correct."""
        assert model.obs_man.data_dim == OBS_MEAN.shape[0]
        assert model.lat_man.data_dim == LAT_MEAN.shape[0]


class TestNormalAnalyticLGMFitting:
    """Test NormalAnalyticLGM fitting and likelihood."""

    @pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
    def test_lgm_matches_normal(
        self,
        rep: Scale | Diagonal | PositiveDefinite,
        samples: Array,
        joint_normal: tuple[Normal, Array],
    ) -> None:
        """Test LGM log-density matches equivalent Normal distribution."""
        obs_dim, lat_dim = OBS_MEAN.shape[0], LAT_MEAN.shape[0]
        lgm = NormalAnalyticLGM(obs_dim, rep, lat_dim)
        nor_man, _ = joint_normal

        # Fit and convert
        means = lgm.average_sufficient_statistic(samples)
        params = lgm.to_natural(means)
        nor_params = lgm.to_normal(params)

        assert lgm.check_natural_parameters(params)

        # Parameter recovery
        recovered = lgm.to_mean(params)
        assert jnp.allclose(recovered, means, rtol=RTOL, atol=ATOL)

        # Log partition functions match
        lgm_lpf = lgm.log_partition_function(params)
        nor_lpf = nor_man.log_partition_function(nor_params)
        assert jnp.allclose(lgm_lpf, nor_lpf, rtol=RTOL, atol=ATOL)

        # Log densities match
        lgm_ll = lgm.average_log_density(params, samples)
        nor_ll = nor_man.average_log_density(nor_params, samples)
        assert jnp.allclose(lgm_ll, nor_ll, rtol=RTOL, atol=ATOL)

        # Compare to scipy
        mean, cov = nor_man.split_mean_covariance(nor_man.to_mean(nor_params))
        scipy_ll = jnp.mean(
            jax.vmap(stats.multivariate_normal.logpdf, in_axes=(0, None, None))(
                samples, mean, nor_man.cov_man.to_matrix(cov)
            )
        )
        assert jnp.allclose(lgm_ll, scipy_ll, rtol=RTOL, atol=ATOL)


class TestNormalAnalyticLGMObservable:
    """Test NormalAnalyticLGM observable distribution."""

    def test_observable_distribution_consistency(self, samples: Array) -> None:
        """Test observable_density matches density of observable_distribution."""
        obs_dim, lat_dim = OBS_MEAN.shape[0], LAT_MEAN.shape[0]
        lgm = NormalAnalyticLGM(obs_dim, Diagonal(), lat_dim)

        means = lgm.average_sufficient_statistic(samples)
        params = lgm.to_natural(means)

        obs_data = samples[:, :obs_dim]
        direct_ll = lgm.average_log_observable_density(params, obs_data)

        nor_man, marginal = lgm.observable_distribution(params)
        marginal_ll = nor_man.average_log_density(marginal, obs_data)

        assert jnp.allclose(direct_ll, marginal_ll, rtol=RTOL, atol=ATOL)


class TestNormalAnalyticLGMRepresentations:
    """Test representation consistency in LGM."""

    def test_representation_consistency(self, samples: Array) -> None:
        """Test embedded representations give same results."""
        obs_dim, lat_dim = OBS_MEAN.shape[0], LAT_MEAN.shape[0]
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
        assert jnp.allclose(iso_lpf, dia_lpf, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(dia_lpf, pod_lpf, rtol=RTOL, atol=ATOL)

        # Log densities match
        iso_ll = iso.average_log_density(iso_natural, samples)
        dia_ll = dia.average_log_density(dia_natural, samples)
        pod_ll = pod.average_log_density(pod_natural, samples)
        assert jnp.allclose(iso_ll, dia_ll, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(dia_ll, pod_ll, rtol=RTOL, atol=ATOL)


class TestNormalAnalyticLGMConjugation:
    """Test conjugation equation for NormalAnalyticLGM."""

    def test_conjugation_equation(self, key: Array) -> None:
        """Test: ψ(θ_X + θ_{XZ}·s_Z(z)) = ρ·s_Z(z) + ψ_X(θ_X)."""
        obs_dim, lat_dim = 3, 2
        model = NormalAnalyticLGM(obs_dim, PositiveDefinite(), lat_dim)

        params = model.initialize(key, location=0.0, shape=1.0)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        for i in range(3):
            z = jax.random.normal(jax.random.fold_in(key, i), (lat_dim,))
            s_z = model.pst_man.sufficient_statistic(z)

            # LHS: ψ(θ_X + θ_{XZ}·s_Z(z))
            conditional_obs = model.lkl_fun_man(lkl_params, z)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            # RHS: ρ·s_Z(z) + ψ_X(θ_X)
            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

            assert jnp.abs(lhs - rhs) < 1e-5


class TestBoltzmannLGMBasics:
    """Test BoltzmannLGM basic properties."""

    @pytest.fixture(params=[(2, 3), (2, 4)])
    def model(self, request: pytest.FixtureRequest) -> BoltzmannLGM:
        """Create BoltzmannLGM model."""
        obs_dim, lat_dim = request.param
        return BoltzmannLGM(obs_dim=obs_dim, obs_rep=PositiveDefinite(), lat_dim=lat_dim)

    @pytest.fixture
    def params(self, model: BoltzmannLGM, key: Array) -> Array:
        """Generate random model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    def test_dimensions(self, model: BoltzmannLGM) -> None:
        """Test dimensions are correct."""
        assert model.obs_man.data_dim == model.obs_dim
        assert model.lat_man.data_dim == model.lat_dim


class TestBoltzmannLGMConjugation:
    """Test BoltzmannLGM conjugation equation."""

    @pytest.fixture(params=[(2, 3), (2, 4)])
    def model(self, request: pytest.FixtureRequest) -> BoltzmannLGM:
        """Create BoltzmannLGM model."""
        obs_dim, lat_dim = request.param
        return BoltzmannLGM(obs_dim=obs_dim, obs_rep=PositiveDefinite(), lat_dim=lat_dim)

    @pytest.fixture
    def params(self, model: BoltzmannLGM, key: Array) -> Array:
        """Generate random model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    def test_conjugation_equation(self, model: BoltzmannLGM, params: Array) -> None:
        """Test: ψ(θ_X + θ_{XZ}·s_Z(z)) = ρ·s_Z(z) + ψ_X(θ_X)."""
        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        states = model.lat_man.states
        n_test = min(8, len(states))
        step = max(1, len(states) // n_test)

        for idx in range(0, len(states), step):
            state = states[idx]
            s_z = model.lat_man.sufficient_statistic(state)

            conditional_obs = model.lkl_fun_man(lkl_params, state)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

            assert jnp.abs(lhs - rhs) < ATOL

    def test_conjugation_parameters_shape(
        self, model: BoltzmannLGM, params: Array
    ) -> None:
        """Test conjugation parameters have correct dimension."""
        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        assert rho.shape[0] == model.prr_man.dim


class TestBoltzmannLGMDensity:
    """Test BoltzmannLGM density computation."""

    @pytest.fixture(params=[(2, 3)])
    def model(self, request: pytest.FixtureRequest) -> BoltzmannLGM:
        """Create BoltzmannLGM model."""
        obs_dim, lat_dim = request.param
        return BoltzmannLGM(obs_dim=obs_dim, obs_rep=PositiveDefinite(), lat_dim=lat_dim)

    @pytest.fixture
    def params(self, model: BoltzmannLGM, key: Array) -> Array:
        """Generate random model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    def test_observable_density_marginalization(
        self, model: BoltzmannLGM, params: Array, key: Array
    ) -> None:
        """Test p(x) = Σ_z p(x|z)p(z)."""
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

        assert jnp.allclose(density_model, density_marginal, rtol=RTOL, atol=ATOL)

    def test_posterior_normalization(
        self, model: BoltzmannLGM, params: Array, key: Array
    ) -> None:
        """Test that p(z|x) sums to 1."""
        obs = jax.random.normal(key, (model.obs_dim,))

        posterior_params = model.posterior_at(params, obs)
        states = model.lat_man.states

        densities = jax.vmap(lambda s: model.lat_man.density(posterior_params, s))(
            states
        )
        total = jnp.sum(densities)

        assert jnp.allclose(total, 1.0, rtol=RTOL, atol=ATOL)

    def test_log_density_formula(
        self, model: BoltzmannLGM, params: Array, key: Array
    ) -> None:
        """Test log p(x) = θ·s(x) + ψ(post) - ψ(prior) - ψ_X(θ_X) + log μ(x)."""
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

        assert jnp.allclose(log_density, manual, rtol=RTOL, atol=ATOL)
