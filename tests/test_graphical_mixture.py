"""Tests for CompleteMixtureOfConjugated.

Tests the graphical mixture model combining harmoniums with categorical latents.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Diagonal
from goal.models import DiagonalNormal, FactorAnalysis, FullNormal
from goal.models.graphical.mixture import CompleteMixtureOfConjugated
from goal.models.harmonium.lgm import NormalLGM

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Tolerances
RTOL = 1e-4
ATOL = 1e-6


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestCompleteMixtureBasics:
    """Test basic CompleteMixtureOfConjugated properties."""

    @pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
    def model(
        self, request: pytest.FixtureRequest
    ) -> CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal]:
        """Create CompleteMixtureOfConjugated with FactorAnalysis base."""
        obs_dim, lat_dim, n_cat = request.param
        base_fa = FactorAnalysis(obs_dim=obs_dim, lat_dim=lat_dim)
        return CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_fa
        )

    @pytest.fixture
    def params(
        self, model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal], key: Array
    ) -> Array:
        """Generate random model parameters."""
        return model.initialize(key, location=0.0, shape=1.0)

    def test_dimension_consistency(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test all manifold dimensions are consistent."""
        assert params.shape[0] == model.dim

        obs, int_params, lat = model.split_coords(params)
        assert obs.shape[0] == model.obs_man.dim
        assert int_params.shape[0] == model.int_man.dim
        assert lat.shape[0] == model.pst_man.dim

    def test_interaction_blocks_sum(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test interaction blocks sum to total."""
        _, int_params, _ = model.split_coords(params)
        xy, xyk, xk = model.int_man.coord_blocks(int_params)
        assert xy.shape[0] + xyk.shape[0] + xk.shape[0] == model.int_man.dim

    def test_domain_codomain_consistency(
        self, model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal]
    ) -> None:
        """Test domain/codomain consistency."""
        assert model.int_man.dom_man.dim == model.pst_man.dim
        assert model.int_man.cod_man.dim == model.obs_man.dim


class TestCompleteMixtureConjugation:
    """Test conjugation parameters."""

    @pytest.fixture(params=[(3, 2, 2)])
    def model(
        self, request: pytest.FixtureRequest
    ) -> CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal]:
        """Create CompleteMixtureOfConjugated."""
        obs_dim, lat_dim, n_cat = request.param
        base_fa = FactorAnalysis(obs_dim=obs_dim, lat_dim=lat_dim)
        return CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_fa
        )

    @pytest.fixture
    def params(
        self, model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal], key: Array
    ) -> Array:
        """Generate random model parameters."""
        return model.initialize(key, location=0.0, shape=1.0)

    def test_conjugation_parameters_shape(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test conjugation parameters have correct shape."""
        obs, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs, int_params)
        rho = model.conjugation_parameters(lkl_params)

        # Conjugation parameters are in prr_man (prior) space
        assert rho.shape[0] == model.prr_man.dim


class TestCompleteMixturePosterior:
    """Test posterior computation."""

    @pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
    def model(
        self, request: pytest.FixtureRequest
    ) -> CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal]:
        """Create CompleteMixtureOfConjugated."""
        obs_dim, lat_dim, n_cat = request.param
        base_fa = FactorAnalysis(obs_dim=obs_dim, lat_dim=lat_dim)
        return CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_fa
        )

    @pytest.fixture
    def params(
        self, model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal], key: Array
    ) -> Array:
        """Generate random model parameters."""
        return model.initialize(key, location=0.0, shape=1.0)

    def test_posterior_dimension(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test posterior has correct dimension."""
        obs = jnp.ones(model.obs_man.data_dim)
        posterior = model.posterior_at(params, obs)
        assert posterior.shape[0] == model.pst_man.dim

    def test_soft_assignments_sum_to_one(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test soft assignments sum to 1."""
        obs = jnp.ones(model.obs_man.data_dim)
        soft = model.posterior_soft_assignments(params, obs)

        assert soft.shape[0] == model.n_categories
        assert jnp.allclose(jnp.sum(soft), 1.0)

    def test_hard_assignment_valid(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test hard assignment is valid index."""
        obs = jnp.ones(model.obs_man.data_dim)
        hard = model.posterior_hard_assignment(params, obs)

        assert 0 <= hard < model.n_categories


class TestCompleteMixtureInteraction:
    """Test interaction block operations."""

    @pytest.fixture
    def model(self) -> CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal]:
        """Create CompleteMixtureOfConjugated."""
        base_fa = FactorAnalysis(obs_dim=3, lat_dim=2)
        return CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal](
            n_categories=2, bas_hrm=base_fa
        )

    def test_interaction_blocks_callable(
        self, model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal]
    ) -> None:
        """Test each interaction block is callable."""
        xy_params = model.xy_man.zeros()
        xyk_params = model.xyk_man.zeros()
        xk_params = model.xk_man.zeros()
        lat_coords = model.pst_man.zeros()

        result_xy = model.xy_man(xy_params, lat_coords)
        result_xyk = model.xyk_man(xyk_params, lat_coords)
        result_xk = model.xk_man(xk_params, lat_coords)

        assert result_xy.shape[0] == model.obs_man.dim
        assert result_xyk.shape[0] == model.obs_man.dim
        assert result_xk.shape[0] == model.obs_man.dim


class TestCompleteMixtureRepresentation:
    """Test mixture representation conversion."""

    @pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
    def model(
        self, request: pytest.FixtureRequest
    ) -> CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal]:
        """Create CompleteMixtureOfConjugated."""
        obs_dim, lat_dim, n_cat = request.param
        base_fa = FactorAnalysis(obs_dim=obs_dim, lat_dim=lat_dim)
        return CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_fa
        )

    @pytest.fixture
    def params(
        self, model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal], key: Array
    ) -> Array:
        """Generate random model parameters."""
        return model.initialize(key, location=0.0, shape=1.0)

    def test_mixture_round_trip(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test conversion to/from CompleteMixture is invertible."""
        mix_params = model.to_mixture_params(params)
        recovered = model.from_mixture_params(mix_params)
        assert jnp.allclose(params, recovered, atol=1e-10)

    def test_mixture_dimension(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test mix_man has correct dimension."""
        mix_params = model.to_mixture_params(params)
        assert mix_params.shape[0] == model.mix_man.dim

    def test_mixture_likelihood(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, FullNormal, FullNormal],
        params: Array,
        key: Array,
    ) -> None:
        """Test likelihood equivalence between representations."""
        mix_params = model.to_mixture_params(params)
        y = jax.random.normal(key, (model.bas_hrm.pst_man.data_dim,))

        comp_params, _ = model.mix_man.split_natural_mixture(mix_params)

        for k in range(model.n_categories):
            hrm_k = model.mix_man.cmp_man.get_replicate(comp_params, k)
            lkl_k = model.bas_hrm.likelihood_at(hrm_k, y)
            assert lkl_k.shape[0] == model.bas_hrm.obs_man.dim


class TestAsymmetricMixture:
    """Test CompleteMixtureOfConjugated with asymmetric base (pst_man != prr_man)."""

    @pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
    def model(
        self, request: pytest.FixtureRequest
    ) -> CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal]:
        """Create CompleteMixtureOfConjugated with NormalLGM[Diagonal, Diagonal] base."""
        obs_dim, lat_dim, n_cat = request.param
        # NormalLGM with diagonal posterior and full prior (asymmetric)
        base_lgm = NormalLGM(
            obs_dim=obs_dim, obs_rep=Diagonal(), lat_dim=lat_dim, pst_rep=Diagonal()
        )
        return CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_lgm
        )

    @pytest.fixture
    def params(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
        key: Array,
    ) -> Array:
        """Generate random model parameters."""
        return model.initialize(key, location=0.0, shape=1.0)

    def test_dimension_consistency(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test all manifold dimensions are consistent."""
        assert params.shape[0] == model.dim

        obs, int_params, lat = model.split_coords(params)
        assert obs.shape[0] == model.obs_man.dim
        assert int_params.shape[0] == model.int_man.dim
        assert lat.shape[0] == model.pst_man.dim

    def test_pst_prr_dimensions_differ(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
    ) -> None:
        """Test that pst_man and prr_man have different dimensions (asymmetric)."""
        # For NormalLGM[Diagonal, Diagonal], pst_man is DiagonalNormal, prr_man is FullNormal
        # DiagonalNormal has dim = 2*lat_dim, FullNormal has dim = lat_dim*(lat_dim+3)/2
        assert model.pst_man.dim != model.prr_man.dim
        # pst_man should be smaller
        assert model.pst_man.dim < model.prr_man.dim

    def test_conjugation_parameters_shape(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test conjugation parameters have correct shape (in prr_man space)."""
        obs, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs, int_params)
        rho = model.conjugation_parameters(lkl_params)

        # Conjugation parameters are in prr_man (prior) space, not pst_man
        assert rho.shape[0] == model.prr_man.dim

    def test_posterior_dimension(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test posterior has correct dimension (in pst_man space)."""
        obs = jnp.ones(model.obs_man.data_dim)
        posterior = model.posterior_at(params, obs)
        # Posterior is in pst_man space
        assert posterior.shape[0] == model.pst_man.dim

    def test_soft_assignments_sum_to_one(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
        params: Array,
    ) -> None:
        """Test soft assignments sum to 1."""
        obs = jnp.ones(model.obs_man.data_dim)
        soft = model.posterior_soft_assignments(params, obs)

        assert soft.shape[0] == model.n_categories
        assert jnp.allclose(jnp.sum(soft), 1.0)

    def test_embedding_roundtrip(
        self,
        model: CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
    ) -> None:
        """Test that pst_prr_emb correctly embeds and projects."""
        emb = model.pst_prr_emb

        # Create test params in pst_man space
        pst_params = model.pst_man.zeros()

        # Embed into prr_man space
        prr_params = emb.embed(pst_params)
        assert prr_params.shape[0] == model.prr_man.dim

        # Project back should recover original (for zero params)
        recovered = emb.project(prr_params)
        assert jnp.allclose(recovered, pst_params)
