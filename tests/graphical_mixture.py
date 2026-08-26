"""Tests for models/graphical/mixture.py.

Covers CompleteMixtureOfSymmetric, CompleteMixtureOfConjugated, and
MixtureOfFactorAnalyzers. Verifies dimension consistency, conjugation parameters,
posterior computation, interaction blocks, mixture representation round-trips,
asymmetric pst/prr handling, and to_natural/to_mean inversion.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import CliqueBlockEmbedding, Diagonal, EmbeddedMap, IdentityEmbedding
from goal.models import (
    DiagonalNormal,
    FullNormal,
    MixtureOfFactorAnalyzers,
    factor_analysis,
)
from goal.models.graphical.mixture import (
    CompleteMixtureOfConjugated,
    CompleteMixtureOfSymmetric,
)
from goal.models.harmonium.lgm import NormalLGM

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


# --- Symmetric mixture (FactorAnalysis base) ---


class TestCompleteMixtureOfSymmetric:
    """Test CompleteMixtureOfSymmetric with FactorAnalysis base."""

    @pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array]:
        obs_dim, lat_dim, n_cat = request.param
        base_fa = factor_analysis(obs_dim=obs_dim, lat_dim=lat_dim)
        model = CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_fa
        )
        params = model.initialize(jax.random.PRNGKey(42), location=0.0, shape=1.0)
        return model, params

    def test_dimension_consistency(
        self,
        model_and_params: tuple[
            CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array
        ],
    ) -> None:
        model, params = model_and_params
        assert params.shape[0] == model.dim
        obs, int_params, lat = model.split_coords(params)
        assert obs.shape[0] == model.obs_man.dim
        assert int_params.shape[0] == model.int_man.dim
        assert lat.shape[0] == model.pst_man.dim

    def test_domain_codomain_consistency(
        self,
        model_and_params: tuple[
            CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array
        ],
    ) -> None:
        model, _ = model_and_params
        assert model.int_man.dom_man.dim == model.pst_man.dim
        assert model.int_man.cod_man.dim == model.obs_man.dim
        assert model.lat_man.dim == model.pst_man.dim
        assert model.lat_man.dim == model.prr_man.dim

    def test_conjugation_parameters_shape(
        self,
        model_and_params: tuple[
            CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array
        ],
    ) -> None:
        model, params = model_and_params
        obs, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs, int_params)
        rho = model.conjugation_parameters(lkl_params)
        assert rho.shape[0] == model.prr_man.dim

    def test_posterior(
        self,
        model_and_params: tuple[
            CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array
        ],
    ) -> None:
        model, params = model_and_params
        obs = jnp.ones(model.obs_man.data_dim)
        posterior = model.posterior_at(params, obs)
        assert posterior.shape[0] == model.pst_man.dim

        soft = model.posterior_soft_assignments(params, obs)
        assert soft.shape[0] == model.n_categories
        assert jnp.allclose(jnp.sum(soft), 1.0)

        hard = model.posterior_hard_assignment(params, obs)
        assert 0 <= hard < model.n_categories

    def test_mixture_round_trip(
        self,
        model_and_params: tuple[
            CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array
        ],
    ) -> None:
        """to_mixture_coords / from_mixture_coords is invertible in both natural and mean coords."""
        model, params = model_and_params
        assert jnp.allclose(
            model.from_mixture_coords(model.to_mixture_coords(params)),
            params,
            atol=1e-10,
        )

        means = model.to_mean(params)
        assert jnp.allclose(
            model.from_mixture_coords(model.to_mixture_coords(means)), means, atol=1e-10
        )

    def test_mixture_means_match(
        self,
        model_and_params: tuple[
            CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array
        ],
    ) -> None:
        """Direct mean conversion matches natural->mean on mix_man."""
        model, params = model_and_params
        means = model.to_mean(params)
        direct = model.to_mixture_coords(means)
        via_natural = model.mix_man.to_mean(model.to_mixture_coords(params))
        assert jnp.allclose(direct, via_natural, atol=1e-10)

    def test_interaction_blocks(
        self,
        model_and_params: tuple[
            CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array
        ],
    ) -> None:
        model, params = model_and_params
        _, int_params, _ = model.split_coords(params)
        xy, xyk, xk = model.int_man.coord_blocks(int_params)
        assert xy.shape[0] + xyk.shape[0] + xk.shape[0] == model.int_man.dim


# --- Asymmetric mixture (NormalLGM base, pst_man != prr_man) ---


class TestCompleteMixtureOfConjugated:
    """Test CompleteMixtureOfConjugated with asymmetric base."""

    @pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[
        CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal], Array
    ]:
        obs_dim, lat_dim, n_cat = request.param
        base_lgm = NormalLGM(
            obs_dim=obs_dim, obs_rep=Diagonal(), lat_dim=lat_dim, pst_rep=Diagonal()
        )
        model = CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_lgm
        )
        params = model.initialize(jax.random.PRNGKey(42), location=0.0, shape=1.0)
        return model, params

    def test_dimension_consistency(
        self,
        model_and_params: tuple[
            CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
            Array,
        ],
    ) -> None:
        model, params = model_and_params
        assert params.shape[0] == model.dim
        obs, int_params, lat = model.split_coords(params)
        assert obs.shape[0] == model.obs_man.dim
        assert int_params.shape[0] == model.int_man.dim
        assert lat.shape[0] == model.pst_man.dim

    def test_pst_prr_dimensions_differ(
        self,
        model_and_params: tuple[
            CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
            Array,
        ],
    ) -> None:
        model, _ = model_and_params
        assert model.pst_man.dim != model.prr_man.dim
        assert model.pst_man.dim < model.prr_man.dim

    def test_conjugation_parameters_in_prr_space(
        self,
        model_and_params: tuple[
            CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
            Array,
        ],
    ) -> None:
        model, params = model_and_params
        obs, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs, int_params)
        rho = model.conjugation_parameters(lkl_params)
        assert rho.shape[0] == model.prr_man.dim

    def test_posterior_and_assignments(
        self,
        model_and_params: tuple[
            CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
            Array,
        ],
    ) -> None:
        model, params = model_and_params
        obs = jnp.ones(model.obs_man.data_dim)
        posterior = model.posterior_at(params, obs)
        assert posterior.shape[0] == model.pst_man.dim

        soft = model.posterior_soft_assignments(params, obs)
        assert soft.shape[0] == model.n_categories
        assert jnp.allclose(jnp.sum(soft), 1.0)

    def test_embedding_roundtrip(
        self,
        model_and_params: tuple[
            CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal],
            Array,
        ],
    ) -> None:
        model, _ = model_and_params
        emb = model.pst_prr_emb
        pst_params = model.pst_man.zeros()
        prr_params = emb.embed(pst_params)
        assert prr_params.shape[0] == model.prr_man.dim
        recovered = emb.project(prr_params)
        assert jnp.allclose(recovered, pst_params)


# --- MFA to_natural round-trip (moved from whitening) ---


def test_mfa_to_natural_round_trip() -> None:
    """MFA to_natural is a left-inverse of to_mean: to_natural(to_mean(params)) == params."""
    fa = factor_analysis(obs_dim=8, lat_dim=3)
    mfa = MixtureOfFactorAnalyzers(n_categories=4, bas_hrm=fa)

    params = mfa.initialize(jax.random.PRNGKey(99), location=0.0, shape=0.5)
    means = mfa.to_mean(params)
    recovered_params = mfa.to_natural(means)

    assert jnp.allclose(params, recovered_params, atol=1e-5), (
        "MFA to_natural(to_mean(params)) != params"
    )


# --- The derived graph ---


class TestMFAGraph:
    """MFA's graph is derived from its coupling pattern, not declared clique by clique.

    The model states only which nodes each interaction block couples --- ``int_members``,
    three tuples. Node count, root count, the biases, the ``(y, k)`` coupling from the
    mixture one level up, the levels, and the block layout all follow from that. These
    tests pin what follows, because a wrong derivation would be silent: every operation
    below reads the level split, which does not consult the graph.
    """

    @staticmethod
    def _mfa(obs_dim: int = 4, lat_dim: int = 2, n_categories: int = 3):
        return MixtureOfFactorAnalyzers(
            n_categories=n_categories,
            bas_hrm=factor_analysis(obs_dim=obs_dim, lat_dim=lat_dim),
        )

    def test_three_nodes_one_root(self) -> None:
        clq = self._mfa().clq_set
        assert clq.n_nodes == 3
        assert clq.n_roots == 1

    def test_all_seven_cliques(self) -> None:
        """Three biases, three couplings, and the triple interaction."""
        assert self._mfa().clq_set.canonical_cliques == (
            (0,),
            (0, 1),
            (0, 1, 2),
            (0, 2),
            (1,),
            (1, 2),
            (2,),
        )

    def test_depth_is_two_not_three(self) -> None:
        """Both $y$ and $k$ are adjacent to $x$, so this is a fork, not a chain.

        This is why ``mix_cut`` cannot use ``levels[-1]``: the deepest level holds $y$ as
        well as $k$, and cutting there would take $y$ along with it.
        """
        assert self._mfa().clq_set.level_sets == ((0,), (1, 2))

    def test_one_block_per_clique(self) -> None:
        mfa = self._mfa()
        assert len(mfa.clique_dims) == len(mfa.clq_set.canonical_cliques)
        assert sum(mfa.clique_dims) == mfa.dim

    def test_block_layout_follows_the_clique_order(self) -> None:
        """obs, then the interaction's three blocks, then the mixture's three."""
        mfa = self._mfa(obs_dim=4, lat_dim=2, n_categories=3)
        xy, xyk, xk = mfa.int_man.block_dims
        expected = (mfa.obs_man.dim, xy, xyk, xk, *mfa.pst_man.clique_dims)
        assert mfa.clique_dims == expected

    def test_cut_still_isolates_the_category_node(self) -> None:
        """The mixture view is a re-view of the derived graph, so it must survive it."""
        mfa = self._mfa()
        params = jax.random.normal(jax.random.PRNGKey(21), (mfa.dim,))
        assert jnp.allclose(
            mfa.from_mixture_coords(mfa.to_mixture_coords(params)), params
        )


class TestDerivedInteractionEmbeddings:
    """MFA's three interaction blocks are derived from members plus selectors.

    All three are one construction, ``CliqueBlockEmbedding``, differing only in which nodes
    they name and how much of each one's statistic they select. These tests pin that
    behaviour directly.

    The mixture's node frame is $y = 0$, $k = 1$.
    """

    @staticmethod
    def _mfa(obs_dim: int = 4, lat_dim: int = 2, n_categories: int = 3):
        return MixtureOfFactorAnalyzers(
            n_categories=n_categories,
            bas_hrm=factor_analysis(obs_dim=obs_dim, lat_dim=lat_dim),
        )

    @classmethod
    def _blocks(cls, **kwargs) -> tuple[EmbeddedMap[Any, Any], ...]:
        blocks = cls._mfa(**kwargs).int_man.blocks
        for block in blocks:
            assert isinstance(block, EmbeddedMap)
        return blocks  # pyright: ignore[reportReturnType]

    @classmethod
    def _dom_embs(cls, **kwargs) -> tuple[CliqueBlockEmbedding[Any], ...]:
        embs = tuple(block.dom_emb for block in cls._blocks(**kwargs))
        for emb in embs:
            assert isinstance(emb, CliqueBlockEmbedding)
        return embs  # pyright: ignore[reportReturnType]

    def test_each_block_addresses_its_own_mixture_clique(self) -> None:
        xy, xyk, xk = self._dom_embs()
        assert xy.members == (0,)
        assert xyk.members == (0, 1)
        assert xk.members == (1,)

    def test_block_dims_are_the_selected_products(self) -> None:
        """obs 4, lat 2, 3 categories: x-location 4, y-location 2, k 2."""
        xy, xyk, xk = self._blocks(obs_dim=4, lat_dim=2, n_categories=3)
        assert xy.dim == 4 * 2  # x_loc (x) y_loc
        assert xyk.dim == 4 * 2 * 2  # x_loc (x) y_loc (x) k
        assert xk.dim == 8 * 2  # full observable (x) k

    def test_the_three_way_block_selects_from_the_joint_yk_block(self) -> None:
        """The point of the whole exercise: it reads a joint block, not two marginals."""
        mfa = self._mfa()
        mix = mfa.pst_man
        emb = self._dom_embs()[1]

        coords = jax.random.normal(jax.random.PRNGKey(30), (mix.dim,))
        _, m_yk, _ = mix.split_level(coords)
        y_sel = mfa.bas_hrm.int_man.dom_emb
        expected = jax.vmap(y_sel.project, in_axes=1, out_axes=1)(
            mix.int_man.to_matrix(m_yk)
        )
        assert jnp.allclose(emb.project(coords), expected.ravel())

    def test_embedding_lands_only_in_that_block(self) -> None:
        """A coupling writes to its own clique and nowhere else."""
        mfa = self._mfa()
        mix = mfa.pst_man
        for idx, emb in enumerate(self._dom_embs()):
            v = jnp.arange(1.0, emb.sub_man.dim + 1)
            spans = mix.split_level(emb.embed(v))
            touched = [i for i, s in enumerate(spans) if jnp.any(s != 0.0)]
            assert touched == [{0: 0, 1: 1, 2: 2}[idx]], (
                f"block {idx} touched {touched}"
            )

    def test_project_after_embed_round_trips(self) -> None:
        for emb in self._dom_embs():
            v = jax.random.normal(jax.random.PRNGKey(31), (emb.sub_man.dim,))
            assert jnp.allclose(emb.project(emb.embed(v)), v)

    def test_a_coupling_to_absent_members_is_rejected(self) -> None:
        """The structural condition: there must be a block holding the joint statistic."""
        mfa = self._mfa()
        mix = mfa.pst_man
        with pytest.raises(ValueError, match="no clique on"):
            CliqueBlockEmbedding(
                mix, (0, 1, 2), (IdentityEmbedding(mix.obs_man),) * 3
            ).project(jnp.zeros(mix.dim))
