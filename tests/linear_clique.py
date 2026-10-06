"""Tests for ``CliqueMap``, ``SubMapEmbedding`` and the clique-indexed layouts in geometry/manifold/clique.py.

A clique manifold stores coordinates as three partitions, ``[root | cross | deep]``. The
tests pin that layout against what ``analytic_hmog`` already produces, so any disagreement
is a real difference and not a change of convention. Every shipped model's crossings are
checked to lie in the blocks they touch.

The last classes test the clique map against the ``MatrixMap`` machinery, and the
sub-block embedding of a map-valued block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueEmbedding,
    CliqueMap,
    CrossTerm,
    Diagonal,
    ExponentialFamily,
    ExponentialFamilyPair,
    IdentityEmbedding,
    InteractionEmbedding,
    MatrixMap,
    ObservableEmbedding,
    PositiveDefinite,
    PosteriorEmbedding,
    Rectangular,
    RecursiveLinearCliques,
    SubMapEmbedding,
)
from goal.models import (
    BoltzmannLGM,
    BoltzmannNormalHarmonium,
    CanonicalCorrelationAnalysis,
    CompleteMixture,
    DiagonalBoltzmann,
    Euclidean,
    MixtureOfFactorAnalyzers,
    Normal,
    Poissons,
    PoissonVonMisesHarmonium,
    analytic_hmog,
    factor_analysis,
    poisson_mixture,
)
from goal.models.harmonium.lgm import GeneralizedGaussianLocationEmbedding

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class _Partitions(RecursiveLinearCliques[ExponentialFamily, ExponentialFamily]):
    """A clique manifold assembled from three explicit partition manifolds.

    The crossing cliques and their maps are those of ``_source``.
    """

    _rot_man: ExponentialFamily
    _dep_man: ExponentialFamily
    _source: RecursiveLinearCliques[Any, Any]

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        return self._source.crs_trms

    @property
    @override
    def rot_man(self) -> ExponentialFamily:
        return self._rot_man

    @property
    @override
    def dep_man(self) -> ExponentialFamily:
        return self._dep_man


def _hmog_partitions():
    """The analytic HMoG alongside a bare clique manifold with the same three partitions."""
    model = analytic_hmog(obs_dim=8, obs_rep=Diagonal(), lat_dim=3, n_components=4)
    partitions = _Partitions(model.obs_man, model.pst_man, model)
    return model, partitions


class TestPartitionLayout:
    """The partition layout reproduces the harmonium's byte layout."""

    def test_span_dims_match_the_model(self) -> None:
        model, partitions = _hmog_partitions()
        assert partitions.rot_man.dim == model.obs_man.dim
        assert partitions.crs_man.dim == model.int_man.dim
        assert partitions.dep_man.dim == model.pst_man.dim

    def test_total_dim_matches_the_model(self) -> None:
        model, partitions = _hmog_partitions()
        assert partitions.dim == model.dim

    def test_deep_span_is_laid_out_as_the_upper_harmonium(self) -> None:
        # The deep partition is byte-identical to what the mixture one level up produces,
        # which is what lets split_coords be applied again to it.
        model, partitions = _hmog_partitions()
        coords = jnp.arange(float(partitions.dim))
        deep = partitions.split_coords(coords)[2]
        assert deep.shape[0] == model.upr_hrm.dim
        y, yk, k = model.upr_hrm.split_coords(deep)
        assert jnp.array_equal(jnp.concatenate([y, yk, k]), deep)
        assert y.shape[0] == model.upr_hrm.obs_man.dim

    def test_split_join_round_trip(self) -> None:
        _, partitions = _hmog_partitions()
        coords = jnp.arange(float(partitions.dim))
        assert jnp.array_equal(
            partitions.join_coords(*partitions.split_coords(coords)), coords
        )


class TestHarmoniumSpans:
    """A harmonium's three partitions are its observable, interaction, and latent sides."""

    def test_hmog_declares_the_three_node_chain(self) -> None:
        model, _ = _hmog_partitions()
        assert model.crs_clqs == (((0,), (0,)),)
        assert model.cliques == ((0,), (0, 1), (1,), (1, 2), (2,))

    def test_spans_are_obs_int_pst(self) -> None:
        """The three partitions are the observable, the interaction, and the posterior.

        The cross partition *is* the interaction --- no wrapper. Its terms are the declared
        crossing cliques, each part in its own partition's numbering.
        """
        model, _ = _hmog_partitions()
        assert model.rot_man == model.obs_man
        assert model.crs_man == model.int_man
        assert model.dep_man == model.pst_man
        assert tuple((trm.cod_clq, trm.dom_clq) for trm in model.int_man.trms) == (
            ((0,), (0,)),
        )
        assert CliqueEmbedding((0, 1), model).sub_man.dim == model.int_man.dim

    def test_a_crossing_maps_between_the_blocks_it_touches(self) -> None:
        """The block of a crossing reads the deep block and writes the root block."""
        model, _ = _hmog_partitions()
        clq_map = CliqueEmbedding((0, 1), model).sub_man
        assert clq_map.cod_man == model.obs_man
        assert clq_map.dom_man == model.lwr_hrm.pst_man

    @pytest.mark.parametrize(
        ("emb_cls", "idx"),
        [(ObservableEmbedding, 0), (InteractionEmbedding, 1), (PosteriorEmbedding, 2)],
    )
    def test_slot_embedding_isolates_its_span(self, emb_cls, idx: int) -> None:
        model, _ = _hmog_partitions()
        emb = emb_cls(model)
        v = jnp.arange(1.0, emb.sub_man.dim + 1)
        assert jnp.array_equal(emb.project(emb.embed(v)), v)
        partitions = model.split_coords(emb.embed(v))
        for other in range(3):
            if other != idx:
                assert jnp.all(partitions[other] == 0.0)


def test_layout_is_jit_static() -> None:
    """Clique manifolds must hash, since models are passed as static jit arguments."""
    _, partitions = _hmog_partitions()

    @jax.jit
    def total(coords: Array, man: _Partitions = partitions) -> Array:
        return jnp.sum(man.split_coords(coords)[0])

    coords = jnp.arange(float(partitions.dim))
    assert jnp.allclose(total(coords), jnp.sum(coords[: partitions.rot_man.dim]))


class TestCliqueLocations:
    """Per-clique coordinates."""

    def test_clique_dims_sum_to_dim(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        assert sum(model.clq_dims) == model.dim

    def test_one_form_per_clique(self) -> None:
        """The composed graph and the parameter layout must agree clique for clique."""
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        assert len(model.clq_dims) == len(model.cliques) == 5

    def test_an_embedding_of_an_absent_clique_is_rejected(self) -> None:
        man = CompleteMixture(Poissons(2), 3)
        with pytest.raises(ValueError, match="not in tuple"):
            CliqueEmbedding((0, 5), man).project(man.zeros())


### Layout Invariants ###


def layout_problems(man: RecursiveLinearCliques[Any, Any]) -> list[str]:
    """Every way a clique manifold's composed graph and its parameter layout can disagree.

    The cliques must tile the coordinate vector, in the partitions' own block sizes, and
    each clique's block must be its partition's block for the same clique in that
    partition's numbering. None of this is checked at construction.
    """
    if sum(man.clq_dims) != man.dim:
        return [f"cliques sum to {sum(man.clq_dims)}, but dim is {man.dim}"]
    parts = man.rot_man.clq_dims + man.crs_man.clq_dims + man.dep_man.clq_dims
    if man.clq_dims != parts:
        return [f"block sizes {man.clq_dims} are not the partitions' {parts}"]
    coords = jnp.arange(float(man.dim))
    root, _, deep = man.split_coords(coords)
    n_rot = man.rot_man.n_nodes
    for clique in man.rot_man.cliques:
        if not jnp.array_equal(
            CliqueEmbedding(clique, man).project(coords),
            CliqueEmbedding(clique, man.rot_man).project(root),
        ):
            return [f"root clique {clique} is misplaced"]
    for clique in man.dep_man.cliques:
        lifted = tuple(i + n_rot for i in clique)
        if not jnp.array_equal(
            CliqueEmbedding(lifted, man).project(coords),
            CliqueEmbedding(clique, man.dep_man).project(deep),
        ):
            return [f"deep clique {clique} is misplaced as {lifted}"]
    return []


def shipped_models() -> list[tuple[str, RecursiveLinearCliques[Any, Any]]]:
    """One instance of every model shape the library ships a graph for."""
    return [
        ("factor_analysis", factor_analysis(obs_dim=4, lat_dim=2)),
        (
            "hmog",
            analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4),
        ),
        ("cca", _cca()),
        ("mfa", _mfa()),
    ]


def _cca() -> CanonicalCorrelationAnalysis[
    PositiveDefinite, PositiveDefinite, PositiveDefinite
]:
    """Asymmetric branches, so a swapped layout is visible in the dimensions."""
    return CanonicalCorrelationAnalysis(
        fst_dim=3,
        fst_rep=PositiveDefinite(),
        snd_dim=2,
        snd_rep=PositiveDefinite(),
        lat_dim=2,
        pst_rep=PositiveDefinite(),
    )


def _mfa() -> MixtureOfFactorAnalyzers:
    return MixtureOfFactorAnalyzers(
        n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
    )


class TestLayoutInvariants:
    """Every shipped model's graph agrees with its parameter layout."""

    @pytest.mark.parametrize("name", [n for n, _ in shipped_models()])
    def test_layout_agrees_with_graph(self, name: str) -> None:
        assert layout_problems(dict(shipped_models())[name]) == []

    @pytest.mark.parametrize("name", [n for n, _ in shipped_models()])
    def test_the_level_split_agrees_with_the_span_dimensions(self, name: str) -> None:
        """The two readings of where a level ends must coincide.

        ``split_coords`` slices at the summed sizes of the root and cross *forms*, while
        the partitions report their own dimensions. Both descriptions exist; the layout is only
        coherent if they agree, and only the form reading is what the split actually uses.
        """
        man = dict(shipped_models())[name]
        coords = jnp.arange(float(man.dim))
        root, cross, deep = man.split_coords(coords)
        partitions = (man.rot_man.dim, man.crs_man.dim, man.dep_man.dim)
        assert (root.size, cross.size, deep.size) == partitions
        assert jnp.array_equal(man.join_coords(root, cross, deep), coords)

    def test_the_three_node_crossing_reads_a_sub_block(self) -> None:
        """MFA's $(x, y, k)$ block maps from a sub-block of the mixture's $(y, k)$ block.

        The sub-block keeps the location of $y$ and all of $k$: two by two, read against
        the four locations of $x$.
        """
        mfa = _mfa()
        clq_map = CliqueEmbedding((0, 1, 2), mfa).sub_man
        assert clq_map.dom_man == mfa.pst_man.clq_man((0, 1))
        assert isinstance(clq_map.dom_emb, SubMapEmbedding)
        assert clq_map.matrix_shape == (4, 4)


### Crossings and blocks ###


def crossing_models() -> list[tuple[str, RecursiveLinearCliques[Any, Any]]]:
    """Every shipped model shape with crossings, including MFA's mixture view."""
    return [
        *shipped_models(),
        ("mixture", poisson_mixture(n_neurons=4, n_components=3)),
        ("mfa_mixture_view", _mfa().mix_man),
        ("boltzmann_lgm", BoltzmannLGM(3, PositiveDefinite(), 2)),
        ("boltzmann_normal", BoltzmannNormalHarmonium(DiagonalBoltzmann(3), 2)),
        ("poisson_von_mises", PoissonVonMisesHarmonium(5, 1)),
    ]


class TestCrossingsLieInTheirBlocks:
    """A crossing maps from a subspace of the deep block it touches to one of the root block."""

    @pytest.mark.parametrize("name", [n for n, _ in crossing_models()])
    def test_each_crossing_reads_and_writes_its_blocks(self, name: str) -> None:
        man = dict(crossing_models())[name]
        for trm in man.crs_trms:
            assert trm.clq_map.cod_man == man.rot_man.clq_man(trm.cod_clq), trm
            assert trm.clq_map.dom_man == man.dep_man.clq_man(trm.dom_clq), trm

    @pytest.mark.parametrize("name", [n for n, _ in crossing_models()])
    def test_a_single_node_block_is_the_node_space(self, name: str) -> None:
        man = dict(crossing_models())[name]
        assert len(man.nod_mans) == man.n_nodes
        for clique, clq_man in zip(man.cliques, man.clq_mans):
            if len(clique) == 1:
                assert clq_man == man.nod_mans[clique[0]]


### Ordering Regressions ###


@dataclass(frozen=True)
class _ReversedCCA(
    CanonicalCorrelationAnalysis[PositiveDefinite, PositiveDefinite, PositiveDefinite]
):
    """CCA declaring its two branches in the other order."""

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        return tuple(reversed(super().crs_trms))


@dataclass(frozen=True)
class _DerivedPartitions(RecursiveLinearCliques[ExponentialFamily, ExponentialFamily]):
    """Two explicit partitions and the crossing cliques between them.

    The interaction is derived as a harmonium derives it.
    """

    _rot_man: ExponentialFamily
    _dep_man: ExponentialFamily
    _crs_clqs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """Each crossing over the whole of both blocks it touches."""
        return tuple(
            CrossTerm(
                near,
                far,
                CliqueMap(
                    Rectangular(),
                    IdentityEmbedding(self._rot_man.clq_man(near)),
                    IdentityEmbedding(self._dep_man.clq_man(far)),
                ),
            )
            for near, far in self._crs_clqs
        )

    @property
    @override
    def rot_man(self) -> ExponentialFamily:
        return self._rot_man

    @property
    @override
    def dep_man(self) -> ExponentialFamily:
        return self._dep_man


def _past_the_root() -> _DerivedPartitions:
    """A crossing clique that couples the observable to the mixture's latent $k$ only.

    The mixture's own root is $y$, but nothing requires a crossing to touch the deep
    partition's root: the crossing names the mixture's clique $(k)$ in the mixture's
    numbering, and the mixture is stored in its own order.
    """
    obs = Normal(3, Diagonal())
    mix = CompleteMixture(Normal(2, Diagonal()), 4)
    return _DerivedPartitions(obs, mix, (((0,), (1,)),))


class TestDeclarationOrderRegressions:
    """A clique's layout slot must hold that clique's parameters."""

    def test_declaration_order_is_storage_order(self) -> None:
        """Crossing cliques are stored in the order they are declared."""
        model = _ReversedCCA(
            fst_dim=3,
            fst_rep=PositiveDefinite(),
            snd_dim=2,
            snd_rep=PositiveDefinite(),
            lat_dim=2,
            pst_rep=PositiveDefinite(),
        )
        assert model.cliques == ((0,), (1,), (1, 2), (0, 2), (2,))
        assert layout_problems(model) == []
        params = jnp.arange(float(model.dim))
        blocks = model.crs_man.coord_blocks(model.split_coords(params)[1])
        assert jnp.array_equal(
            CliqueEmbedding((1, 2), model).project(params), blocks[0]
        )
        assert jnp.array_equal(
            CliqueEmbedding((0, 2), model).project(params), blocks[1]
        )

    def test_a_crossing_past_the_deep_root_composes(self) -> None:
        """The deep partition is stored in its own order whatever the crossings touch."""
        man = _past_the_root()
        assert man.cliques == ((0,), (0, 2), (1,), (1, 2), (2,))
        assert layout_problems(man) == []


@dataclass(frozen=True)
class _Pair(ExponentialFamilyPair[Any, Any]):
    """A pair of two given families, for testing how a pair composes its components."""

    _fst: ExponentialFamily
    _snd: ExponentialFamily

    @property
    @override
    def fst_man(self) -> Any:
        return self._fst

    @property
    @override
    def snd_man(self) -> Any:
        return self._snd


class TestPairComposition:
    """A pair places its components' graphs side by side and keeps their cliques."""

    def test_single_node_components_make_two_nodes(self) -> None:
        pair = _Pair(Normal(2, PositiveDefinite()), Normal(3, Diagonal()))
        assert pair.cliques == ((0,), (1,))
        assert pair.clq_mans == (pair.fst_man, pair.snd_man)
        assert pair.nod_mans == (pair.fst_man, pair.snd_man)

    def test_a_multi_clique_component_keeps_its_cliques(self) -> None:
        cca = _cca()
        pair = _Pair(cca, Normal(2, PositiveDefinite()))
        n = cca.n_nodes
        assert pair.cliques == (*cca.cliques, (n,))
        assert pair.clq_mans == (*cca.clq_mans, pair.snd_man)
        assert sum(pair.clq_dims) == pair.dim
        params = jnp.arange(float(pair.dim))
        fst, snd = pair.split_coords(params)
        for clique in cca.cliques:
            assert jnp.array_equal(
                CliqueEmbedding(clique, pair).project(params),
                CliqueEmbedding(clique, cca).project(fst),
            )
        assert jnp.array_equal(CliqueEmbedding((n,), pair).project(params), snd)


### Clique maps ###


def _key(seed: int) -> Array:
    return jax.random.PRNGKey(seed)


def _clq_map(cod_dim: int, dom_dim: int) -> CliqueMap:
    """A clique map over the whole of two ``Euclidean`` blocks."""
    return CliqueMap(
        Rectangular(),
        IdentityEmbedding(Euclidean(cod_dim)),
        IdentityEmbedding(Euclidean(dom_dim)),
    )


class TestCliqueMapMatchesMatrixMap:
    """A clique map over whole blocks reproduces the matrix machinery."""

    @staticmethod
    def _pair(cod_dim: int, dom_dim: int):
        emb_map = MatrixMap(Rectangular(), Euclidean(cod_dim), Euclidean(dom_dim))
        return emb_map, _clq_map(cod_dim, dom_dim)

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6), (6, 1)])
    def test_dim_matches(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, clq_map = self._pair(cod_dim, dom_dim)
        assert clq_map.dim == emb_map.dim

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6)])
    def test_outer_product_matches(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, clq_map = self._pair(cod_dim, dom_dim)
        w = jax.random.normal(_key(0), (cod_dim,))
        v = jax.random.normal(_key(1), (dom_dim,))
        assert jnp.array_equal(clq_map.outer_product(w, v), emb_map.outer_product(w, v))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (6, 1)])
    def test_application_matches(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, clq_map = self._pair(cod_dim, dom_dim)
        params = jax.random.normal(_key(2), (clq_map.dim,))
        v = jax.random.normal(_key(3), (dom_dim,))
        assert jnp.allclose(clq_map(params, v), emb_map(params, v))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6)])
    def test_transpose_matches(self, cod_dim: int, dom_dim: int) -> None:
        """The transpose swaps the two embeddings."""
        emb_map, clq_map = self._pair(cod_dim, dom_dim)
        params = jax.random.normal(_key(4), (clq_map.dim,))
        w = jax.random.normal(_key(5), (cod_dim,))
        trn = clq_map.trn_man
        assert trn.cod_emb == clq_map.dom_emb
        assert trn.matrix_shape == clq_map.matrix_shape[::-1]
        assert jnp.allclose(
            trn(clq_map.transpose(params), w), emb_map.transpose_apply(params, w)
        )

    def test_storage_order_is_row_major_over_codomain_then_domain(self) -> None:
        emb_map, clq_map = self._pair(2, 3)
        params = jnp.arange(6.0)
        assert jnp.array_equal(clq_map.to_matrix(params), emb_map.to_matrix(params))

    def test_trn_man_is_an_involution(self) -> None:
        clq_map = _clq_map(2, 3)
        params = jax.random.normal(_key(6), (clq_map.dim,))
        assert clq_map.trn_man.trn_man == clq_map
        assert jnp.allclose(
            clq_map.trn_man.transpose(clq_map.transpose(params)), params
        )


class TestSubMapEmbedding:
    """A sub-block of a map-valued block: the location rows of a Gaussian, all columns."""

    @staticmethod
    def _emb() -> SubMapEmbedding:
        nor = Normal(2, PositiveDefinite())
        amb = CliqueMap(
            Rectangular(), IdentityEmbedding(nor), IdentityEmbedding(Euclidean(3))
        )
        return SubMapEmbedding(
            amb,
            GeneralizedGaussianLocationEmbedding(nor),
            IdentityEmbedding(Euclidean(3)),
        )

    def test_the_sub_block_is_a_map_on_nested_subspaces(self) -> None:
        emb = self._emb()
        assert emb.sub_man.cod_man == emb.amb_man.cod_man
        assert emb.sub_man.dom_man == emb.amb_man.dom_man
        assert emb.sub_man.matrix_shape == (2, 3)

    def test_project_inverts_embed(self) -> None:
        emb = self._emb()
        sub = jax.random.normal(_key(7), (emb.sub_man.dim,))
        assert jnp.allclose(emb.project(emb.embed(sub)), sub)

    def test_project_is_the_transpose_of_embed(self) -> None:
        emb = self._emb()
        sub = jax.random.normal(_key(8), (emb.sub_man.dim,))
        amb = jax.random.normal(_key(9), (emb.amb_man.dim,))
        assert jnp.allclose(
            jnp.dot(emb.embed(sub), amb), jnp.dot(sub, emb.project(amb))
        )

    def test_the_sub_block_acts_as_its_embedding(self) -> None:
        """Applying the sub-block equals applying the ambient block at its embedding."""
        emb = self._emb()
        sub = jax.random.normal(_key(10), (emb.sub_man.dim,))
        v = jax.random.normal(_key(11), (3,))
        assert jnp.allclose(emb.sub_man(sub, v), emb.amb_man(emb.embed(sub), v))
