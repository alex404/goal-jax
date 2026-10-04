"""Tests for ``CliqueMap`` and the clique-indexed layouts in geometry/manifold/clique.py.

A clique manifold stores coordinates as three partitions, ``[root | cross | deep]``. The tests pin that layout against what ``analytic_hmog`` already
produces, so any disagreement is a real difference and not
a change of convention.

The last four classes test a clique's *form algebra* rather than its scope. The
decisive ones are at arity 2: they pin the contraction against the ``MatrixMap``
machinery every interaction in the library already runs on, so the arity-$n$
generalization is verified against working code rather than against a fresh derivation.
The arity-3 tests then check the two properties that make higher arity usable --- that
contraction order does not matter, and that partial contraction composes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueEmbedding,
    CliqueMap,
    Diagonal,
    ExponentialFamily,
    ExponentialFamilyPair,
    IdentityEmbedding,
    InteractionEmbedding,
    LinearEmbedding,
    Manifold,
    MatrixMap,
    MatrixRep,
    ObservableEmbedding,
    PositiveDefinite,
    PosteriorEmbedding,
    Rectangular,
    RecursiveLinearCliques,
)
from goal.models import (
    CanonicalCorrelationAnalysis,
    CompleteMixture,
    Euclidean,
    MixtureOfFactorAnalyzers,
    Normal,
    Poissons,
    analytic_hmog,
    factor_analysis,
)

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
    def crs_cliques(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return self._source.crs_cliques

    @override
    def crs_rep(self, crossing: tuple[tuple[int, ...], tuple[int, ...]]) -> MatrixRep:
        return self._source.crs_rep(crossing)

    @override
    def crs_emb_constructors(
        self, crossing: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> tuple[
        tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...],
        tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...],
    ]:
        return self._source.crs_emb_constructors(crossing)

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
        assert model.crs_cliques == (((0,), (0,)),)
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
        assert tuple((cod, dom) for cod, dom, _ in model.int_man.terms) == (
            ((0,), (0,)),
        )
        assert CliqueEmbedding((0, 1), model).sub_man.dim == model.int_man.dim

    def test_the_reading_is_derived_from_the_graph(self) -> None:
        """A crossing's parts fix its order, arity and reading.

        A model states which clique of each partition a coupling touches and what it uses
        at each node. The output group is the root part, the contracted group the deep
        part, and the arity is how many nodes the two hold.
        """
        model, _ = _hmog_partitions()
        form = CliqueEmbedding((0, 1), model).sub_man
        assert len(form.cod_embs) == 1, "the output group is the root node"
        assert len(form.dom_embs) == 1
        assert form.cod_man.mans == (model.obs_man,)
        assert form.dom_man.mans == (model.lwr_hrm.pst_man,)

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

    def test_the_arity_three_clique_has_three_axes(self) -> None:
        """MFA's $(x,y,k)$ clique is a three-way interaction, so it has three axes.

        The domain sub-statistic is the joint $(y,k)$ form, which contributes one axis per
        node rather than one for the pair --- which is what makes the axis count the arity.
        """
        mfa = _mfa()
        form = CliqueEmbedding((0, 1, 2), mfa).sub_man
        assert len(form.embs) == 3
        assert tuple(emb.sub_man.dim for emb in form.embs) == (4, 2, 2)


### Ordering Regressions ###


@dataclass(frozen=True)
class _ReversedCCA(
    CanonicalCorrelationAnalysis[PositiveDefinite, PositiveDefinite, PositiveDefinite]
):
    """CCA declaring its two branches in the other order."""

    @property
    @override
    def crs_cliques(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return (((1,), (0,)), ((0,), (0,)))


@dataclass(frozen=True)
class _DerivedPartitions(RecursiveLinearCliques[ExponentialFamily, ExponentialFamily]):
    """Two explicit partitions and the crossing cliques between them.

    The interaction is derived as a harmonium derives it.
    """

    _rot_man: ExponentialFamily
    _dep_man: ExponentialFamily
    _crs_cliques: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]

    @property
    @override
    def crs_cliques(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return self._crs_cliques

    @override
    def crs_rep(self, crossing: tuple[tuple[int, ...], tuple[int, ...]]) -> MatrixRep:
        return Rectangular()

    @override
    def crs_emb_constructors(
        self, crossing: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> tuple[
        tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...],
        tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...],
    ]:
        near, far = crossing
        return (IdentityEmbedding,) * len(near), (IdentityEmbedding,) * len(far)

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
        assert pair.clq_maps == (
            pair.fst_man.clq_maps[0],
            pair.snd_man.clq_maps[0],
        )

    def test_a_multi_clique_component_keeps_its_cliques(self) -> None:
        cca = _cca()
        pair = _Pair(cca, Normal(2, PositiveDefinite()))
        n = cca.n_nodes
        assert pair.cliques == (*cca.cliques, (n,))
        assert pair.clq_maps == (*cca.clq_maps, *pair.snd_man.clq_maps)
        assert sum(pair.clq_dims) == pair.dim
        params = jnp.arange(float(pair.dim))
        fst, snd = pair.split_coords(params)
        for clique, emb in zip(cca.cliques, cca.clq_embs, strict=True):
            assert jnp.array_equal(
                CliqueEmbedding(clique, pair).project(params), emb.project(fst)
            )
        assert jnp.array_equal(CliqueEmbedding((n,), pair).project(params), snd)


### Form algebra ###


def _key(seed: int) -> Array:
    return jax.random.PRNGKey(seed)


def _product(*parts: Array) -> Array:
    """The flat outer product of one vector per contracted axis.

    What a ``CliqueMap`` reads as its input joint when every contracted axis is given
    separately --- exact for observed axes, and *not* what a dependent joint looks like,
    which is the distinction ``tests/interaction.py`` turns on.
    """
    out = jnp.ones(1)
    for part in parts:
        out = jnp.tensordot(out, part, axes=0)
    return out.reshape(-1)


def _axes(cod_dims: tuple[int, ...], dom_dims: tuple[int, ...]) -> CliqueMap:
    """A form with the given output and input axis sizes, over ``Euclidean`` nodes."""
    return CliqueMap(
        Rectangular(),
        tuple(IdentityEmbedding(Euclidean(d)) for d in cod_dims),
        tuple(IdentityEmbedding(Euclidean(d)) for d in dom_dims),
    )


class TestArityTwoMatchesMatrixMap:
    """At arity 2 a clique's form must reproduce the existing matrix machinery."""

    @staticmethod
    def _pair(cod_dim: int, dom_dim: int):
        emb_map = MatrixMap(Rectangular(), Euclidean(cod_dim), Euclidean(dom_dim))
        return emb_map, _axes((cod_dim,), (dom_dim,))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6), (6, 1)])
    def test_dim_matches(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, form = self._pair(cod_dim, dom_dim)
        assert form.dim == emb_map.dim

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6)])
    def test_outer_product_matches(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, form = self._pair(cod_dim, dom_dim)
        w = jax.random.normal(_key(0), (cod_dim,))
        v = jax.random.normal(_key(1), (dom_dim,))
        assert jnp.array_equal(form.outer_product(w, v), emb_map.outer_product(w, v))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (6, 1)])
    def test_application_matches(self, cod_dim: int, dom_dim: int) -> None:
        """Applying the map is matrix-vector multiplication."""
        emb_map, form = self._pair(cod_dim, dom_dim)
        params = jax.random.normal(_key(2), (form.dim,))
        v = jax.random.normal(_key(3), (dom_dim,))
        assert jnp.allclose(form(params, v), emb_map(params, v))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6)])
    def test_transpose_matches(self, cod_dim: int, dom_dim: int) -> None:
        """The transposed reading is the two embedding groups swapped."""
        emb_map, form = self._pair(cod_dim, dom_dim)
        params = jax.random.normal(_key(4), (form.dim,))
        w = jax.random.normal(_key(5), (cod_dim,))
        trn = form.trn_man
        assert trn.cod_embs == form.dom_embs
        assert trn.matrix_shape == form.matrix_shape[::-1]
        assert jnp.allclose(
            trn(form.transpose(params), w),
            emb_map.transpose_apply(params, w),
        )
        assert jnp.allclose(
            form.transpose_apply(params, w),
            emb_map.transpose_apply(params, w),
        )

    def test_storage_order_is_row_major_over_codomain_then_domain(self) -> None:
        """The layout every ``int_man`` in the library already stores in."""
        emb_map, form = self._pair(2, 3)
        params = jnp.arange(6.0)
        assert jnp.array_equal(form.to_matrix(params), emb_map.to_matrix(params))


class TestHigherArity:
    """Properties that make arity 3 usable, which arity 2 cannot distinguish."""

    form: CliqueMap = _axes((2,), (3, 4))

    def test_dim_is_the_product(self) -> None:
        assert self.form.dim == 24
        assert len(self.form.embs) == 3

    def test_every_reading_of_one_tensor_contracts_correctly(self) -> None:
        """Contracting a rank-one tensor against its own axes rescales the kept one.

        A reading is a clique with the kept axis as its output group, and its tensor is in
        its own (out, in) axis order, so each reading takes the base tensor permuted
        accordingly.
        """
        u = jax.random.normal(_key(6), (2,))
        v = jax.random.normal(_key(7), (3,))
        w = jax.random.normal(_key(8), (4,))
        tensor = u[:, None, None] * v[None, :, None] * w[None, None, :]
        cases = (
            (_axes((2,), (3, 4)), (0, 1, 2), (v, w), u * (v @ v) * (w @ w)),
            (_axes((3,), (2, 4)), (1, 0, 2), (u, w), v * (u @ u) * (w @ w)),
            (_axes((4,), (2, 3)), (2, 0, 1), (u, v), w * (u @ u) * (v @ v)),
        )
        for view, perm, inputs, want in cases:
            params = jnp.transpose(tensor, perm).reshape(-1)
            assert jnp.allclose(view(params, _product(*inputs)), want)

    def test_partial_contraction_composes(self) -> None:
        """Contracting axes one at a time equals contracting them together.

        This is what lets the partition split contract the root axes and leave the deep ones:
        the result must not depend on the order the root axes are taken in.
        """
        params = jax.random.normal(_key(9), (self.form.dim,))
        u = jax.random.normal(_key(10), (2,))
        v = jax.random.normal(_key(11), (3,))
        keep_last = _axes((4,), (2, 3))
        both = keep_last(
            jnp.transpose(params.reshape(2, 3, 4), (2, 0, 1)).reshape(-1),
            _product(u, v),
        )

        # axis 0 first, leaving a (3, 4) tensor read with its last axis kept
        step = _axes((4,), (3,))
        after_u = jnp.tensordot(params.reshape(2, 3, 4), u, axes=([0], [0]))
        stepwise = step(after_u.T.reshape(-1), v)
        assert jnp.allclose(both, stepwise)

    def test_the_input_joint_spans_both_contracted_axes(self) -> None:
        """The contracted group is one joint space, not two separate arguments."""
        assert self.form.dom_man.dim == 12


class TestArityOne:
    """A bias is a form with an empty input group; nothing should special-case it."""

    def test_outer_product_and_application_are_identity(self) -> None:
        form = _axes((5,), ())
        v = jax.random.normal(_key(13), (5,))
        assert form.dim == 5
        assert form.matrix_shape == (5, 1)
        assert form.dom_embs == ()
        # The empty input group's joint is the constant 1.
        assert jnp.array_equal(form.outer_product(v, jnp.ones(1)), v)
        assert jnp.array_equal(form(v, jnp.ones(1)), v)


class TestTranspose:
    """Direction is the (out, in) split of the embeddings, so the transpose is a swap."""

    def test_trn_man_is_an_involution(self) -> None:
        form = _axes((2,), (3, 4))
        assert form.trn_man.trn_man == form
        assert form.trn_man.cod_embs == form.dom_embs
        assert form.trn_man.dom_embs == form.cod_embs

    def test_transpose_round_trips_parameters(self) -> None:
        form = _axes((2,), (3, 4))
        params = jax.random.normal(_key(14), (form.dim,))
        assert jnp.allclose(form.trn_man.transpose(form.transpose(params)), params)

    def test_a_bias_transposes_to_a_functional(self) -> None:
        """The old design raised here; the two-group shape handles it for free."""
        form = _axes((5,), ())
        trn = form.trn_man
        assert trn.matrix_shape == (1, 5)
        v = jax.random.normal(_key(15), (5,))
        assert jnp.allclose(trn(v, v), jnp.dot(v, v)[None])
