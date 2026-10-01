"""Tests for ``CliqueMap`` and the clique-indexed layouts in geometry/manifold/clique.py.

A clique manifold stores coordinates as the three partitions of one level ascent,
``[root | cross | deep]``. The tests pin that layout against what ``analytic_hmog`` and
``differentiable_hmog`` already produce, so any disagreement is a real difference and not
a change of convention. The decisive layout case is the embedding one: a hierarchical
model's posterior-to-prior embedding must transform the root partition and leave the other two
untouched, which is what lets a difference deep in the graph be expressed by nesting.

The last four classes test a clique's *form algebra* rather than its scope. The
decisive ones are at arity 2: they pin the contraction against the ``MatrixMap``
machinery every interaction in the library already runs on, so the arity-$n$
generalization is verified against working code rather than against a fresh derivation.
The arity-3 tests then check the two properties that make higher arity usable --- that
contraction order does not matter, and that partial contraction composes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, override

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueMap,
    Diagonal,
    ExponentialFamily,
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
    RootEmbedding,
    Scale,
)
from goal.models import (
    CanonicalCorrelationAnalysis,
    CompleteMixture,
    Euclidean,
    MixtureOfFactorAnalyzers,
    Normal,
    Poissons,
    analytic_hmog,
    differentiable_hmog,
    factor_analysis,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class _Partitions(RecursiveLinearCliques[ExponentialFamily, ExponentialFamily]):
    """A clique manifold assembled from three explicit partition manifolds.

    The graph and the crossing cliques' maps are those of ``_source``.
    """

    _rot_man: ExponentialFamily
    _dep_man: ExponentialFamily
    _source: RecursiveLinearCliques[Any, Any]

    @property
    @override
    def raw_cliques(self) -> tuple[tuple[int, ...], ...]:
        return self._source.raw_cliques

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return self._source.root_nodes

    @override
    def crs_rep(self, clique: tuple[int, ...]) -> MatrixRep:
        return self._source.crs_rep(clique)

    @override
    def crs_emb_constructors(
        self, clique: tuple[int, ...]
    ) -> tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...]:
        return self._source.crs_emb_constructors(clique)

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
        # which is what lets split_level be applied again to it.
        model, partitions = _hmog_partitions()
        coords = jnp.arange(float(partitions.dim))
        deep = partitions.split_level(coords)[2]
        assert deep.shape[0] == model.upr_hrm.dim
        y, yk, k = model.upr_hrm.split_level(deep)
        assert jnp.array_equal(jnp.concatenate([y, yk, k]), deep)
        assert y.shape[0] == model.upr_hrm.obs_man.dim

    def test_split_join_round_trip(self) -> None:
        _, partitions = _hmog_partitions()
        coords = jnp.arange(float(partitions.dim))
        assert jnp.array_equal(
            partitions.join_coords(*partitions.split_coords(coords)), coords
        )

    def test_split_level_is_split_coords(self) -> None:
        _, partitions = _hmog_partitions()
        coords = jnp.arange(float(partitions.dim))
        for a, b in zip(
            partitions.split_level(coords), partitions.split_coords(coords), strict=True
        ):
            assert jnp.array_equal(a, b)


class TestHarmoniumSpans:
    """A harmonium's three partitions are its observable, interaction, and latent sides."""

    def test_hmog_declares_the_three_node_chain(self) -> None:
        model, _ = _hmog_partitions()
        assert model.root_nodes == frozenset({0})
        assert model.cliques == ((0,), (0, 1), (1,), (1, 2), (2,))
        assert model.level_sets == ((0,), (1,), (2,))

    def test_spans_are_obs_int_pst(self) -> None:
        """The three partitions are the observable, the interaction, and the posterior.

        The cross partition *is* the interaction --- no wrapper. Which nodes its pieces couple
        is read off the model's graph, as the crossing group of its level split.
        """
        model, _ = _hmog_partitions()
        assert model.rot_man == model.obs_man
        assert model.crs_man == model.int_man
        assert model.dep_man == model.pst_man
        assert model.level_split()[1] == ((0, 1),)
        assert model.clq_map((0, 1)).dim == model.int_man.dim

    def test_the_reading_is_derived_from_the_graph(self) -> None:
        """Nothing positional is declared: the node set fixes order, arity and reading.

        A model states which nodes a coupling touches and what it uses at each. The axis
        order is those nodes ascending, the arity is how many there are, and the output
        group is the root nodes among them.
        """
        model, _ = _hmog_partitions()
        form = model.clq_map((0, 1))
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
        partitions = model.split_level(emb.embed(v))
        for other in range(3):
            if other != idx:
                assert jnp.all(partitions[other] == 0.0)


class TestRootEmbedding:
    """Transforms the root partition; the cross and deep partitions pass through untouched."""

    @staticmethod
    def _asymmetric_pair():
        model = differentiable_hmog(
            obs_dim=6, obs_rep=Diagonal(), lat_dim=2, pst_rep=Scale(), n_components=3
        )
        return model, model.pst_prr_emb

    def test_dims_line_up(self) -> None:
        model, emb = self._asymmetric_pair()
        assert emb.sub_man.dim == model.pst_upr_hrm.dim
        assert emb.amb_man.dim == model.prr_upr_hrm.dim

    def test_root_span_is_the_lower_embedding(self) -> None:
        model, emb = self._asymmetric_pair()
        v = jax.random.normal(jax.random.PRNGKey(2), (model.pst_upr_hrm.dim,))
        pst_root, _, _ = model.pst_upr_hrm.split_level(v)
        prr_root, _, _ = model.prr_upr_hrm.split_level(emb.embed(v))
        assert jnp.array_equal(prr_root, model.lwr_hrm.pst_prr_emb.embed(pst_root))

    def test_cross_and_deep_spans_pass_through(self) -> None:
        model, emb = self._asymmetric_pair()
        v = jax.random.normal(jax.random.PRNGKey(5), (model.pst_upr_hrm.dim,))
        _, sub_cross, sub_deep = model.pst_upr_hrm.split_level(v)
        _, amb_cross, amb_deep = model.prr_upr_hrm.split_level(emb.embed(v))
        assert jnp.array_equal(amb_cross, sub_cross)
        assert jnp.array_equal(amb_deep, sub_deep)

        w = jax.random.normal(jax.random.PRNGKey(6), (model.prr_upr_hrm.dim,))
        _, amb_cross, amb_deep = model.prr_upr_hrm.split_level(w)
        _, sub_cross, sub_deep = model.pst_upr_hrm.split_level(emb.project(w))
        assert jnp.array_equal(sub_cross, amb_cross)
        assert jnp.array_equal(sub_deep, amb_deep)

    def test_translate_is_additive_on_the_root_span(self) -> None:
        model, emb = self._asymmetric_pair()
        key_p, key_q = jax.random.split(jax.random.PRNGKey(4))
        p = jax.random.normal(key_p, (model.prr_upr_hrm.dim,))
        q = jax.random.normal(key_q, (model.pst_upr_hrm.dim,))
        assert jnp.allclose(emb.translate(p, q), p + emb.embed(q))

    def test_rejects_mismatched_graphs(self) -> None:
        """The two sides must present the same cover, not merely the same dimensions.

        Since the merge, a layout's graph is read off its forms, so a mismatch cannot be
        faked by declaring one --- it has to come from partitions that genuinely cover
        differently. Here the ambient is the whole three-node model rather than its upper
        harmonium, so its cover has a node the sub's does not.
        """
        model, _ = self._asymmetric_pair()
        pst = model.pst_upr_hrm
        assert not pst.same_graph(model)
        with pytest.raises(ValueError, match="differ only in the root partition"):
            RootEmbedding(model.lwr_hrm.pst_prr_emb, pst, model)


def test_layout_is_jit_static() -> None:
    """Clique manifolds must hash, since models are passed as static jit arguments."""
    _, partitions = _hmog_partitions()

    @jax.jit
    def total(coords: Array, man: _Partitions = partitions) -> Array:
        return jnp.sum(man.split_level(coords)[0])

    coords = jnp.arange(float(partitions.dim))
    assert jnp.allclose(total(coords), jnp.sum(coords[: partitions.rot_man.dim]))


class TestCliqueLocations:
    """Per-clique coordinates."""

    def test_clique_dims_sum_to_dim(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        assert sum(model.clq_dims) == model.dim

    def test_one_form_per_clique(self) -> None:
        """The declared graph and the parameter layout must agree clique for clique."""
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        assert len(model.clq_dims) == sum(len(group) for group in model.level_cliques)

    def test_an_embedding_of_an_absent_clique_is_rejected(self) -> None:
        man = CompleteMixture(Poissons(2), 3)
        with pytest.raises(ValueError, match="not in tuple"):
            man.clq_emb((0, 5))


### Layout Invariants ###


def layout_problems(man: RecursiveLinearCliques[Any, Any]) -> list[str]:
    """Every way a clique manifold's graph and its parameter layout can disagree.

    Blocks are stored in the graph's canonical order, so a recursive deep partition must
    store its own blocks in the order canonical order gives its segment. The cliques must
    also tile the coordinate vector. Neither is checked at construction.
    """
    if sum(man.clq_dims) != man.dim:
        return [f"cliques sum to {sum(man.clq_dims)}, but dim is {man.dim}"]
    dep = man.dep_man
    if isinstance(dep, RecursiveLinearCliques):
        segment = tuple(chain.from_iterable(man.level_cliques[2:]))
        own = dep.cliques
        if segment != own:
            return [f"deep storage {own} is not canonical order {segment}"]
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
        root, cross, deep = man.split_level(coords)
        partitions = (man.rot_man.dim, man.crs_man.dim, man.dep_man.dim)
        assert (root.size, cross.size, deep.size) == partitions
        assert jnp.array_equal(man.join_level(root, cross, deep), coords)

    def test_the_arity_three_clique_has_three_axes(self) -> None:
        """MFA's $(x,y,k)$ clique is a three-way interaction, so it has three axes.

        The domain sub-statistic is the joint $(y,k)$ form, which contributes one axis per
        node rather than one for the pair --- which is what makes the axis count the arity.
        """
        mfa = _mfa()
        form = mfa.clq_emb((0, 1, 2)).sub_man
        assert len(form.embs) == 3
        assert tuple(emb.sub_man.dim for emb in form.embs) == (4, 2, 2)


### Ordering Regressions ###


@dataclass(frozen=True)
class _ReversedCCA(
    CanonicalCorrelationAnalysis[PositiveDefinite, PositiveDefinite, PositiveDefinite]
):
    """CCA declaring its cliques in non-canonical order.

    Nothing forbids this: the two branches are symmetric, and a model author has no reason
    to know that ``RecursiveCliques`` will sort them.
    """

    _raw_cliques: tuple[tuple[int, ...], ...] = field(
        default=((2,), (1, 2), (1,), (0, 2), (0,)), kw_only=True
    )


@dataclass(frozen=True)
class _DerivedPartitions(RecursiveLinearCliques[Manifold, Manifold]):
    """Two explicit partitions and the graph over them, all in full.

    The interaction is derived as a harmonium derives it.
    """

    _rot_man: Manifold
    _dep_man: Manifold
    _raw_cliques: tuple[tuple[int, ...], ...]
    _root_nodes: frozenset[int]

    @property
    @override
    def raw_cliques(self) -> tuple[tuple[int, ...], ...]:
        return self._raw_cliques

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return self._root_nodes

    @override
    def crs_rep(self, clique: tuple[int, ...]) -> MatrixRep:
        return Rectangular()

    @override
    def crs_emb_constructors(
        self, clique: tuple[int, ...]
    ) -> tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...]:
        return (IdentityEmbedding,) * len(clique)

    @property
    @override
    def rot_man(self) -> Manifold:
        return self._rot_man

    @property
    @override
    def dep_man(self) -> Manifold:
        return self._dep_man


def _misrooted() -> _DerivedPartitions:
    """A level whose cross clique reaches past the deep partition's own root.

    The deep partition is a mixture rooted at $y$, but the crossing clique couples the
    observable to $k$. The glued graph therefore reroots at $k$, while the deep partition still
    lays itself out $y$-first --- so $k$ lands at level 1 and $y$ at level 2, and levels
    descend with the node index. Canonical order and the mixture's own storage then disagree
    on the deep segment, so this shape is inconsistent.
    """
    obs = Normal(3, Diagonal())
    mix = CompleteMixture(Normal(2, Diagonal()), 4).impose(
        ((1,), (1, 2), (2,)), frozenset({1})
    )
    return _DerivedPartitions(
        obs, mix, ((0,), (0, 2), (1,), (1, 2), (2,)), frozenset({0})
    )


class TestDeclarationOrderRegressions:
    """A clique's layout slot must hold that clique's parameters."""

    def test_declaration_order_does_not_change_storage(self) -> None:
        """Crossing cliques are stored in canonical order, however they are declared."""
        model = _ReversedCCA(
            fst_dim=3,
            fst_rep=PositiveDefinite(),
            snd_dim=2,
            snd_rep=PositiveDefinite(),
            lat_dim=2,
            pst_rep=PositiveDefinite(),
        )
        assert model.level_cliques == _cca().level_cliques
        params = jnp.arange(float(model.dim))
        blocks = model.int_man.coord_blocks(model.split_level(params)[1])
        found = model.clq_emb((0, 2)).project(params)
        assert jnp.array_equal(found, blocks[0])

    def test_misrooted_deep_span_is_still_a_graph(self) -> None:
        """Labels need not ascend with level, so this is a graph like any other.

        Node 2 is at level 1 and node 1 at level 2. Nothing renumbers between levels ---
        labels are global and the graph reads membership rather than comparing indices ---
        so the level structure comes out as the connectivity dictates rather than as the
        labels suggest.
        """
        man = _misrooted()
        assert man.level_sets == ((0,), (2,), (1,))

    def test_misrooted_deep_span_disagrees_with_canonical_order(self) -> None:
        """Canonical order is the storage order, and the mixture does not follow it here."""
        assert layout_problems(_misrooted()) != []


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
        emb_map = MatrixMap(Rectangular(), Euclidean(dom_dim), Euclidean(cod_dim))
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

        This is what lets a level split contract the root axes and leave the deep ones:
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
